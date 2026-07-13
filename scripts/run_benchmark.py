from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.chains.langchain_rag import build_chat_model
from src.evaluation.answer_writer import (
    append_answer,
    append_graph_trace,
    append_retrieved_docs,
)
from src.evaluation.benchmark_report import matches_source_type
from src.graphs.workflow import build_stage26_rag_graph
from src.ingestion.parse_documents import read_manifest
from src.retrieval.embeddings import build_embeddings
from src.retrieval.entity_links import build_entity_link_index
from src.retrieval.vector_retriever import (
    build_hybrid_candidate_retriever,
    build_parent_document_store,
)
from src.utils.config import load_config
from src.utils.logging import setup_logging


def iter_questions(
    path: str | Path,
    limit: int | None = None,
    source_type: str | None = None,
    question_type: str | None = None,
    include_mixed_sources: bool = False,
):
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            question = json.loads(line)
            if not matches_source_type(
                question,
                source_type,
                include_mixed_sources=include_mixed_sources,
            ):
                continue
            if question_type and question.get("question_type") != question_type:
                continue
            yield question
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def prepare_run_dir(config: dict) -> Path:
    run_dir = Path(config["output"]["runs_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["answers.jsonl", "retrieved_docs.jsonl", "graph_traces.jsonl"]:
        target = run_dir / filename
        if target.exists():
            target.unlink()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive RAG benchmark graph.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-source-type",
        default=None,
        help="Only run questions whose source_types exactly match this value.",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        help="Select a benchmark question type for a focused experiment only.",
    )
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument(
        "--include-mixed-source-questions",
        action="store_true",
        help="Include questions that combine the selected source with other sources.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    run_dir = prepare_run_dir(config)
    default_source_type = config["data"].get("default_source_type", "github")
    question_source_type = (
        None if args.all_questions else args.question_source_type or default_source_type
    )

    embeddings = build_embeddings(config["embedding"])
    llm = build_chat_model(config["llm"])
    manifest_documents = read_manifest(config["data"]["manifest_file"])
    parent_documents = build_parent_document_store(manifest_documents)
    entity_index = build_entity_link_index(
        parent_documents,
        max_document_frequency=int(
            config["retrieval"].get("entity_max_document_frequency", 20)
        ),
    )
    retrieval_config = config["retrieval"]
    logging.info("Building local BM25 index from %d chunks", len(manifest_documents))
    retriever = build_hybrid_candidate_retriever(
        config["qdrant"],
        embeddings,
        manifest_documents,
        dense_k=int(retrieval_config.get("dense_candidate_k", 30)),
        bm25_k=int(retrieval_config.get("bm25_candidate_k", 30)),
        candidate_k=int(retrieval_config.get("hybrid_candidate_k", 40)),
        rrf_k=int(retrieval_config.get("channel_rrf_k", 60)),
        text_section_weight=float(
            retrieval_config.get("text_section_weight", 0.8)
        ),
    )
    logging.info("Local BM25 index is ready")
    logging.info("Entity link index contains %d reusable identifiers", len(entity_index))
    graph = build_stage26_rag_graph(
        retriever,
        llm,
        parent_documents,
        retrieval_config,
        entity_index=entity_index,
    )

    answers_file = run_dir / "answers.jsonl"
    retrieved_file = run_dir / "retrieved_docs.jsonl"
    trace_file = run_dir / "graph_traces.jsonl"
    logging.info("Question source filter: %s", question_source_type or "all")

    for question in iter_questions(
        config["data"]["questions_file"],
        limit=args.limit,
        source_type=question_source_type,
        question_type=args.question_type,
        include_mixed_sources=args.include_mixed_source_questions,
    ):
        question_id = question["question_id"]
        logging.info("Answering %s", question_id)
        state = graph.invoke(
            {"question": question["question"]},
            config={"recursion_limit": int(config["graph"].get("recursion_limit", 20))},
        )
        append_answer(
            answers_file,
            question_id,
            state.get("answer", ""),
            state.get("document_ids", []),
        )
        append_retrieved_docs(
            retrieved_file,
            question_id,
            state.get("retrieved_docs", []),
        )
        append_graph_trace(trace_file, question_id, state)

    logging.info("Wrote answers to %s", answers_file)


if __name__ == "__main__":
    main()
