from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.chains.langchain_rag import answer_with_retriever, build_chat_model
from src.evaluation.answer_writer import append_answer, append_retrieved_docs
from src.graphs.rag_graph import build_simple_rag_graph
from src.ingestion.parse_documents import read_manifest
from src.retrieval.embeddings import build_embeddings
from src.retrieval.vector_retriever import build_parent_document_store, build_retriever
from src.utils.config import load_config
from src.utils.logging import setup_logging


def iter_questions(
    path: str | Path,
    limit: int | None = None,
    source_type: str | None = None,
):
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            question = json.loads(line)
            if source_type and source_type not in question.get("source_types", []):
                continue

            yield question
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def prepare_run_dir(config: dict, config_path: str | None) -> Path:
    run_dir = Path(config["output"]["runs_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["answers.jsonl", "retrieved_docs.jsonl"]:
        target = run_dir / filename
        if target.exists():
            target.unlink()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark answers.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-source-type",
        default=None,
        help="Only run questions whose source_types include this value. Defaults to data.default_source_type.",
    )
    parser.add_argument("--all-questions", action="store_true", help="Disable source_type filtering.")
    parser.add_argument(
        "--mode",
        choices=["chain", "graph"],
        default="chain",
        help="chain uses LangChain directly; graph uses the simple LangGraph workflow.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    run_dir = prepare_run_dir(config, args.config)
    default_source_type = config["data"].get("default_source_type", "github")
    question_source_type = None if args.all_questions else args.question_source_type or default_source_type

    embeddings = build_embeddings(config["embedding"])
    llm = build_chat_model(config["llm"])
    parent_documents = build_parent_document_store(
        read_manifest(config["data"]["manifest_file"])
    )
    retriever = build_retriever(
        config["qdrant"],
        embeddings,
        llm,
        candidate_k=int(config["retrieval"].get("candidate_k", 40)),
        candidate_documents=int(config["retrieval"].get("candidate_documents", 12)),
        max_documents=int(config["retrieval"].get("max_documents", 8)),
        chunks_per_document=int(config["retrieval"].get("chunks_per_document", 2)),
        rerank_chunk_chars=int(config["retrieval"].get("rerank_chunk_chars", 800)),
        fallback_documents=int(config["retrieval"].get("fallback_documents", 6)),
        parent_documents=parent_documents,
        expanded_documents=int(config["retrieval"].get("expanded_documents", 3)),
        max_parent_chunks=int(config["retrieval"].get("max_parent_chunks", 8)),
    )
    graph = build_simple_rag_graph(retriever, llm) if args.mode == "graph" else None

    answers_file = run_dir / "answers.jsonl"
    retrieved_file = run_dir / "retrieved_docs.jsonl"

    logging.info("Question source filter: %s", question_source_type or "all")
    for question in iter_questions(
        config["data"]["questions_file"],
        limit=args.limit,
        source_type=question_source_type,
    ):
        question_id = question["question_id"]
        question_text = question["question"]
        logging.info("Answering %s", question_id)

        if graph is not None:
            # LangGraph 模式会显式保留状态流转，后续可在这里扩展多步检索。
            state = graph.invoke({"question": question_text, "step_count": 0})
            answer = state.get("answer", "")
            document_ids = state.get("document_ids", [])
            documents = state.get("retrieved_docs", [])
        else:
            result = answer_with_retriever(question_text, retriever, llm)
            answer = result.answer
            document_ids = result.document_ids
            documents = result.documents

        append_answer(answers_file, question_id, answer, document_ids)
        append_retrieved_docs(retrieved_file, question_id, documents)

    logging.info("Wrote answers to %s", answers_file)


if __name__ == "__main__":
    main()
