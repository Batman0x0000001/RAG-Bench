from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.graphs.nodes import (
    normalize_plan,
    parse_document_ids,
    prioritize_evidence_documents,
    route_after_assessment,
)
from src.graphs.rag_graph import build_adaptive_rag_graph
from src.ingestion.parse_documents import (
    EnterpriseRagLoader,
    read_manifest,
    split_documents,
    write_manifest,
)
from src.indexing.qdrant_index import stable_document_ids
from src.retrieval.vector_retriever import (
    SimilarityCandidateRetriever,
    build_parent_document_store,
    expand_selected_documents,
    reciprocal_rank_fuse,
)


def test_loader_and_splitter_use_standard_documents(tmp_path: Path) -> None:
    source = tmp_path / "github" / "sample.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "dataset_doc_uuid": "dsid_test",
                "title": "Multipart limits",
                "description": "A" * 5_000,
                "content_field_names": ["description"],
            }
        ),
        encoding="utf-8",
    )
    loaded = EnterpriseRagLoader(source, tmp_path).load()
    chunks = split_documents(loaded)
    description_chunks = [
        document for document in chunks if document.metadata["section"] == "description"
    ]
    assert all(isinstance(document, Document) for document in loaded)
    assert len(description_chunks) >= 2
    assert description_chunks[0].metadata["chunk_id"] == "dsid_test::description::0"
    assert description_chunks[1].metadata["chunk_id"] == "dsid_test::description::1"


def test_manifest_round_trip_uses_document_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    document = Document(
        page_content="content",
        metadata={"dsid": "dsid_test", "chunk_id": "dsid_test::text::0"},
    )
    write_manifest([document], path)
    assert read_manifest(path) == [document]


def test_old_manifest_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"dsid":"dsid_test","content":"legacy"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="run scripts.ingest again"):
        read_manifest(path)


def test_chunk_id_produces_stable_qdrant_uuid() -> None:
    document = Document(
        page_content="content",
        metadata={"chunk_id": "dsid_test::description::0"},
    )
    assert stable_document_ids([document]) == stable_document_ids([document])


def test_similarity_candidate_retriever_uses_configured_k() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [Document(page_content="a", metadata={"dsid": "a"})]

    store = FakeVectorStore()
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=store,
        candidate_k=30,
    )
    assert retriever.invoke("query")[0].page_content == "a"
    assert store.call == {"query": "query", "k": 30}


def test_rrf_rewards_documents_retrieved_by_multiple_queries() -> None:
    a1 = Document(page_content="a1", metadata={"dsid": "a", "chunk_id": "a1"})
    a2 = Document(page_content="a2", metadata={"dsid": "a", "chunk_id": "a2"})
    b = Document(page_content="b", metadata={"dsid": "b", "chunk_id": "b1"})
    c = Document(page_content="c", metadata={"dsid": "c", "chunk_id": "c1"})

    fused = reciprocal_rank_fuse(
        [[b, a1], [c, a2]],
        max_documents=3,
        chunks_per_document=2,
        rrf_k=60,
    )

    assert list(fused)[0] == "a"
    assert [document.page_content for document in fused["a"]] == ["a1", "a2"]


def test_normalize_plan_applies_adaptive_budget_and_keeps_original_query() -> None:
    plan = normalize_plan(
        {
            "strategy": "completeness",
            "queries": ["subquery one", "subquery two"],
            "document_budget": 2,
            "requirements": ["all rollout artifacts"],
        },
        question="original question",
        max_queries=5,
        max_documents=10,
    )

    assert plan["queries"][0] == "original question"
    assert plan["document_budget"] == 6
    assert plan["minimum_documents"] == 6

    single = normalize_plan(
        {
            "strategy": "single",
            "queries": ["unneeded rewrite"],
            "document_budget": 3,
        },
        question="exact lookup",
        max_queries=5,
        max_documents=10,
    )
    assert single["queries"] == ["exact lookup"]
    assert single["document_budget"] == 1


def test_parse_document_ids_filters_unknown_and_fills_minimum() -> None:
    selected = parse_document_ids(
        '{"document_ids":["b","unknown"]}',
        available_ids=["a", "b", "c"],
        budget=3,
        minimum=2,
    )
    assert selected == ["b", "a"]


def test_expand_selected_documents_adds_missing_parent_sections() -> None:
    description = Document(
        page_content="description",
        metadata={"dsid": "a", "chunk_id": "a::description::0"},
    )
    discussion = Document(
        page_content="answer in discussion",
        metadata={"dsid": "a", "chunk_id": "a::discussion::0"},
    )
    store = build_parent_document_store([description, discussion])
    selected = expand_selected_documents(
        ["a"],
        {"a": [description]},
        parent_documents=store,
        expanded_documents=1,
        max_parent_chunks=8,
    )
    assert [document.page_content for document in selected] == [
        "description",
        "answer in discussion",
    ]


def test_evidence_route_is_bounded() -> None:
    state = {
        "evidence_sufficient": False,
        "can_retry": True,
        "pending_queries": ["follow up"],
        "retrieval_round": 1,
    }
    assert route_after_assessment(state, max_retrieval_rounds=2) == "retrieve_queries"
    state["retrieval_round"] = 2
    assert route_after_assessment(state, max_retrieval_rounds=2) == "generate_answer"


def test_evidence_prioritization_keeps_parent_context() -> None:
    first = Document(page_content="first", metadata={"chunk_id": "a"})
    second = Document(page_content="second", metadata={"chunk_id": "b"})

    prioritized = prioritize_evidence_documents([first, second], ["b"])

    assert [document.page_content for document in prioritized] == ["second", "first"]


def test_langchain_components_run_inside_unified_langgraph() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            return [
                Document(
                    page_content="The limit is 10 MiB.",
                    metadata={
                        "dsid": "a",
                        "chunk_id": "a::description::0",
                        "title": "Upload limit",
                    },
                )
            ]

    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","queries":[],"document_budget":1,"requirements":["limit"]}',
            '{"document_ids":["a"]}',
            '{"sufficient":true,"relevant_chunk_ids":["a::description::0"],"missing_evidence":[],"follow_up_queries":[]}',
            "The limit is 10 MiB.",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        build_parent_document_store(retriever.invoke("seed")),
        {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "max_retrieval_rounds": 2,
        },
    )

    state = graph.invoke({"question": "What is the upload limit?"})

    assert state["answer"] == "The limit is 10 MiB."
    assert state["document_ids"] == ["a"]
    assert state["executed_queries"] == ["What is the upload limit?"]


def test_unified_graph_runs_bounded_follow_up_retrieval() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            dsid = "b" if query == "follow up detail" else "a"
            return [
                Document(
                    page_content=f"evidence {dsid}",
                    metadata={
                        "dsid": dsid,
                        "chunk_id": f"{dsid}::description::0",
                        "title": dsid,
                    },
                )
            ]

    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    parent_documents = build_parent_document_store(
        retriever.batch(["original", "follow up detail"])
        [0]
        + retriever.batch(["follow up detail"])[0]
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"semantic","queries":["alternate wording"],'
            '"document_budget":2,"requirements":["detail"]}',
            '{"document_ids":["a"]}',
            '{"sufficient":false,"relevant_chunk_ids":["a::description::0"],'
            '"missing_evidence":["detail"],"follow_up_queries":["follow up detail"]}',
            '{"document_ids":["b","a"]}',
            '{"sufficient":true,"relevant_chunk_ids":["b::description::0"],'
            '"missing_evidence":[],"follow_up_queries":[]}',
            "combined answer",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        parent_documents,
        {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "max_retrieval_rounds": 2,
        },
    )

    state = graph.invoke({"question": "original"})

    assert state["answer"] == "combined answer"
    assert state["retrieval_round"] == 2
    assert state["executed_queries"] == [
        "original",
        "alternate wording",
        "follow up detail",
    ]
    assert state["document_ids"] == ["b", "a"]
