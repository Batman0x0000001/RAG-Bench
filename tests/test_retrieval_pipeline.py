from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.ingestion.parse_documents import (
    EnterpriseRagLoader,
    read_manifest,
    split_documents,
    write_manifest,
)
from src.indexing.qdrant_index import stable_document_ids
from src.retrieval.vector_retriever import (
    DocumentRerankingRetriever,
    build_parent_document_store,
    expand_selected_documents,
    group_scored_documents,
    parse_reranked_document_ids,
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
    assert loaded
    assert all(isinstance(document, Document) for document in loaded)

    chunks = split_documents(loaded)
    description_chunks = [
        document for document in chunks if document.metadata["section"] == "description"
    ]
    assert len(description_chunks) >= 2
    assert description_chunks[0].metadata["chunk_id"] == "dsid_test::description::0"
    assert description_chunks[1].metadata["chunk_id"] == "dsid_test::description::1"
    assert all("Section: description" in document.page_content for document in description_chunks)


def test_manifest_round_trip_uses_document_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    document = Document(
        page_content="content",
        metadata={"dsid": "dsid_test", "chunk_id": "dsid_test::text::0"},
    )

    write_manifest([document], path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "page_content": "content",
        "metadata": {"dsid": "dsid_test", "chunk_id": "dsid_test::text::0"},
    }
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


def test_group_scored_documents_limits_files_and_chunks() -> None:
    scored_documents = [
        (Document(page_content="a1", metadata={"dsid": "a"}), 0.9),
        (Document(page_content="a2", metadata={"dsid": "a"}), 0.8),
        (Document(page_content="a3", metadata={"dsid": "a"}), 0.7),
        (Document(page_content="b1", metadata={"dsid": "b"}), 0.6),
        (Document(page_content="c1", metadata={"dsid": "c"}), 0.5),
    ]

    grouped = group_scored_documents(
        scored_documents,
        max_documents=2,
        chunks_per_document=2,
    )

    assert list(grouped) == ["a", "b"]
    assert [document.page_content for document in grouped["a"]] == ["a1", "a2"]


def test_parse_reranked_document_ids_filters_unknown_and_duplicate_ids() -> None:
    response = '{"document_ids":["b","unknown","b","a","c"]}'

    selected = parse_reranked_document_ids(response, ["a", "b", "c"], 2)

    assert selected == ["b", "a"]


def test_expand_selected_documents_adds_missing_parent_sections_for_top_rank() -> None:
    description = Document(
        page_content="description",
        metadata={"dsid": "a", "chunk_id": "a::description::0"},
    )
    discussion = Document(
        page_content="answer in discussion",
        metadata={"dsid": "a", "chunk_id": "a::discussion::0"},
    )
    other = Document(
        page_content="other",
        metadata={"dsid": "b", "chunk_id": "b::description::0"},
    )
    store = build_parent_document_store([description, discussion, other])

    selected = expand_selected_documents(
        ["a", "b"],
        {"a": [description], "b": [other]},
        parent_documents=store,
        expanded_documents=1,
        max_parent_chunks=8,
    )

    assert [document.page_content for document in selected] == [
        "description",
        "answer in discussion",
        "other",
    ]


def test_document_reranking_retriever_uses_similarity_and_llm_order() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search_with_score(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [
                (Document(page_content="a1", metadata={"dsid": "a"}), 0.9),
                (Document(page_content="a2", metadata={"dsid": "a"}), 0.8),
                (Document(page_content="b1", metadata={"dsid": "b"}), 0.7),
            ]

    vector_store = FakeVectorStore()
    retriever = DocumentRerankingRetriever.model_construct(
        vector_store=vector_store,
        llm=FakeListChatModel(responses=['{"document_ids":["b","a"]}']),
        candidate_k=30,
        candidate_documents=10,
        max_documents=2,
        chunks_per_document=1,
        rerank_chunk_chars=100,
        fallback_documents=2,
    )

    selected = retriever.invoke("test query")

    assert vector_store.call == {"query": "test query", "k": 30}
    assert [document.page_content for document in selected] == ["b1", "a1"]


def test_document_reranking_retriever_falls_back_on_invalid_json() -> None:
    class FakeVectorStore:
        def similarity_search_with_score(self, query: str, **kwargs):
            return [
                (Document(page_content="a1", metadata={"dsid": "a"}), 0.9),
                (Document(page_content="b1", metadata={"dsid": "b"}), 0.8),
            ]

    retriever = DocumentRerankingRetriever.model_construct(
        vector_store=FakeVectorStore(),
        llm=FakeListChatModel(responses=["not json"]),
        candidate_k=30,
        candidate_documents=10,
        max_documents=2,
        chunks_per_document=1,
        rerank_chunk_chars=100,
        fallback_documents=1,
    )

    selected = retriever.invoke("test query")

    assert [document.page_content for document in selected] == ["a1"]
