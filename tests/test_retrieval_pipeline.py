from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.parse_documents import (
    EnterpriseRagLoader,
    read_manifest,
    split_documents,
    write_manifest,
)
from src.indexing.qdrant_index import stable_document_ids


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
