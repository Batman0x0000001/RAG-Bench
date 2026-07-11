from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class SimilarityCandidateRetriever(BaseRetriever):
    """只负责高召回候选检索，规划、融合和重排由 LangGraph 编排。"""

    vector_store: QdrantVectorStore
    candidate_k: int = 30

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.candidate_k)


def reciprocal_rank_fuse(
    query_results: list[list[Document]],
    max_documents: int,
    chunks_per_document: int,
    rrf_k: int = 60,
) -> dict[str, list[Document]]:
    """按文档执行 RRF；同一查询中的重复 chunk 只贡献该文档的最佳排名。"""
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, list[Document]] = defaultdict(list)
    seen_chunks: dict[str, set[str]] = defaultdict(set)

    for documents in query_results:
        best_rank: dict[str, int] = {}
        for rank, document in enumerate(documents, start=1):
            dsid = str(document.metadata.get("dsid") or "")
            if not dsid:
                continue
            best_rank.setdefault(dsid, rank)
            chunk_id = str(document.metadata.get("chunk_id") or document.page_content)
            if (
                chunk_id not in seen_chunks[dsid]
                and len(chunks[dsid]) < chunks_per_document
            ):
                seen_chunks[dsid].add(chunk_id)
                chunks[dsid].append(document)
        for dsid, rank in best_rank.items():
            scores[dsid] += 1.0 / (rrf_k + rank)

    ordered_ids = sorted(scores, key=lambda dsid: scores[dsid], reverse=True)
    return {dsid: chunks[dsid] for dsid in ordered_ids[:max_documents]}


def build_parent_document_store(
    documents: list[Document],
) -> dict[str, list[Document]]:
    store: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        dsid = document.metadata.get("dsid")
        if dsid:
            store[str(dsid)].append(document)
    return dict(store)


def expand_selected_documents(
    selected_ids: list[str],
    candidate_groups: dict[str, list[Document]],
    parent_documents: dict[str, list[Document]],
    expanded_documents: int,
    max_parent_chunks: int,
) -> list[Document]:
    """为最高排名文件补齐其他语义 section，并保持候选 chunk 优先。"""
    selected: list[Document] = []
    for rank, dsid in enumerate(selected_ids):
        candidates = candidate_groups[dsid]
        if rank >= expanded_documents or dsid not in parent_documents:
            selected.extend(candidates)
            continue

        combined = candidates + parent_documents[dsid]
        seen: set[str] = set()
        parent_chunks: list[Document] = []
        for index, document in enumerate(combined):
            chunk_id = str(document.metadata.get("chunk_id") or f"{dsid}::{index}")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            parent_chunks.append(document)
            if len(parent_chunks) >= max_parent_chunks:
                break
        selected.extend(parent_chunks)
    return selected


def build_qdrant_client(config: dict[str, Any]) -> QdrantClient:
    api_key = config.get("api_key") or None
    return QdrantClient(url=config["url"], api_key=api_key)


def build_vector_store(
    qdrant_config: dict[str, Any],
    embeddings: Embeddings,
) -> QdrantVectorStore:
    client = build_qdrant_client(qdrant_config)
    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_config["collection"],
        embedding=embeddings,
    )


def build_candidate_retriever(
    qdrant_config: dict[str, Any],
    embeddings: Embeddings,
    candidate_k: int = 30,
) -> BaseRetriever:
    return SimilarityCandidateRetriever(
        vector_store=build_vector_store(qdrant_config, embeddings),
        candidate_k=candidate_k,
    )
