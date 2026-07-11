from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.retrieval.lexical_retriever import build_bm25_retriever


class SimilarityCandidateRetriever(BaseRetriever):
    """只负责高召回候选检索，规划、融合和重排由 LangGraph 编排。"""

    vector_store: QdrantVectorStore
    candidate_k: int = 30

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        documents = self.vector_store.similarity_search(query, k=self.candidate_k)
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_channels": ["dense"],
                    "dense_rank": rank,
                },
            )
            for rank, document in enumerate(documents, start=1)
        ]


class HybridCandidateRetriever(BaseRetriever):
    """对 Dense 与 BM25 两路标准检索器执行 chunk 级 RRF。"""

    dense_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    candidate_k: int = 40
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        channel_results = {
            "dense": self.dense_retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            ),
            "bm25": self.bm25_retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            ),
        }
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        channels: dict[str, list[str]] = defaultdict(list)
        channel_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for channel, results in channel_results.items():
            for rank, document in enumerate(results, start=1):
                chunk_id = str(
                    document.metadata.get("chunk_id")
                    or f"{document.metadata.get('dsid', '')}::{document.page_content}"
                )
                scores[chunk_id] += 1.0 / (self.rrf_k + rank)
                documents.setdefault(chunk_id, document)
                channels[chunk_id].append(channel)
                channel_ranks[chunk_id][channel] = rank

        ordered_ids = sorted(scores, key=scores.get, reverse=True)[: self.candidate_k]
        return [
            Document(
                page_content=documents[chunk_id].page_content,
                metadata={
                    **documents[chunk_id].metadata,
                    "retrieval_channels": channels[chunk_id],
                    "retrieval_channel_ranks": channel_ranks[chunk_id],
                    "hybrid_rrf_score": scores[chunk_id],
                },
            )
            for chunk_id in ordered_ids
        ]


def reciprocal_rank_fuse(
    query_results: list[list[Document]],
    max_documents: int,
    chunks_per_document: int,
    rrf_k: int = 60,
) -> dict[str, list[Document]]:
    """按文档执行 RRF；同一查询中的重复 chunk 只贡献该文档的最佳排名。"""
    scores: dict[str, float] = defaultdict(float)
    query_hits: dict[str, int] = defaultdict(int)
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
            query_hits[dsid] += 1

    ordered_ids = sorted(scores, key=lambda dsid: scores[dsid], reverse=True)
    groups: dict[str, list[Document]] = {}
    for document_rank, dsid in enumerate(ordered_ids[:max_documents], start=1):
        groups[dsid] = [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "document_rrf_rank": document_rank,
                    "document_rrf_score": scores[dsid],
                    "query_hit_count": query_hits[dsid],
                },
            )
            for document in chunks[dsid]
        ]
    return groups


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


def build_hybrid_candidate_retriever(
    qdrant_config: dict[str, Any],
    embeddings: Embeddings,
    documents: list[Document],
    dense_k: int = 30,
    bm25_k: int = 30,
    candidate_k: int = 40,
    rrf_k: int = 60,
) -> BaseRetriever:
    return HybridCandidateRetriever(
        dense_retriever=build_candidate_retriever(
            qdrant_config,
            embeddings,
            candidate_k=dense_k,
        ),
        bm25_retriever=build_bm25_retriever(documents, k=bm25_k),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
