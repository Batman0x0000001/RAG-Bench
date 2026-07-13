from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

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
    text_section_weight: float = 0.8
    mode: Literal["dense", "bm25", "rank_sum", "rrf"] = "rrf"

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        channel_results: dict[str, list[Document]] = {}
        if self.mode != "bm25":
            channel_results["dense"] = self.dense_retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
        if self.mode != "dense":
            channel_results["bm25"] = self.bm25_retriever.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        channels: dict[str, list[str]] = defaultdict(list)
        channel_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for channel, results in channel_results.items():
            result_count = max(len(results), 1)
            for rank, document in enumerate(results, start=1):
                chunk_id = str(
                    document.metadata.get("chunk_id")
                    or f"{document.metadata.get('dsid', '')}::{document.page_content}"
                )
                section_weight = (
                    self.text_section_weight
                    if document.metadata.get("section") == "text"
                    else 1.0
                )
                if self.mode == "rank_sum":
                    scores[chunk_id] += section_weight * (
                        (result_count - rank + 1) / result_count
                    )
                else:
                    scores[chunk_id] += section_weight / (self.rrf_k + rank)
                documents.setdefault(chunk_id, document)
                channels[chunk_id].append(channel)
                channel_ranks[chunk_id][channel] = rank

        ordered_ids = sorted(scores, key=scores.get, reverse=True)[: self.candidate_k]
        union_ids = list(
            dict.fromkeys(
                str(
                    document.metadata.get("chunk_id")
                    or f"{document.metadata.get('dsid', '')}::{document.page_content}"
                )
                for results in channel_results.values()
                for document in results
            )
        )

        def trace_rows(ids: list[str]) -> list[dict[str, Any]]:
            return [
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "dsid": documents[chunk_id].metadata.get("dsid"),
                }
                for rank, chunk_id in enumerate(ids, start=1)
                if chunk_id in documents
            ]

        retrieval_trace = {
            "query": query,
            "mode": self.mode,
            "stages": {
                channel: [
                    {
                        "rank": rank,
                        "chunk_id": document.metadata.get("chunk_id"),
                        "dsid": document.metadata.get("dsid"),
                    }
                    for rank, document in enumerate(results, start=1)
                ]
                for channel, results in channel_results.items()
            }
            | {
                "union": trace_rows(union_ids),
                "channel_fusion": trace_rows(ordered_ids),
            },
        }
        return [
            Document(
                page_content=documents[chunk_id].page_content,
                metadata={
                    **documents[chunk_id].metadata,
                    "retrieval_channels": channels[chunk_id],
                    "retrieval_channel_ranks": channel_ranks[chunk_id],
                    "hybrid_rrf_score": scores[chunk_id],
                    # 只挂在第一条结果上，LangGraph 节点提取后立即移除。
                    **({"_retrieval_trace": retrieval_trace} if index == 0 else {}),
                },
            )
            for index, chunk_id in enumerate(ordered_ids)
        ]


def reciprocal_rank_fuse(
    query_results: list[list[Document]],
    max_documents: int,
    chunks_per_document: int,
    rrf_k: int = 60,
    reserved_documents_per_result: int = 0,
    reserve_entity_link_results: bool = True,
) -> dict[str, list[Document]]:
    """按文档执行 RRF；同一查询中的重复 chunk 只贡献该文档的最佳排名。"""
    scores: dict[str, float] = defaultdict(float)
    query_hits: dict[str, int] = defaultdict(int)
    chunks: dict[str, list[Document]] = defaultdict(list)
    seen_chunks: dict[str, set[str]] = defaultdict(set)
    task_ids: dict[str, set[str]] = defaultdict(set)
    slots: dict[str, set[str]] = defaultdict(set)
    reservation_lists: list[list[str]] = []

    for documents in query_results:
        best_rank: dict[str, int] = {}
        for rank, document in enumerate(documents, start=1):
            dsid = str(document.metadata.get("dsid") or "")
            if not dsid:
                continue
            best_rank.setdefault(dsid, rank)
            task_ids[dsid].update(document.metadata.get("retrieval_task_ids", []))
            if document.metadata.get("retrieval_slot"):
                slots[dsid].add(str(document.metadata["retrieval_slot"]))
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
        is_entity_result = bool(documents) and all(
            document.metadata.get("retrieval_slot") == "entity_link"
            for document in documents
        )
        if reserve_entity_link_results or not is_entity_result:
            reservation_lists.append(
                list(best_rank)[:reserved_documents_per_result]
            )

    ranked_ids = sorted(scores, key=lambda dsid: scores[dsid], reverse=True)
    reserved_ids: list[str] = []
    for offset in range(reserved_documents_per_result):
        for reservation in reservation_lists:
            if offset < len(reservation) and reservation[offset] not in reserved_ids:
                reserved_ids.append(reservation[offset])
    ordered_ids = reserved_ids[:max_documents]
    ordered_ids.extend(
        dsid for dsid in ranked_ids if dsid not in ordered_ids
    )
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
                    "retrieval_task_ids": sorted(task_ids[dsid]),
                    "retrieval_slots": sorted(slots[dsid]),
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
    text_section_weight: float = 0.8,
    mode: Literal["dense", "bm25", "rank_sum", "rrf"] = "rrf",
) -> BaseRetriever:
    return HybridCandidateRetriever(
        dense_retriever=build_candidate_retriever(
            qdrant_config,
            embeddings,
            candidate_k=dense_k,
        ),
        bm25_retriever=build_bm25_retriever(
            documents,
            k=bm25_k,
            text_section_weight=text_section_weight,
        ),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        text_section_weight=text_section_weight,
        mode=mode,
    )
