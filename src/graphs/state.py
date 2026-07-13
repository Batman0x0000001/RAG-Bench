from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document


class RagState(TypedDict, total=False):
    """Stage 26 工作流在 LangGraph 节点之间传递的状态。"""

    question: str
    plan: dict[str, Any]
    pending_queries: list[str]
    pending_tasks: list[dict[str, str]]
    executed_queries: list[str]
    executed_tasks: list[dict[str, str]]
    query_results: list[list[Document]]
    base_query_results: list[list[Document]]
    result_tasks: list[dict[str, str]]
    entity_expansions: list[dict[str, Any]]
    candidate_groups: dict[str, list[Document]]
    candidate_archive: dict[str, list[Document]]
    selected_document_ids: list[str]
    rerank_history: list[dict[str, Any]]
    fusion_guard_history: list[dict[str, Any]]
    retrieval_stage_history: list[dict[str, Any]]
    parent_expansion_history: list[dict[str, Any]]
    retrieved_docs: list[Document]
    answer_docs: list[Document]
    guard_documents: list[Document]
    guard_parent_documents: list[Document]
    missing_evidence: list[str]
    evidence_sufficient: bool
    evidence_status: str
    judge_attempts: list[dict[str, Any]]
    can_retry: bool
    retrieval_round: int
    answer: str
    original_answer: str
    answer_repaired: bool
    document_ids: list[str]
    model_calls: list[dict[str, Any]]
    node_metrics: list[dict[str, Any]]
