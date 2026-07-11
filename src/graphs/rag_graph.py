from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph

from src.graphs.nodes import (
    RagState,
    assess_evidence_node,
    expand_parents_node,
    fuse_and_rerank_node,
    generate_answer_node,
    plan_question_node,
    retrieve_queries_node,
    route_after_assessment,
)


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
):
    """LangGraph 管控制流，节点内部复用标准 LangChain 组件。"""
    graph = StateGraph(RagState)
    graph.add_node(
        "plan_question",
        plan_question_node(
            llm,
            max_queries=int(config.get("max_queries", 5)),
            max_documents=int(config.get("max_documents", 10)),
        ),
    )
    graph.add_node(
        "retrieve_queries",
        retrieve_queries_node(
            retriever,
            max_concurrency=int(config.get("query_parallelism", 3)),
        ),
    )
    graph.add_node(
        "fuse_and_rerank",
        fuse_and_rerank_node(
            llm,
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_documents=int(config.get("candidate_documents", 24)),
            chunks_per_document=int(config.get("chunks_per_document", 2)),
            rerank_chunk_chars=int(config.get("rerank_chunk_chars", 800)),
        ),
    )
    graph.add_node(
        "expand_parents",
        expand_parents_node(
            parent_documents,
            max_parent_chunks=int(config.get("max_parent_chunks", 8)),
        ),
    )
    graph.add_node(
        "assess_evidence",
        assess_evidence_node(
            llm,
            max_follow_up_queries=int(config.get("max_follow_up_queries", 2)),
            evidence_chunk_chars=int(config.get("evidence_chunk_chars", 2_500)),
        ),
    )
    graph.add_node("generate_answer", generate_answer_node(llm))

    graph.set_entry_point("plan_question")
    graph.add_edge("plan_question", "retrieve_queries")
    graph.add_edge("retrieve_queries", "fuse_and_rerank")
    graph.add_edge("fuse_and_rerank", "expand_parents")
    graph.add_edge("expand_parents", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence",
        lambda state: route_after_assessment(
            state,
            max_retrieval_rounds=int(config.get("max_retrieval_rounds", 2)),
        ),
        {
            "retrieve_queries": "retrieve_queries",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("generate_answer", END)
    return graph.compile(name="adaptive_enterprise_rag")
