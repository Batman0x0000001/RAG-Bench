from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from src.graphs.nodes import (
    assess_evidence_node,
    expand_entity_links_node,
    expand_parents_node,
    fuse_and_rerank_node,
    generate_answer_node,
    plan_question_node,
    repair_answer_node,
    retrieve_queries_node,
    route_after_assessment,
    route_after_generation,
)
from src.graphs.retrieval_policy import build_retrieval_policies
from src.graphs.state import RagState


def build_stage26_rag_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
    entity_index: dict[str, list[str]] | None = None,
):
    """构建并编译经过 n39 验证的 Stage 26 LangGraph 工作流。"""
    graph = StateGraph(RagState)
    _add_stage26_nodes(
        graph,
        retriever=retriever,
        llm=llm,
        parent_documents=parent_documents,
        config=config,
        entity_index=entity_index or {},
    )
    _add_stage26_edges(graph, max_retrieval_rounds=int(config.get("max_retrieval_rounds", 2)))
    return graph.compile(name="adaptive_enterprise_rag")


def _add_stage26_nodes(
    graph: StateGraph,
    *,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
    entity_index: dict[str, list[str]],
) -> None:
    """集中装配节点依赖，节点内部继续复用 LangChain 组件。"""
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
        "expand_entity_links",
        expand_entity_links_node(
            parent_documents,
            entity_index,
            seed_documents=int(config.get("entity_seed_documents", 8)),
            max_linked_documents=int(config.get("max_linked_documents", 8)),
            chunks_per_document=int(config.get("entity_chunks_per_document", 1)),
        ),
    )
    graph.add_node(
        "fuse_and_rerank",
        fuse_and_rerank_node(
            llm,
            rrf_k=int(config.get("rrf_k", 60)),
            chunks_per_document=int(config.get("chunks_per_document", 2)),
            rerank_chunk_chars=int(config.get("rerank_chunk_chars", 800)),
            policies=build_retrieval_policies(config),
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
    graph.add_node("repair_answer", repair_answer_node(llm))


def _add_stage26_edges(graph: StateGraph, max_retrieval_rounds: int) -> None:
    """声明稳定流程拓扑；条件边只负责路由，不承载业务逻辑。"""
    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve_queries")
    graph.add_edge("retrieve_queries", "expand_entity_links")
    graph.add_edge("expand_entity_links", "fuse_and_rerank")
    graph.add_edge("fuse_and_rerank", "expand_parents")
    graph.add_edge("expand_parents", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence",
        lambda state: route_after_assessment(state, max_retrieval_rounds),
        {
            "retrieve_queries": "retrieve_queries",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "generate_answer",
        route_after_generation,
        {
            "repair_answer": "repair_answer",
            "end": END,
        },
    )
    graph.add_edge("repair_answer", END)
