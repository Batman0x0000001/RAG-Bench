from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from src.graphs.nodes import (
    assess_evidence_p0_node,
    assess_evidence_node,
    expand_entity_links_node,
    expand_parents_node,
    fuse_and_rerank_node,
    generate_answer_node,
    generate_answer_p0_node,
    plan_question_node,
    repair_answer_node,
    retrieve_queries_node,
    route_after_assessment,
    route_after_generation,
)
from src.observability.telemetry import instrument_node
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


def build_p0_candidate_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
    entity_index: dict[str, list[str]] | None = None,
):
    """构建具备安全 Judge、消融开关和完整遥测的 P0 候选工作流。"""
    graph = StateGraph(RagState)
    features = config.get("features", {})
    retrieval = config.get("retrieval", config)

    def add(name: str, node) -> None:
        graph.add_node(name, instrument_node(name, node))

    add(
        "plan_question",
        plan_question_node(
            llm,
            max_queries=int(retrieval.get("max_queries", 5)),
            max_documents=int(retrieval.get("max_documents", 10)),
            adaptive=bool(features.get("adaptive_planning", True)),
            fixed_document_budget=int(retrieval.get("fixed_document_budget", 4)),
        ),
    )
    add(
        "retrieve_queries",
        retrieve_queries_node(
            retriever,
            max_concurrency=int(retrieval.get("query_parallelism", 3)),
        ),
    )
    add(
        "expand_entity_links",
        expand_entity_links_node(
            parent_documents,
            entity_index or {},
            seed_documents=int(retrieval.get("entity_seed_documents", 8)),
            max_linked_documents=int(retrieval.get("max_linked_documents", 8)),
            chunks_per_document=int(retrieval.get("entity_chunks_per_document", 1)),
        ),
    )
    add(
        "fuse_and_rerank",
        fuse_and_rerank_node(
            llm,
            rrf_k=int(retrieval.get("rrf_k", 60)),
            chunks_per_document=int(retrieval.get("chunks_per_document", 2)),
            rerank_chunk_chars=int(retrieval.get("rerank_chunk_chars", 800)),
            policies=build_retrieval_policies(retrieval),
            enabled=bool(features.get("llm_rerank", True)),
            fusion_guard_enabled=bool(features.get("fusion_rank_guard", False)),
            fusion_guard_top_k=int(retrieval.get("fusion_guard_top_k", 6)),
            fusion_guard_chunk_chars=int(
                retrieval.get("fusion_guard_chunk_chars", 800)
            ),
        ),
    )
    add(
        "expand_parents",
        expand_parents_node(
            parent_documents,
            max_parent_chunks=int(retrieval.get("max_parent_chunks", 8)),
            enabled=bool(features.get("parent_expansion", True)),
            fusion_guard_enabled=bool(features.get("fusion_rank_guard", False)),
        ),
    )
    add(
        "assess_evidence",
        assess_evidence_p0_node(
            llm,
            max_follow_up_queries=int(retrieval.get("max_follow_up_queries", 2)),
            guard_max_promotions=(
                int(retrieval.get("fusion_guard_max_promotions", 2))
                if features.get("fusion_rank_guard", False)
                else 0
            ),
        ),
    )
    add("generate_answer", generate_answer_p0_node(llm))
    add("repair_answer", repair_answer_node(llm))
    _add_candidate_edges(
        graph,
        max_retrieval_rounds=int(retrieval.get("max_retrieval_rounds", 2)),
        followup_enabled=bool(features.get("evidence_followup", True)),
        repair_enabled=bool(features.get("answer_repair", True)),
    )
    return graph.compile(name="adaptive_enterprise_rag_p0_candidate")


def build_rag_graph(
    profile: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
    entity_index: dict[str, list[str]] | None = None,
):
    if profile == "stage26":
        retrieval = config.get("retrieval", config)
        return build_stage26_rag_graph(
            retriever, llm, parent_documents, retrieval, entity_index=entity_index
        )
    if profile == "p0_candidate":
        return build_p0_candidate_graph(
            retriever, llm, parent_documents, config, entity_index=entity_index
        )
    raise ValueError(f"Unknown workflow profile: {profile}")


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
    def add(name: str, node) -> None:
        graph.add_node(name, instrument_node(name, node))

    add(
        "plan_question",
        plan_question_node(
            llm,
            max_queries=int(config.get("max_queries", 5)),
            max_documents=int(config.get("max_documents", 10)),
        ),
    )
    add(
        "retrieve_queries",
        retrieve_queries_node(
            retriever,
            max_concurrency=int(config.get("query_parallelism", 3)),
        ),
    )
    add(
        "expand_entity_links",
        expand_entity_links_node(
            parent_documents,
            entity_index,
            seed_documents=int(config.get("entity_seed_documents", 8)),
            max_linked_documents=int(config.get("max_linked_documents", 8)),
            chunks_per_document=int(config.get("entity_chunks_per_document", 1)),
        ),
    )
    add(
        "fuse_and_rerank",
        fuse_and_rerank_node(
            llm,
            rrf_k=int(config.get("rrf_k", 60)),
            chunks_per_document=int(config.get("chunks_per_document", 2)),
            rerank_chunk_chars=int(config.get("rerank_chunk_chars", 800)),
            policies=build_retrieval_policies(config),
        ),
    )
    add(
        "expand_parents",
        expand_parents_node(
            parent_documents,
            max_parent_chunks=int(config.get("max_parent_chunks", 8)),
        ),
    )
    add(
        "assess_evidence",
        assess_evidence_node(
            llm,
            max_follow_up_queries=int(config.get("max_follow_up_queries", 2)),
            evidence_chunk_chars=int(config.get("evidence_chunk_chars", 2_500)),
        ),
    )
    add("generate_answer", generate_answer_node(llm))
    add("repair_answer", repair_answer_node(llm))


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


def _add_candidate_edges(
    graph: StateGraph,
    *,
    max_retrieval_rounds: int,
    followup_enabled: bool,
    repair_enabled: bool,
) -> None:
    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve_queries")
    graph.add_edge("retrieve_queries", "expand_entity_links")
    graph.add_edge("expand_entity_links", "fuse_and_rerank")
    graph.add_edge("fuse_and_rerank", "expand_parents")
    graph.add_edge("expand_parents", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence",
        lambda state: (
            route_after_assessment(state, max_retrieval_rounds)
            if followup_enabled
            else "generate_answer"
        ),
        {
            "retrieve_queries": "retrieve_queries",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "generate_answer",
        lambda state: route_after_generation(state) if repair_enabled else "end",
        {"repair_answer": "repair_answer", "end": END},
    )
    graph.add_edge("repair_answer", END)
