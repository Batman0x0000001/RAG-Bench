from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from src.graphs.rag_graph import build_adaptive_rag_graph
from src.graphs.workflow import build_stage26_rag_graph
from src.utils.config import DEFAULT_CONFIG


class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        return []


def test_stage26_workflow_has_the_stable_topology() -> None:
    graph = build_stage26_rag_graph(
        EmptyRetriever(),
        FakeListChatModel(responses=[]),
        {},
        {},
    )

    drawable = graph.get_graph()
    assert set(drawable.nodes) == {
        "__start__",
        "plan_question",
        "retrieve_queries",
        "expand_entity_links",
        "fuse_and_rerank",
        "expand_parents",
        "assess_evidence",
        "generate_answer",
        "repair_answer",
        "__end__",
    }
    assert {
        (edge.source, edge.target, edge.conditional)
        for edge in drawable.edges
    } == {
        ("__start__", "plan_question", False),
        ("plan_question", "retrieve_queries", False),
        ("retrieve_queries", "expand_entity_links", False),
        ("expand_entity_links", "fuse_and_rerank", False),
        ("fuse_and_rerank", "expand_parents", False),
        ("expand_parents", "assess_evidence", False),
        ("assess_evidence", "retrieve_queries", True),
        ("assess_evidence", "generate_answer", True),
        ("generate_answer", "repair_answer", True),
        ("generate_answer", "__end__", True),
        ("repair_answer", "__end__", False),
    }


def test_legacy_builder_uses_stage26_workflow() -> None:
    graph = build_adaptive_rag_graph(
        EmptyRetriever(),
        FakeListChatModel(responses=[]),
        {},
        {},
    )

    assert graph.name == "adaptive_enterprise_rag"
    assert set(graph.get_graph().nodes) == set(
        build_stage26_rag_graph(
            EmptyRetriever(),
            FakeListChatModel(responses=[]),
            {},
            {},
        ).get_graph().nodes
    )


def test_stable_config_keeps_stage26_retrieval_defaults() -> None:
    retrieval = DEFAULT_CONFIG["retrieval"]

    assert retrieval["dense_candidate_k"] == 40
    assert retrieval["bm25_candidate_k"] == 40
    assert retrieval["hybrid_candidate_k"] == 40
    assert retrieval["rrf_k"] == 60
    assert retrieval["candidate_documents"] == 24
    assert retrieval["candidate_archive_documents"] == 32
    assert retrieval["max_retrieval_rounds"] == 2
    assert "enable_multi_query" not in retrieval
    assert "enable_hyde" not in retrieval
