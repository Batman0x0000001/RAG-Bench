from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from src.graphs.nodes import (
    EvidenceStatus,
    assess_evidence_p0_node,
    build_fusion_guard_candidates,
    expand_parents_node,
    generate_answer_p0_node,
    parse_evidence_assessment,
    route_after_generation,
)
from src.graphs.workflow import build_rag_graph
from src.retrieval.vector_retriever import build_parent_document_store
from src.retrieval.vector_retriever import HybridCandidateRetriever


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        return self.documents


def _judge_state() -> dict:
    document = Document(
        page_content="The exact limit is 10 MiB.",
        metadata={"dsid": "gold", "chunk_id": "gold::description::0"},
    )
    return {
        "question": "What is the exact limit?",
        "plan": {
            "requirements": ["exact limit"],
            "retrieval_tasks": [],
        },
        "retrieved_docs": [document],
        "executed_queries": ["What is the exact limit?"],
        "retrieval_round": 1,
        "model_calls": [],
    }


def test_strict_judge_parser_rejects_legacy_or_incomplete_payload() -> None:
    try:
        parse_evidence_assessment('{"sufficient":true}')
    except Exception:
        pass
    else:
        raise AssertionError("Legacy Judge payload must not be accepted by P0")


def test_judge_retry_can_recover_valid_structured_output() -> None:
    llm = FakeListChatModel(
        responses=[
            "not-json",
            '{"status":"SUFFICIENT","relevant_chunk_ids":['
            '"gold::description::0"],"missing_evidence":[],"follow_up_queries":[]}',
        ]
    )

    state = assess_evidence_p0_node(llm, 2)(_judge_state())

    assert state["evidence_status"] == EvidenceStatus.SUFFICIENT.value
    assert state["evidence_sufficient"] is True
    assert len(state["judge_attempts"]) == 2
    assert len(state["model_calls"]) == 2


def test_invalid_judge_output_fails_closed_to_unknown_and_followup() -> None:
    llm = FakeListChatModel(responses=["not-json", "still-not-json"])

    state = assess_evidence_p0_node(llm, 2)(_judge_state())

    assert state["evidence_status"] == EvidenceStatus.UNKNOWN.value
    assert state["evidence_sufficient"] is False
    assert state["can_retry"] is True
    assert state["pending_queries"]


def test_judge_fallback_marks_missing_direct_signal_insufficient() -> None:
    llm = FakeListChatModel(responses=["not-json", "still-not-json"])
    state = _judge_state()
    state["question"] = "Which HTTP status code is returned?"
    state["plan"]["requirements"] = ["HTTP status code"]

    result = assess_evidence_p0_node(llm, 2)(state)

    assert result["evidence_status"] == EvidenceStatus.INSUFFICIENT.value
    assert result["evidence_sufficient"] is False
    assert result["can_retry"] is True
    assert result["judge_attempts"][-1]["fallback_reason"] == (
        "deterministic_missing_signals"
    )


def test_no_documents_still_generates_a_conservative_followup() -> None:
    state = _judge_state()
    state["retrieved_docs"] = []

    result = assess_evidence_p0_node(
        FakeListChatModel(responses=["must not be used"]), 2
    )(state)

    assert result["evidence_status"] == EvidenceStatus.INSUFFICIENT.value
    assert result["evidence_sufficient"] is False
    assert result["can_retry"] is True
    assert result["pending_queries"]


def test_unknown_evidence_refuses_without_calling_generator_or_repair() -> None:
    llm = FakeListChatModel(responses=["must not be used"])
    state = {
        **_judge_state(),
        "evidence_status": EvidenceStatus.UNKNOWN.value,
        "evidence_sufficient": False,
        "missing_evidence": ["exact limit"],
    }

    result = generate_answer_p0_node(llm)(state)

    assert "insufficient" in result["answer"].lower()
    assert result["answer_repaired"] is False
    assert result.get("model_calls", []) == []
    assert route_after_generation(result) == "end"


def test_insufficient_evidence_refuses_without_calling_generator_or_repair() -> None:
    llm = FakeListChatModel(responses=["must not be used"])
    state = {
        **_judge_state(),
        "evidence_status": EvidenceStatus.INSUFFICIENT.value,
        "evidence_sufficient": False,
        "missing_evidence": ["exact limit"],
    }

    result = generate_answer_p0_node(llm)(state)

    assert "insufficient" in result["answer"].lower()
    assert result["answer_repaired"] is False
    assert result.get("model_calls", []) == []
    assert route_after_generation(result) == "end"


def test_p0_judge_receives_the_full_answer_evidence(monkeypatch) -> None:
    tail_marker = "PHASED_ROLLOUT_DETAILS_AT_DOCUMENT_END"
    state = _judge_state()
    state["retrieved_docs"][0].page_content = "x" * 2_500 + tail_marker
    captured: dict[str, str] = {}

    def fake_invoke_text_model(llm, prompt, **kwargs):
        captured["prompt"] = str(prompt)
        return (
            '{"status":"SUFFICIENT","relevant_chunk_ids":['
            '"gold::description::0"],"missing_evidence":[],"follow_up_queries":[]}',
            [],
        )

    monkeypatch.setattr(
        "src.graphs.nodes.invoke_text_model",
        fake_invoke_text_model,
    )

    result = assess_evidence_p0_node(FakeListChatModel(responses=[]), 2)(state)

    assert result["evidence_status"] == EvidenceStatus.SUFFICIENT.value
    assert tail_marker in captured["prompt"]


def test_fusion_guard_collects_channel_and_document_candidates() -> None:
    groups = {
        dsid: [
            Document(
                page_content=(dsid + " evidence ") * 100,
                metadata={"dsid": dsid, "chunk_id": f"{dsid}::0"},
            )
        ]
        for dsid in ("primary", "document-guard", "channel-guard")
    }
    state = {
        "retrieval_stage_history": [
            {
                "stages": {
                    "channel_fusion": [
                        {"dsid": "primary"},
                        {"dsid": "channel-guard"},
                    ]
                }
            }
        ]
    }

    documents, trace = build_fusion_guard_candidates(
        state,
        groups,
        selected_ids=["primary"],
        top_k=2,
        chunk_chars=80,
    )

    assert {document.metadata["dsid"] for document in documents} == {
        "channel-guard",
        "document-guard",
    }
    assert all(len(document.page_content) == 80 for document in documents)
    assert trace["candidate_document_ids"] == [
        "channel-guard",
        "document-guard",
    ]


def test_guarded_judge_promotes_only_relevant_parent_document() -> None:
    primary = Document(
        page_content="primary full evidence",
        metadata={"dsid": "primary", "chunk_id": "primary::0"},
    )
    relevant_guard = Document(
        page_content="relevant guard excerpt",
        metadata={
            "dsid": "guard-1",
            "chunk_id": "guard-1::0",
            "fusion_guard_candidate": True,
        },
    )
    irrelevant_guard = Document(
        page_content="irrelevant guard excerpt",
        metadata={
            "dsid": "guard-2",
            "chunk_id": "guard-2::0",
            "fusion_guard_candidate": True,
        },
    )
    state = {
        **_judge_state(),
        "retrieved_docs": [primary, relevant_guard, irrelevant_guard],
        "answer_docs": [primary],
        "selected_document_ids": ["primary"],
        "document_ids": ["primary"],
        "guard_documents": [relevant_guard, irrelevant_guard],
        "guard_parent_documents": [
            Document(
                page_content="guard-1 complete parent evidence",
                metadata={"dsid": "guard-1", "chunk_id": "guard-1::0"},
            ),
            Document(
                page_content="guard-2 complete parent evidence",
                metadata={"dsid": "guard-2", "chunk_id": "guard-2::0"},
            ),
        ],
        "fusion_guard_history": [
            {
                "candidate_document_ids": ["guard-1", "guard-2"],
                "promoted_document_ids": [],
            }
        ],
    }
    llm = FakeListChatModel(
        responses=[
            '{"status":"SUFFICIENT","relevant_chunk_ids":['
            '"primary::0","guard-1::0"],"missing_evidence":[],'
            '"follow_up_queries":[]}'
        ]
    )

    result = assess_evidence_p0_node(llm, 2, guard_max_promotions=1)(state)

    assert result["selected_document_ids"] == ["primary", "guard-1"]
    assert result["fusion_guard_history"][-1]["promoted_document_ids"] == [
        "guard-1"
    ]
    assert {document.metadata["dsid"] for document in result["answer_docs"]} == {
        "primary",
        "guard-1",
    }
    assert any(
        document.page_content == "guard-1 complete parent evidence"
        for document in result["answer_docs"]
    )


def test_guard_parent_content_is_hidden_until_promotion() -> None:
    primary = Document(
        page_content="primary chunk",
        metadata={"dsid": "primary", "chunk_id": "primary::0"},
    )
    guard_chunk = Document(
        page_content="guard full candidate chunk",
        metadata={"dsid": "guard", "chunk_id": "guard::0"},
    )
    guard_excerpt = Document(
        page_content="guard short excerpt",
        metadata={
            "dsid": "guard",
            "chunk_id": "guard::0",
            "fusion_guard_candidate": True,
        },
    )
    state = {
        "retrieval_round": 1,
        "selected_document_ids": ["primary"],
        "candidate_groups": {
            "primary": [primary],
            "guard": [guard_chunk],
        },
        "guard_documents": [guard_excerpt],
    }
    parent_documents = {
        "primary": [primary],
        "guard": [
            guard_chunk,
            Document(
                page_content="guard hidden parent detail",
                metadata={"dsid": "guard", "chunk_id": "guard::1"},
            ),
        ],
    }

    result = expand_parents_node(
        parent_documents,
        max_parent_chunks=8,
        enabled=True,
        fusion_guard_enabled=True,
    )(state)

    assert [document.page_content for document in result["retrieved_docs"]] == [
        "primary chunk",
        "guard short excerpt",
    ]
    assert any(
        document.page_content == "guard hidden parent detail"
        for document in result["guard_parent_documents"]
    )


def test_hybrid_retriever_preserves_channel_union_and_fusion_trace() -> None:
    dense = StaticRetriever(
        documents=[
            Document(page_content="dense", metadata={"dsid": "a", "chunk_id": "a::0"})
        ]
    )
    bm25 = StaticRetriever(
        documents=[
            Document(page_content="bm25", metadata={"dsid": "b", "chunk_id": "b::0"})
        ]
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        candidate_k=10,
        mode="rrf",
    )

    documents = retriever.invoke("query")
    trace = documents[0].metadata["_retrieval_trace"]

    assert [row["dsid"] for row in trace["stages"]["dense"]] == ["a"]
    assert [row["dsid"] for row in trace["stages"]["bm25"]] == ["b"]
    assert {row["dsid"] for row in trace["stages"]["union"]} == {"a", "b"}
    assert {row["dsid"] for row in trace["stages"]["channel_fusion"]} == {"a", "b"}


def test_dense_only_does_not_call_bm25() -> None:
    class FailingRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager):
            raise AssertionError("disabled channel was invoked")

    dense = StaticRetriever(
        documents=[
            Document(page_content="dense", metadata={"dsid": "a", "chunk_id": "a::0"})
        ]
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=dense,
        bm25_retriever=FailingRetriever(),
        mode="dense",
    )

    assert retriever.invoke("query")[0].metadata["dsid"] == "a"


def test_p0_graph_retries_unknown_then_refuses_with_node_telemetry() -> None:
    document = Document(
        page_content="Ambiguous evidence.",
        metadata={"dsid": "a", "chunk_id": "a::0"},
    )
    retriever = StaticRetriever(documents=[document])
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","requirements":["exact value"]}',
            '{"document_ids":["a"]}',
            "invalid judge",
            "invalid retry",
            '{"document_ids":["a"]}',
            "invalid judge again",
            "invalid retry again",
        ]
    )
    config = {
        "retrieval": {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "max_retrieval_rounds": 2,
        },
        "features": {
            "adaptive_planning": True,
            "llm_rerank": True,
            "parent_expansion": True,
            "evidence_followup": True,
            "answer_repair": True,
        },
    }
    graph = build_rag_graph(
        "p0_candidate",
        retriever,
        llm,
        build_parent_document_store([document]),
        config,
    )

    state = graph.invoke({"question": "What is the exact value?"})

    assert state["retrieval_round"] == 2
    assert state["evidence_status"] == EvidenceStatus.UNKNOWN.value
    assert "insufficient" in state["answer"].lower()
    assert state["answer_repaired"] is False
    assert {event["node"] for event in state["node_metrics"]} >= {
        "plan_question",
        "retrieve_queries",
        "assess_evidence",
        "generate_answer",
    }


def test_guarded_rerank_graph_promotes_a_protected_document() -> None:
    primary = Document(
        page_content="Similar but incomplete evidence.",
        metadata={"dsid": "primary", "chunk_id": "primary::0"},
    )
    protected = Document(
        page_content="The exact limit is 10 MiB.",
        metadata={"dsid": "protected", "chunk_id": "protected::0"},
    )
    retriever = StaticRetriever(documents=[primary, protected])
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","requirements":["exact limit"]}',
            '{"document_ids":["primary"],"coverage":['
            '{"task_id":"r1","document_ids":["primary"]}]}',
            '{"status":"SUFFICIENT","relevant_chunk_ids":['
            '"protected::0"],"missing_evidence":[],"follow_up_queries":[]}',
            "The exact limit is 10 MiB.",
        ]
    )
    config = {
        "retrieval": {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "fusion_guard_top_k": 6,
            "fusion_guard_max_promotions": 2,
            "fusion_guard_chunk_chars": 800,
            "max_retrieval_rounds": 2,
        },
        "features": {
            "adaptive_planning": True,
            "llm_rerank": True,
            "fusion_rank_guard": True,
            "parent_expansion": True,
            "evidence_followup": True,
            "answer_repair": True,
        },
    }
    graph = build_rag_graph(
        "p0_candidate",
        retriever,
        llm,
        build_parent_document_store([primary, protected]),
        config,
    )

    state = graph.invoke({"question": "What is the exact limit?"})

    assert state["evidence_status"] == EvidenceStatus.SUFFICIENT.value
    assert state["selected_document_ids"] == ["primary", "protected"]
    assert state["fusion_guard_history"][-1]["promoted_document_ids"] == [
        "protected"
    ]
    assert state["answer"] == "The exact limit is 10 MiB."
