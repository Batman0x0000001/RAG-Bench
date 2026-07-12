from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from src.graphs.nodes import (
    apply_direct_evidence_guardrail,
    format_candidates,
    format_retrieval_guidance,
    merge_candidate_archives,
    normalize_plan,
    preserve_query_identifiers,
    parse_coverage_document_ids,
    parse_document_ids,
    prioritize_evidence_documents,
    route_after_assessment,
    route_after_generation,
)
from src.graphs.rag_graph import build_adaptive_rag_graph
from src.ingestion.parse_documents import (
    EnterpriseRagLoader,
    read_manifest,
    split_documents,
    write_manifest,
)
from src.indexing.qdrant_index import stable_document_ids
from src.retrieval.vector_retriever import (
    HybridCandidateRetriever,
    SimilarityCandidateRetriever,
    build_parent_document_store,
    expand_selected_documents,
    reciprocal_rank_fuse,
)
from src.retrieval.lexical_retriever import LocalBM25Retriever, tokenize_technical_text
from src.retrieval.entity_links import (
    build_entity_link_index,
    expand_documents_by_entity_links,
    extract_link_entities,
)


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
    chunks = split_documents(loaded)
    description_chunks = [
        document for document in chunks if document.metadata["section"] == "description"
    ]
    assert all(isinstance(document, Document) for document in loaded)
    assert len(description_chunks) >= 2
    assert description_chunks[0].metadata["chunk_id"] == "dsid_test::description::0"
    assert description_chunks[1].metadata["chunk_id"] == "dsid_test::description::1"


def test_loader_keeps_all_declared_content_fields(tmp_path: Path) -> None:
    source = tmp_path / "github" / "declared-fields.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "dataset_doc_uuid": "dsid_declared",
                "title": "Alert review",
                "conversation": ["Warn keeps workspace_id; page removes it."],
                "notes_for_ops": "Private deployments use the server setting.",
                "content_field_names": ["conversation", "notes_for_ops"],
            }
        ),
        encoding="utf-8",
    )

    loaded = EnterpriseRagLoader(source, tmp_path).load()
    by_section = {document.metadata["section"]: document for document in loaded}

    assert "Warn keeps workspace_id" in by_section["discussion"].page_content
    assert "Private deployments" in by_section["text"].page_content
    assert by_section["text"].metadata["field_names"] == ["notes_for_ops"]


def test_manifest_round_trip_uses_document_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    document = Document(
        page_content="content",
        metadata={"dsid": "dsid_test", "chunk_id": "dsid_test::text::0"},
    )
    write_manifest([document], path)
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


def test_similarity_candidate_retriever_uses_configured_k() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.call: dict = {}

        def similarity_search(self, query: str, **kwargs):
            self.call = {"query": query, **kwargs}
            return [Document(page_content="a", metadata={"dsid": "a"})]

    store = FakeVectorStore()
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=store,
        candidate_k=30,
    )
    assert retriever.invoke("query")[0].page_content == "a"
    assert store.call == {"query": "query", "k": 30}


def test_technical_tokenizer_keeps_identifier_and_parts() -> None:
    assert tokenize_technical_text("workspace_id v2.1") == [
        "workspace_id",
        "workspace",
        "id",
        "v2.1",
        "v2",
        "1",
    ]


def test_local_bm25_retrieves_exact_technical_identifier() -> None:
    retriever = LocalBM25Retriever(
        documents=[
            Document(
                page_content="The workspace_id controls tenant isolation.",
                metadata={"dsid": "exact", "chunk_id": "exact::0"},
            ),
            Document(
                page_content="Workspace setup and user permissions.",
                metadata={"dsid": "broad", "chunk_id": "broad::0"},
            ),
        ],
        k=2,
    )

    results = retriever.invoke("workspace_id")

    assert results[0].metadata["dsid"] == "exact"
    assert results[0].metadata["retrieval_channels"] == ["bm25"]


def test_hybrid_retriever_fuses_dense_and_bm25_channels() -> None:
    class StaticRetriever(BaseRetriever):
        documents: list[Document]

        def _get_relevant_documents(self, query: str, *, run_manager):
            return self.documents

    shared = Document(page_content="shared", metadata={"dsid": "a", "chunk_id": "a::0"})
    dense_only = Document(
        page_content="dense", metadata={"dsid": "b", "chunk_id": "b::0"}
    )
    bm25_only = Document(
        page_content="bm25", metadata={"dsid": "c", "chunk_id": "c::0"}
    )
    retriever = HybridCandidateRetriever(
        dense_retriever=StaticRetriever(documents=[dense_only, shared]),
        bm25_retriever=StaticRetriever(documents=[bm25_only, shared]),
        candidate_k=3,
        rrf_k=60,
    )

    results = retriever.invoke("query")

    assert results[0].metadata["dsid"] == "a"
    assert results[0].metadata["retrieval_channels"] == ["dense", "bm25"]
    assert results[0].metadata["retrieval_channel_ranks"] == {
        "dense": 2,
        "bm25": 2,
    }


def test_rrf_rewards_documents_retrieved_by_multiple_queries() -> None:
    a1 = Document(page_content="a1", metadata={"dsid": "a", "chunk_id": "a1"})
    a2 = Document(page_content="a2", metadata={"dsid": "a", "chunk_id": "a2"})
    b = Document(page_content="b", metadata={"dsid": "b", "chunk_id": "b1"})
    c = Document(page_content="c", metadata={"dsid": "c", "chunk_id": "c1"})

    fused = reciprocal_rank_fuse(
        [[b, a1], [c, a2]],
        max_documents=3,
        chunks_per_document=2,
        rrf_k=60,
    )

    assert list(fused)[0] == "a"
    assert [document.page_content for document in fused["a"]] == ["a1", "a2"]
    assert fused["a"][0].metadata["document_rrf_rank"] == 1
    assert fused["a"][0].metadata["query_hit_count"] == 2


def test_rrf_reserves_top_documents_for_each_requirement() -> None:
    first = [
        Document(page_content=f"a{i}", metadata={"dsid": f"a{i}", "chunk_id": f"a{i}"})
        for i in range(3)
    ]
    second = [
        Document(page_content=f"b{i}", metadata={"dsid": f"b{i}", "chunk_id": f"b{i}"})
        for i in range(3)
    ]

    fused = reciprocal_rank_fuse(
        [first, second],
        max_documents=4,
        chunks_per_document=1,
        reserved_documents_per_result=2,
    )

    assert list(fused) == ["a0", "b0", "a1", "b1"]


def test_rrf_does_not_reserve_entity_links_for_completeness() -> None:
    base = [
        Document(
            page_content=f"base{i}",
            metadata={"dsid": f"base{i}", "chunk_id": f"base{i}"},
        )
        for i in range(3)
    ]
    entity = [
        Document(
            page_content=f"entity{i}",
            metadata={
                "dsid": f"entity{i}",
                "chunk_id": f"entity{i}",
                "retrieval_slot": "entity_link",
            },
        )
        for i in range(3)
    ]

    fused = reciprocal_rank_fuse(
        [entity, base],
        max_documents=2,
        chunks_per_document=1,
        reserved_documents_per_result=2,
        reserve_entity_link_results=False,
    )

    assert list(fused) == ["base0", "base1"]


def test_bm25_downweights_generic_text_sections() -> None:
    retriever = LocalBM25Retriever(
        documents=[
            Document(
                page_content="origin verification release index",
                metadata={"dsid": "direct", "section": "description"},
            ),
            Document(
                page_content="origin verification release index",
                metadata={"dsid": "notes", "section": "text"},
            ),
        ],
        k=2,
        text_section_weight=0.8,
    )

    results = retriever.invoke("origin verification release index")

    assert [document.metadata["dsid"] for document in results] == [
        "direct",
        "notes",
    ]


def test_entity_link_expansion_finds_documents_sharing_ticket() -> None:
    seed = Document(
        page_content="Root cause tracked in ENG-1234.",
        metadata={"dsid": "seed", "chunk_id": "seed::0"},
    )
    linked = Document(
        page_content="ENG-1234 rollout and mitigation details.",
        metadata={"dsid": "linked", "chunk_id": "linked::0"},
    )
    unrelated = Document(
        page_content="No shared identifier.",
        metadata={"dsid": "other", "chunk_id": "other::0"},
    )
    parents = build_parent_document_store([seed, linked, unrelated])
    index = build_entity_link_index(parents)

    expanded, trace = expand_documents_by_entity_links(
        [seed],
        parents,
        index,
        max_linked_documents=2,
    )

    assert extract_link_entities(seed.page_content) == {"ENG-1234"}
    assert [document.metadata["dsid"] for document in expanded] == ["seed", "linked"]
    assert expanded[1].metadata["retrieval_channels"] == ["entity_link"]
    assert trace == {"linked": ["ENG-1234"]}
    assert extract_link_entities("PR #157 and pr-157") == {"PR-157"}


def test_coverage_selection_preserves_each_requirement() -> None:
    groups = {
        "noise": [Document(page_content="noise", metadata={"retrieval_task_ids": []})],
        "a": [Document(page_content="a", metadata={"retrieval_task_ids": ["r1"]})],
        "b": [Document(page_content="b", metadata={"retrieval_task_ids": ["r2"]})],
    }
    tasks = [
        {"task_id": "r1", "requirement": "first", "slot": "general", "query": "a"},
        {"task_id": "r2", "requirement": "second", "slot": "general", "query": "b"},
    ]

    selected = parse_coverage_document_ids(
        '{"document_ids":["noise"]}',
        groups,
        tasks,
        strategy="multi_document",
        source_scope="multiple_sources",
        budget=2,
        minimum=2,
    )

    assert set(selected) == {"a", "b"}


def test_coverage_selection_reuses_one_document_for_all_requirements() -> None:
    groups = {
        "complete": [
            Document(
                page_content="covers both states",
                metadata={"retrieval_task_ids": ["r1", "r2"]},
            )
        ],
        "extra": [Document(page_content="analogous behavior", metadata={})],
    }
    tasks = [
        {"task_id": "r1", "requirement": "null", "slot": "general", "query": "null"},
        {"task_id": "r2", "requirement": "omitted", "slot": "general", "query": "omitted"},
    ]

    selected = parse_coverage_document_ids(
        '{"document_ids":["complete","extra"],"coverage":['
        '{"task_id":"r1","document_ids":["complete"]},'
        '{"task_id":"r2","document_ids":["complete"]}]}',
        groups,
        tasks,
        strategy="multi_document",
        source_scope="multiple_sources",
        budget=2,
        minimum=2,
    )

    assert selected == ["complete"]


def test_candidate_archive_keeps_previous_round_documents() -> None:
    previous = {
        "gold": [Document(page_content="gold", metadata={"chunk_id": "gold::0"})]
    }
    current = {
        "new": [Document(page_content="new", metadata={"chunk_id": "new::0"})],
    }

    merged = merge_candidate_archives(previous, current, max_documents=2)

    assert list(merged) == ["gold", "new"]


def test_candidate_format_prioritizes_direct_time_evidence() -> None:
    overview = Document(
        page_content="General batching migration guidance.",
        metadata={
            "dsid": "a",
            "document_rrf_rank": 1,
            "query_hit_count": 2,
            "retrieval_channels": ["dense"],
            "retrieval_task_ids": ["r1"],
        },
    )
    direct = Document(
        page_content="Dedicated defaults to 10ms and Hosted defaults to 5ms.",
        metadata={
            "dsid": "a",
            "document_rrf_rank": 1,
            "query_hit_count": 2,
            "retrieval_channels": ["bm25"],
            "retrieval_task_ids": ["r1"],
        },
    )

    formatted = format_candidates(
        {"a": [overview, direct]},
        chunk_chars=800,
        question="What is the default wait time?",
        requirements=["Compare Dedicated and Hosted"],
    )

    assert formatted.index("10ms") < formatted.index("General batching")
    assert "Retrieval channels: bm25, dense" in formatted
    assert "Task matches: r1" in formatted


def test_direct_evidence_guardrail_recovers_time_value() -> None:
    groups = {
        "gold": [Document(page_content="Dedicated 10ms; Hosted 5ms.")],
        "wrong": [Document(page_content="Migration from max_wait_ms.")],
    }

    selected, reason = apply_direct_evidence_guardrail(
        "What is the default wait time for each tier?",
        strategy="single",
        budget=1,
        groups=groups,
        selected_ids=["wrong"],
    )

    assert selected == ["gold"]
    assert reason == "direct_evidence:time_value"


def test_time_limit_context_does_not_request_a_time_value() -> None:
    groups = {
        "gold": [Document(page_content="Added metric stream.timebox_finalized.")],
        "wrong": [Document(page_content="The default timebox is 30 seconds.")],
    }

    selected, reason = apply_direct_evidence_guardrail(
        "What is the metric name for sessions finalized due to the time limit?",
        strategy="single",
        budget=1,
        groups=groups,
        selected_ids=["gold"],
    )

    assert selected == ["gold"]
    assert reason is None


def test_direct_evidence_guardrail_recovers_three_named_modes() -> None:
    groups = {
        "selected": [Document(page_content="Preferred precision: fp32, bf16, int8.")],
        "unrelated": [
            Document(
                page_content=(
                    "KVPrefetchAdapter supports three modes: exact-layout, "
                    "dequant-inline, and universal."
                )
            )
        ],
        "gold": [
            Document(
                page_content=(
                    "The runtime precision toggle has three modes: strict (fp32), "
                    "balanced (fp16 with validators), and aggressive (fp16/int8 "
                    "once warmed)."
                )
            )
        ],
    }

    selected, reason = apply_direct_evidence_guardrail(
        "What are the three runtime precision settings and what does each allow?",
        strategy="single",
        budget=1,
        groups=groups,
        selected_ids=["selected"],
    )

    assert selected == ["gold"]
    assert reason == "direct_evidence:three_item_enumeration"


def test_retrieval_guidance_drops_overridden_llm_reason() -> None:
    guidance = format_retrieval_guidance(
        {
            "rerank_history": [
                {
                    "selection_reason": "The wrong tolerant-schema document is best.",
                    "guardrail_reason": "direct_evidence:http_status",
                }
            ]
        }
    )

    assert "wrong tolerant-schema" not in guidance
    assert "http_status" in guidance


def test_retrieval_guidance_keeps_unoverridden_reason() -> None:
    guidance = format_retrieval_guidance(
        {
            "rerank_history": [
                {
                    "selection_reason": "Both omitted and unset values use defaults.",
                    "guardrail_reason": None,
                }
            ]
        }
    )

    assert guidance == "Both omitted and unset values use defaults."


def test_direct_evidence_guardrail_requires_http_context() -> None:
    groups = {
        "gold": [Document(page_content="We return a 422 structured error payload.")],
        "wrong": [Document(page_content="612 additions and 174 deletions.")],
    }

    selected, reason = apply_direct_evidence_guardrail(
        "What HTTP status does schema validation return?",
        strategy="single",
        budget=1,
        groups=groups,
        selected_ids=["wrong"],
    )

    assert selected == ["gold"]
    assert reason == "direct_evidence:http_status"


def test_direct_evidence_guardrail_does_not_override_semantic_strategy() -> None:
    groups = {
        "gold": [Document(page_content="We return HTTP status 422.")],
        "selected": [Document(page_content="Schema validation behavior.")],
    }

    selected, reason = apply_direct_evidence_guardrail(
        "What HTTP status does validation return?",
        strategy="semantic",
        budget=1,
        groups=groups,
        selected_ids=["selected"],
    )

    assert selected == ["selected"]
    assert reason is None


def test_normalize_plan_applies_adaptive_budget_and_keeps_original_query() -> None:
    plan = normalize_plan(
        {
            "strategy": "completeness",
            "queries": ["subquery one", "subquery two"],
            "document_budget": 2,
            "requirements": ["all rollout artifacts"],
        },
        question="original question",
        max_queries=5,
        max_documents=10,
    )

    assert plan["queries"][0] == "original question"
    assert plan["document_budget"] == 10
    assert plan["minimum_documents"] == 6

    single = normalize_plan(
        {
            "strategy": "single",
            "queries": ["unneeded rewrite"],
            "document_budget": 3,
        },
        question="exact lookup",
        max_queries=5,
        max_documents=10,
    )
    assert single["queries"] == ["exact lookup"]
    assert single["document_budget"] == 1


def test_normalize_plan_builds_requirement_tasks_and_conflict_slots() -> None:
    completeness = normalize_plan(
        {
            "strategy": "multi_document",
            "document_budget": 6,
            "requirements": ["Go tickets", "Python tickets", "TypeScript tickets"],
            "retrieval_tasks": [
                {"requirement": "Go tickets", "slot": "go", "query": "Go auth tickets"},
                {
                    "requirement": "Python tickets",
                    "slot": "python",
                    "query": "Python auth tickets",
                },
                {
                    "requirement": "TypeScript tickets",
                    "slot": "typescript",
                    "query": "TypeScript auth tickets",
                },
            ],
        },
        question="Across all SDKs, which has the highest number and all corresponding tickets?",
        max_queries=10,
        max_documents=10,
    )

    assert completeness["strategy"] == "completeness"
    assert [task["task_id"] for task in completeness["retrieval_tasks"]] == [
        "r1",
        "r2",
        "r3",
    ]
    assert completeness["document_budget"] == 10

    conflicting = normalize_plan(
        {"strategy": "single", "requirements": ["compare behavior"]},
        question="What are the previous and current thresholds?",
        max_queries=10,
        max_documents=10,
    )

    assert conflicting["strategy"] == "conflicting"
    assert [task["slot"] for task in conflicting["retrieval_tasks"]] == [
        "previous",
        "current",
    ]

    same_document = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "single_source",
            "requirements": ["file limit", "request limit"],
            "retrieval_tasks": [
                {"requirement": "file limit", "query": "file limit"},
                {"requirement": "request limit", "query": "request limit"},
            ],
        },
        question="What are the file and request limits for multipart upload?",
        max_queries=10,
        max_documents=10,
    )

    assert same_document["strategy"] == "single"
    assert same_document["document_budget"] == 1
    assert same_document["queries"] == [
        "What are the file and request limits for multipart upload?"
    ]

    parameter_states = normalize_plan(
        {
            "strategy": "conflicting",
            "source_scope": "multiple_sources",
            "requirements": ["explicit null", "omitted field"],
            "retrieval_tasks": [
                {"requirement": "explicit null", "query": "max_tokens null"},
                {"requirement": "omitted field", "query": "max_tokens omitted"},
            ],
        },
        question=(
            "How does the normalizer handle max_tokens as null compared to leaving it "
            "out entirely?"
        ),
        max_queries=10,
        max_documents=10,
    )

    assert parameter_states["strategy"] == "single"
    assert parameter_states["source_scope"] == "single_source"
    assert parameter_states["document_budget"] == 1

    release_components = normalize_plan(
        {
            "strategy": "multi_document",
            "source_scope": "multiple_sources",
            "requirements": ["config flag", "other release components"],
        },
        question=(
            "What config flag enables the TP fanout planner, and what other new "
            "components are called out in the release notes?"
        ),
        max_queries=10,
        max_documents=10,
    )

    assert release_components["strategy"] == "single"
    assert release_components["source_scope"] == "single_source"
    assert release_components["document_budget"] == 1


def test_follow_up_query_preserves_exact_question_identifiers() -> None:
    query = preserve_query_identifiers(
        "fanout planner configuration setting",
        "Which flag controls TP allreduce in runtime.bandwidth_fanout.enabled?",
    )

    assert query.endswith("TP runtime.bandwidth_fanout.enabled")


def test_parse_document_ids_filters_unknown_and_fills_minimum() -> None:
    selected = parse_document_ids(
        '{"document_ids":["b","unknown"]}',
        available_ids=["a", "b", "c"],
        budget=3,
        minimum=2,
    )
    assert selected == ["b", "a"]


def test_expand_selected_documents_adds_missing_parent_sections() -> None:
    description = Document(
        page_content="description",
        metadata={"dsid": "a", "chunk_id": "a::description::0"},
    )
    discussion = Document(
        page_content="answer in discussion",
        metadata={"dsid": "a", "chunk_id": "a::discussion::0"},
    )
    store = build_parent_document_store([description, discussion])
    selected = expand_selected_documents(
        ["a"],
        {"a": [description]},
        parent_documents=store,
        expanded_documents=1,
        max_parent_chunks=8,
    )
    assert [document.page_content for document in selected] == [
        "description",
        "answer in discussion",
    ]


def test_evidence_route_is_bounded() -> None:
    state = {
        "evidence_sufficient": False,
        "can_retry": True,
        "pending_queries": ["follow up"],
        "retrieval_round": 1,
    }
    assert route_after_assessment(state, max_retrieval_rounds=2) == "retrieve_queries"
    state["retrieval_round"] = 2
    assert route_after_assessment(state, max_retrieval_rounds=2) == "generate_answer"


def test_evidence_prioritization_keeps_parent_context() -> None:
    first = Document(page_content="first", metadata={"chunk_id": "a"})
    second = Document(page_content="second", metadata={"chunk_id": "b"})

    prioritized = prioritize_evidence_documents([first, second], ["b"])

    assert [document.page_content for document in prioritized] == ["second", "first"]


def test_langchain_components_run_inside_unified_langgraph() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            return [
                Document(
                    page_content="The limit is 10 MiB.",
                    metadata={
                        "dsid": "a",
                        "chunk_id": "a::description::0",
                        "title": "Upload limit",
                    },
                )
            ]

    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","queries":[],"document_budget":1,"requirements":["limit"]}',
            '{"document_ids":["a"]}',
            '{"sufficient":true,"relevant_chunk_ids":["a::description::0"],"missing_evidence":[],"follow_up_queries":[]}',
            "The limit is 10 MiB.",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        build_parent_document_store(retriever.invoke("seed")),
        {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "max_retrieval_rounds": 2,
        },
    )

    state = graph.invoke({"question": "What is the upload limit?"})

    assert state["answer"] == "The limit is 10 MiB."
    assert state["document_ids"] == ["a"]
    assert state["executed_queries"] == ["What is the upload limit?"]


def test_graph_repairs_answer_that_conflicts_with_sufficient_evidence() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            return [
                Document(
                    page_content=(
                        "Explicit null is unset and falls back to model defaults."
                    ),
                    metadata={
                        "dsid": "a",
                        "chunk_id": "a::discussion::0",
                        "title": "Parameter normalization",
                    },
                )
            ]

    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"single","source_scope":"single_source",'
            '"requirements":["null and omission behavior"]}',
            '{"document_ids":["a"]}',
            '{"sufficient":true,"relevant_chunk_ids":["a::discussion::0"],'
            '"missing_evidence":[],"follow_up_queries":[]}',
            "The provided context does not specify omission behavior.",
            "Null and omission are both unset and fall back to model defaults.",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        build_parent_document_store(retriever.invoke("seed")),
        {"max_queries": 5, "max_documents": 10, "candidate_documents": 10},
    )

    state = graph.invoke({"question": "How do null and omission differ?"})

    assert route_after_generation(
        {"evidence_sufficient": True, "answer": "The context does not specify it."}
    ) == "repair_answer"
    assert state["original_answer"] == (
        "The provided context does not specify omission behavior."
    )
    assert state["answer"] == (
        "Null and omission are both unset and fall back to model defaults."
    )
    assert state["answer_repaired"] is True


def test_langgraph_expands_linked_entities_between_retrieval_and_rerank() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            return [
                Document(
                    page_content="Incident references ENG-1234.",
                    metadata={"dsid": "seed", "chunk_id": "seed::0", "title": "Seed"},
                )
            ]

    seed = Document(
        page_content="Incident references ENG-1234.",
        metadata={"dsid": "seed", "chunk_id": "seed::0", "title": "Seed"},
    )
    linked = Document(
        page_content="ENG-1234 contains the rollout answer.",
        metadata={"dsid": "linked", "chunk_id": "linked::0", "title": "Linked"},
    )
    parents = build_parent_document_store([seed, linked])
    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"multi_document","source_scope":"multiple_sources",'
            '"document_budget":2,"requirements":["rollout"],'
            '"retrieval_tasks":[{"requirement":"rollout","slot":"general",'
            '"query":"ENG-1234 rollout"}]}',
            '{"document_ids":["linked"],"coverage":['
            '{"task_id":"r1","document_ids":["linked"]}]}',
            '{"sufficient":true,"relevant_chunk_ids":["linked::0"],'
            '"missing_evidence":[],"follow_up_queries":[]}',
            "linked answer",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        parents,
        {"max_queries": 10, "candidate_documents": 10},
        entity_index=build_entity_link_index(parents),
    )

    state = graph.invoke({"question": "What was the rollout?"})

    assert state["document_ids"][0] == "linked"
    assert state["entity_expansions"][0]["linked_documents"] == {
        "linked": ["ENG-1234"]
    }
    assert state["executed_tasks"][0]["task_id"] == "r1"
    assert [document.metadata["dsid"] for document in state["query_results"][0]] == [
        "seed"
    ]
    assert state["query_results"][1][0].metadata["dsid"] == "linked"
    assert state["answer"] == "linked answer"


def test_unified_graph_runs_bounded_follow_up_retrieval() -> None:
    class FakeVectorStore:
        def similarity_search(self, query: str, **kwargs):
            dsid = "b" if query == "follow up detail" else "a"
            return [
                Document(
                    page_content=f"evidence {dsid}",
                    metadata={
                        "dsid": dsid,
                        "chunk_id": f"{dsid}::description::0",
                        "title": dsid,
                    },
                )
            ]

    retriever = SimilarityCandidateRetriever.model_construct(
        vector_store=FakeVectorStore(),
        candidate_k=10,
    )
    parent_documents = build_parent_document_store(
        retriever.batch(["original", "follow up detail"])
        [0]
        + retriever.batch(["follow up detail"])[0]
    )
    llm = FakeListChatModel(
        responses=[
            '{"strategy":"semantic","queries":["alternate wording"],'
            '"document_budget":2,"requirements":["detail"]}',
            '{"document_ids":["a"]}',
            '{"sufficient":false,"relevant_chunk_ids":["a::description::0"],'
            '"missing_evidence":["detail"],"follow_up_queries":["follow up detail"]}',
            '{"document_ids":["b","a"]}',
            '{"sufficient":true,"relevant_chunk_ids":["b::description::0"],'
            '"missing_evidence":[],"follow_up_queries":[]}',
            "combined answer",
        ]
    )
    graph = build_adaptive_rag_graph(
        retriever,
        llm,
        parent_documents,
        {
            "max_queries": 5,
            "max_documents": 10,
            "candidate_documents": 10,
            "max_retrieval_rounds": 2,
        },
    )

    state = graph.invoke({"question": "original"})

    assert state["answer"] == "combined answer"
    assert state["retrieval_round"] == 2
    assert state["executed_queries"] == [
        "original",
        "alternate wording",
        "follow up detail",
    ]
    assert state["document_ids"] == ["b", "a"]
