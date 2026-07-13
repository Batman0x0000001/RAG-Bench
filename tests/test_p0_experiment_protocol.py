from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.experiment_protocol import (
    aggregate_official_repetitions,
    build_question_metrics,
    build_judge_confusion_matrix,
    build_recall_funnel,
    build_run_manifest,
    prepare_blind_dataset,
    summarize_question_metrics,
    summarize_repetitions,
)
from src.evaluation.experiment_config import (
    ABLATION_VARIANTS,
    DatasetSpec,
    apply_experiment_variant,
    changed_variant_paths,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_blind_dataset_excludes_all_gold_fields(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions,
        [
            {
                "question_id": "q1",
                "question": "What changed?",
                "question_type": "basic",
                "source_types": ["confluence"],
                "gold_answer": "secret",
                "expected_doc_ids": ["gold-doc"],
            },
            {
                "question_id": "q2",
                "question": "Ignored",
                "source_types": ["github"],
                "gold_answer": "other",
            },
        ],
    )
    blind_file = tmp_path / "frozen" / "questions.blind.jsonl"
    dataset = DatasetSpec(
        name="test_frozen",
        source_type="confluence",
        expected_questions=1,
        role="frozen_test",
        blind_questions_file=str(blind_file),
        gold_questions_file=str(questions),
        manifest_file="unused.jsonl",
        qdrant_collection="unused",
    )

    manifest = prepare_blind_dataset(questions, dataset)
    blind = json.loads(blind_file.read_text(encoding="utf-8"))

    assert blind == {
        "question_id": "q1",
        "question": "What changed?",
        "question_type": "basic",
        "source_types": ["confluence"],
    }
    assert manifest["gold_is_excluded"] is True


def test_frozen_dataset_cannot_be_silently_replaced(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions,
        [{"question_id": "q1", "question": "one", "source_types": ["confluence"]}],
    )
    dataset = DatasetSpec(
        "frozen",
        "confluence",
        1,
        "frozen_test",
        str(tmp_path / "blind.jsonl"),
        str(questions),
        "unused",
        "unused",
    )
    prepare_blind_dataset(questions, dataset)
    _write_jsonl(
        questions,
        [{"question_id": "q1", "question": "changed", "source_types": ["confluence"]}],
    )

    with pytest.raises(RuntimeError, match="different content"):
        prepare_blind_dataset(questions, dataset)


def test_every_registered_ablation_changes_at_most_one_factor() -> None:
    for name in ABLATION_VARIANTS:
        changed = changed_variant_paths(name)
        if name in {"p0_full", "hybrid_rrf"}:
            assert changed == set()
        else:
            assert len(changed) == 1


def test_variant_application_does_not_mutate_base_config() -> None:
    base = {"retrieval": {}, "features": {}}
    changed = apply_experiment_variant(base, "no_rerank")

    assert base == {"retrieval": {}, "features": {}}
    assert changed["features"]["llm_rerank"] is False
    assert changed["retrieval"]["mode"] == "rrf"


def test_guarded_rerank_changes_only_the_guard_switch() -> None:
    base = {"retrieval": {}, "features": {}}

    changed = apply_experiment_variant(base, "guarded_rerank")

    assert changed_variant_paths("guarded_rerank") == {
        "features.fusion_rank_guard"
    }
    assert changed["features"]["llm_rerank"] is True
    assert changed["features"]["fusion_rank_guard"] is True


def test_run_manifest_redacts_secrets_and_hashes_inputs(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    documents = tmp_path / "documents.jsonl"
    questions.write_text("{}\n", encoding="utf-8")
    documents.write_text("{}\n", encoding="utf-8")
    config = {
        "data": {
            "questions_file": str(questions),
            "manifest_file": str(documents),
        },
        "llm": {"api_key": "never-write-this", "model": "fake"},
    }

    first = build_run_manifest(config, project_root=tmp_path, prompt_texts={"p": "same"})
    second = build_run_manifest(config, project_root=tmp_path, prompt_texts={"p": "same"})

    assert first["config"]["llm"]["api_key"] == "<redacted>"
    assert first["config_sha256"] == second["config_sha256"]
    assert first["prompt_hashes"] == second["prompt_hashes"]
    assert first["file_hashes"] == second["file_hashes"]


def test_question_metrics_keep_unknown_cost_as_null() -> None:
    state = {
        "model_calls": [
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 50, "output_tokens": 10},
        ],
        "node_metrics": [{"node": "plan", "duration_ms": 12.5}],
        "retrieval_round": 2,
    }
    row = build_question_metrics("q1", state, {})

    assert row["total_tokens"] == 180
    assert row["cost"] is None
    assert row["followup_used"] is True
    assert summarize_question_metrics([row])["duration_ms"]["p95"] == 12.5


def test_repetition_summary_reports_mean_std_and_ci() -> None:
    summary = summarize_repetitions([80.0, 82.0, 84.0])

    assert summary["count"] == 3
    assert summary["mean"] == 82.0
    assert summary["std"] == 2.0
    assert summary["ci95_low"] < 82.0 < summary["ci95_high"]


def test_official_aggregation_requires_three_runs_and_tracks_flips(tmp_path: Path) -> None:
    paths = []
    outcomes = (True, False, True)
    for index, outcome in enumerate(outcomes):
        path = tmp_path / f"result-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question_id": "q1",
                            "answer_correct": outcome,
                            "correctness_pct": 100.0 if outcome else 0.0,
                            "completeness_pct": 100.0,
                            "document_recall_pct": 100.0,
                            "invalid_extra_docs": 0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    with pytest.raises(ValueError, match="three repetitions"):
        aggregate_official_repetitions(paths[:2])
    report = aggregate_official_repetitions(paths)

    assert report["unstable_questions"] == ["q1"]
    assert report["metrics"]["correctness_pct"]["count"] == 3


def test_recall_funnel_supports_multi_gold_and_rerank_attribution() -> None:
    questions = [
        {
            "question_id": "q1",
            "expected_doc_ids": ["gold-a", "gold-b"],
        }
    ]
    traces = [
        {
            "question_id": "q1",
            "retrieval_stage_history": [
                {
                    "stages": {
                        "dense": [{"dsid": "gold-a"}],
                        "bm25": [{"dsid": "gold-b"}],
                        "union": [{"dsid": "gold-a"}, {"dsid": "gold-b"}],
                        "channel_fusion": [
                            {"dsid": "gold-a"},
                            {"dsid": "gold-b"},
                        ],
                    }
                }
            ],
            "rerank_history": [
                {
                    "candidate_document_ids": ["gold-a", "gold-b"],
                    "selected_document_ids": ["gold-a"],
                }
            ],
            "selected_document_ids": ["gold-a"],
        }
    ]
    official = {"questions": [{"question_id": "q1", "answer_correct": False}]}

    report = build_recall_funnel(questions, traces, official)

    assert report["overall"]["union"]["recall"] == 1.0
    assert report["overall"]["rerank"]["recall"] == 0.5
    assert report["questions"][0]["primary_failure_cause"] == "rerank_elimination"


def test_recall_funnel_records_guard_recovery() -> None:
    questions = [{"question_id": "q1", "expected_doc_ids": ["gold"]}]
    traces = [
        {
            "question_id": "q1",
            "rerank_history": [
                {
                    "candidate_document_ids": ["noise", "gold"],
                    "selected_document_ids": ["noise"],
                }
            ],
            "fusion_guard_history": [
                {
                    "candidate_document_ids": ["gold"],
                    "promoted_document_ids": ["gold"],
                }
            ],
            "selected_document_ids": ["noise", "gold"],
        }
    ]
    official = {"questions": [{"question_id": "q1", "answer_correct": True}]}

    report = build_recall_funnel(questions, traces, official)

    question = report["questions"][0]
    assert question["metrics"]["rerank"]["hit"] == 0.0
    assert question["metrics"]["guard_promoted"]["hit"] == 1.0
    assert question["guard_recovered"] is True
    assert question["primary_failure_cause"] == "success_or_unscored"


def test_judge_confusion_matrix_reports_precision_and_recall() -> None:
    annotations = [
        {"question_id": "q1", "expected_status": "SUFFICIENT"},
        {"question_id": "q2", "expected_status": "INSUFFICIENT"},
        {"question_id": "q3", "expected_status": "UNKNOWN"},
    ]
    traces = [
        {"question_id": "q1", "evidence_status": "SUFFICIENT"},
        {"question_id": "q2", "evidence_status": "UNKNOWN"},
        {"question_id": "q3", "evidence_status": "UNKNOWN"},
    ]

    report = build_judge_confusion_matrix(annotations, traces)

    assert report["accuracy"] == 0.6667
    assert report["confusion_matrix"]["INSUFFICIENT"]["UNKNOWN"] == 1
    assert report["per_status"]["UNKNOWN"]["precision"] == 0.5
