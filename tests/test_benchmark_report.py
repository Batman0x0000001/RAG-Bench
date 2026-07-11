from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.benchmark_report import (
    build_failed_questions,
    build_supplementary_metrics,
    write_question_subset,
)


def test_question_subset_only_contains_requested_source(tmp_path: Path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "q1", "source_types": ["github"]}),
                json.dumps({"question_id": "q2", "source_types": ["slack"]}),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "github.jsonl"

    rows = write_question_subset(source, output, "github")

    assert [row["question_id"] for row in rows] == ["q1"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_question_subset_can_be_limited_to_answered_ids(tmp_path: Path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "q1", "source_types": ["github"]}),
                json.dumps({"question_id": "q2", "source_types": ["github"]}),
            ]
        ),
        encoding="utf-8",
    )

    rows = write_question_subset(
        source,
        tmp_path / "answered.jsonl",
        "github",
        question_ids={"q2"},
    )

    assert [row["question_id"] for row in rows] == ["q2"]


def test_supplementary_metrics_use_unique_document_rank() -> None:
    questions = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "expected_doc_ids": ["d1", "d2"],
        }
    ]
    retrieved = [
        {
            "question_id": "q1",
            "documents": [
                {"dsid": "d1"},
                {"dsid": "d1"},
                {"dsid": "noise"},
                {"dsid": "d2"},
            ],
        }
    ]

    result = build_supplementary_metrics(questions, retrieved, k_values=(1, 5))

    assert result["average_unique_documents"] == 3.0
    assert result["overall"]["recall_at_1"] == 0.5
    assert result["overall"]["recall_at_5"] == 1.0
    assert result["overall"]["mrr_at_5"] == 1.0


def test_failed_question_contains_engineering_diagnosis() -> None:
    questions = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "question": "question",
            "gold_answer": "gold",
            "expected_doc_ids": ["d1"],
        }
    ]
    answers = [{"question_id": "q1", "answer": "candidate"}]
    retrieved = [{"question_id": "q1", "documents": [{"dsid": "noise"}]}]
    official = {
        "questions": [
            {
                "question_id": "q1",
                "answer_correct": False,
                "correctness_reasoning": "missing evidence",
                "completeness_pct": 0.0,
                "document_recall_pct": 0.0,
                "invalid_extra_docs": 1,
            }
        ]
    }

    failures = build_failed_questions(questions, answers, retrieved, official)

    assert failures[0]["missing_expected_doc_ids"] == ["d1"]
    assert failures[0]["failure_tags"] == [
        "retrieval_miss",
        "incomplete_answer",
        "retrieval_noise",
    ]
