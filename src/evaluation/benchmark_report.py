from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_K_VALUES = (1, 5, 10, 20)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def matches_source_type(
    row: dict[str, Any],
    source_type: str | None,
    include_mixed_sources: bool = False,
) -> bool:
    if source_type is None:
        return True
    source_types = row.get("source_types", [])
    if include_mixed_sources:
        return source_type in source_types
    return source_types == [source_type]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_question_subset(
    questions_file: str | Path,
    output_file: str | Path,
    source_type: str,
    question_ids: set[str] | None = None,
    include_mixed_sources: bool = False,
) -> list[dict[str, Any]]:
    questions = [
        row
        for row in load_jsonl(questions_file)
        if matches_source_type(row, source_type, include_mixed_sources)
        and (question_ids is None or row.get("question_id") in question_ids)
    ]
    write_jsonl(questions, output_file)
    return questions


def _unique_document_ids(retrieved_row: dict[str, Any] | None) -> list[str]:
    unique: list[str] = []
    for document in (retrieved_row or {}).get("documents", []):
        dsid = document.get("dsid")
        if dsid and dsid not in unique:
            unique.append(str(dsid))
    return unique


def _question_retrieval_metrics(
    expected_ids: list[str],
    retrieved_ids: list[str],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    expected = set(expected_ids)
    metrics: dict[str, float] = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        relevant = sum(document_id in expected for document_id in top_k)
        metrics[f"hit_at_{k}"] = float(relevant > 0)
        metrics[f"recall_at_{k}"] = relevant / len(expected)
        metrics[f"precision_at_{k}"] = relevant / k

        first_rank = next(
            (rank for rank, document_id in enumerate(top_k, start=1) if document_id in expected),
            None,
        )
        metrics[f"mrr_at_{k}"] = 1.0 / first_rank if first_rank else 0.0

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, document_id in enumerate(top_k, start=1)
            if document_id in expected
        )
        ideal_count = min(len(expected), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
    return metrics


def _average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 4)
        for key in rows[0]
    }


def build_supplementary_metrics(
    questions: list[dict[str, Any]],
    retrieved_rows: list[dict[str, Any]],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    retrieved_by_id = {row["question_id"]: row for row in retrieved_rows}
    metric_rows: list[dict[str, float]] = []
    metrics_by_type: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    unique_document_counts: list[int] = []

    for question in questions:
        expected_ids = list(dict.fromkeys(question.get("expected_doc_ids") or []))
        retrieved_ids = _unique_document_ids(retrieved_by_id.get(question["question_id"]))
        unique_document_counts.append(len(retrieved_ids))
        if not expected_ids:
            continue
        metrics = _question_retrieval_metrics(expected_ids, retrieved_ids, k_values)
        metric_rows.append(metrics)
        metrics_by_type[str(question.get("question_type", "unknown"))].append(metrics)

    return {
        "question_count": len(questions),
        "questions_with_gold_documents": len(metric_rows),
        "average_unique_documents": round(
            sum(unique_document_counts) / len(unique_document_counts), 2
        )
        if unique_document_counts
        else 0.0,
        "overall": _average_metrics(metric_rows),
        "by_question_type": {
            question_type: {
                "count": len(rows),
                **_average_metrics(rows),
            }
            for question_type, rows in sorted(metrics_by_type.items())
        },
    }


def build_failed_questions(
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    retrieved_rows: list[dict[str, Any]],
    official_results: dict[str, Any],
) -> list[dict[str, Any]]:
    questions_by_id = {row["question_id"]: row for row in questions}
    answers_by_id = {row["question_id"]: row for row in answers}
    retrieved_by_id = {row["question_id"]: row for row in retrieved_rows}
    failures: list[dict[str, Any]] = []

    for result in official_results.get("questions", []):
        if result.get("answer_correct"):
            continue
        question_id = result["question_id"]
        question = questions_by_id[question_id]
        answer = answers_by_id.get(question_id, {})
        expected_ids = list(dict.fromkeys(question.get("expected_doc_ids") or []))
        retrieved_ids = _unique_document_ids(retrieved_by_id.get(question_id))
        expected_set = set(expected_ids)
        retrieved_set = set(retrieved_ids)
        missing_ids = [document_id for document_id in expected_ids if document_id not in retrieved_set]
        unexpected_ids = [document_id for document_id in retrieved_ids if document_id not in expected_set]
        gold_best_rank = next(
            (
                rank
                for rank, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected_set
            ),
            None,
        )

        tags: list[str] = []
        recall = result.get("document_recall_pct")
        if expected_ids and (recall is None or recall < 100):
            tags.append("retrieval_miss")
        if expected_ids and recall == 100:
            tags.append("generation_failure")
        if result.get("completeness_pct", 100) < 100:
            tags.append("incomplete_answer")
        if result.get("invalid_extra_docs", 0) > 0:
            tags.append("retrieval_noise")
        if gold_best_rank is not None and gold_best_rank > 10:
            tags.append("low_rank_gold")
        if not expected_ids:
            tags.append("unanswerable_failure")

        failures.append(
            {
                "question_id": question_id,
                "question_type": question.get("question_type"),
                "question": question.get("question"),
                "gold_answer": question.get("gold_answer"),
                "candidate_answer": answer.get("answer", ""),
                "answer_correct": False,
                "correctness_reasoning": result.get("correctness_reasoning", ""),
                "completeness_pct": result.get("completeness_pct"),
                "document_recall_pct": recall,
                "invalid_extra_docs": result.get("invalid_extra_docs"),
                "expected_doc_ids": expected_ids,
                "retrieved_document_ids": retrieved_ids,
                "missing_expected_doc_ids": missing_ids,
                "unexpected_document_ids": unexpected_ids,
                "gold_best_rank": gold_best_rank,
                "failure_tags": tags,
            }
        )
    return failures


def write_evaluation_outputs(
    questions_file: str | Path,
    answers_file: str | Path,
    retrieved_file: str | Path,
    official_results_file: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions = load_jsonl(questions_file)
    answers = load_jsonl(answers_file)
    retrieved_rows = load_jsonl(retrieved_file)
    with Path(official_results_file).open("r", encoding="utf-8-sig") as file:
        official_results = json.load(file)

    supplementary = build_supplementary_metrics(questions, retrieved_rows)
    failures = build_failed_questions(questions, answers, retrieved_rows, official_results)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "supplementary_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(supplementary, file, ensure_ascii=False, indent=2)
    write_jsonl(failures, target / "failed_questions.jsonl")
    return supplementary, failures
