from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.benchmark_report import load_jsonl, matches_source_type, write_jsonl
from src.evaluation.experiment_config import DatasetSpec


BLIND_QUESTION_FIELDS = ("question_id", "question", "question_type", "source_types")
TRACE_SCHEMA_VERSION = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_blind_dataset(
    questions_file: str | Path,
    dataset: DatasetSpec,
) -> dict[str, Any]:
    """生成不含 Gold 的运行文件；冻结集已存在时禁止静默改写。"""
    selected = [
        row
        for row in load_jsonl(questions_file)
        if matches_source_type(row, dataset.source_type)
    ]
    if len(selected) != dataset.expected_questions:
        raise ValueError(
            f"{dataset.name} expected {dataset.expected_questions} questions, "
            f"found {len(selected)}"
        )
    blind_rows = [
        {field: row.get(field) for field in BLIND_QUESTION_FIELDS if field in row}
        for row in selected
    ]
    blind_path = Path(dataset.blind_questions_file)
    serialized = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows
    )
    if blind_path.exists() and blind_path.read_text(encoding="utf-8") != serialized:
        if dataset.role == "frozen_test":
            raise RuntimeError(
                f"Frozen dataset {dataset.name} already exists with different content"
            )
    else:
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        blind_path.write_text(serialized, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "dataset": dataset.name,
        "role": dataset.role,
        "source_type": dataset.source_type,
        "question_count": len(blind_rows),
        "blind_questions_file": str(blind_path),
        "blind_questions_sha256": sha256_file(blind_path),
        "gold_is_excluded": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = blind_path.parent / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _redact_secrets(value: Any, key: str = "") -> Any:
    if key.lower() in {"api_key", "authorization", "password", "secret"}:
        return "<redacted>" if value else value
    if isinstance(value, dict):
        return {name: _redact_secrets(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_manifest(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    prompt_texts: dict[str, str],
) -> dict[str, Any]:
    root = Path(project_root)
    data_config = config.get("data", {})
    tracked_files = {
        "questions": data_config.get("questions_file"),
        "documents": data_config.get("manifest_file"),
    }
    file_hashes = {
        name: sha256_file(root / path)
        for name, path in tracked_files.items()
        if path and (root / path).exists()
    }
    packages = {}
    for package in ("langchain", "langchain-core", "langgraph", "qdrant-client"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    safe_config = _redact_secrets(config)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "python_version": sys.version.split()[0],
        "packages": packages,
        "config": safe_config,
        "config_sha256": sha256_text(
            json.dumps(safe_config, ensure_ascii=False, sort_keys=True)
        ),
        "file_hashes": file_hashes,
        "prompt_hashes": {
            name: sha256_text(text) for name, text in sorted(prompt_texts.items())
        },
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def calculate_model_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any],
) -> float | None:
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    if input_price is None or output_price is None:
        return None
    return round(
        input_tokens * float(input_price) / 1_000_000
        + output_tokens * float(output_price) / 1_000_000,
        8,
    )


def build_question_metrics(
    question_id: str,
    state: dict[str, Any],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    calls = state.get("model_calls", [])
    input_tokens = sum(int(call.get("input_tokens", 0)) for call in calls)
    output_tokens = sum(int(call.get("output_tokens", 0)) for call in calls)
    node_events = state.get("node_metrics", [])
    return {
        "schema_version": 1,
        "question_id": question_id,
        "total_duration_ms": round(
            sum(float(event.get("duration_ms", 0.0)) for event in node_events), 3
        ),
        "node_metrics": node_events,
        "model_calls": calls,
        "llm_call_count": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": calculate_model_cost(input_tokens, output_tokens, pricing),
        "retrieval_rounds": int(state.get("retrieval_round", 0)),
        "followup_used": int(state.get("retrieval_round", 0)) > 1,
        "evidence_status": state.get("evidence_status"),
        "judge_attempts": state.get("judge_attempts", []),
        "answer_repaired": bool(state.get("answer_repaired", False)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_question_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row.get("total_duration_ms", 0.0)) for row in rows]
    node_totals: dict[str, float] = {}
    for row in rows:
        for event in row.get("node_metrics", []):
            name = str(event.get("node", "unknown"))
            node_totals[name] = node_totals.get(name, 0.0) + float(
                event.get("duration_ms", 0.0)
            )
    costs = [row.get("cost") for row in rows]
    known_costs = [float(cost) for cost in costs if cost is not None]
    followup_rows = [row for row in rows if row.get("followup_used")]
    single_rows = [row for row in rows if not row.get("followup_used")]

    def average(rows_to_average: list[dict[str, Any]], key: str) -> float | None:
        values = [row.get(key) for row in rows_to_average]
        known = [float(value) for value in values if value is not None]
        return round(statistics.fmean(known), 4) if len(known) == len(values) and known else None

    followup_increment = {
        "duration_ms": (
            round(
                average(followup_rows, "total_duration_ms")
                - average(single_rows, "total_duration_ms"),
                3,
            )
            if followup_rows and single_rows
            else None
        ),
        "total_tokens": (
            round(
                average(followup_rows, "total_tokens")
                - average(single_rows, "total_tokens"),
                3,
            )
            if followup_rows and single_rows
            else None
        ),
        "cost": (
            round(average(followup_rows, "cost") - average(single_rows, "cost"), 8)
            if followup_rows
            and single_rows
            and average(followup_rows, "cost") is not None
            and average(single_rows, "cost") is not None
            else None
        ),
    }
    judge_attempts = [
        attempt for row in rows for attempt in row.get("judge_attempts", [])
    ]
    return {
        "question_count": len(rows),
        "duration_ms": {
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
        },
        "node_duration_ms": {
            name: round(value, 3) for name, value in sorted(node_totals.items())
        },
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
        "total_cost": round(sum(known_costs), 8) if len(known_costs) == len(rows) else None,
        "followup_questions": sum(bool(row.get("followup_used")) for row in rows),
        "followup_increment": followup_increment,
        "judge": {
            "attempts": len(judge_attempts),
            "parse_failures": sum(
                attempt.get("status") == "parse_error" for attempt in judge_attempts
            ),
            "fallbacks": sum(bool(attempt.get("fallback")) for attempt in judge_attempts),
            "unknown_questions": sum(
                row.get("evidence_status") == "UNKNOWN" for row in rows
            ),
        },
    }


def write_question_metrics_outputs(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    target = Path(output_dir)
    write_jsonl(rows, target / "question_metrics.jsonl")
    summary = summarize_question_metrics(rows)
    (target / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def summarize_repetitions(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("At least one repetition is required")
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    if len(samples) == 1:
        margin = 0.0
    else:
        critical = _T_95.get(len(samples) - 1, 1.96)
        margin = critical * std / math.sqrt(len(samples))
    return {
        "count": len(samples),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_low": round(mean - margin, 4),
        "ci95_high": round(mean + margin, 4),
    }


def aggregate_official_repetitions(
    result_files: list[str | Path],
) -> dict[str, Any]:
    if len(result_files) < 3:
        raise ValueError("The evaluation protocol requires at least three repetitions")
    results = [json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in result_files]
    metric_names = (
        "correctness_pct",
        "completeness_pct",
        "combined_score",
        "document_recall_pct",
        "invalid_extra_docs",
    )
    aggregate_names = {
        "correctness_pct": "average_correctness_pct",
        "completeness_pct": "average_completeness_pct",
        "combined_score": "combined_correctness_completeness_score",
        "document_recall_pct": "average_recall_pct",
        "invalid_extra_docs": "average_invalid_extra_docs",
    }
    metric_values: dict[str, list[float]] = {name: [] for name in metric_names}
    question_outcomes: dict[str, list[bool]] = {}
    expected_question_ids: set[str] | None = None
    for result in results:
        questions = result.get("questions", [])
        question_ids = {str(row["question_id"]) for row in questions}
        if expected_question_ids is None:
            expected_question_ids = question_ids
        elif question_ids != expected_question_ids:
            raise ValueError("All repetitions must contain the same question IDs")
        aggregate = result.get("aggregate_stats", {})
        for row in questions:
            question_outcomes.setdefault(str(row["question_id"]), []).append(
                bool(row.get("answer_correct", False))
            )
        for name in metric_names:
            aggregate_value = aggregate.get(aggregate_names[name])
            if aggregate_value is not None:
                metric_values[name].append(float(aggregate_value))
                continue
            if name == "correctness_pct":
                values = [100.0 if row.get("answer_correct") else 0.0 for row in questions]
            elif name == "combined_score":
                values = []
            else:
                values = [row.get(name) for row in questions if row.get(name) is not None]
            if values:
                metric_values[name].append(statistics.fmean(float(value) for value in values))
    return {
        "repetitions": len(results),
        "metrics": {
            name: summarize_repetitions(values)
            for name, values in metric_values.items()
            if values
        },
        "question_outcomes": question_outcomes,
        "unstable_questions": sorted(
            question_id
            for question_id, outcomes in question_outcomes.items()
            if len(set(outcomes)) > 1
        ),
    }


def build_judge_confusion_matrix(
    annotations: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = ("SUFFICIENT", "INSUFFICIENT", "UNKNOWN")
    trace_by_id = {str(row["question_id"]): row for row in traces}
    matrix = {
        expected: {predicted: 0 for predicted in statuses} for expected in statuses
    }
    evaluated = 0
    correct = 0
    for annotation in annotations:
        question_id = str(annotation["question_id"])
        expected = str(annotation["expected_status"]).upper()
        predicted = str(trace_by_id.get(question_id, {}).get("evidence_status", "UNKNOWN")).upper()
        if expected not in statuses:
            raise ValueError(f"Invalid expected Judge status for {question_id}: {expected}")
        if predicted not in statuses:
            predicted = "UNKNOWN"
        matrix[expected][predicted] += 1
        evaluated += 1
        correct += expected == predicted

    per_status = {}
    for status in statuses:
        true_positive = matrix[status][status]
        predicted_total = sum(matrix[expected][status] for expected in statuses)
        expected_total = sum(matrix[status].values())
        per_status[status] = {
            "precision": round(true_positive / predicted_total, 4) if predicted_total else 0.0,
            "recall": round(true_positive / expected_total, 4) if expected_total else 0.0,
        }
    return {
        "count": evaluated,
        "accuracy": round(correct / evaluated, 4) if evaluated else 0.0,
        "confusion_matrix": matrix,
        "per_status": per_status,
    }


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _stage_document_ids(trace: dict[str, Any]) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    for query_trace in trace.get("retrieval_stage_history", []):
        for stage, rows in query_trace.get("stages", {}).items():
            current = stages.setdefault(stage, [])
            current.extend(
                str(row.get("dsid")) if isinstance(row, dict) else str(row)
                for row in rows
                if (isinstance(row, str) and row) or (isinstance(row, dict) and row.get("dsid"))
            )
    stages = {name: _unique(ids) for name, ids in stages.items()}
    rerank = trace.get("rerank_history", [])
    if rerank:
        stages["document_fusion"] = _unique(rerank[-1].get("candidate_document_ids", []))
        stages["rerank"] = _unique(rerank[-1].get("selected_document_ids", []))
    guard = trace.get("fusion_guard_history", [])
    if guard:
        reranked = stages.get("rerank", [])
        stages["guard_candidates"] = _unique(
            reranked + guard[-1].get("candidate_document_ids", [])
        )
        stages["guard_promoted"] = _unique(
            reranked + guard[-1].get("promoted_document_ids", [])
        )
    parent_expansion = trace.get("parent_expansion_history", [])
    if parent_expansion:
        stages["parent_expansion"] = _unique(
            parent_expansion[-1].get("document_ids", [])
        )
    stages["final"] = _unique(trace.get("selected_document_ids", []))
    return stages


def build_recall_funnel(
    questions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    official_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_by_id = {str(row["question_id"]): row for row in traces}
    result_by_id = {
        str(row["question_id"]): row
        for row in (official_results or {}).get("questions", [])
    }
    stage_rows: dict[str, list[dict[str, float]]] = {}
    per_question: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        expected = _unique(str(item) for item in question.get("expected_doc_ids", []))
        if not expected:
            continue
        stages = _stage_document_ids(trace_by_id.get(question_id, {}))
        metrics: dict[str, dict[str, float]] = {}
        expected_set = set(expected)
        for name, ids in stages.items():
            relevant = [item for item in ids if item in expected_set]
            first = next(
                (rank for rank, item in enumerate(ids, start=1) if item in expected_set),
                None,
            )
            metric = {
                "hit": float(bool(relevant)),
                "recall": len(set(relevant)) / len(expected),
                "mrr": 1.0 / first if first else 0.0,
            }
            metrics[name] = metric
            stage_rows.setdefault(name, []).append(metric)

        def stage_recall(stage: str) -> float:
            return len(expected_set & set(stages.get(stage, []))) / len(expected_set)

        answer_correct = bool(result_by_id.get(question_id, {}).get("answer_correct"))
        guard_recovered = (
            "guard_promoted" in stages
            and stage_recall("guard_promoted") > stage_recall("rerank")
        )
        if "union" in stages and stage_recall("union") < 1.0:
            cause = "initial_recall_miss"
        elif (
            "union" in stages
            and "channel_fusion" in stages
            and stage_recall("channel_fusion") < stage_recall("union")
        ):
            cause = "channel_fusion_drop"
        elif (
            "channel_fusion" in stages
            and "document_fusion" in stages
            and stage_recall("document_fusion") < stage_recall("channel_fusion")
        ):
            cause = "document_fusion_drop"
        elif (
            "document_fusion" in stages
            and "rerank" in stages
            and stage_recall("rerank") < stage_recall("document_fusion")
            and stage_recall("final") < stage_recall("document_fusion")
        ):
            cause = "rerank_elimination"
        elif (
            "rerank" in stages
            and "final" in stages
            and stage_recall("final") < stage_recall("rerank")
        ):
            cause = "post_processing_drop"
        elif result_by_id and not answer_correct:
            cause = "answer_failure"
        else:
            cause = "success_or_unscored"
        per_question.append(
            {
                "question_id": question_id,
                "expected_doc_ids": expected,
                "stages": stages,
                "metrics": metrics,
                "guard_recovered": guard_recovered,
                "primary_failure_cause": cause,
            }
        )

    overall = {
        stage: {
            metric: round(statistics.fmean(row[metric] for row in rows), 4)
            for metric in ("hit", "recall", "mrr")
        }
        for stage, rows in stage_rows.items()
    }
    guard_runs = [
        trace.get("fusion_guard_history", [])[-1]
        for trace in traces
        if trace.get("fusion_guard_history")
    ]
    guard_summary = {
        "enabled_questions": len(guard_runs),
        "candidate_documents": sum(
            len(row.get("candidate_document_ids", [])) for row in guard_runs
        ),
        "promoted_documents": sum(
            len(row.get("promoted_document_ids", [])) for row in guard_runs
        ),
        "recovered_questions": sum(
            bool(row.get("guard_recovered")) for row in per_question
        ),
    }
    return {
        "schema_version": 2,
        "overall": overall,
        "fusion_guard": guard_summary,
        "questions": per_question,
    }
