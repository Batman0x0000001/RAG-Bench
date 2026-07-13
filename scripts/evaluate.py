from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.evaluation.benchmark_report import (
    load_jsonl,
    write_evaluation_outputs,
    write_question_subset,
)
from src.evaluation.experiment_protocol import build_recall_funnel
from src.evaluation.experiment_config import get_dataset
from src.utils.config import load_config


def run_official_evaluation(
    official_repo: Path,
    answers_file: Path,
    questions_file: Path,
    results_file: Path,
    updated_questions_file: Path,
    parallelism: int,
    correction: bool,
    resume: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.scripts.answer_evaluation.metrics_based_eval",
        "--answers-file",
        str(answers_file),
        "--questions-file",
        str(questions_file),
        "--results-file",
        str(results_file),
        "--parallelism",
        str(parallelism),
    ]
    if correction:
        command.extend(
            [
                "--updated-questions-file",
                str(updated_questions_file),
                "--uuid-index-cache-file",
                str((official_repo / "generated_data" / "uuid_index.json").resolve()),
            ]
        )
    else:
        command.append("--no-correction")
    if resume:
        command.append("--resume")

    child_env = os.environ.copy()
    # 官方读取器未显式指定 encoding；Windows 上需在解释器启动前启用 UTF-8 模式。
    child_env["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=official_repo, env=child_env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official EnterpriseRAG-Bench evaluation and local diagnostics."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--source-type", default="github")
    parser.add_argument("--dataset", default=None, choices=("github_dev", "confluence_frozen"))
    parser.add_argument(
        "--frozen-run-dirs",
        nargs="+",
        default=None,
        help="All completed repetition directories required before frozen Gold is opened.",
    )
    parser.add_argument(
        "--official-repo",
        default="external/EnterpriseRAG-Bench",
    )
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--official-correction",
        action="store_true",
        help="Enable the official three-judge gold-document correction flow.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only-answered",
        action="store_true",
        help="Evaluate only question IDs present in answers.jsonl.",
    )
    parser.add_argument(
        "--include-mixed-source-questions",
        action="store_true",
        help="Evaluate questions that combine the selected source with other sources.",
    )
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Only analyze an existing official results file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    questions_gold_file = config["data"]["questions_file"]
    source_type = args.source_type
    if args.dataset:
        dataset = get_dataset(args.dataset)
        questions_gold_file = dataset.gold_questions_file
        source_type = dataset.source_type
    project_root = Path.cwd().resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else (project_root / config["output"]["runs_dir"] / config["run_name"]).resolve()
    )
    if args.dataset and dataset.role == "frozen_test":
        frozen_runs = [Path(path).resolve() for path in (args.frozen_run_dirs or [])]
        if len(frozen_runs) < 3:
            raise ValueError(
                "Frozen evaluation requires at least three completed --frozen-run-dirs."
            )
        if run_dir not in frozen_runs:
            raise ValueError("The evaluated run must be included in --frozen-run-dirs.")
        protocol_signatures: list[str] = []
        answer_id_sets: list[set[str]] = []
        for frozen_run in frozen_runs:
            for filename in ("answers.jsonl", "run_manifest.json"):
                if not (frozen_run / filename).exists():
                    raise FileNotFoundError(
                        f"Frozen repetition is incomplete: {frozen_run / filename}"
                    )
            manifest = json.loads(
                (frozen_run / "run_manifest.json").read_text(encoding="utf-8")
            )
            manifest_config = manifest.get("config", {})
            signature = {
                "prompt_hashes": manifest.get("prompt_hashes"),
                "file_hashes": manifest.get("file_hashes"),
                "workflow": manifest_config.get("workflow"),
                "dataset": manifest_config.get("experiment", {}).get("dataset"),
                "variant": manifest_config.get("experiment", {}).get("variant"),
                "retrieval": manifest_config.get("retrieval"),
                "features": manifest_config.get("features"),
                "llm": manifest_config.get("llm"),
                "embedding": manifest_config.get("embedding"),
            }
            protocol_signatures.append(json.dumps(signature, sort_keys=True))
            answer_id_sets.append(
                {
                    str(row.get("question_id"))
                    for row in load_jsonl(frozen_run / "answers.jsonl")
                }
            )
        if len(set(protocol_signatures)) != 1:
            raise ValueError("Frozen repetitions do not share one locked protocol.")
        if any(ids != answer_id_sets[0] for ids in answer_id_sets[1:]):
            raise ValueError("Frozen repetitions do not contain the same question IDs.")
    official_repo = (project_root / args.official_repo).resolve()
    answers_file = run_dir / "answers.jsonl"
    retrieved_file = run_dir / "retrieved_docs.jsonl"
    questions_subset_file = run_dir / f"{source_type}_questions.jsonl"
    official_results_file = run_dir / "official_results.json"
    updated_questions_file = run_dir / f"{source_type}_questions_corrected.jsonl"

    for required in (answers_file, retrieved_file):
        if not required.exists():
            raise FileNotFoundError(f"Required benchmark output does not exist: {required}")
    if not official_repo.exists():
        raise FileNotFoundError(f"Official benchmark repository does not exist: {official_repo}")

    answer_rows = load_jsonl(answers_file)
    answer_ids = [str(row.get("question_id", "")) for row in answer_rows]
    if any(not question_id for question_id in answer_ids):
        raise ValueError("Every answer row must contain question_id.")
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("answers.jsonl contains duplicate question_id values.")

    questions = write_question_subset(
        questions_gold_file,
        questions_subset_file,
        source_type,
        question_ids=set(answer_ids) if args.only_answered else None,
        include_mixed_sources=args.include_mixed_source_questions,
    )
    answer_count = len(answer_rows)
    if answer_count != len(questions):
        mode_hint = "" if args.only_answered else " Use --only-answered for a partial run."
        raise ValueError(
            f"Answer count ({answer_count}) does not match {source_type} question "
            f"count ({len(questions)}). Run the benchmark for the complete source subset first."
            f"{mode_hint}"
        )

    if not args.skip_official:
        run_official_evaluation(
            official_repo=official_repo,
            answers_file=answers_file,
            questions_file=questions_subset_file,
            results_file=official_results_file,
            updated_questions_file=updated_questions_file,
            parallelism=args.parallelism,
            correction=args.official_correction,
            resume=args.resume,
        )
    if not official_results_file.exists():
        raise FileNotFoundError(
            f"Official results do not exist: {official_results_file}. "
            "Run without --skip-official first."
        )

    supplementary, failures = write_evaluation_outputs(
        questions_file=questions_subset_file,
        answers_file=answers_file,
        retrieved_file=retrieved_file,
        official_results_file=official_results_file,
        output_dir=run_dir,
    )
    trace_file = run_dir / "graph_traces.jsonl"
    if trace_file.exists():
        with official_results_file.open("r", encoding="utf-8-sig") as file:
            official_results = json.load(file)
        funnel = build_recall_funnel(
            questions,
            load_jsonl(trace_file),
            official_results,
        )
        (run_dir / "recall_funnel.json").write_text(
            json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Questions evaluated: {supplementary['question_count']}")
    print(f"Incorrect answers: {len(failures)}")
    print(f"Official results: {official_results_file}")
    print(f"Supplementary metrics: {run_dir / 'supplementary_metrics.json'}")
    print(f"Failure list: {run_dir / 'failed_questions.jsonl'}")
    if trace_file.exists():
        print(f"Recall funnel: {run_dir / 'recall_funnel.json'}")


if __name__ == "__main__":
    main()
