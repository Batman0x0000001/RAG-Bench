from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.evaluation.benchmark_report import (
    load_jsonl,
    write_evaluation_outputs,
    write_question_subset,
)
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
    project_root = Path.cwd().resolve()
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else (project_root / config["output"]["runs_dir"] / config["run_name"]).resolve()
    )
    official_repo = (project_root / args.official_repo).resolve()
    answers_file = run_dir / "answers.jsonl"
    retrieved_file = run_dir / "retrieved_docs.jsonl"
    questions_subset_file = run_dir / f"{args.source_type}_questions.jsonl"
    official_results_file = run_dir / "official_results.json"
    updated_questions_file = run_dir / f"{args.source_type}_questions_corrected.jsonl"

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
        config["data"]["questions_file"],
        questions_subset_file,
        args.source_type,
        question_ids=set(answer_ids) if args.only_answered else None,
        include_mixed_sources=args.include_mixed_source_questions,
    )
    answer_count = len(answer_rows)
    if answer_count != len(questions):
        mode_hint = "" if args.only_answered else " Use --only-answered for a partial run."
        raise ValueError(
            f"Answer count ({answer_count}) does not match {args.source_type} question "
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
    print(f"Questions evaluated: {supplementary['question_count']}")
    print(f"Incorrect answers: {len(failures)}")
    print(f"Official results: {official_results_file}")
    print(f"Supplementary metrics: {run_dir / 'supplementary_metrics.json'}")
    print(f"Failure list: {run_dir / 'failed_questions.jsonl'}")


if __name__ == "__main__":
    main()
