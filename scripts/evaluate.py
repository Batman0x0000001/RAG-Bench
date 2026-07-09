from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.result_analyzer import summarize_by_question_type
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lightweight local run summary.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config override file.")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else Path(config["output"]["runs_dir"]) / config["run_name"]
    summary = summarize_by_question_type(
        config["data"]["questions_file"],
        run_dir / "answers.jsonl",
    )

    results_file = run_dir / "results.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Wrote lightweight summary to {results_file}")
    print("Official metrics should still be computed with EnterpriseRAG-Bench answer_evaluation scripts.")


if __name__ == "__main__":
    main()
