from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark_report import load_jsonl
from src.evaluation.experiment_protocol import build_judge_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the three-state evidence Judge.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_judge_confusion_matrix(
        load_jsonl(args.annotations), load_jsonl(args.traces)
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Judge report to {target}")


if __name__ == "__main__":
    main()
