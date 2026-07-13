from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.experiment_protocol import aggregate_official_repetitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three or more official evaluations.")
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = aggregate_official_repetitions(args.result_files)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote repeated-run report to {target}")


if __name__ == "__main__":
    main()
