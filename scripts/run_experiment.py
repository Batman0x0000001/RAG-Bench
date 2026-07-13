from __future__ import annotations

import argparse
import subprocess
import sys

from src.evaluation.experiment_config import ABLATION_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered P0 variants repeatedly.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="github_dev")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=tuple(ABLATION_VARIANTS), default=["p0_full"])
    parser.add_argument("--name-prefix", default="p0")
    args = parser.parse_args()
    if args.repetitions < 3:
        raise SystemExit("P0 protocol requires at least three repetitions")

    for variant in args.variants:
        for repetition in range(1, args.repetitions + 1):
            run_name = f"{args.name_prefix}_{variant}_{args.dataset}_r{repetition:02d}"
            command = [
                sys.executable,
                "-m",
                "scripts.run_benchmark",
                "--workflow-profile",
                "p0_candidate",
                "--variant",
                variant,
                "--dataset",
                args.dataset,
                "--run-name",
                run_name,
            ]
            if args.config:
                command.extend(["--config", args.config])
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
