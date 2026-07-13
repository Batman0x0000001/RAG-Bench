from __future__ import annotations

import argparse

from src.evaluation.experiment_protocol import prepare_blind_dataset
from src.evaluation.experiment_config import get_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Gold-free benchmark question file.")
    parser.add_argument("--dataset", required=True, choices=("github_dev", "confluence_frozen"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = prepare_blind_dataset(
        config["data"]["questions_file"], get_dataset(args.dataset)
    )
    print(
        f"Prepared {manifest['dataset']} with {manifest['question_count']} blind questions; "
        f"sha256={manifest['blind_questions_sha256']}"
    )


if __name__ == "__main__":
    main()
