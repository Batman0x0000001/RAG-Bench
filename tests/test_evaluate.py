from pathlib import Path
from unittest.mock import patch

from scripts.evaluate import run_official_evaluation


def test_official_evaluation_subprocess_uses_utf8_mode() -> None:
    with patch("scripts.evaluate.subprocess.run") as run:
        run_official_evaluation(
            official_repo=Path("official"),
            answers_file=Path("answers.jsonl"),
            questions_file=Path("questions.jsonl"),
            results_file=Path("results.json"),
            updated_questions_file=Path("corrected.jsonl"),
            parallelism=1,
            correction=False,
            resume=False,
        )

    assert run.call_args.kwargs["env"]["PYTHONUTF8"] == "1"
    assert run.call_args.kwargs["check"] is True
