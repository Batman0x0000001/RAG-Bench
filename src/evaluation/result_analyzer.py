from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_by_question_type(questions_file: str | Path, answers_file: str | Path) -> dict[str, Any]:
    questions = load_jsonl(questions_file)
    answers = {row["question_id"]: row for row in load_jsonl(answers_file)}
    totals: Counter[str] = Counter()
    answered: Counter[str] = Counter()
    doc_counts: defaultdict[str, list[int]] = defaultdict(list)

    for question in questions:
        question_type = question.get("question_type", "unknown")
        totals[question_type] += 1
        answer = answers.get(question["question_id"])
        if answer:
            answered[question_type] += 1
            doc_counts[question_type].append(len(answer.get("document_ids", [])))

    # 这是轻量分析，不替代官方评测；用于快速确认每类问题是否都产生了答案。
    return {
        question_type: {
            "total": totals[question_type],
            "answered": answered[question_type],
            "avg_returned_docs": (
                sum(doc_counts[question_type]) / len(doc_counts[question_type])
                if doc_counts[question_type]
                else 0
            ),
        }
        for question_type in sorted(totals)
    }
