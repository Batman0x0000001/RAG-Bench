from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def append_answer(
    answers_file: str | Path,
    question_id: str,
    answer: str,
    document_ids: list[str],
) -> None:
    Path(answers_file).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "question_id": question_id,
        "answer": answer,
        "document_ids": document_ids,
    }
    with Path(answers_file).open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_retrieved_docs(
    retrieved_file: str | Path,
    question_id: str,
    documents: list[Document],
) -> None:
    # 检索日志用于复盘 document recall：先看有没有召回，再看答案生成是否用对证据。
    Path(retrieved_file).parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "question_id": question_id,
        "documents": [
            {
                "rank": rank,
                "dsid": document.metadata.get("dsid"),
                "chunk_id": document.metadata.get("chunk_id"),
                "source_type": document.metadata.get("source_type"),
                "relative_path": document.metadata.get("relative_path"),
            }
            for rank, document in enumerate(documents, start=1)
        ],
    }
    with Path(retrieved_file).open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
