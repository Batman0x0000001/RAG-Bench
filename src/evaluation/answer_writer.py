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


def append_graph_trace(
    trace_file: str | Path,
    question_id: str,
    state: dict[str, Any],
) -> None:
    """保存规划和循环状态，不序列化大段文档正文。"""
    Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "question_id": question_id,
        "plan": state.get("plan", {}),
        "executed_queries": state.get("executed_queries", []),
        "retrieval_round": state.get("retrieval_round", 0),
        "selected_document_ids": state.get("selected_document_ids", []),
        "candidate_document_ids": list(state.get("candidate_groups", {})),
        "rerank_history": state.get("rerank_history", []),
        "query_candidates": [
            {
                "query": query,
                "document_ids": list(
                    dict.fromkeys(
                        str(document.metadata.get("dsid"))
                        for document in documents
                        if document.metadata.get("dsid")
                    )
                ),
                "channels_by_document": {
                    dsid: sorted(
                        {
                            channel
                            for document in documents
                            if str(document.metadata.get("dsid")) == dsid
                            for channel in document.metadata.get(
                                "retrieval_channels", []
                            )
                        }
                    )
                    for dsid in dict.fromkeys(
                        str(document.metadata.get("dsid"))
                        for document in documents
                        if document.metadata.get("dsid")
                    )
                },
            }
            for query, documents in zip(
                state.get("executed_queries", []),
                state.get("query_results", []),
            )
        ],
        "evidence_sufficient": state.get("evidence_sufficient", False),
        "can_retry": state.get("can_retry", False),
        "missing_evidence": state.get("missing_evidence", []),
        "answer_chunk_ids": [
            document.metadata.get("chunk_id")
            for document in state.get("answer_docs", [])
        ],
    }
    with Path(trace_file).open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
