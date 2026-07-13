from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from src.graphs.workflow import build_stage26_rag_graph


def build_adaptive_rag_graph(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    parent_documents: dict[str, list[Document]],
    config: dict[str, Any],
    entity_index: dict[str, list[str]] | None = None,
):
    """兼容原有调用方；当前稳定实现固定为 Stage 26。"""
    return build_stage26_rag_graph(
        retriever,
        llm,
        parent_documents,
        config,
        entity_index=entity_index,
    )
