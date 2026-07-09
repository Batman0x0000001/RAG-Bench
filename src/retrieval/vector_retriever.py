from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def build_qdrant_client(config: dict[str, Any]) -> QdrantClient:
    api_key = config.get("api_key") or None
    return QdrantClient(url=config["url"], api_key=api_key)


def build_vector_store(
    qdrant_config: dict[str, Any],
    embeddings: Embeddings,
) -> QdrantVectorStore:
    # QdrantVectorStore 负责把 LangChain retriever 和底层 Qdrant collection 接起来。
    client = build_qdrant_client(qdrant_config)
    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_config["collection"],
        embedding=embeddings,
    )


def build_retriever(
    qdrant_config: dict[str, Any],
    embeddings: Embeddings,
    top_k: int,
):
    vector_store = build_vector_store(qdrant_config, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
