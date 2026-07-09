from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def recreate_collection(config: dict[str, Any]) -> None:
    client = QdrantClient(url=config["url"], api_key=config.get("api_key") or None)
    distance = Distance[config.get("distance", "Cosine").upper()]
    client.recreate_collection(
        collection_name=config["collection"],
        vectors_config=VectorParams(
            size=int(config.get("vector_size", 1024)),
            distance=distance,
        ),
    )


def index_documents(
    documents: list[Document],
    embeddings: Embeddings,
    qdrant_config: dict[str, Any],
    batch_size: int = 64,
) -> None:
    # 这里直接使用 LangChain 的 QdrantVectorStore，保证后续 retriever 使用同一套 metadata。
    client = QdrantClient(url=qdrant_config["url"], api_key=qdrant_config.get("api_key") or None)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_config["collection"],
        embedding=embeddings,
    )
    for start in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[start : start + batch_size])
