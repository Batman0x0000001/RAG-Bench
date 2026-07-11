from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


def build_embeddings(config: dict[str, Any]) -> Embeddings:
    provider = config.get("provider", "openai_compatible")
    model = config.get("model", "BAAI/bge-m3")

    if provider in {"openai", "openai_compatible", "siliconflow"}:
        # SiliconFlow 的 embedding API 兼容 OpenAI 格式，因此复用 LangChain 的 OpenAIEmbeddings。
        from langchain_openai import OpenAIEmbeddings

        api_key = config.get("api_key") or None
        base_url = config.get("base_url") or None
        if not api_key:
            raise ValueError("Missing embedding api_key. Please set SILICONFLOW_API_KEY in .env.")

        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            # OpenAI-compatible 服务应让服务端使用目标模型自己的 tokenizer。
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
