from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise RAG assistant. Answer only from the provided context. "
            "Cover every stated requirement with all explicit qualifiers, names, numbers, "
            "versions, labels, and exceptions supported by the evidence. Resolve old/new or "
            "conflicting information explicitly. Answer every side of comparisons and every "
            "item in a list; do not stop after the first supported fact. You may state a minimal "
            "logical inference when it follows directly from documented semantics, and label it "
            "as an inference. For parameter normalization, a missing field is unset; if the "
            "context says unset values fall back to defaults, omission does too unless the "
            "context states an exception. If evidence is insufficient after checking all "
            "requirements and context, identify what is unavailable instead of inventing it. "
            "Before responding, silently turn the required coverage into a checklist. For each "
            "item, include every explicitly supported mechanism, input signal, stored state, "
            "lifetime, processing step, and outcome that explains the answer. Then verify that "
            "the answer covers every checklist item and all supported labels or qualifiers.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRequired coverage:\n{requirements}\n\n"
            "Retrieval assessment (guidance only; verify it against context):\n"
            "{retrieval_guidance}\n\n"
            "Context:\n{context}\n\nAnswer clearly and cite no external knowledge.",
        ),
    ]
)


def build_chat_model(llm_config: dict[str, Any]) -> BaseChatModel:
    provider = llm_config.get("provider", "anthropic_compatible")
    if provider not in {"anthropic", "anthropic_compatible", "deepseek"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # DeepSeek 的 Anthropic-compatible API 使用 ChatAnthropic，但需要显式传入 base url 和 key。
    api_key = llm_config.get("api_key") or None
    base_url = llm_config.get("base_url") or None
    return init_chat_model(
        model=llm_config["model"],
        model_provider="anthropic",
        api_key=api_key,
        base_url=base_url,
        temperature=float(llm_config.get("temperature", 0.0)),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
    )


def format_context(documents: list[Document]) -> str:
    # 把 dsid 放进上下文，方便模型理解每段证据对应哪个原始文档。
    chunks = []
    for index, document in enumerate(documents, start=1):
        dsid = document.metadata.get("dsid", "unknown")
        source = document.metadata.get("source_type", "unknown")
        path = document.metadata.get("relative_path", "")
        chunks.append(
            f"[Document {index} | {dsid} | source={source} | path={path}]\n"
            f"{document.page_content}"
        )
    return "\n\n---\n\n".join(chunks)
