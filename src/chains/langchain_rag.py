from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise RAG assistant. Answer only from the provided context. "
            "If the context is insufficient, say that the answer is not available in the documents.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer clearly and cite no external knowledge.",
        ),
    ]
)


@dataclass(frozen=True)
class RagResult:
    answer: str
    document_ids: list[str]
    documents: list[Document]


def build_chat_model(llm_config: dict[str, Any]) -> ChatAnthropic:
    provider = llm_config.get("provider", "anthropic_compatible")
    if provider not in {"anthropic", "anthropic_compatible", "deepseek"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # DeepSeek 的 Anthropic-compatible API 使用 ChatAnthropic，但需要显式传入 base url 和 key。
    api_key = llm_config.get("api_key") or None
    base_url = llm_config.get("base_url") or None
    return ChatAnthropic(
        model=llm_config["model"],
        anthropic_api_key=api_key,
        anthropic_api_url=base_url,
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


def extract_document_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []
    for document in documents:
        dsid = document.metadata.get("dsid")
        if dsid and dsid not in ids:
            ids.append(str(dsid))
    return ids


def answer_with_retriever(question: str, retriever: Any, llm: ChatAnthropic) -> RagResult:
    documents = retriever.invoke(question)
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda _: format_context(documents)),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return RagResult(
        answer=answer,
        document_ids=extract_document_ids(documents),
        documents=documents,
    )
