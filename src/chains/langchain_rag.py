from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
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


def extract_document_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []
    for document in documents:
        dsid = document.metadata.get("dsid")
        if dsid and dsid not in ids:
            ids.append(str(dsid))
    return ids


def answer_with_retriever(
    question: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
) -> RagResult:
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
