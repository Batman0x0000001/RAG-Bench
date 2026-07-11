from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from src.chains.langchain_rag import ANSWER_PROMPT, extract_document_ids, format_context
from langchain_core.output_parsers import StrOutputParser


class RagState(TypedDict, total=False):
    question: str
    retrieved_docs: list[Document]
    answer: str
    document_ids: list[str]
    step_count: int


def retrieve_node(retriever: BaseRetriever):
    def _node(state: RagState) -> RagState:
        # 检索节点只负责找证据，不生成答案；这样后续可以替换成多路检索或混合检索。
        question = state["question"]
        docs = retriever.invoke(question)
        return {
            **state,
            "retrieved_docs": docs,
            "document_ids": extract_document_ids(docs),
            "step_count": state.get("step_count", 0) + 1,
        }

    return _node


def generate_answer_node(llm: BaseChatModel):
    def _node(state: RagState) -> RagState:
        # 生成节点只消费已检索文档，保持答案生成和检索逻辑解耦。
        prompt_value = ANSWER_PROMPT.invoke(
            {
                "question": state["question"],
                "context": format_context(state.get("retrieved_docs", [])),
            }
        )
        answer = (llm | StrOutputParser()).invoke(prompt_value)
        return {**state, "answer": answer}

    return _node
