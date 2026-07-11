from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, StateGraph

from src.graphs.nodes import RagState, generate_answer_node, retrieve_node


def build_simple_rag_graph(retriever: BaseRetriever, llm: BaseChatModel):
    # 第一版图保持极简：先把可观测的 LangGraph 流程跑通，再逐步扩展复杂节点。
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node(retriever))
    graph.add_node("generate_answer", generate_answer_node(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
