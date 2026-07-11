from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*")
TOKEN_SEPARATOR = re.compile(r"[._:/-]+")


def tokenize_technical_text(text: str) -> list[str]:
    """同时保留完整技术标识符和组成词，兼顾精确匹配与普通关键词检索。"""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(match)
        parts = [part for part in TOKEN_SEPARATOR.split(match) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class LocalBM25Retriever(BaseRetriever):
    """基于 manifest chunks 的进程内 BM25 检索器，不修改持久化向量库。"""

    documents: list[Document]
    k: int = 30
    k1: float = 1.5
    b: float = 0.75

    _postings: dict[str, list[tuple[int, int]]] = PrivateAttr(default_factory=dict)
    _document_lengths: list[int] = PrivateAttr(default_factory=list)
    _average_length: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context: object) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for index, document in enumerate(self.documents):
            metadata_text = " ".join(
                str(document.metadata.get(key, ""))
                for key in ("title", "relative_path", "section")
            )
            terms = tokenize_technical_text(f"{metadata_text}\n{document.page_content}")
            lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append((index, frequency))
        self._postings = dict(postings)
        self._document_lengths = lengths
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        query_terms = set(tokenize_technical_text(query))
        if not query_terms or not self.documents or self._average_length == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        total_documents = len(self.documents)
        for term in sorted(query_terms):
            postings = self._postings.get(term, [])
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_frequency = math.log(
                1.0
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, term_frequency in postings:
                length_ratio = self._document_lengths[index] / self._average_length
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[index] += inverse_frequency * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )

        ranked = sorted(scores, key=lambda index: (-scores[index], index))[: self.k]
        results: list[Document] = []
        for index in ranked:
            document = self.documents[index]
            results.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "retrieval_channels": ["bm25"],
                        "bm25_score": scores[index],
                    },
                )
            )
        return results


def build_bm25_retriever(documents: list[Document], k: int = 30) -> BaseRetriever:
    return LocalBM25Retriever(documents=documents, k=k)
