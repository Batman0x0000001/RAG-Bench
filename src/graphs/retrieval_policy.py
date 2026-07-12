from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal


Strategy = Literal[
    "single",
    "semantic",
    "multi_document",
    "conflicting",
    "completeness",
]


@dataclass(frozen=True)
class RetrievalPolicy:
    max_queries: int
    document_budget: int
    minimum_documents: int
    candidate_documents: int = 24
    candidate_archive_documents: int = 32
    reserved_documents_per_task: int = 2
    reserve_entity_link_results: bool = True


DEFAULT_RETRIEVAL_POLICIES: dict[Strategy, RetrievalPolicy] = {
    "single": RetrievalPolicy(1, 1, 1),
    "semantic": RetrievalPolicy(3, 4, 1),
    "multi_document": RetrievalPolicy(8, 8, 3),
    "conflicting": RetrievalPolicy(4, 4, 2),
    "completeness": RetrievalPolicy(
        10,
        10,
        6,
        candidate_documents=48,
        candidate_archive_documents=64,
        reserved_documents_per_task=10,
        reserve_entity_link_results=False,
    ),
}


def build_retrieval_policies(config: dict[str, Any]) -> dict[Strategy, RetrievalPolicy]:
    policies = dict(DEFAULT_RETRIEVAL_POLICIES)
    shared = {
        "candidate_documents": int(config.get("candidate_documents", 24)),
        "candidate_archive_documents": int(
            config.get("candidate_archive_documents", 32)
        ),
        "reserved_documents_per_task": int(
            config.get("reserved_documents_per_task", 2)
        ),
    }
    for strategy in ("single", "semantic", "multi_document", "conflicting"):
        policies[strategy] = replace(policies[strategy], **shared)
    policies["completeness"] = replace(
        policies["completeness"],
        candidate_documents=int(config.get("completeness_candidate_documents", 48)),
        candidate_archive_documents=int(
            config.get("completeness_candidate_archive_documents", 64)
        ),
        reserved_documents_per_task=int(
            config.get("completeness_reserved_documents_per_task", 10)
        ),
    )
    return policies
