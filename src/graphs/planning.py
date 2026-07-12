from __future__ import annotations

import re
from typing import Any

from src.graphs.retrieval_policy import DEFAULT_RETRIEVAL_POLICIES, Strategy


PLAN_PROMPT = """You plan retrieval for an enterprise RAG system.
Classify the question as one strategy: single, semantic, multi_document,
conflicting, or completeness.

- single: one source document should contain the answer.
- semantic: one source is likely, but terminology is indirect or ambiguous.
- multi_document: several changes, projects, incidents, or artifacts are required.
- conflicting: versions, old/new behavior, or contradictory sources must be compared.
- completeness: the question asks for an exhaustive list or broad coverage.

Also classify source_scope as single_source or multiple_sources. Multiple requested
facts do not imply multiple sources: limits, labels, tiers, or enum values from one
feature/PR remain single_source. Comparisons between null, omitted, default, enabled,
or disabled states of the same parameter or feature also remain single_source. Use
multiple_sources only when independent projects, artifacts, incidents, versions, SDKs,
or linked records must each supply evidence.

Create one retrieval task per independent evidence requirement. Preserve exact names,
numbers, API terms, quoted phrases, SDKs, repositories, tickets, and named entities.
For conflicting questions, create separate previous and current slots. For exhaustive
or cross-project counting questions, use completeness and create a task for every named
group. Do not answer the question.

Return JSON only:
{{"strategy":"multi_document","source_scope":"multiple_sources","document_budget":6,
"requirements":["fact that the final answer must cover"],
"retrieval_tasks":[{{"requirement":"...","slot":"general",
"query":"focused search query"}}]}}

Question:
{question}
"""


EXACT_IDENTIFIER_PATTERN = re.compile(
    r"--[A-Za-z0-9][A-Za-z0-9_-]*|"
    r"[A-Za-z][A-Za-z0-9_-]*(?:[._/:][A-Za-z0-9_-]+)+|"
    r"\b[A-Z]{2,}[A-Z0-9-]*\b"
)


def extract_exact_query_identifiers(question: str) -> list[str]:
    identifiers = re.findall(r"`([^`]+)`", question)
    identifiers.extend(EXACT_IDENTIFIER_PATTERN.findall(question))
    return list(dict.fromkeys(identifier for identifier in identifiers if identifier))


def preserve_query_identifiers(query: str, question: str) -> str:
    missing = [
        identifier
        for identifier in extract_exact_query_identifiers(question)
        if identifier.lower() not in query.lower()
    ]
    if not missing:
        return query
    return f"{query} {' '.join(missing)}"


def normalize_plan(
    payload: dict[str, Any],
    question: str,
    max_queries: int,
    max_documents: int,
) -> dict[str, Any]:
    strategy = str(payload.get("strategy", "single"))
    if strategy not in DEFAULT_RETRIEVAL_POLICIES:
        strategy = "single"
    lowered_question = question.lower()
    same_source_state_comparison = (
        any(
            term in lowered_question
            for term in (" null", "omitted", "leaving it out", "left out")
        )
        and any(term in lowered_question for term in ("compared to", "versus", " vs "))
        and not any(
            term in lowered_question
            for term in ("previous version", "current version", "old version", "new version")
        )
    )
    same_release_components = (
        "release notes" in lowered_question
        and any(phrase in lowered_question for phrase in ("config flag", "configuration flag"))
        and not any(
            term in lowered_question
            for term in ("across ", "different projects", "multiple releases")
        )
    )
    single_change_with_observed_result = (
        bool(
            re.search(
                r"\bwhat (?:change|mechanism|proposal|update)\b",
                lowered_question,
            )
        )
        and any(
            phrase in lowered_question
            for phrase in ("was observed", "were observed", "measured", "benchmark")
        )
        and not any(
            phrase in lowered_question
            for phrase in (
                "different projects",
                "separate projects",
                "each sdk",
                "each project",
                "respectively",
            )
        )
        and not re.search(
            r"\bacross (?:the )?(?:python|typescript|go|java|sdk|project|"
            r"repository|version|release)s?\b",
            lowered_question,
        )
    )
    if (
        "complete list" in lowered_question
        or "all corresponding" in lowered_question
        or ("across " in lowered_question and "highest number" in lowered_question)
    ):
        strategy = "completeness"
    elif strategy == "single" and any(
        phrase in lowered_question
        for phrase in ("previous and current", "old and new", "no longer needed")
    ):
        strategy = "conflicting"
    raw_source_scope = payload.get("source_scope")
    source_scope = (
        raw_source_scope
        if raw_source_scope in {"single_source", "multiple_sources"}
        else (
            "multiple_sources"
            if strategy in {"conflicting", "completeness"}
            else "single_source"
        )
    )
    if (
        same_source_state_comparison
        or same_release_components
        or single_change_with_observed_result
    ):
        strategy = "single"
        source_scope = "single_source"
    elif strategy in {"conflicting", "completeness"}:
        source_scope = "multiple_sources"
    elif source_scope == "single_source" and strategy == "multi_document":
        strategy = "single"
    policy = DEFAULT_RETRIEVAL_POLICIES[strategy]

    raw_budget = payload.get("document_budget", policy.document_budget)
    try:
        budget = int(raw_budget)
    except (TypeError, ValueError):
        budget = policy.document_budget
    budget = max(
        policy.minimum_documents,
        min(budget, policy.document_budget, max_documents),
    )

    requirements = payload.get("requirements", [])
    if not isinstance(requirements, list):
        requirements = []
    requirements = [
        item.strip() for item in requirements if isinstance(item, str) and item.strip()
    ]
    if not requirements:
        requirements = [question]
    task_limit = min(max_queries, policy.max_queries)
    tasks: list[dict[str, str]] = []
    raw_tasks = payload.get("retrieval_tasks", [])
    if isinstance(raw_tasks, list):
        for item in raw_tasks:
            if len(tasks) >= task_limit:
                break
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            requirement = item.get("requirement")
            slot = item.get("slot", "general")
            tasks.append(
                {
                    "task_id": f"r{len(tasks) + 1}",
                    "requirement": (
                        requirement.strip()
                        if isinstance(requirement, str) and requirement.strip()
                        else requirements[min(len(tasks), len(requirements) - 1)]
                    ),
                    "slot": slot.strip() if isinstance(slot, str) and slot.strip() else "general",
                    "query": query.strip(),
                }
            )
    raw_queries = payload.get("queries", [])
    if not tasks and isinstance(raw_queries, list):
        for query in [question, *raw_queries]:
            if len(tasks) >= task_limit:
                break
            if isinstance(query, str) and query.strip() and all(
                task["query"] != query.strip() for task in tasks
            ):
                tasks.append(
                    {
                        "task_id": f"r{len(tasks) + 1}",
                        "requirement": requirements[min(len(tasks), len(requirements) - 1)],
                        "slot": "general",
                        "query": query.strip(),
                    }
                )
    if not tasks:
        tasks = [
            {
                "task_id": "r1",
                "requirement": requirements[0],
                "slot": "general",
                "query": question,
            }
        ]
    if strategy == "conflicting" and len(tasks) < 2:
        tasks = [
            {
                **tasks[0],
                "task_id": "r1",
                "slot": "previous",
                "query": f"{question} previous behavior",
            },
            {
                "task_id": "r2",
                "requirement": requirements[-1],
                "slot": "current",
                "query": f"{question} current behavior",
            },
        ]
    if strategy == "completeness":
        budget = min(policy.document_budget, max_documents)
    if strategy == "single":
        tasks = [
            {
                "task_id": "r1",
                "requirement": "\n".join(requirements),
                "slot": "general",
                "query": question,
            }
        ]
        budget = 1
    return {
        "strategy": strategy,
        "source_scope": source_scope,
        "queries": [task["query"] for task in tasks],
        "retrieval_tasks": tasks,
        "document_budget": budget,
        "minimum_documents": min(policy.minimum_documents, budget),
        "requirements": requirements,
    }
