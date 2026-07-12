from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.documents import Document


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{1,11}-\d{2,}|[Pp][Rr][-_ ]?#?\d{2,})\b",
)


def extract_link_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in ENTITY_PATTERN.findall(text):
        normalized = match.upper().replace(" ", "-").replace("_", "-")
        if normalized.startswith("PR"):
            number = re.search(r"\d+", normalized)
            if number:
                normalized = f"PR-{number.group()}"
        entities.add(normalized)
    return entities


def build_entity_link_index(
    parent_documents: dict[str, list[Document]],
    max_document_frequency: int = 20,
) -> dict[str, list[str]]:
    entity_documents: dict[str, set[str]] = defaultdict(set)
    for dsid, documents in parent_documents.items():
        text = "\n".join(document.page_content for document in documents)
        for entity in extract_link_entities(text):
            entity_documents[entity].add(dsid)
    return {
        entity: sorted(dsids)
        for entity, dsids in entity_documents.items()
        if 1 < len(dsids) <= max_document_frequency
    }


def expand_documents_by_entity_links(
    documents: list[Document],
    parent_documents: dict[str, list[Document]],
    entity_index: dict[str, list[str]],
    seed_documents: int = 8,
    max_linked_documents: int = 8,
    chunks_per_document: int = 1,
) -> tuple[list[Document], dict[str, list[str]]]:
    existing_ids = {
        str(document.metadata.get("dsid"))
        for document in documents
        if document.metadata.get("dsid")
    }
    linked_entities: dict[str, set[str]] = defaultdict(set)
    for document in documents[:seed_documents]:
        for entity in extract_link_entities(document.page_content):
            for dsid in entity_index.get(entity, []):
                if dsid not in existing_ids:
                    linked_entities[dsid].add(entity)

    ranked_ids = sorted(
        linked_entities,
        key=lambda dsid: (-len(linked_entities[dsid]), dsid),
    )[:max_linked_documents]
    expanded = list(documents[:seed_documents])
    trace: dict[str, list[str]] = {}
    for dsid in ranked_ids:
        entities = sorted(linked_entities[dsid])
        trace[dsid] = entities
        candidates = parent_documents.get(dsid, [])
        ranked_chunks = sorted(
            candidates,
            key=lambda document: -sum(
                entity.lower() in document.page_content.lower() for entity in entities
            ),
        )
        for document in ranked_chunks[:chunks_per_document]:
            expanded.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "retrieval_channels": ["entity_link"],
                        "linked_entities": entities,
                    },
                )
            )
    expanded.extend(documents[seed_documents:])
    return expanded, trace
