from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DSID_PATTERN = re.compile(r"(dsid_[A-Za-z0-9]+)")
CONTROL_KEYS = {"dataset_doc_uuid", "title_field_name", "content_field_names"}
METADATA_KEYS = {
    "repo",
    "pr_number",
    "author",
    "created_at",
    "updated_at",
    "merged_at",
    "state",
    "base_branch",
    "head_branch",
    "reviewers",
    "labels",
    "linked_linear",
    "linked_jira",
    "ci_status",
    "files_changed_count",
    "additions",
    "deletions",
    "breaking_change",
    "merge_method",
    "merge_commit",
    "merge_commit_sha",
    "pr_url",
    "risk_level",
    "original_location",
}
HEADER_KEYS = [
    "repo",
    "pr_number",
    "author",
    "state",
    "labels",
    "reviewers",
]
CONTENT_FIELD_WHITELIST = {
    "description",
    "body",
    "review_comments",
    "review_thread",
    "review_timeline",
    "review_conversation",
    "discussion",
    "discussion_snippets",
    "release_notes",
    "notes_for_release",
    "notes_for_release_team",
    "changelog_entry",
    "file_summaries",
    "files_changed",
    "files_changed_summary",
    "files_changed_list",
    "changed_files",
    "changed_areas",
    "commits",
    "commit_summary",
    "commits_summary",
    "commit_summaries",
    "ci_checks",
    "ci_summary",
    "ci_log_summary",
    "post_merge_actions",
    "post_merge_tasks",
    "post_merge_notes",
    "merge_outcome",
    "merge_result",
    "merge_summary",
    "merge_details",
    "merge_info",
}
SECTION_FIELDS = {
    "description": {"description", "body"},
    "discussion": {
        "review_comments",
        "review_thread",
        "review_timeline",
        "review_conversation",
        "discussion",
        "discussion_snippets",
    },
    "release": {
        "release_notes",
        "notes_for_release",
        "notes_for_release_team",
        "changelog_entry",
    },
    "changes": {
        "files_changed",
        "files_changed_summary",
        "files_changed_list",
        "changed_files",
        "changed_areas",
        "file_summaries",
        "commits",
        "commit_summary",
        "commits_summary",
        "commit_summaries",
    },
    "ci": {"ci_status", "ci_checks", "ci_summary", "ci_log_summary"},
    "post_merge": {
        "post_merge_actions",
        "post_merge_tasks",
        "post_merge_notes",
        "merge_outcome",
        "merge_result",
        "merge_summary",
        "merge_details",
        "merge_info",
    },
}
OVERVIEW_KEYS = [
    "repo",
    "pr_number",
    "author",
    "state",
    "base_branch",
    "head_branch",
    "reviewers",
    "labels",
    "linked_linear",
    "linked_jira",
    "created_at",
    "updated_at",
    "merged_at",
    "merge_method",
    "files_changed_count",
    "additions",
    "deletions",
    "breaking_change",
]


class EnterpriseRagLoader(BaseLoader):
    """将 EnterpriseRAG-Bench 源文件加载为标准 LangChain Document。"""

    def __init__(
        self,
        path: str | Path,
        documents_dir: str | Path,
        source_type_hint: str = "github",
    ) -> None:
        self.path = Path(path)
        self.documents_dir = Path(documents_dir)
        self.source_type_hint = source_type_hint

    def lazy_load(self) -> Iterator[Document]:
        if self.path.suffix.lower() == ".json":
            yield from _load_json_documents(
                self.path,
                self.documents_dir,
                self.source_type_hint,
            )
            return
        if self.path.suffix.lower() == ".txt":
            yield _load_text_document(
                self.path,
                self.documents_dir,
                self.source_type_hint,
            )
            return
        raise ValueError(f"Unsupported document file type: {self.path}")


def parse_dsid_from_filename(filename: str) -> str:
    match = DSID_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot find dsid in filename: {filename}")
    return match.group(1)


def parse_source_type(relative_path: Path) -> str:
    # EnterpriseRAG-Bench 的目录顶层通常是 slack/gmail/github 等 source type。
    if len(relative_path.parts) <= 1:
        return "unknown"
    return relative_path.parts[0]


def iter_document_files(documents_dir: Path, extensions: tuple[str, ...]) -> Iterable[Path]:
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def extract_archives(archives_dir: str | Path, documents_dir: str | Path) -> int:
    archive_root = Path(archives_dir)
    output_root = Path(documents_dir)
    if not archive_root.exists():
        raise FileNotFoundError(f"Archives directory does not exist: {archive_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for archive in sorted(archive_root.glob("*.zip")):
        # 使用 zipfile 解压，避免依赖系统 unzip 命令。
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(output_root)
            extracted += len(
                [
                    name
                    for name in zip_file.namelist()
                    if name.endswith((".json", ".txt"))
                ]
            )
    return extracted


def resolve_document_path(
    documents_dir: str | Path,
    path: str | Path | None = None,
    source_type: str = "github",
) -> Path:
    root = Path(documents_dir)
    if path is not None:
        candidate = Path(path)
        if candidate.is_absolute() or candidate.exists():
            return candidate
        return root / candidate
    return root / source_type


def _relative_path(path: Path, documents_dir: Path) -> Path:
    try:
        return path.relative_to(documents_dir)
    except ValueError:
        return Path(path.name)


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {_format_value(item)}" for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return str(value)


def _compact_value(value: Any, max_chars: int = 180) -> str:
    text = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _simple_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"raw_format": "json"}
    for key in METADATA_KEYS:
        if key not in doc:
            continue
        value = doc[key]
        if isinstance(value, str | int | float | bool) or value is None:
            metadata[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, str | int | float | bool) or item is None for item in value
        ):
            metadata[key] = value
    return metadata


def _title(doc: dict[str, Any]) -> tuple[str, str]:
    title_field = doc.get("title_field_name")
    if not isinstance(title_field, str) or title_field not in doc:
        title_field = "title"
    return title_field, _format_value(doc.get(title_field, ""))


def _header(doc: dict[str, Any], section: str) -> str:
    _, title = _title(doc)
    lines = [f"title: {title}", f"section: {section}"]
    for key in HEADER_KEYS:
        if key in doc:
            lines.append(f"{key}: {_compact_value(doc[key])}")
    return "\n".join(lines)


def _content_field_names(doc: dict[str, Any], title_field: str) -> list[str]:
    fields: list[str] = []
    configured = doc.get("content_field_names")
    if isinstance(configured, list):
        fields.extend(
            field
            for field in configured
            if isinstance(field, str) and field in doc and field != title_field
        )

    for field in CONTENT_FIELD_WHITELIST:
        if field in doc and field not in fields and field != title_field:
            fields.append(field)
    return fields


def _field_to_items(field_name: str, value: Any) -> list[str]:
    if isinstance(value, list):
        return [f"{field_name}:\n{_format_value(item)}" for item in value]
    return [f"{field_name}:\n{_format_value(value)}"]


def _section_for_field(field_name: str) -> str | None:
    for section, fields in SECTION_FIELDS.items():
        if field_name in fields:
            return section
    return None


def _overview_text(doc: dict[str, Any]) -> str:
    _, title = _title(doc)
    lines = [f"title: {title}"]
    for key in OVERVIEW_KEYS:
        if key in doc:
            lines.append(f"{key}: {_format_value(doc[key])}")
    return "\n".join(lines)


def _collect_sections(doc: dict[str, Any], title_field: str) -> dict[str, dict[str, Any]]:
    overview_field_names = ["title"] + [key for key in OVERVIEW_KEYS if key in doc]
    sections: dict[str, dict[str, Any]] = {
        "overview": {"field_names": overview_field_names, "items": [_overview_text(doc)]}
    }
    for field_name in _content_field_names(doc, title_field):
        section = _section_for_field(field_name)
        if section is None:
            continue
        sections.setdefault(section, {"field_names": [], "items": []})
        sections[section]["field_names"].append(field_name)
        sections[section]["items"].extend(_field_to_items(field_name, doc[field_name]))
    if "ci_status" in doc:
        sections.setdefault("ci", {"field_names": [], "items": []})
        if "ci_status" not in sections["ci"]["field_names"]:
            sections["ci"]["field_names"].insert(0, "ci_status")
            sections["ci"]["items"].insert(0, f"ci_status: {_format_value(doc['ci_status'])}")
    return sections


def _json_documents(
    doc: dict[str, Any],
    dsid: str,
    source_type: str,
    relative_path: Path,
    filename: str,
) -> list[Document]:
    title_field, title = _title(doc)
    base_metadata = {
        **_simple_metadata(doc),
        "parent_doc_id": dsid,
        "title": title,
        "title_field_name": title_field,
    }
    sections = _collect_sections(doc, title_field)
    documents: list[Document] = []

    for section, section_data in sections.items():
        field_names = list(dict.fromkeys(section_data["field_names"]))
        section_text = "\n\n".join(
            item for item in section_data["items"] if item.strip()
        ).strip()
        if not section_text:
            continue
        documents.append(
            Document(
                page_content=section_text,
                metadata={
                    **base_metadata,
                    "dsid": dsid,
                    "source_type": source_type,
                    "relative_path": relative_path.as_posix(),
                    "filename": filename,
                    "field_name": section,
                    "section": section,
                    "field_names": field_names,
                    "context_header": _header(doc, section),
                },
            )
        )
    return documents


def _load_json_documents(
    path: Path,
    documents_dir: Path,
    source_type_hint: str = "github",
) -> list[Document]:
    relative_path = _relative_path(path, documents_dir)
    parsed_source_type = parse_source_type(relative_path)
    source_type = parsed_source_type if parsed_source_type != "unknown" else source_type_hint
    with path.open("r", encoding="utf-8") as file:
        doc = json.load(file)
    if not isinstance(doc, dict):
        raise ValueError(f"JSON document must be an object: {path}")

    dsid = doc.get("dataset_doc_uuid") or parse_dsid_from_filename(path.name)
    if not isinstance(dsid, str):
        raise ValueError(f"dataset_doc_uuid must be a string: {path}")

    return _json_documents(doc, dsid, source_type, relative_path, path.name)


def _load_text_document(
    path: Path,
    documents_dir: Path,
    source_type_hint: str = "github",
) -> Document:
    relative_path = _relative_path(path, documents_dir)
    parsed_source_type = parse_source_type(relative_path)
    source_type = parsed_source_type if parsed_source_type != "unknown" else source_type_hint
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    dsid = parse_dsid_from_filename(path.name)
    return Document(
        page_content=content,
        metadata={
            "dsid": dsid,
            "source_type": source_type,
            "relative_path": relative_path.as_posix(),
            "filename": path.name,
            "field_name": "text",
            "raw_format": "txt",
            "parent_doc_id": dsid,
            "section": "text",
            "field_names": ["text"],
        },
    )


def parse_document_file(
    path: Path,
    documents_dir: Path,
    source_type_hint: str = "github",
) -> list[Document]:
    return EnterpriseRagLoader(path, documents_dir, source_type_hint).load()


def split_documents(documents: Iterable[Document]) -> list[Document]:
    """按语义分区选择参数，并通过官方 TextSplitter API 生成可检索块。"""
    splitters = {
        "description": RecursiveCharacterTextSplitter(chunk_size=4_500, chunk_overlap=500),
        "discussion": RecursiveCharacterTextSplitter(chunk_size=4_500, chunk_overlap=300),
        "release": RecursiveCharacterTextSplitter(chunk_size=3_800, chunk_overlap=200),
        "changes": RecursiveCharacterTextSplitter(chunk_size=3_800, chunk_overlap=0),
        "text": RecursiveCharacterTextSplitter(chunk_size=3_800, chunk_overlap=200),
    }
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=3_800, chunk_overlap=200)
    chunks: list[Document] = []
    section_counts: Counter[tuple[str, str]] = Counter()
    for document in documents:
        section = str(document.metadata.get("section", "text"))
        splitter = splitters.get(section, default_splitter)
        for chunk in splitter.split_documents([document]):
            dsid = str(chunk.metadata["dsid"])
            key = (dsid, section)
            section_index = section_counts[key]
            section_counts[key] += 1
            chunk.metadata.update(
                {
                    "chunk_id": f"{dsid}::{section}::{section_index}",
                    "chunk_index": len(chunks),
                    "chunk_field_index": section_index,
                    "content_chars": len(chunk.page_content),
                }
            )
            context_header = chunk.metadata.get("context_header")
            if context_header:
                chunk.page_content = (
                    f"{context_header}\n\n"
                    f"Section: {section}\n"
                    f"Chunk: {section_index}\n\n"
                    f"{chunk.page_content}"
                )
            chunks.append(chunk)
    return chunks


def parse_documents(
    documents_dir: str | Path,
    limit: int | None = None,
    path: str | Path | None = None,
    source_type: str = "github",
    extensions: tuple[str, ...] = (".json", ".txt"),
) -> list[Document]:
    documents_root = Path(documents_dir)
    source_root = resolve_document_path(documents_root, path=path, source_type=source_type)
    if not source_root.exists():
        raise FileNotFoundError(f"Document path does not exist: {source_root}")

    loaded: list[Document] = []
    files = [source_root] if source_root.is_file() else iter_document_files(source_root, extensions)
    for file_index, file_path in enumerate(files):
        if limit is not None and file_index >= limit:
            break
        loaded.extend(EnterpriseRagLoader(file_path, documents_root, source_type).load())
    return split_documents(loaded)


def summarize_documents(documents: list[Document]) -> dict[str, Any]:
    total_files = len({document.metadata.get("relative_path") for document in documents})
    lengths = [len(document.page_content) for document in documents]
    sorted_lengths = sorted(lengths)

    def percentile(percent: float) -> int:
        if not sorted_lengths:
            return 0
        index = min(int((len(sorted_lengths) - 1) * percent), len(sorted_lengths) - 1)
        return sorted_lengths[index]

    section_counts = Counter(
        str(document.metadata.get("section") or "unknown")
        for document in documents
    )
    return {
        "total_files": total_files,
        "total_chunks": len(documents),
        "avg_chunks_per_pr": round(len(documents) / total_files, 2) if total_files else 0,
        "section_counts": dict(sorted(section_counts.items())),
        "length_distribution": {
            "min": min(lengths) if lengths else 0,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "max": max(lengths) if lengths else 0,
            "avg": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "chunks_under_200_chars": sum(1 for length in lengths if length < 200),
    }


def write_manifest(documents: list[Document], manifest_file: str | Path) -> None:
    path = Path(manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            row = {"page_content": document.page_content, "metadata": document.metadata}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(manifest_file: str | Path, limit: int | None = None) -> list[Document]:
    path = Path(manifest_file)
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if set(row) != {"page_content", "metadata"}:
                    raise ValueError(
                        "Manifest is not in standard Document format; run scripts.ingest again."
                    )
                documents.append(Document(**row))
            if limit is not None and len(documents) >= limit:
                break
    return documents
