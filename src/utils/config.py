from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$")


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "${RUN_NAME:RAG_Bench}",
    "data": {
        "archives_dir": "data/raw/archives",
        "documents_dir": "data/raw/generated_data/sources",
        "default_source_type": "github",
        "questions_file": "data/raw/questions.jsonl",
        "manifest_file": "data/processed/github_documents.jsonl",
    },
    "qdrant": {
        "url": "${QDRANT_URL:http://localhost:6333}",
        "api_key": "${QDRANT_API_KEY:}",
        "collection": "${QDRANT_COLLECTION:enterprise_rag_bench}",
        "vector_size": "${EMBEDDING_VECTOR_SIZE:1024}",
        "distance": "Cosine",
    },
    "embedding": {
        "provider": "openai_compatible",
        "model": "${SILICONFLOW_EMBEDDING_MODEL:Qwen/Qwen3-Embedding-0.6B}",
        "api_key": "${SILICONFLOW_API_KEY:}",
        "base_url": "${SILICONFLOW_BASE_URL:https://api.siliconflow.com/v1}",
    },
    "llm": {
        "provider": "anthropic_compatible",
        "model": "${DEEPSEEK_MODEL:deepseek-chat}",
        "api_key": "${DEEPSEEK_API_KEY:}",
        "base_url": "${DEEPSEEK_ANTHROPIC_BASE_URL:https://api.deepseek.com/anthropic}",
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    "retrieval": {
        "candidate_k": 40,
        "candidate_documents": 12,
        "max_documents": 8,
        "chunks_per_document": 2,
        "rerank_chunk_chars": 800,
        "fallback_documents": 6,
        "expanded_documents": 3,
        "max_parent_chunks": 8,
    },
    "graph": {
        "max_steps": 3,
    },
    "output": {
        "runs_dir": "runs",
    },
}


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    match = ENV_PATTERN.match(value)
    if not match:
        return value

    name, default = match.groups()
    return os.getenv(name, default or "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取项目配置：默认值来自代码，环境变量来自 .env，可选配置文件只做局部覆盖。"""
    load_dotenv()
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() == ".json":
                override = json.load(file) or {}
            else:
                override = yaml.safe_load(file) or {}
        config = _deep_merge(config, override)

    return _expand_env(config)


def project_path(path: str | Path) -> Path:
    """把配置中的相对路径解析为当前项目下的绝对路径。"""
    return Path(path).expanduser().resolve()
