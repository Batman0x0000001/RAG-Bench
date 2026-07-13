from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    """评测数据集的公开注册信息，不包含 Gold 内容。"""

    name: str
    source_type: str
    expected_questions: int
    role: str
    blind_questions_file: str
    gold_questions_file: str
    manifest_file: str
    qdrant_collection: str


DATASETS: dict[str, DatasetSpec] = {
    "github_dev": DatasetSpec(
        name="github_dev",
        source_type="github",
        expected_questions=39,
        role="development",
        blind_questions_file="data/evaluation/github_dev/questions.blind.jsonl",
        gold_questions_file="data/raw/questions.jsonl",
        manifest_file="data/processed/github_documents.jsonl",
        qdrant_collection="enterprise_rag_bench",
    ),
    "confluence_frozen": DatasetSpec(
        name="confluence_frozen",
        source_type="confluence",
        expected_questions=64,
        role="frozen_test",
        blind_questions_file="data/evaluation/confluence_frozen/questions.blind.jsonl",
        gold_questions_file="data/raw/questions.jsonl",
        manifest_file="data/processed/confluence_documents.jsonl",
        qdrant_collection="enterprise_rag_bench_confluence_frozen",
    ),
}


P0_FEATURE_DEFAULTS: dict[str, Any] = {
    "adaptive_planning": True,
    "llm_rerank": True,
    "fusion_rank_guard": False,
    "parent_expansion": True,
    "evidence_followup": True,
    "answer_repair": True,
}


ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "p0_full": {},
    "dense_only": {"retrieval.mode": "dense"},
    "bm25_only": {"retrieval.mode": "bm25"},
    "hybrid_rank_sum": {"retrieval.mode": "rank_sum"},
    "hybrid_rrf": {},
    "fixed_planning": {"features.adaptive_planning": False},
    "no_rerank": {"features.llm_rerank": False},
    "guarded_rerank": {"features.fusion_rank_guard": True},
    "no_parent_expansion": {"features.parent_expansion": False},
    "no_followup": {"features.evidence_followup": False},
    "no_answer_repair": {"features.answer_repair": False},
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def apply_experiment_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    """应用一个预注册消融；禁止自由组合，保证实验可归因。"""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    resolved = copy.deepcopy(config)
    resolved["features"] = copy.deepcopy(P0_FEATURE_DEFAULTS)
    resolved.setdefault("retrieval", {})["mode"] = "rrf"
    for path, value in ABLATION_VARIANTS[variant].items():
        _set_path(resolved, path, value)
    resolved.setdefault("experiment", {})["variant"] = variant
    return resolved


def changed_variant_paths(variant: str) -> set[str]:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return set(ABLATION_VARIANTS[variant])


def validate_ablation_registry() -> None:
    """除完整方案及其显式 RRF 别名外，每个消融只能改变一个因素。"""
    for name, overrides in ABLATION_VARIANTS.items():
        if name in {"p0_full", "hybrid_rrf"}:
            if overrides:
                raise ValueError(f"{name} must use the complete P0 configuration")
            continue
        if len(overrides) != 1:
            raise ValueError(f"{name} must change exactly one configuration path")


validate_ablation_registry()
