from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if value is not None:
            return int(value)
    return 0


def invoke_text_model(
    llm: BaseChatModel,
    prompt: Any,
    *,
    node: str,
    state: dict[str, Any],
    retry: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """调用模型并从标准 AIMessage 元数据提取 Token 与模型标识。"""
    started_at = _utc_now()
    started = perf_counter()
    response = llm.invoke(prompt)
    duration_ms = (perf_counter() - started) * 1000
    ended_at = _utc_now()
    usage = dict(getattr(response, "usage_metadata", None) or {})
    response_metadata = dict(getattr(response, "response_metadata", None) or {})
    if not usage and isinstance(response_metadata.get("usage"), dict):
        usage = dict(response_metadata["usage"])
    input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    event = {
        "node": node,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "duration_ms": round(duration_ms, 3),
        "model": response_metadata.get("model_name")
        or response_metadata.get("model")
        or getattr(llm, "model", None),
        "prompt_sha256": sha256(prompt_text.encode("utf-8")).hexdigest(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "retry": retry,
        "status": "success",
    }
    calls = list(state.get("model_calls", []))
    calls.append(event)
    return StrOutputParser().invoke(response), calls


def instrument_node(
    name: str,
    node: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """记录 LangGraph 节点端到端耗时，不改变节点业务结果。"""

    def _instrumented(state: dict[str, Any]) -> dict[str, Any]:
        started_at = _utc_now()
        started = perf_counter()
        updated = node(state)
        event = {
            "node": name,
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "candidate_documents": len(updated.get("candidate_groups", {})),
            "selected_documents": len(updated.get("selected_document_ids", [])),
            "status": "success",
        }
        events = list(updated.get("node_metrics", state.get("node_metrics", [])))
        events.append(event)
        return {**updated, "node_metrics": events}

    return _instrumented
