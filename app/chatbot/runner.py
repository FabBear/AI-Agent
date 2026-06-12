"""Shared chat tool-loop helpers for non-stream and SSE paths."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.chatbot.postprocess import (
    _confidence,
    _enforce_forecast_boundary,
    _pick_ui,
    _sanitize_answer,
    _validate_numbers,
)
from app.chatbot.rag import dedupe_sources

logger = logging.getLogger(__name__)


def content_text(ai: Any) -> str:
    """Convert LangChain message/chunk content to text."""
    text = getattr(ai, "content", None) if ai is not None else None
    if isinstance(text, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    return text or ""


def tool_name(call: Any) -> str:
    return str(call.get("name", "")) if isinstance(call, dict) else ""


def tool_call_id(call: Any) -> str:
    return str(call.get("id", "")) if isinstance(call, dict) else ""


def tool_args(call: Any) -> dict:
    args = call.get("args", {}) if isinstance(call, dict) else {}
    return args if isinstance(args, dict) else {}


async def run_tool_calls(agent: dict, tool_calls: list, messages: list) -> None:
    for call in tool_calls:
        name = tool_name(call)
        fn = agent["tool_fns"].get(name)
        if fn:
            agent["tools_used"].append(name)
        try:
            result = fn(**tool_args(call)) if fn else "알 수 없는 도구입니다."
            if inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001 - one tool failure should not kill the whole chat turn.
            logger.exception("chat tool call failed: %s", name or "<unknown>")
            result = f"{name or '도구'} 실행 중 오류가 발생했습니다. 해당 데이터는 현재 사용할 수 없습니다."
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id(call)))


def finalize_payload(
    message: str,
    answer: str,
    spoken: str,
    followups: list[str],
    ui_type: str,
    agent: dict,
    messages: list,
) -> dict:
    """Apply deterministic post-processing and build the final chat payload."""
    answer = _enforce_forecast_boundary(message, answer)
    answer, guard_warnings = _sanitize_answer(answer)
    sources = dedupe_sources(agent["used_hits"])
    suspects = _validate_numbers(answer, messages)
    confidence, warnings = _confidence(agent["tools_used"], sources, suspects)
    if guard_warnings:
        confidence = "LOW"
        warnings = guard_warnings + warnings
    return {
        "answer": answer,
        "sources": sources,
        "followUps": followups,
        "spokenSummary": spoken,
        "ui": _pick_ui(agent["ui_cards"], ui_type, agent["tools_used"]),
        "toolsUsed": list(agent["tools_used"]),
        "confidence": confidence,
        "warnings": warnings,
    }
