"""Agent Task 룰베이스 빌더가 공유하는 컨텍스트 접근/포매팅 헬퍼."""

from typing import Any


def num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pct(value: Any) -> str:
    return f"{num(value) * 100:.1f}%"


def name_of(obj: dict[str, Any] | None, fallback: str = "선택 대상") -> str:
    if not obj:
        return fallback
    for key in ("tgName", "tgCode", "toolCode", "name", "label"):
        if obj.get(key):
            return str(obj[key])
    return fallback


def backend_context(context: dict[str, Any]) -> dict[str, Any]:
    backend = context.get("backendContext")
    return backend if isinstance(backend, dict) else {}


def backend_list(context: dict[str, Any], section: str, key: str) -> list[dict[str, Any]]:
    backend = backend_context(context)
    payload = backend.get(section)
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [item for item in payload[key] if isinstance(item, dict)]
    return []
