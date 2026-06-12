"""Small coercion helpers for LLM-provided tool parameters."""

from __future__ import annotations

import re
from typing import Any


def safe_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    """Coerce tool args like "6시간" or "5개" without letting ValueError abort chat."""
    try:
        if isinstance(value, bool):
            parsed = default
        elif isinstance(value, int | float):
            parsed = int(value)
        else:
            match = re.search(r"-?\d+", str(value or ""))
            parsed = int(match.group(0)) if match else default
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
