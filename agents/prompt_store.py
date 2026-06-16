"""DB에 등록된 활성 프롬프트를 읽어오는 작은 스토어.

DB 조회가 실패하거나 값이 비어 있으면 호출자가 넘긴 코드 기본 프롬프트를 사용한다.
매 호출마다 DB를 왕복하지 않도록 짧은 TTL 캐시를 둔다.
"""

from __future__ import annotations

import os
import time

from agents.logger import get_logger

_log = get_logger(__name__)
_TTL_SEC = float(os.getenv("PROMPT_CACHE_TTL_SEC", "30"))
_cache: dict[str, tuple[str, float]] = {}

_ACTIVE_BODY_SQL = """
    SELECT v.prompt_body
    FROM th_prompt_version v
    JOIN tm_prompt_template t ON t.template_id = v.template_id
    WHERE t.template_category = :category
      AND v.is_active = TRUE
    LIMIT 1
"""


def _fetch_active_body(category: str) -> str | None:
    try:
        from sqlalchemy import text

        from agents.sim_runner.db_connector import get_session

        with get_session() as session:
            row = session.execute(text(_ACTIVE_BODY_SQL), {"category": category}).fetchone()
    except Exception as exc:
        _log.warning(f"[prompt_store] '{category}' 조회 실패({type(exc).__name__}) — 기본 프롬프트 사용")
        return None

    if not row or not row[0]:
        return None

    body = str(row[0]).strip()
    if not body or (body.startswith("[") and "프롬프트 초안" in body):
        return None
    return body


def get_active_prompt(category: str, default: str) -> str:
    now = time.monotonic()
    cached = _cache.get(category)
    if cached is not None and (now - cached[1]) < _TTL_SEC:
        return cached[0]

    resolved = _fetch_active_body(category) or default
    _cache[category] = (resolved, now)
    return resolved


def invalidate(category: str | None = None) -> None:
    if category is None:
        _cache.clear()
    else:
        _cache.pop(category, None)
