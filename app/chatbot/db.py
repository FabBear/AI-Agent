"""챗봇 전용 DB 접근 — 가능하면 read-only 풀로 도구 가드레일을 DB 권한으로도 강제한다.

CHATBOT_DB_URL이 설정되면(=SELECT만 가능한 read-only 계정) 그 풀을 쓰고,
없으면 공용 풀로 폴백한다(현재 단일 계정 환경). 챗봇 도구는 모두 조회 전용이므로
read-only 계정을 분리하면 코드 가드레일이 뚫려도 DB가 한 번 더 막아준다."""

import asyncio
import logging
import os

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)

_chat_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_chat_pool() -> asyncpg.Pool:
    """챗봇 조회 전용 풀. CHATBOT_DB_URL 미설정 시 공용 풀로 폴백."""
    readonly_dsn = os.getenv("CHATBOT_DB_URL")
    if not readonly_dsn:
        from app.db.pool import get_pool
        return await get_pool()

    global _chat_pool
    if _chat_pool is None:
        async with _pool_lock:
            if _chat_pool is None:
                settings = get_settings()
                logger.info("챗봇 read-only DB 풀 생성(CHATBOT_DB_URL)")
                _chat_pool = await asyncpg.create_pool(
                    dsn=readonly_dsn,
                    min_size=1,
                    max_size=int(os.getenv("CHATBOT_DB_POOL_MAX", "5")),
                    command_timeout=settings.agent_step_timeout_sec,
                )
    assert _chat_pool is not None
    return _chat_pool


async def close_chat_pool() -> None:
    global _chat_pool
    async with _pool_lock:
        if _chat_pool is not None:
            await _chat_pool.close()
            _chat_pool = None
