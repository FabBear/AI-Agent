"""챗봇 공용 설정 — 모델/RAG/출처 필터 상수와 .env 로딩, OpenAI 키 존재 여부."""

import os

from dotenv import load_dotenv

from agents.agent_task.llm import _ROOT, llm_tuning_kwargs

load_dotenv(_ROOT / ".env")

__all__ = [
    "llm_tuning_kwargs",
    "chat_enabled",
    "DEFAULT_MODEL",
    "MAX_HISTORY",
    "MAX_CONTEXT_CHARS",
    "EMBED_MODEL",
    "RAG_TOP_K",
    "RAG_MIN_SCORE",
    "RAG_CHUNK_CHARS",
    "SOURCE_MAX",
    "SOURCE_TOP_GAP",
    "SOURCE_MIN_SCORE",
]

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_HISTORY = 12
MAX_CONTEXT_CHARS = int(os.getenv("CHAT_AGENT_MAX_CONTEXT_CHARS", "24000"))

EMBED_MODEL = "text-embedding-3-small"
RAG_TOP_K = 4
RAG_MIN_SCORE = 0.28
RAG_CHUNK_CHARS = 900
SOURCE_MAX = 2
SOURCE_TOP_GAP = 0.08
SOURCE_MIN_SCORE = 0.30


def chat_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
