"""호환 shim — 챗봇 구현은 `app.chatbot.*` vertical slice로 이전됐다.

기존 import 경로(`from agents.chat_agent import ...`)를 깨지 않기 위해 공개/내부 심볼을
재노출만 한다. 신규 코드는 `app.chatbot.service` 등 실제 모듈을 직접 import할 것."""

from app.chatbot.config import DEFAULT_MODEL, chat_enabled  # noqa: F401
from app.chatbot.postprocess import (  # noqa: F401
    _confidence,
    _enforce_forecast_boundary,
    _is_future_wip_forecast,
    _parse_trailers,
    _pick_ui,
    _sanitize_answer,
    _validate_numbers,
)
from app.chatbot.prompts import (  # noqa: F401
    CARD_RULE_MSG,
    _routing_hint,
    _system_prompt,
    _to_messages,
)
from app.chatbot.rag import dedupe_sources as _dedupe_sources  # noqa: F401
from app.chatbot.rag import knowledge_block as _knowledge_block  # noqa: F401
from app.chatbot.rag import rag_search as _rag_search  # noqa: F401
from app.chatbot.service import (  # noqa: F401
    answer_chat,
    answer_chat_stream,
    build_agent as _build_agent,
    generate_title,
)
from app.chatbot.tools.registry import TOOL_LABELS as _TOOL_LABELS  # noqa: F401

__all__ = [
    "answer_chat",
    "answer_chat_stream",
    "chat_enabled",
    "generate_title",
    "DEFAULT_MODEL",
]
