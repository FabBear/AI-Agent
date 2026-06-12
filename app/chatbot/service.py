"""챗봇 오케스트레이션 — 에이전트 빌드, 멀티턴 도구 루프, 응답 마감, 스트리밍.

answer_chat(단발)과 answer_chat_stream(SSE)이 공유하는 도구 실행/응답 마감을 헬퍼로 통합했다.
입력 가드레일(실행성 명령 차단)은 LLM 호출 전에 단락(short-circuit)한다."""

import logging
import os
import re
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.chatbot.config import DEFAULT_MODEL, chat_enabled, llm_tuning_kwargs
from app.chatbot.context import ChatContext
from app.chatbot.guardrails import detect_execution_intent, execution_block_response
from app.chatbot.postprocess import (
    FOLLOWUPS_MARKER,
    SPOKEN_MARKER,
    UI_MARKER,
    _parse_trailers,
)
from app.chatbot.prompts import CARD_RULE_MSG, _to_messages
from app.chatbot.runner import content_text, finalize_payload, run_tool_calls, tool_name
from app.chatbot.tools.registry import TOOL_LABELS, build_tools

logger = logging.getLogger(__name__)

_END_TOOLS_MSG = (
    "도구 조회는 여기서 충분하다. 더 이상 도구를 호출하지 말고, 지금까지 받은 도구 결과만 근거로 "
    "최종 답변을 작성하라. 트레일러 규칙도 지켜라."
)
_INCOMPLETE_RETRY_MSG = (
    "방금 답변은 미완성이다. '기다려 달라'는 답은 금지 — 지금 즉시 필요한 도구를 호출하고, "
    "그 결과 수치로 완결된 답변을 작성하라. 트레일러 규칙도 지켜라."
)
_PROMISE_RE = re.compile(r"(잠시만\s*기다|확인해\s*(보겠|드리겠)|불러오겠|조회해\s*보겠)")


def generate_title(message: str, model: str | None = None) -> str | None:
    """사용자 첫 메시지로 짧은 대화 제목 생성. 키 없음/오류 시 None."""
    if not chat_enabled() or not (message or "").strip():
        return None
    try:
        model_name = model or os.getenv("CHAT_AGENT_MODEL", DEFAULT_MODEL)
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=15,
            max_retries=0,
            max_completion_tokens=300,
            **llm_tuning_kwargs(model_name, 0.2),
        )
        system = (
            "사용자의 첫 메시지를 보고 대화 제목을 한국어로 아주 짧게 지어라. "
            "공백 포함 18자 이내, 따옴표·마침표·이모지 없이 명사구로. 반도체 FAB 운영 맥락. 제목만 출력."
        )
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=message[:500])])
        text = getattr(response, "content", "")
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        title = (text or "").strip().strip("\"'").splitlines()[0].strip() if text else ""
        return title[:30] or None
    except Exception:  # noqa: BLE001
        return None


def build_agent(live_status: str | None, fab_id: UUID | None, model: str | None) -> dict:
    """ChatContext + 도구(registry) + LLM 바인딩. ui_cards/used_hits/tools_used는 ctx 객체를 공유한다."""
    ctx = ChatContext(live_status=live_status, fab_id=fab_id)
    tools, tool_fns = build_tools(ctx)
    model_name = model or os.getenv("CHAT_AGENT_MODEL", DEFAULT_MODEL)
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=float(os.getenv("CHAT_AGENT_TIMEOUT_SEC", "30")),
        max_retries=int(os.getenv("CHAT_AGENT_MAX_RETRIES", "1")),
        max_completion_tokens=int(os.getenv("CHAT_AGENT_MAX_TOKENS", "1600")),
        **llm_tuning_kwargs(model_name, float(os.getenv("CHAT_AGENT_TEMPERATURE", "0.3"))),
    ).bind_tools(tools)
    return {
        "llm": llm,
        "tool_fns": tool_fns,
        "ctx": ctx,
        "ui_cards": ctx.ui_cards,
        "used_hits": ctx.used_hits,
        "tools_used": ctx.tools_used,
    }


def _guard_intent(message: str) -> bool:
    return os.getenv("CHATBOT_EXECUTION_GUARD", "1") != "0"


async def answer_chat(
    message: str,
    history: list[dict] | None = None,
    context: str | None = None,
    live_status: str | None = None,
    fab_id: UUID | None = None,
    model: str | None = None,
) -> dict | None:
    """LLM 답변 + 참고 출처 + 후속 질문(같은 호출에서) 생성. 키 없음/오류 시 None."""
    if not chat_enabled():
        return None
    if _guard_intent(message) and (category := detect_execution_intent(message)):
        return execution_block_response(category)

    agent = build_agent(live_status, fab_id, model)
    llm, ui_cards = agent["llm"], agent["ui_cards"]
    messages = _to_messages(message, history or [], context)
    ai = None
    card_rule_added = False
    for _ in range(int(os.getenv("CHAT_AGENT_MAX_TOOL_STEPS", "4"))):
        ai = await llm.ainvoke(messages)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break
        await run_tool_calls(agent, tool_calls, messages)
        if ui_cards and not card_rule_added:
            card_rule_added = True
            messages.append(SystemMessage(content=CARD_RULE_MSG))
    if ai is not None and (getattr(ai, "tool_calls", None) or []):
        messages.append(SystemMessage(content=_END_TOOLS_MSG))
        ai = await llm.ainvoke(messages)
        messages.append(ai)

    text = content_text(ai).strip()
    if text and _PROMISE_RE.search(text):
        messages.append(SystemMessage(content=_INCOMPLETE_RETRY_MSG))
        for _ in range(2):
            ai = await llm.ainvoke(messages)
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                break
            await run_tool_calls(agent, tool_calls, messages)
        retry_text = content_text(ai).strip()
        if retry_text:
            text = retry_text

    if not text:
        return None
    answer, spoken, followups, ui_type = _parse_trailers(text)
    if not answer:
        return None
    payload = finalize_payload(message, answer, spoken, followups, ui_type, agent, messages)
    if not payload["answer"]:
        return None
    return payload


async def answer_chat_stream(
    message: str,
    history: list[dict] | None = None,
    context: str | None = None,
    live_status: str | None = None,
    fab_id: UUID | None = None,
    model: str | None = None,
):
    """SSE 스트리밍 제너레이터. yield: stage/token/meta/error."""
    if not chat_enabled():
        yield {"type": "error", "message": "LLM 미구성(OPENAI_API_KEY 없음)"}
        return
    if _guard_intent(message) and (category := detect_execution_intent(message)):
        block = execution_block_response(category)
        yield {"type": "token", "text": block["answer"]}
        yield {"type": "meta", **block}
        return

    agent = build_agent(live_status, fab_id, model)
    llm, ui_cards = agent["llm"], agent["ui_cards"]
    messages = _to_messages(message, history or [], context)
    card_rule_added = False
    emitted = 0
    full_text = ""

    def _visible(txt: str, final: bool) -> str:
        cut = len(txt)
        for m in (SPOKEN_MARKER, FOLLOWUPS_MARKER, UI_MARKER):
            i = txt.find(m)
            if i != -1:
                cut = min(cut, i)
        if not final:
            cut = min(cut, max(0, len(txt) - 14))
        return txt[:cut]

    acc = None
    for _ in range(int(os.getenv("CHAT_AGENT_MAX_TOOL_STEPS", "4"))):
        acc = None
        async for chunk in llm.astream(messages):
            acc = chunk if acc is None else acc + chunk
            if getattr(acc, "tool_call_chunks", None) or getattr(acc, "tool_calls", None):
                continue
            vis = _visible(content_text(acc), final=False)
            if len(vis) > emitted:
                yield {"type": "token", "text": vis[emitted:]}
                emitted = len(vis)
        if acc is None:
            break
        messages.append(acc)
        tool_calls = getattr(acc, "tool_calls", None) or []
        if not tool_calls:
            full_text = content_text(acc).strip()
            break
        for call in tool_calls:
            name = tool_name(call)
            yield {"type": "stage", "tool": name, "label": TOOL_LABELS.get(name, "조회 중")}
        await run_tool_calls(agent, tool_calls, messages)
        if ui_cards and not card_rule_added:
            card_rule_added = True
            messages.append(SystemMessage(content=CARD_RULE_MSG))

    if not full_text and messages and isinstance(messages[-1], ToolMessage):
        messages.append(SystemMessage(content=_END_TOOLS_MSG))
        acc = None
        async for chunk in llm.astream(messages):
            acc = chunk if acc is None else acc + chunk
            if getattr(acc, "tool_call_chunks", None) or getattr(acc, "tool_calls", None):
                continue
            vis = _visible(content_text(acc), final=False)
            if len(vis) > emitted:
                yield {"type": "token", "text": vis[emitted:]}
                emitted = len(vis)
        if acc is not None:
            messages.append(acc)
            if not (getattr(acc, "tool_calls", None) or []):
                full_text = content_text(acc).strip()

    if not full_text:
        yield {"type": "error", "message": "응답 생성 실패"}
        return
    vis = _visible(full_text, final=True)
    if len(vis) > emitted:
        yield {"type": "token", "text": vis[emitted:]}
    answer, spoken, followups, ui_type = _parse_trailers(full_text)
    payload = finalize_payload(message, answer, spoken, followups, ui_type, agent, messages)
    yield {"type": "meta", **payload}
