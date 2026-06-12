"""Agent Task용 tool-calling 결정 루프.

LLM이 도구를 스스로 골라 호출 → 결과 보고 다음 행동 결정 → structured 결과 작성.
도구 호출마다 progress step을 남겨 '에이전트가 판단하는 과정'을 화면에 노출한다.
LLM 미사용/실패 시 결정론적 baseline으로 graceful degrade한다."""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.agent_task.context import TaskContext
from agents.agent_task.llm import llm_model_name, make_chat_llm, should_use_llm
from agents.agent_task.schemas import AgentTaskAgentResponse, AgentTaskResult

logger = logging.getLogger(__name__)

BaselineBuilder = Callable[..., AgentTaskResult]

COMPOSE_INSTRUCTION = (
    "지금까지 도구로 조회한 결과만 근거로 최종 결과를 작성하라. "
    "숫자·ID·TG명은 조회 결과에 나온 값만 인용하고 절대 지어내지 마라(없으면 '확인 필요'). "
    "확정 스케줄/Lot 순서 변경/실행·승인 지시는 출력하지 마라. "
    "structured schema(summary·evidence·propagation·responseDirections·references·followUpPrompts·artifacts"
    "·watchToolGroups)를 채워라."
)


async def run_agent_loop(
    ctx: TaskContext,
    *,
    tools: list,
    tool_fns: dict[str, Callable[..., Awaitable[str] | str]],
    tool_labels: dict[str, str],
    system_prompt: str,
    task_prompt: str,
    baseline_builder: BaselineBuilder,
    max_steps: int = 5,
) -> AgentTaskAgentResponse:
    """도구 기반 에이전트 루프를 돌려 AgentTaskAgentResponse를 만든다."""
    if not should_use_llm(ctx.req):
        ctx.trace("RULE_BASED", "로컬 룰베이스 모드로 현황을 집계했습니다.")
        return _respond(ctx, baseline_builder(ctx.req))

    try:
        model = llm_model_name(ctx.req)
        bound = make_chat_llm(ctx.req).bind_tools(tools)
        messages: list = [SystemMessage(content=system_prompt), HumanMessage(content=task_prompt)]
        ctx.trace("PLAN", f"{model}가 작업을 분석하고 필요한 조회를 계획합니다.")

        for _ in range(max(1, max_steps)):
            ai = await bound.ainvoke(messages)
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                break
            for call in tool_calls:
                await _exec_and_trace(ctx, tool_fns, tool_labels, call, messages)

        structured = make_chat_llm(ctx.req).with_structured_output(AgentTaskResult)
        result = await structured.ainvoke(messages + [HumanMessage(content=COMPOSE_INSTRUCTION)])
        if not isinstance(result, AgentTaskResult):
            result = AgentTaskResult.model_validate(result)
        if not (result.summary or "").strip():
            raise ValueError("빈 결과")
        ctx.trace("COMPOSE", "조회 결과를 종합해 결과를 작성했습니다.")
        return _respond(ctx, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent loop failed; falling back to baseline")
        ctx.trace("FALLBACK", f"에이전트 실행 실패로 룰베이스 결과를 반환합니다: {_short(exc)}")
        return _respond(ctx, baseline_builder(ctx.req))


async def _exec_and_trace(ctx, tool_fns, tool_labels, call, messages) -> None:
    name = str(call.get("name", "")) if isinstance(call, dict) else ""
    args = call.get("args", {}) if isinstance(call, dict) else {}
    args = args if isinstance(args, dict) else {}
    call_id = str(call.get("id", "")) if isinstance(call, dict) else ""
    fn = tool_fns.get(name)
    if fn:
        ctx.tools_used.append(name)
    try:
        result = await _call_tool(fn, args) if fn else "알 수 없는 도구입니다."
        if inspect.isawaitable(result):
            result = await result
    except Exception:  # noqa: BLE001
        logger.exception("agent tool call failed: %s", name or "<unknown>")
        result = f"{name or '도구'} 조회 실패 — 해당 데이터는 현재 사용할 수 없습니다."
    text = str(result)
    label = tool_labels.get(name, name or "조회")
    ctx.trace("TOOL_CALL", f"{label} — {_snippet(text)}")
    messages.append(ToolMessage(content=text, tool_call_id=call_id))


async def _call_tool(fn, args: dict) -> object:
    if inspect.iscoroutinefunction(fn):
        return await fn(**args)
    return await asyncio.to_thread(fn, **args)


def _respond(ctx: TaskContext, result: AgentTaskResult) -> AgentTaskAgentResponse:
    return AgentTaskAgentResponse(status="SUCCEEDED", progress=list(ctx.progress), result=result)


def _snippet(text: str, limit: int = 80) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _short(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message if len(message) <= 160 else message[:157] + "..."
