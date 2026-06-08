"""글로벌 플랜을 LLM으로 다듬어 엔지니어용 실행 지침(expected_effect)을 채운다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.schemas.solution import GlobalSolutionPlan
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 600


def _build_prompt(
    plan: GlobalSolutionPlan,
    alerts: list[BottleneckAlert],
    cause_map: dict[str, CauseReport],
) -> str:
    tg_lines = "\n".join(
        f"  - {a.toolgroup}: {cause_map[a.toolgroup].cause_summary[:100]}"
        for a in alerts
        if a.toolgroup in cause_map
    )
    return f"""당신은 반도체 FAB 운영 전문가입니다. 반드시 한국어로만 답변하세요.

[Lot Release 테이블 조정안 — 플랜 {plan.plan_id}]
- Release Interval: {plan.release_interval_minutes:.1f}분
- 투입 우선순위: {plan.lot_priority_rule or "변경 없음"}
- SUPERHOTLOT: {"활성화" if plan.superhotlot_enable else "비활성화"}

Critical 병목 TG 및 원인 요약:
{tg_lines if tg_lines else "  (없음)"}

이 파라미터로 시뮬레이션 실행 시 아래 두 가지를 2~3문장으로 요약하세요.
1. 기대 효과 (WIP / wait_ratio / Cycle Time 개선 관점)
2. 주의사항 또는 부작용

답변:"""


def refine_plan(
    plan: GlobalSolutionPlan,
    alerts: list[BottleneckAlert],
    cause_map: dict[str, CauseReport],
) -> GlobalSolutionPlan:
    """LLM으로 expected_effect 보강. 실패 시 description으로 fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return plan.model_copy(update={"expected_effect": plan.description})

    try:
        from openai import APIError, OpenAI, RateLimitError

        client = OpenAI(api_key=api_key)
        prompt = _build_prompt(plan, alerts, cause_map)

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIError)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call():
            return client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_MAX_TOKENS,
                temperature=0.2,
            )

        response = _call()
        if response.usage:
            _record_tokens("llm_generator", response.usage.prompt_tokens, response.usage.completion_tokens)

        llm_text = response.choices[0].message.content.strip()
        if not llm_text:
            _log.warning("[llm_generator] 빈 응답 — description으로 fallback")
            return plan.model_copy(update={"expected_effect": plan.description})

        return plan.model_copy(update={"expected_effect": llm_text})

    except Exception as e:
        _log.warning(f"[llm_generator] {type(e).__name__} — description으로 fallback")
        return plan.model_copy(update={"expected_effect": plan.description})
