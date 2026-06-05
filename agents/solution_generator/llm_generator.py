"""규칙 기반 대응안 후보를 LLM으로 다듬어 엔지니어용 실행 지침으로 변환한다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.schemas.solution import SolutionCandidate
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 800


def _build_prompt(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidate: SolutionCandidate,
) -> str:
    cause_summary = cause_report.cause_summary
    forecast_text = ""
    if cause_report.sim_forecast:
        wip = cause_report.sim_forecast.kpi_delta.get("wip")
        wr = cause_report.sim_forecast.kpi_delta.get("wait_ratio")
        if wip:
            forecast_text = f"2시간 후 WIP {wip.now}→{wip.future} ({wip.pct_change:+.1f}%)"
        if wr:
            forecast_text += f", wait_ratio {wr.now}→{wr.future} ({wr.pct_change:+.1f}%)"

    return f"""당신은 반도체 FAB 운영 전문가입니다. 반드시 한국어로만 답변하세요.

병목 공정: {alert.toolgroup}  심각도: {alert.severity.value}  확률: {alert.probability:.1%}
원인 요약: {cause_summary}
{f"시뮬 예측: {forecast_text}" if forecast_text else ""}

아래 대응안에 대해:
1. 구체적인 실행 방법 (어떤 파라미터를 얼마나 바꿀 것인가)
2. 예상 효과와 주의사항
을 2~3문장으로 정리해주세요.

[대응안] {candidate.name}
  파라미터: {candidate.params.model_dump(exclude_none=True)}
  근거: {candidate.rationale}

답변:"""


def refine_candidates(
    alert: BottleneckAlert,
    cause_report: CauseReport,
    candidates: list[SolutionCandidate],
) -> list[SolutionCandidate]:
    """LLM으로 대응안 설명 보강. 실패 시 원본 반환."""
    if not candidates:
        return candidates
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return candidates

    try:
        from openai import OpenAI, RateLimitError, APIError

        client = OpenAI(api_key=api_key)
        prompt = _build_prompt(alert, cause_report, candidates[0])

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
        usage = response.usage
        if usage:
            _record_tokens("llm_generator", usage.prompt_tokens, usage.completion_tokens)
        llm_text = response.choices[0].message.content.strip()

        if not llm_text:
            _log.warning("[llm_generator] 빈 응답 수신 — 규칙 기반 유지")
            return candidates

        enriched = candidates[0].model_copy(update={"expected_effect": llm_text})
        return [enriched] + candidates[1:]

    except Exception as e:
        _log.warning(f"[llm_generator] {type(e).__name__} — 규칙 기반 유지")
        return candidates
