"""LLM 원인 판정 에이전트: evidence_bundle → CauseJudgment.

LLM은 4개 분석의 수렴 증거를 받아 primary cause를 판정한다.
SHAP 1위라도 다른 분석이 반박하면 기각 가능.
evidence가 불충분하면 needs_more_data=True로 재시도를 요청한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents import config
from agents.cause_analyzer.g_star_loader import GStarResult
from agents.logger import get_logger
from agents.schemas.cause import CauseJudgment, FeatureEvidence, SimForecast
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MAX_TOKENS = 1000

_SYSTEM = """당신은 반도체 FAB 병목 원인 판정 전문가입니다.
SHAP, 트렌드, 업스트림, G* 4가지 분석의 수렴 증거를 보고 실제 root cause를 판정합니다.
반드시 한국어로 답하고, 요청한 JSON 형식만 출력하세요."""


def _build_prompt(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
    g_star: GStarResult | None,
    retry_n: int,
) -> str:
    retry_note = f"\n※ 재시도 #{retry_n} — 더 긴 window 데이터가 추가되었습니다.\n" if retry_n > 0 else ""

    evidence_lines = []
    for ev in evidence_bundle:
        parts = [f"  {ev.feature} | votes={ev.votes} | confidence={ev.confidence}"]
        if ev.shap_rank is not None:
            direction = "↑병목" if (ev.shap_value or 0) > 0 else "↓완화"
            parts.append(f"    SHAP: rank={ev.shap_rank}, value={ev.shap_value:+.4f} ({direction})")
        if ev.trend_slope is not None:
            sig = "★유의" if ev.trend_significant else "미유의"
            parts.append(f"    트렌드: slope={ev.trend_slope:+.4f}/h, R²={ev.trend_r2:.3f} [{sig}]")
        if ev.upstream_match:
            parts.append(f"    업스트림: capacity 피처 해당 (과부하 TG: {', '.join(upstream_suspects[:3])})")
        if ev.g_star_p_value is not None:
            sig = "★통계유의" if ev.g_star_significant else "비유의"
            parts.append(f"    G* t-test: p={ev.g_star_p_value:.4f} [{sig}]")
        evidence_lines.append("\n".join(parts))

    evidence_text = "\n".join(evidence_lines) if evidence_lines else "  (증거 없음)"

    forecast_text = ""
    if sim_forecast:
        lines = [
            f"  {kpi}: {c.now:.3f} → {c.future:.3f} ({c.pct_change:+.1f}%)"
            for kpi, c in sim_forecast.kpi_delta.items()
        ]
        forecast_text = "\n[2시간 후 시뮬레이션 예측]\n" + "\n".join(lines)

    g_star_text = ""
    if g_star:
        confirmed = toolgroup in (g_star.toolgroups or [])
        g_star_text = f"\n[G* 분석] TG 포함여부: {'포함(통계 확인)' if confirmed else '미포함'}"

    return f"""반도체 FAB '{toolgroup}' 공정의 병목 원인을 아래 증거를 토대로 판정하세요.
{retry_note}
[피처별 4가지 분석 증거] (votes=0~4, HIGH≥3, MEDIUM=2, LOW≤1)
{evidence_text}
{forecast_text}
{g_star_text}

판정 규칙:
1. votes가 가장 높은 피처를 primary_cause로 선택하세요.
2. SHAP 1위라도 다른 3개 분석이 모두 반박하면 기각하고 dismissed에 추가하세요.
3. votes≤1이고 증거가 불명확하면 needs_more_data=true로 설정하세요.
4. cause_summary는 [주요 원인] / [악화 추세] / [업스트림(있을때)] / [2시간 전망(있을때)] 형식으로 작성하세요.

반드시 아래 JSON만 출력하세요:
{{
  "primary_cause": "피처명",
  "primary_confidence": "HIGH|MEDIUM|LOW",
  "primary_reasoning": "왜 이 피처가 primary인지 1~2문장",
  "secondary_causes": ["피처명", ...],
  "dismissed": ["피처명", ...],
  "dismissed_reason": "기각 이유 (없으면 빈 문자열)",
  "needs_more_data": false,
  "cause_summary": "[주요 원인] ... [악화 추세] ..."
}}"""


def judge(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
    g_star: GStarResult | None = None,
    retry_n: int = 0,
) -> CauseJudgment:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        result = _call_openai(toolgroup, evidence_bundle, upstream_suspects, sim_forecast, g_star, retry_n, api_key)
        if result:
            return result
    return _rule_based_judgment(toolgroup, evidence_bundle, upstream_suspects, sim_forecast)


def _call_openai(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
    g_star: GStarResult | None,
    retry_n: int,
    api_key: str,
) -> CauseJudgment | None:
    prompt = _build_prompt(toolgroup, evidence_bundle, upstream_suspects, sim_forecast, g_star, retry_n)
    try:
        from openai import APIError, OpenAI, RateLimitError

        client = OpenAI(api_key=api_key)

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIError)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call():
            return client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )

        response = _call()
        if response.usage:
            _record_tokens("llm_judge", response.usage.prompt_tokens, response.usage.completion_tokens)

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        raw_confidence = str(data.get("primary_confidence", "LOW")).upper()
        confidence = raw_confidence if raw_confidence in ("HIGH", "MEDIUM", "LOW") else "LOW"
        return CauseJudgment(
            primary_cause=str(data.get("primary_cause", "")),
            primary_confidence=confidence,
            primary_reasoning=str(data.get("primary_reasoning", "")),
            secondary_causes=list(data.get("secondary_causes", [])),
            dismissed=list(data.get("dismissed", [])),
            dismissed_reason=str(data.get("dismissed_reason", "")),
            needs_more_data=str(data.get("needs_more_data", "")).lower() in ("true", "1"),
            cause_summary=str(data.get("cause_summary", "")),
        )
    except Exception as e:
        _log.warning(f"[llm_judge] {type(e).__name__}: {e} — 규칙 기반으로 대체")
        return None


def _rule_based_judgment(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
) -> CauseJudgment:
    if not evidence_bundle:
        return CauseJudgment(
            primary_cause="unknown",
            primary_confidence="LOW",
            primary_reasoning="분석 데이터 없음",
            needs_more_data=True,
            cause_summary=f"{toolgroup}: 데이터 부족으로 원인 판정 불가",
        )

    top = evidence_bundle[0]
    secondary = [e.feature for e in evidence_bundle[1:] if e.votes >= 1]
    dismissed = [e.feature for e in evidence_bundle if e.votes == 0 and e.shap_rank is not None and e.shap_rank <= 2]

    needs_more = top.votes <= 1 and top.confidence == "LOW"

    parts = []
    direction = "상승" if (top.shap_value or 0) > 0 else "하락"
    parts.append(
        f"[주요 원인] {top.feature} (votes={top.votes}, confidence={top.confidence}) — "
        f"SHAP {(top.shap_value or 0):+.3f}, {direction}하며 병목 기여."
    )
    trend_sig = [e for e in evidence_bundle if e.trend_significant]
    if trend_sig:
        worst = trend_sig[0]
        parts.append(f"[악화 추세] {worst.feature}: {(worst.trend_slope or 0):+.4f}/h (R²={worst.trend_r2:.2f})")
    if upstream_suspects:
        parts.append(f"[업스트림] 과부하 TG: {', '.join(upstream_suspects[:2])}")
    if sim_forecast and sim_forecast.gets_worse:
        worst_kpi = max(sim_forecast.kpi_delta.items(), key=lambda x: abs(x[1].pct_change))
        parts.append(f"[2시간 전망] {worst_kpi[0]} {worst_kpi[1].pct_change:+.1f}% 악화 예측")

    return CauseJudgment(
        primary_cause=top.feature,
        primary_confidence=top.confidence,
        primary_reasoning=f"votes={top.votes}으로 4개 분석 중 가장 많은 지지. confidence={top.confidence}.",
        secondary_causes=secondary,
        dismissed=dismissed,
        dismissed_reason="SHAP 기여 있으나 트렌드·G* 미확인" if dismissed else "",
        needs_more_data=needs_more,
        cause_summary=" ".join(parts),
    )
