"""SHAP + 트렌드 + 업스트림 결과를 LLM으로 한국어 원인 요약문으로 변환한다."""

import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.logger import get_logger
from agents.schemas.cause import SHAPFeature, SimForecast, TrendInsight
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MODEL = "gpt-4o-mini"
_MAX_TOKENS = 800


def _build_prompt(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
) -> str:
    shap_text = "\n".join(
        f"  - {s.feature}: SHAP={s.shap_value:+.3f} (현재값={s.kpi_value:.3f})" for s in shap_top
    )
    trend_text = "\n".join(
        f"  - {t.feature}: {t.slope_per_hour:+.3f}/h  최근값={t.values}" for t in trend_top
    )
    upstream_text = ", ".join(upstream_suspects) if upstream_suspects else "없음"

    forecast_section = ""
    if sim_forecast:
        lines = []
        for kpi, comp in sim_forecast.kpi_delta.items():
            arrow = "↑" if comp.delta > 0 else "↓"
            lines.append(f"  - {kpi}: {comp.now} → {comp.future} ({comp.pct_change:+.1f}% {arrow})")
        forecast_section = f"""
[2시간 후 시뮬레이션 예측]
{chr(10).join(lines)}
"""

    return f"""당신은 반도체 FAB 병목 원인 분석 전문가입니다. 반드시 한국어로만 답변하세요.
아래 분석 결과를 바탕으로 '{toolgroup}' 공정에 대해 다음 두 가지를 3~4문장으로 설명하세요.
1. 지금 병목이 발생한 원인
2. 조치하지 않으면 2시간 후 어떤 상황이 되는지 (시뮬 결과 기반)
운영 엔지니어가 즉시 판단할 수 있도록 구체적으로 한국어로 작성하세요.

[SHAP 기여도 상위]
{shap_text}

[KPI 트렌드 (악화 속도 순)]
{trend_text}

[과부하 업스트림 공정]
{upstream_text}
{forecast_section}
원인 및 전망:"""


def summarize(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
) -> str:
    """OpenAI LLM 호출. 실패 시 규칙 기반으로 폴백."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        return _call_openai(toolgroup, shap_top, trend_top, upstream_suspects, api_key, sim_forecast)
    return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast)


def _call_openai(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    api_key: str,
    sim_forecast: SimForecast | None = None,
) -> str:
    prompt = _build_prompt(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast)
    try:
        from openai import OpenAI, RateLimitError, APIError

        client = OpenAI(api_key=api_key)

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
            _record_tokens("llm_summarizer", usage.prompt_tokens, usage.completion_tokens)
        result = response.choices[0].message.content.strip()
        if not result:
            _log.warning("[llm_summarizer] 빈 응답 수신 — 규칙 기반으로 대체")
            return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast)
        return result
    except Exception as e:
        _log.warning(f"[llm_summarizer] {type(e).__name__} — 규칙 기반으로 대체")
        return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast)


def _rule_based_summary(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
) -> str:
    parts = []

    if shap_top:
        top = shap_top[0]
        direction = "상승" if top.shap_value > 0 else "하락"
        parts.append(
            f"병목 예측에 가장 크게 기여한 지표는 {top.feature} (SHAP {top.shap_value:+.3f}, "
            f"현재값 {top.kpi_value:.3f})으로 해당 값이 {direction}하며 병목 위험을 높이고 있습니다."
        )

    if trend_top:
        worst = trend_top[0]
        if abs(worst.slope_per_hour) > 0.01:
            parts.append(
                f"{worst.feature}이(가) 시간당 {worst.slope_per_hour:+.3f} 속도로 변화하고 있어 "
                f"지속적인 악화가 관찰됩니다."
            )

    if upstream_suspects:
        parts.append(
            f"업스트림 공정 {', '.join(upstream_suspects[:2])}의 가동률이 높아 "
            f"해당 공정으로의 WIP 과공급이 원인으로 의심됩니다."
        )

    if sim_forecast and sim_forecast.gets_worse:
        worst_kpi = max(
            sim_forecast.kpi_delta.items(),
            key=lambda x: abs(x[1].pct_change),
        )
        parts.append(
            f"조치 없이 방치 시 2시간 후 {worst_kpi[0]}가 "
            f"{worst_kpi[1].pct_change:+.1f}% 변화하여 상황이 악화될 것으로 예측됩니다."
        )

    return " ".join(parts) if parts else f"{toolgroup}: 복합적 원인으로 병목 위험이 감지됨."
