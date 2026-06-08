"""SHAP + 트렌드 + 업스트림 + G* 결과를 LLM으로 한국어 원인 요약문으로 변환한다."""

import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.cause_analyzer.consensus_checker import ConsensusReport
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
    consensus: ConsensusReport | None = None,
) -> str:
    shap_text = "\n".join(
        f"  - {s.feature}: SHAP={s.shap_value:+.3f} (현재값={s.kpi_value:.3f})" for s in shap_top
    )
    trend_text = "\n".join(
        f"  - {t.feature}: {t.slope_per_hour:+.3f}/h  최근값={t.values}" for t in trend_top
    )

    forecast_section = ""
    if sim_forecast:
        lines = [
            f"  - {kpi}: {comp.now} → {comp.future} ({comp.pct_change:+.1f}% {'↑' if comp.delta > 0 else '↓'})"
            for kpi, comp in sim_forecast.kpi_delta.items()
        ]
        forecast_section = "\n[2시간 후 시뮬레이션 예측]\n" + "\n".join(lines)

    upstream_section = f"\n[업스트림 과부하 공정]\n{', '.join(upstream_suspects)}" if upstream_suspects else ""

    consensus_section = ""
    if consensus:
        g_star_text = "확인됨" if consensus.g_star_confirmed else "미확인"
        g_star_up = f" / G* 업스트림: {', '.join(consensus.g_star_upstream_confirmed)}" if consensus.g_star_upstream_confirmed else ""

        sig_kpi_text = ""
        if consensus.g_star_sig_kpis:
            confirmed_causes = [e for e in consensus.g_star_sig_kpis if e.significant]
            normal_kpis = [e for e in consensus.g_star_sig_kpis if not e.significant]
            lines = []
            for e in confirmed_causes:
                lines.append(f"  - {e.kpi}: Δ={e.delta_mean:+.1f}, p={e.t_p_adj:.4f} → ★ 통계적 원인 확인")
            for e in normal_kpis:
                lines.append(f"  - {e.kpi}: Δ={e.delta_mean:+.1f}, p={e.t_p_adj:.4f} → 정상 범위")
            conclusion = (
                f"통계적 원인 KPI: {', '.join(e.kpi for e in confirmed_causes)}"
                if confirmed_causes else
                "통계적으로 확인된 영구 원인 없음 → 일시적 과부하 가능성"
            )
            sig_kpi_text = f"\n[G* T-test — 병목 원인 검증 (FDR 보정)]\n" + "\n".join(lines) + f"\n  결론: {conclusion}"

        consensus_section = f"""
[분석 간 합의]
- 신뢰도: {consensus.confidence_level}
- G* 통계 검정: {g_star_text}{g_star_up}
- SHAP·트렌드 합의 피처: {', '.join(consensus.agreed_features) if consensus.agreed_features else '없음'}
- 불일치 피처: {', '.join(consensus.conflicted_features) if consensus.conflicted_features else '없음'}{sig_kpi_text}"""

    g_star_instruction = ""
    if consensus and consensus.g_star_confirmed:
        g_star_tg_list = ", ".join(consensus.g_star_toolgroups_all or [])
        confirmed_causes = [e for e in (consensus.g_star_sig_kpis or []) if e.significant]
        if confirmed_causes:
            cause_str = ", ".join(f"{e.kpi}(p={e.t_p_adj:.4f})" for e in confirmed_causes)
            g_star_instruction = f"※ G* T-test: 통계적 원인 KPI 확인됨 — {cause_str}. [주요 원인]에 이 KPI들이 통계적으로 비정상 상승이 확인된 원인임을 명시하세요."
        else:
            g_star_tg_str = f"({g_star_tg_list})" if g_star_tg_list else ""
            g_star_instruction = f"※ G* T-test: 통계적으로 확인된 영구 원인 KPI 없음{g_star_tg_str}. [주요 원인]에 'T-test 결과 일시적 과부하로 판단, 2시간 내 자연 해소 가능성'을 명시하세요."
    elif consensus:
        g_star_instruction = "※ 이 TG는 G* 분석 풀에 포함되지 않음 — [주요 원인]에 'G* 분석 대상 외, SHAP 기반 추정'임을 명시하세요."

    return f"""당신은 반도체 FAB 병목 원인 분석 전문가입니다. 반드시 한국어로만 답변하세요.
{g_star_instruction}

아래 데이터를 참고해서 '{toolgroup}' 공정의 병목 원인을 분석하세요.
반드시 아래 4개 항목만 출력하세요. 입력 데이터를 그대로 복사하지 마세요.
데이터가 없는 항목은 생략하세요. 각 항목은 1~2문장으로 작성하세요.

--- 참고 데이터 ---
[SHAP 기여도 상위]
{shap_text}

[KPI 트렌드]
{trend_text}
{upstream_section}
{forecast_section}
{consensus_section}
--- 참고 데이터 끝 ---

출력 형식:
[주요 원인] (원인 1~2문장, G* 결과 수치 포함)
[악화 추세] (트렌드 1~2문장)
[업스트림] (있을 때만)
[2시간 전망] (있을 때만)"""


def summarize(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
    consensus: ConsensusReport | None = None,
) -> str:
    """OpenAI LLM 호출. 실패 시 규칙 기반으로 폴백."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        return _call_openai(toolgroup, shap_top, trend_top, upstream_suspects, api_key, sim_forecast, consensus)
    return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast, consensus)


def _call_openai(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    api_key: str,
    sim_forecast: SimForecast | None = None,
    consensus: ConsensusReport | None = None,
) -> str:
    prompt = _build_prompt(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast, consensus)
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
            return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast, consensus)
        return result
    except Exception as e:
        _log.warning(f"[llm_summarizer] {type(e).__name__} — 규칙 기반으로 대체")
        return _rule_based_summary(toolgroup, shap_top, trend_top, upstream_suspects, sim_forecast, consensus)


def _rule_based_summary(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
    consensus: ConsensusReport | None = None,
) -> str:
    parts = []

    if shap_top:
        top = shap_top[0]
        direction = "상승" if top.shap_value > 0 else "하락"
        g_star_note = " (G* 통계 확인)" if consensus and consensus.g_star_confirmed else ""
        parts.append(
            f"병목 예측에 가장 크게 기여한 지표는 {top.feature}{g_star_note} (SHAP {top.shap_value:+.3f}, "
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
        worst_kpi = max(sim_forecast.kpi_delta.items(), key=lambda x: abs(x[1].pct_change))
        parts.append(
            f"조치 없이 방치 시 2시간 후 {worst_kpi[0]}가 "
            f"{worst_kpi[1].pct_change:+.1f}% 변화하여 상황이 악화될 것으로 예측됩니다."
        )

    return " ".join(parts) if parts else f"{toolgroup}: 복합적 원인으로 병목 위험이 감지됨."
