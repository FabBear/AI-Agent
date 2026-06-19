"""LLM 원인 판정 에이전트: raw evidence + rubric → CauseJudgment.

루브릭의 4개 차원을 raw 수치에 직접 적용해 평가하고,
차원 평가를 종합해 최종 원인을 판정한다.
코드가 미리 점수를 매기지 않고, 평가·판단은 LLM이 수행한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agents import config
from agents.cause_analyzer.g_star_loader import GStarResult, KpiEvidence
from agents.cause_analyzer.rubric import RUBRIC
from agents.logger import get_logger
from agents.schemas.cause import (
    CauseCategory,
    CauseJudgment,
    FeatureEvidence,
    SHAPFeature,
    SimForecast,
    TrendInsight,
)
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MAX_TOKENS = 1200
_MAX_INPUT_TOKENS_ESTIMATE = 3000

_SYSTEM = """당신은 반도체 FAB 병목 원인 판정 전문가입니다.

작업 순서:
1. 루브릭의 각 차원 관점을 raw 수치에 직접 적용해 차원별 평가를 수행한다.
2. 네 차원의 평가를 종합해 증거가 얼마나 일관되게 하나의 원인을 가리키는지
   전문가 관점에서 판단하고 최종 원인과 신뢰도를 결정한다.

주의:
- 수치 임계값을 기계적으로 적용하지 말 것. 맥락을 보고 판단한다.
- 신뢰도는 수식이 아니라 증거 수렴 정도에 대한 전문가적 판단이다.
- 차원 간 모순이 있을 경우 어느 쪽이 더 신뢰할 만한지 이유를 설명한다.
- 반드시 한국어로 답하고, 아래 JSON 형식만 출력한다."""

_OUTPUT_SCHEMA = """\
반드시 아래 JSON만 출력하세요:
{
  "rubric_eval": {
    "shap_dominance":                      "HIGH|MEDIUM|LOW",
    "shap_dominance_reason":               "수치를 직접 인용한 판단 근거",
    "trend_convergence":                   "HIGH|MEDIUM|LOW",
    "trend_convergence_reason":            "수치를 직접 인용한 판단 근거",
    "statistical_confirmation":            "HIGH|MEDIUM|LOW",
    "statistical_confirmation_reason":     "수치를 직접 인용한 판단 근거",
    "future_outlook":                      "HIGH|MEDIUM|LOW",
    "future_outlook_reason":               "수치를 직접 인용한 판단 근거"
  },
  "synthesis": "네 차원 평가를 종합했을 때 증거 수렴 정도와 불확실성에 대한 전문가 판단 2~3문장",
  "primary_category": "설비_포화|대기_누적|WIP_누적|공급_부족",
  "primary_cause": "카테고리 내 SHAP 기여가 가장 큰 대표 피처명",
  "primary_confidence": "HIGH|MEDIUM|LOW",
  "primary_reasoning": "synthesis 기반 최종 판정 근거 1~2문장",
  "secondary_causes": [],
  "dismissed": [],
  "dismissed_reason": "",
  "needs_more_data": false,
  "cause_summary": "[주요 원인] ... [트렌드] ... [통계] ... [전망] ..."
}"""


def _format_raw_evidence(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    g_star_evidence: list[KpiEvidence],
    sim_forecast: SimForecast | None,
    upstream_suspects: list[str],
    retry_n: int,
) -> str:
    retry_note = f"\n※ 재시도 #{retry_n} — 더 긴 window 데이터 반영됨.\n" if retry_n > 0 else ""

    total_pos = sum(abs(f.shap_value) for f in shap_top if f.shap_value > 0) or 1.0

    shap_lines = "\n".join(
        f"  {f.feature:<32} SHAP={f.shap_value:+.4f}  비율={abs(f.shap_value)/total_pos*100:.1f}%"
        for f in shap_top
    ) or "  (없음)"

    trend_lines = "\n".join(
        f"  {t.feature:<32} slope={t.slope_per_hour:+.3f}/h  R²={t.r2:.3f}  "
        f"{'★유의' if t.significant else '비유의'}"
        for t in trend_top
    ) or "  (없음)"

    if g_star_evidence:
        g_star_lines = "\n".join(
            f"  {e.kpi:<32} delta={e.delta_mean:+.4f}  p={e.t_p_adj:.4f}  "
            f"{'★유의' if e.significant else '비유의'}"
            for e in sorted(g_star_evidence, key=lambda x: (0 if x.significant else 1, x.t_p_adj))
        )
        g_star_note = "  ※ delta 양수=처리군이 베이스라인보다 악화, available_tool_ratio는 음수=악화"
        g_star_section = g_star_lines + "\n" + g_star_note
    else:
        g_star_section = "  (데이터 없음 — 해당 시점 포워드 시뮬 미수행)"

    if sim_forecast:
        sim_lines = "\n".join(
            f"  {kpi:<32} 현재={c.now:.3f}  →  예측={c.future:.3f}  ({c.pct_change:+.1f}%)"
            for kpi, c in sim_forecast.kpi_delta.items()
        )
    else:
        sim_lines = "  (데이터 없음)"

    upstream_text = ", ".join(upstream_suspects) if upstream_suspects else "없음"

    return f"""{retry_note}[raw 증거 — {toolgroup}]

SHAP 기여 (양수=병목 방향 기여, 전체 양수 SHAP 대비 비율 포함):
{shap_lines}

트렌드 분석 (slope: 시간당 변화량):
{trend_lines}

G* 통계 검정 (포워드 시뮬 vs 베이스라인 t-검정):
{g_star_section}

2시간 시뮬레이션 전망:
{sim_lines}

업스트림 의심 TG: {upstream_text}"""


def _build_prompt(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    g_star_evidence: list[KpiEvidence],
    sim_forecast: SimForecast | None,
    upstream_suspects: list[str],
    retry_n: int,
) -> str:
    evidence = _format_raw_evidence(
        toolgroup, shap_top, trend_top,
        g_star_evidence, sim_forecast, upstream_suspects, retry_n,
    )
    return f"{RUBRIC}\n{evidence}\n\n{_OUTPUT_SCHEMA}"


def judge(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    g_star_evidence: list[KpiEvidence],
    sim_forecast: SimForecast | None,
    upstream_suspects: list[str],
    retry_n: int = 0,
    # 룰 기반 fallback용 (evidence_aggregator 결과)
    evidence_bundle: list[FeatureEvidence] | None = None,
    categories: list[CauseCategory] | None = None,
) -> CauseJudgment:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        result = _call_openai(
            toolgroup, shap_top, trend_top,
            g_star_evidence, sim_forecast, upstream_suspects, retry_n, api_key,
        )
        if result:
            return result
    return _rule_based_judgment(toolgroup, shap_top, evidence_bundle or [], categories or [])


def _call_openai(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    g_star_evidence: list[KpiEvidence],
    sim_forecast: SimForecast | None,
    upstream_suspects: list[str],
    retry_n: int,
    api_key: str,
) -> CauseJudgment | None:
    prompt = _build_prompt(
        toolgroup, shap_top, trend_top,
        g_star_evidence, sim_forecast, upstream_suspects, retry_n,
    )

    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(config.LLM_MODEL)
        estimated_tokens = len(encoding.encode(prompt))
    except Exception:
        estimated_tokens = int(len(prompt) * 1.2)

    if estimated_tokens > _MAX_INPUT_TOKENS_ESTIMATE:
        _log.warning(
            f"[llm_judge] {toolgroup} 프롬프트 추정 토큰 {estimated_tokens} > "
            f"{_MAX_INPUT_TOKENS_ESTIMATE} — 룰 기반으로 대체"
        )
        return None

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

        raw_conf = str(data.get("primary_confidence", "LOW")).upper()
        confidence = raw_conf if raw_conf in ("HIGH", "MEDIUM", "LOW") else "LOW"

        # synthesis를 primary_reasoning 앞에 붙여 보존
        synthesis = str(data.get("synthesis", ""))
        reasoning = str(data.get("primary_reasoning", ""))
        combined_reasoning = f"{synthesis}\n{reasoning}".strip() if synthesis else reasoning

        _log.info(
            f"[llm_judge] {toolgroup} 루브릭 판정 완료 — "
            f"category={data.get('primary_category')}  confidence={confidence}"
        )

        return CauseJudgment(
            primary_category=str(data.get("primary_category", "")),
            primary_cause=str(data.get("primary_cause", "")),
            primary_confidence=confidence,
            primary_reasoning=combined_reasoning,
            secondary_causes=list(data.get("secondary_causes", [])),
            dismissed=list(data.get("dismissed", [])),
            dismissed_reason=str(data.get("dismissed_reason", "")),
            needs_more_data=str(data.get("needs_more_data", "")).lower() in ("true", "1"),
            cause_summary=str(data.get("cause_summary", "")),
        )

    except Exception as e:
        _log.warning(f"[llm_judge] {type(e).__name__}: {e} — 룰 기반으로 대체")
        return None


def _rule_based_judgment(
    toolgroup: str,
    shap_top: list[SHAPFeature],
    evidence_bundle: list[FeatureEvidence],
    categories: list[CauseCategory],
) -> CauseJudgment:
    """LLM 호출 실패 시 SHAP 기반 룰 판정."""
    _CATEGORY_MAP = {
        "설비_포화": {"utilization_avg", "max_util", "utilization_avg_delta_120", "max_util_delta_120"},
        "대기_누적": {"q_time_min", "wait_ratio", "q_time_min_delta_120", "wait_ratio_delta_120"},
        "WIP_누적":  {"wip", "wip_delta_120"},
        "공급_부족": {"available_tool_ratio"},
    }

    if not shap_top:
        return CauseJudgment(
            primary_category="",
            primary_cause="unknown",
            primary_confidence="LOW",
            primary_reasoning="SHAP 데이터 없음 — 판정 불가",
            needs_more_data=True,
            cause_summary=f"{toolgroup}: 분석 데이터 불충분",
        )

    # 카테고리별 SHAP 합산
    total_pos = sum(abs(f.shap_value) for f in shap_top if f.shap_value > 0) or 1.0
    cat_scores: dict[str, float] = {c: 0.0 for c in _CATEGORY_MAP}
    cat_top_feat: dict[str, tuple[str, float]] = {}

    for feat in shap_top:
        if feat.shap_value <= 0:
            continue
        for cat, members in _CATEGORY_MAP.items():
            if feat.feature in members:
                cat_scores[cat] += feat.shap_value / total_pos
                if cat not in cat_top_feat or feat.shap_value > cat_top_feat[cat][1]:
                    cat_top_feat[cat] = (feat.feature, feat.shap_value)

    top_cat = max(cat_scores, key=lambda c: cat_scores[c])
    top_score = cat_scores[top_cat]
    primary_feature = cat_top_feat.get(top_cat, (shap_top[0].feature, 0))[0]

    confidence: str
    if top_score >= 0.45:
        confidence = "HIGH"
    elif top_score >= 0.25:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # categories 결과도 참고 (있을 경우)
    if categories:
        top_cat_obj = next((c for c in categories if c.name == top_cat), None)
        if top_cat_obj and top_cat_obj.g_star_confirmed:
            confidence = "HIGH"

    return CauseJudgment(
        primary_category=top_cat,
        primary_cause=primary_feature,
        primary_confidence=confidence,
        primary_reasoning=(
            f"[룰 기반 판정] {top_cat} SHAP 기여율 {top_score*100:.1f}% "
            f"(LLM 호출 실패로 대체)"
        ),
        secondary_causes=[c for c, s in cat_scores.items() if c != top_cat and s > 0.1],
        dismissed=[c for c, s in cat_scores.items() if s <= 0.05],
        dismissed_reason="SHAP 기여율 낮음" if any(s <= 0.05 for s in cat_scores.values()) else "",
        needs_more_data=confidence == "LOW",
        cause_summary=f"[주요 원인] {top_cat} — {primary_feature} (SHAP {top_score*100:.1f}%)",
    )
