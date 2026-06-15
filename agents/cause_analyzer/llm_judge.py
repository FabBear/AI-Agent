"""LLM 원인 판정 에이전트: categories + evidence_bundle → CauseJudgment.

카테고리 단위로 수렴 증거를 집계한 뒤 LLM이 판정한다.
SHAP 1위 피처라도 카테고리 total_score가 낮으면 기각 가능.
needs_more_data=True 반환 시 window 확장 후 재시도.
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
from agents.schemas.cause import CauseCategory, CauseJudgment, FeatureEvidence, SimForecast
from agents.token_tracker import record as _record_tokens

_log = get_logger(__name__)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

_MAX_TOKENS = 1000
_MAX_INPUT_TOKENS_ESTIMATE = 2500  # 초과 시 프롬프트 축약
_MAX_CATEGORIES_IN_PROMPT = 4
_MAX_FEATS_PER_CATEGORY = 3

_SYSTEM = """당신은 반도체 FAB 병목 원인 판정 전문가입니다.
카테고리별 수렴 증거를 보고 실제 root cause 카테고리를 판정합니다.
반드시 한국어로 답하고, 요청한 JSON 형식만 출력하세요."""


def _build_prompt(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    categories: list[CauseCategory],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
    g_star: GStarResult | None,
    retry_n: int,
) -> str:
    retry_note = f"\n※ 재시도 #{retry_n} — 더 긴 window 데이터 반영됨.\n" if retry_n > 0 else ""

    # 입력 토큰 가드레일: categories 수 상한
    if len(categories) > _MAX_CATEGORIES_IN_PROMPT:
        _log.debug(
            f"[llm_judge] {toolgroup} categories {len(categories)}개 → 상위 {_MAX_CATEGORIES_IN_PROMPT}개로 축약"
        )
        categories = categories[:_MAX_CATEGORIES_IN_PROMPT]

    # 카테고리 섹션 (핵심 판정 근거)
    cat_lines = []
    symbols = "①②③④⑤⑥"
    feat_map = {e.feature: e for e in evidence_bundle}

    for i, cat in enumerate(categories):
        sym = symbols[i] if i < len(symbols) else f"({i+1})"
        extras = []
        if cat.n_trend_significant:
            extras.append(f"트렌드유의={cat.n_trend_significant}개")
        extras.append("G*=확인" if cat.g_star_confirmed else "G*=미확인")
        if cat.upstream_match:
            extras.append("업스트림=있음")

        feat_details = []
        for fn in cat.features[:_MAX_FEATS_PER_CATEGORY]:  # 카테고리당 피처 상한
            ev = feat_map.get(fn)
            if not ev:
                continue
            parts = []
            if ev.shap_value is not None:
                d = "↑병목" if ev.shap_value > 0 else "↓완화"
                parts.append(f"SHAP{ev.shap_value:+.3f}{d}")
            if ev.trend_significant and ev.trend_slope is not None:
                parts.append(f"Trend★{ev.trend_slope:+.3f}/h")
            if ev.g_star_significant:
                parts.append(f"G*(p={ev.g_star_p_value:.3f})")
            if parts:
                feat_details.append(f"{fn}({', '.join(parts)})")

        feat_str = "  피처: " + " / ".join(feat_details) if feat_details else ""
        cat_lines.append(
            f"  {sym} {cat.name}  SHAP기여={cat.shap_share_pct:.1f}%  "
            f"{' | '.join(extras)}  score={cat.total_score:.3f}  [{cat.confidence}]\n"
            + (f"  {feat_str}" if feat_str else "")
        )

    cat_text = "\n".join(cat_lines) if cat_lines else "  (분류 없음)"

    forecast_text = ""
    if sim_forecast:
        lines = [
            f"  {kpi}: {c.now:.2f}→{c.future:.2f} ({c.pct_change:+.1f}%)"
            for kpi, c in sim_forecast.kpi_delta.items()
        ]
        forecast_text = "\n[2시간 예측]\n" + "\n".join(lines)

    g_star_text = ""
    if g_star:
        confirmed = toolgroup in (g_star.toolgroups or [])
        conf_str = "포함(통계 확인)" if confirmed else "미포함"
        kpi_evs = g_star.kpi_evidence.get(toolgroup, []) if g_star.kpi_evidence else []
        if kpi_evs:
            kpi_lines = []
            for e in sorted(kpi_evs, key=lambda x: (0 if x.significant else 1, x.t_p_adj)):
                verdict = "★유의" if e.significant else "비유의"
                kpi_lines.append(f"  {e.kpi}: Δ={e.delta_mean:+.3f}  p={e.t_p_adj:.4f}  {verdict}")
            note = "※ p≈0=포워드 시뮬에서 베이스라인 대비 유의미하게 증가(병목 증거), p≈1=오히려 감소 또는 정상(병목 무관)"
            g_star_text = f"\n[G* 통계 검정] TG={conf_str}\n{note}\n" + "\n".join(kpi_lines)
        else:
            g_star_text = f"\n[G* 통계 검정]: {conf_str}  (KPI 증거 없음)"

    return f"""반도체 FAB '{toolgroup}' 공정 병목 원인 판정.
{retry_note}
[카테고리별 원인 수렴 분석] — 판정의 핵심 근거
{cat_text}
{forecast_text}
{g_star_text}

판정 기준:
1. total_score 가장 높은 카테고리 → primary_category
2. SHAP기여율 50% 이상이면 score 낮아도 우선 고려
3. G* 통계 검정에서 ★유의(p<0.05) KPI가 존재하면 → 해당 KPI가 속한 카테고리를 최우선 고려하고, primary_reasoning에 반드시 유의 KPI명과 p값을 인용할 것
4. 1·2위 score 차이 < 0.15이고 최고 score < 0.4 → needs_more_data=true
5. secondary_causes: primary_category와 score 차이가 0.10 미만인 카테고리만 포함 (보통 빈 배열)
6. cause_summary: [주요 원인 카테고리 + 대표피처] / [G* 통계: ★유의 KPI명(p값) — 있을 때만] / [악화 추세 — 있을 때만] / [2시간 전망 — 있을 때만]

반드시 아래 JSON만 출력하세요:
{{
  "primary_category": "카테고리명",
  "primary_cause": "카테고리 내 대표 피처명 (SHAP 기여 가장 큰 것)",
  "primary_confidence": "HIGH|MEDIUM|LOW",
  "primary_reasoning": "판정 근거 1~2문장 (수치 포함, G* ★유의 KPI 있으면 반드시 인용)",
  "secondary_causes": ["primary와 score 차이 0.10 미만인 카테고리만, 보통 빈 배열"],
  "dismissed": ["기각 카테고리명"],
  "dismissed_reason": "기각 이유 (없으면 빈 문자열)",
  "needs_more_data": false,
  "cause_summary": "[주요 원인] ... [G* 통계] ... [악화 추세] ..."
}}"""


def judge(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    categories: list[CauseCategory],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None = None,
    g_star: GStarResult | None = None,
    retry_n: int = 0,
) -> CauseJudgment:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        result = _call_openai(
            toolgroup, evidence_bundle, categories,
            upstream_suspects, sim_forecast, g_star, retry_n, api_key,
        )
        if result:
            return result
    return _rule_based_judgment(toolgroup, evidence_bundle, categories, upstream_suspects, sim_forecast)


def _call_openai(
    toolgroup: str,
    evidence_bundle: list[FeatureEvidence],
    categories: list[CauseCategory],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
    g_star: GStarResult | None,
    retry_n: int,
    api_key: str,
) -> CauseJudgment | None:
    prompt = _build_prompt(
        toolgroup, evidence_bundle, categories,
        upstream_suspects, sim_forecast, g_star, retry_n,
    )

    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(config.LLM_MODEL)
        estimated_tokens = len(encoding.encode(prompt))
    except Exception:
        estimated_tokens = int(len(prompt) * 1.2)
    if estimated_tokens > _MAX_INPUT_TOKENS_ESTIMATE:
        _log.warning(
            f"[llm_judge] {toolgroup} 프롬프트 추정 토큰 {estimated_tokens} > {_MAX_INPUT_TOKENS_ESTIMATE} "
            f"— 룰 기반으로 대체"
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
        return CauseJudgment(
            primary_category=str(data.get("primary_category", "")),
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
    categories: list[CauseCategory],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
) -> CauseJudgment:
    if not categories:
        top_feat = evidence_bundle[0] if evidence_bundle else None
        return CauseJudgment(
            primary_category="",
            primary_cause=top_feat.feature if top_feat else "unknown",
            primary_confidence="LOW",
            primary_reasoning="카테고리 분류 불가, 최고 score 피처로 대체",
            needs_more_data=True,
            cause_summary=f"{toolgroup}: 분석 데이터 불충분",
        )

    top = categories[0]

    # 카테고리 내 대표 피처: SHAP 기여 가장 큰 양수 피처
    feat_map = {e.feature: e for e in evidence_bundle}
    cat_evs = sorted(
        [feat_map[f] for f in top.features if f in feat_map and (feat_map[f].shap_value or 0) > 0],
        key=lambda e: -(e.shap_value or 0),
    )
    primary_feature = cat_evs[0].feature if cat_evs else (top.features[0] if top.features else "unknown")

    secondary = [c.name for c in categories[1:] if c.total_score > 0.1]
    dismissed = [c.name for c in categories if c.total_score <= 0.05]

    # 1·2위 score 차이 작고 전반적 score 낮으면 추가 데이터 요청
    needs_more = (
        len(categories) >= 2
        and abs(categories[0].total_score - categories[1].total_score) < 0.15
        and top.confidence == "LOW"
    )

    parts = [
        f"[주요 원인] {top.name} — SHAP 기여율 {top.shap_share_pct:.1f}%, score={top.total_score:.3f} [{top.confidence}]"
    ]
    if top.n_trend_significant:
        trend_feats = [f for f in top.features if feat_map.get(f) and feat_map[f].trend_significant]
        parts.append(f"[악화 추세] {', '.join(trend_feats)} 유의미 상승 추세")
    if upstream_suspects:
        parts.append(f"[업스트림] {', '.join(upstream_suspects[:2])} 과부하가 {top.name}에 영향")
    if sim_forecast and sim_forecast.gets_worse:
        worst = max(sim_forecast.kpi_delta.items(), key=lambda x: abs(x[1].pct_change))
        parts.append(f"[2시간 전망] {worst[0]} {worst[1].pct_change:+.1f}% 변화 예측")

    return CauseJudgment(
        primary_category=top.name,
        primary_cause=primary_feature,
        primary_confidence=top.confidence,
        primary_reasoning=(
            f"{top.name} 카테고리가 SHAP {top.shap_share_pct:.1f}% 기여로 가장 높음 "
            f"(score={top.total_score:.3f})."
        ),
        secondary_causes=secondary,
        dismissed=dismissed,
        dismissed_reason="SHAP 기여율 및 추가 증거 부족" if dismissed else "",
        needs_more_data=needs_more,
        cause_summary=" ".join(parts),
    )
