"""4개 분석 결과를 피처별 증거 번들(FeatureEvidence)로 집계한다."""

from __future__ import annotations

from agents.cause_analyzer.g_star_loader import KpiEvidence
from agents.schemas.cause import FeatureEvidence, SHAPFeature, TrendInsight

# 업스트림 과부하가 영향을 미치는 피처: upstream_suspects가 있으면 이 피처들에 1표
_UPSTREAM_CAPACITY_FEATURES = {"wip", "wait_ratio", "available_tool_ratio", "q_time_min"}


def aggregate_evidence(
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    g_star_kpi_evidence: list[KpiEvidence],
) -> list[FeatureEvidence]:
    """4개 분석 결과를 피처별로 집계해 FeatureEvidence 리스트를 반환한다.

    votes 기준:
    - SHAP: shap_value > 0인 경우 (병목 방향 기여)
    - 트렌드: significant == True
    - 업스트림: upstream_suspects가 있고 capacity/flow 피처인 경우
    - G*: t-test significant == True
    """
    all_features: set[str] = set()
    for f in shap_top:
        all_features.add(f.feature)
    for t in trend_top:
        all_features.add(t.feature)
    for g in g_star_kpi_evidence:
        all_features.add(g.kpi)

    shap_map = {f.feature: (i + 1, f) for i, f in enumerate(shap_top)}
    trend_map = {t.feature: t for t in trend_top}
    g_star_map = {e.kpi: e for e in g_star_kpi_evidence}
    upstream_nonempty = bool(upstream_suspects)

    result: list[FeatureEvidence] = []
    for feat in all_features:
        votes = 0
        shap_rank: int | None = None
        shap_value: float | None = None
        trend_slope: float | None = None
        trend_r2: float | None = None
        trend_significant = False
        g_star_p_value: float | None = None
        g_star_significant = False

        if feat in shap_map:
            rank, sf = shap_map[feat]
            shap_rank = rank
            shap_value = sf.shap_value
            if sf.shap_value > 0:
                votes += 1

        if feat in trend_map:
            t = trend_map[feat]
            trend_slope = t.slope_per_hour
            trend_r2 = t.r2
            trend_significant = t.significant
            if t.significant:
                votes += 1

        upstream_match = upstream_nonempty and feat in _UPSTREAM_CAPACITY_FEATURES
        if upstream_match:
            votes += 1

        if feat in g_star_map:
            e = g_star_map[feat]
            g_star_p_value = e.t_p_adj
            g_star_significant = e.significant
            if e.significant:
                votes += 1

        confidence = "HIGH" if votes >= 3 else "MEDIUM" if votes == 2 else "LOW"

        result.append(
            FeatureEvidence(
                feature=feat,
                votes=votes,
                shap_rank=shap_rank,
                shap_value=shap_value,
                trend_slope=trend_slope,
                trend_r2=trend_r2,
                trend_significant=trend_significant,
                upstream_match=upstream_match,
                g_star_p_value=g_star_p_value,
                g_star_significant=g_star_significant,
                confidence=confidence,
            )
        )

    result.sort(key=lambda e: (-e.votes, e.shap_rank if e.shap_rank is not None else 999))
    return result
