"""4개 분석 결과를 피처별 증거 + 카테고리별 수렴으로 집계한다.

점수 체계:
  SHAP    : abs(shap_value) / total_positive_shap  (병목 방향인 경우만)
  트렌드  : R² × 0.4  (유의미 + 악화 방향인 경우만)
  업스트림: 0.1  (간접 증거)
  G*      : 0.5  (통계 확인 시)
"""

from __future__ import annotations

from agents.cause_analyzer.g_star_loader import KpiEvidence
from agents.schemas.cause import CauseCategory, FeatureEvidence, SHAPFeature, SimForecast, TrendInsight

_INVERSE_KPI = {"available_tool_ratio"}
_UPSTREAM_CAPACITY_FEATURES = {"wip", "wait_ratio", "available_tool_ratio", "q_time_min"}

_CATEGORY_MAP: dict[str, list[str]] = {
    "설비_포화": [
        "max_util", "max_util_delta_120",
        "utilization_avg", "utilization_avg_delta_120",
    ],
    "대기_누적": [
        "wait_ratio", "wait_ratio_delta_120",
        "q_time_min", "q_time_min_delta_120",
    ],
    "WIP_누적": [
        "wip", "wip_delta_120",
    ],
    "공급_부족": [
        "available_tool_ratio",
    ],
}

# G* evidence CSV의 KPI 이름 → 모델 피처 이름 정규화
_G_STAR_KPI_ALIAS: dict[str, str] = {
    "avg_qtime_min": "q_time_min",
    "avg_q_time_min": "q_time_min",
    "wip_count": "wip",
    "utilization_rate": "utilization_avg",
    "wait_ratio": "wait_ratio",
    "available_tool_ratio": "available_tool_ratio",
    "max_util": "max_util",
    "setup_ratio": "setup_ratio_avg",
}


def _norm_g_star(name: str) -> str:
    return _G_STAR_KPI_ALIAS.get(name, name)


def aggregate_evidence(
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    g_star_kpi_evidence: list[KpiEvidence],
    sim_forecast: SimForecast | None = None,
) -> tuple[list[FeatureEvidence], list[CauseCategory]]:
    """
    Returns:
        features  : 피처별 FeatureEvidence (세부 증거, score 내림차순)
        categories: 카테고리별 CauseCategory (수렴 요약, total_score 내림차순)
    """
    g_star_norm = [
        KpiEvidence(
            kpi=_norm_g_star(e.kpi),
            delta_mean=e.delta_mean,
            t_p_adj=e.t_p_adj,
            significant=e.significant,
        )
        for e in g_star_kpi_evidence
    ]

    total_shap_pos = sum(abs(f.shap_value) for f in shap_top if f.shap_value > 0) or 1.0

    shap_map = {f.feature: (i + 1, f) for i, f in enumerate(shap_top)}
    trend_map = {t.feature: t for t in trend_top}
    g_star_map = {e.kpi: e for e in g_star_norm}
    upstream_nonempty = bool(upstream_suspects)

    all_features: set[str] = set()
    for f in shap_top:        all_features.add(f.feature)
    for t in trend_top:       all_features.add(t.feature)
    for e in g_star_norm:     all_features.add(e.kpi)
    if sim_forecast:
        for feat in sim_forecast.kpi_delta:
            all_features.add(feat)

    features: list[FeatureEvidence] = []
    for feat in all_features:
        score = 0.0
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
                score += abs(sf.shap_value) / total_shap_pos
                votes += 1

        if feat in trend_map:
            t = trend_map[feat]
            trend_slope = t.slope_per_hour
            trend_r2 = t.r2
            trend_significant = t.significant
            if t.significant:
                is_worsening = (
                    t.slope_per_hour < 0 if feat in _INVERSE_KPI else t.slope_per_hour > 0
                )
                if is_worsening:
                    score += t.r2 * 0.4
                    votes += 1

        upstream_match = upstream_nonempty and feat in _UPSTREAM_CAPACITY_FEATURES
        if upstream_match:
            score += 0.1
            votes += 1

        if feat in g_star_map:
            e = g_star_map[feat]
            g_star_p_value = e.t_p_adj
            g_star_significant = e.t_p_adj <= 0.05
            if g_star_significant:
                direction_ok = (
                    e.delta_mean < 0 if feat in _INVERSE_KPI else e.delta_mean > 0
                )
                if direction_ok:
                    score += (0.05 - e.t_p_adj) / 0.05 * 0.5
                    votes += 1

        if sim_forecast and feat in sim_forecast.kpi_delta:
            cmp = sim_forecast.kpi_delta[feat]
            is_worsening = (
                cmp.pct_change < 0 if feat in _INVERSE_KPI else cmp.pct_change > 0
            )
            if is_worsening and abs(cmp.pct_change) > 5.0:
                score += min(abs(cmp.pct_change) / 100.0, 0.3)
                votes += 1

        confidence = "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.15 else "LOW"

        features.append(
            FeatureEvidence(
                feature=feat,
                votes=votes,
                score=round(score, 4),
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

    features.sort(key=lambda e: (-e.score, e.shap_rank if e.shap_rank is not None else 999))

    # ── 카테고리 집계
    feat_map = {e.feature: e for e in features}
    categories: list[CauseCategory] = []
    assigned: set[str] = set()

    for cat_name, cat_feats in _CATEGORY_MAP.items():
        matched = [f for f in cat_feats if f in feat_map]
        if not matched:
            continue

        cat_shap_pos = sum(
            abs(feat_map[f].shap_value or 0)
            for f in matched
            if (feat_map[f].shap_value or 0) > 0
        )
        shap_share = cat_shap_pos / total_shap_pos
        shap_share_pct = shap_share * 100

        n_trend_sig = sum(1 for f in matched if feat_map[f].trend_significant)
        ups_match = any(feat_map[f].upstream_match for f in matched)
        g_star_conf = any(feat_map[f].g_star_significant for f in matched)

        total_score = (
            shap_share
            + n_trend_sig * 0.15
            + (0.4 if g_star_conf else 0.0)
            + (0.08 if ups_match else 0.0)
        )

        confidence = (
            "HIGH"
            if (
                total_score >= 0.8
                or (shap_share >= 0.5 and (n_trend_sig > 0 or g_star_conf))
                or g_star_conf
            )
            else "MEDIUM" if total_score >= 0.25
            else "LOW"
        )

        for f in matched:
            assigned.add(f)

        categories.append(
            CauseCategory(
                name=cat_name,
                features=matched,
                shap_share_pct=round(shap_share_pct, 1),
                n_trend_significant=n_trend_sig,
                upstream_match=ups_match,
                g_star_confirmed=g_star_conf,
                total_score=round(total_score, 4),
                confidence=confidence,
            )
        )

    # 미분류 피처 (카테고리에 없는 것)
    unassigned = [e for e in features if e.feature not in assigned and e.score > 0]
    if unassigned:
        u_shap_pos = sum(abs(e.shap_value or 0) for e in unassigned if (e.shap_value or 0) > 0)
        u_n_trend = sum(1 for e in unassigned if e.trend_significant)
        u_score = u_shap_pos / total_shap_pos + u_n_trend * 0.15
        if u_score > 0.05:
            categories.append(
                CauseCategory(
                    name="기타",
                    features=[e.feature for e in unassigned],
                    shap_share_pct=round(u_shap_pos / total_shap_pos * 100, 1),
                    n_trend_significant=u_n_trend,
                    upstream_match=any(e.upstream_match for e in unassigned),
                    g_star_confirmed=any(e.g_star_significant for e in unassigned),
                    total_score=round(u_score, 4),
                    confidence="LOW",
                )
            )

    categories.sort(key=lambda c: -c.total_score)
    return features, categories
