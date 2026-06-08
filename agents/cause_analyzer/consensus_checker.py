"""4가지 분석 결과 + G* 통계 검정 간 합의/불일치를 점수화한다."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.cause_analyzer.g_star_loader import GStarResult, KpiEvidence
from agents.schemas.cause import SHAPFeature, SimForecast, TrendInsight

_INVERSE_KPI = {"available_tool_ratio"}


@dataclass
class ConsensusReport:
    agreed_features: list[str]
    conflicted_features: list[str]
    upstream_aligns: bool
    sim_aligns: bool
    g_star_confirmed: bool
    g_star_upstream_confirmed: list[str]
    g_star_sig_kpis: list[KpiEvidence]
    g_star_proba: float = 0.0
    g_star_n_total: int = 0
    g_star_n_alarm: int = 0
    g_star_toolgroups_all: list[str] = field(default_factory=list)  # G* 전체 TG 목록
    confidence_level: str = "LOW"
    summary: str = ""


def check_consensus(
    shap_top: list[SHAPFeature],
    trend_top: list[TrendInsight],
    upstream_suspects: list[str],
    sim_forecast: SimForecast | None,
    toolgroup: str = "",
    g_star_toolgroups: list[str] | None = None,
    g_star_evidence: dict[str, list[KpiEvidence]] | None = None,
    g_star_tg_proba: dict[str, float] | None = None,
    g_star_n_total: int = 0,
) -> ConsensusReport:
    # SHAP에서 병목 위험 증가 피처
    shap_bad = {s.feature for s in shap_top if s.shap_value > 0}

    # 트렌드에서 악화 중인 피처
    trend_bad: set[str] = set()
    for t in trend_top:
        is_worsening = (
            -t.slope_per_hour > 0 if t.feature in _INVERSE_KPI else t.slope_per_hour > 0
        )
        if is_worsening:
            trend_bad.add(t.feature)

    agreed = sorted(shap_bad & trend_bad)
    conflicted = sorted(shap_bad ^ trend_bad)

    capacity_features = {"wip", "wait_ratio", "q_time_min"}
    shap_top_feature = shap_top[0].feature if shap_top else ""
    upstream_aligns = bool(upstream_suspects) and shap_top_feature in capacity_features
    sim_aligns = bool(sim_forecast and sim_forecast.gets_worse)

    # G* 통계 검정 확인
    g_star_set = set(g_star_toolgroups or [])
    g_star_confirmed = toolgroup in g_star_set
    g_star_upstream_confirmed = sorted(set(upstream_suspects) & g_star_set)

    # KPI별 t-test 결과 (유의미/비유의미 모두 포함)
    g_star_sig_kpis: list[KpiEvidence] = []
    if g_star_confirmed and g_star_evidence:
        tg_evidence = g_star_evidence.get(toolgroup, [])
        # 유의미한 것 먼저, 나머지는 delta_mean 절대값 순
        g_star_sig_kpis = sorted(
            tg_evidence,
            key=lambda e: (0 if e.significant else 1, -abs(e.delta_mean)),
        )

    # 신뢰도 산출 (G* 통계 검정은 가중치 2배)
    score = len(agreed)
    score += 1 if upstream_aligns else 0
    score += 1 if sim_aligns else 0
    score += 2 if g_star_confirmed else 0
    score += 1 if g_star_sig_kpis else 0  # KPI 수준 유의성 추가 보너스

    if score >= 4:
        confidence_level = "HIGH"
    elif score >= 2:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    parts = []
    if g_star_confirmed:
        parts.append("G* 통계 확인")
    if agreed:
        parts.append(f"SHAP·트렌드 합의: {', '.join(agreed)}")
    if conflicted:
        parts.append(f"불일치: {', '.join(conflicted)}")
    if upstream_aligns:
        parts.append("업스트림 보강")
    if g_star_upstream_confirmed:
        parts.append(f"G* 업스트림: {', '.join(g_star_upstream_confirmed)}")
    if sim_aligns:
        parts.append("시뮬 확인")
    summary = " / ".join(parts) if parts else "분석 간 방향 불명확"

    return ConsensusReport(
        agreed_features=agreed,
        conflicted_features=conflicted,
        upstream_aligns=upstream_aligns,
        sim_aligns=sim_aligns,
        g_star_confirmed=g_star_confirmed,
        g_star_upstream_confirmed=g_star_upstream_confirmed,
        g_star_sig_kpis=g_star_sig_kpis,
        g_star_proba=g_star_tg_proba.get(toolgroup, 0.0) if g_star_tg_proba else 0.0,
        g_star_n_total=g_star_n_total,
        g_star_n_alarm=len(g_star_set),
        g_star_toolgroups_all=sorted(g_star_set),
        confidence_level=confidence_level,
        summary=summary,
    )
