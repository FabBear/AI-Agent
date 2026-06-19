"""report_agent JSON v2 결정론적 빌더 (순수 함수).

표 / 카드 / 차트 데이터는 모두 이 모듈이 코드로 만든다.
LLM은 narrative(문장)만 담당. 환각·포맷 변형 가능성을 원천 차단.

원칙
====
- 업스트림에 있는 데이터만 채운다. 없는 건 None / [] / available=False.
- 임계값·토큰 매핑은 모두 도메인 규칙 표(상수)로 명시한다.
- 모든 함수는 외부 상태에 의존하지 않는 순수 함수다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI

from agents.report_agent.schema import (
    ActionCandidate,
    ActionParams,
    Actions,
    AffectedProcess,
    Approval,
    ApprovalToken,
    Cause,
    CauseCategoryItem,
    CauseDismissed,
    CauseGStar,
    CausePrimary,
    Confidence,
    ConfidenceToken,
    ConsensusAxes,
    DataQuality,
    DataQualityWarning,
    DecisionStatusToken,
    Diffusion,
    DirectionToken,
    EvidenceMatrixItem,
    ForwardSimResult,
    ForwardSimulation,
    GStarMonteCarlo,
    GStarSignificantKpi,
    IfNoAction,
    ImmediateAction,
    KpiCard,
    KpiChange,
    KpiImpactItem,
    Meta,
    MonitoringCheck,
    OperationalAttrs,
    Playbook,
    Provenance,
    Recommendation,
    ReleaseIntervalParam,
    ReliabilityToken,
    ReportV2,
    Risk,
    SeverityToken,
    ShapTopItem,
    SimulationStats,
    ThresholdStateToken,
    TrendPoint,
    TrendSeries,
    TrendSeriesFeature,
    VerdictToken,
    WhyNotOther,
)


# ── 도메인 규칙 상수 ─────────────────────────────────────────────────────────

# severity → 토큰·우선순위 (CauseAnalyzer writer 시스템 프롬프트 기준)
_SEVERITY_TOKEN: dict[str, SeverityToken] = {
    "Critical": SeverityToken.DANGER,
    "High":     SeverityToken.WARN,
    "Medium":   SeverityToken.CAUTION,
    "Low":      SeverityToken.OK,
}
_SEVERITY_PRIORITY: dict[str, int] = {
    "Critical": 0, "High": 1, "Medium": 2, "Low": 3,
}

_CONFIDENCE_TOKEN: dict[str, ConfidenceToken] = {
    "HIGH":   ConfidenceToken.HIGH,
    "MEDIUM": ConfidenceToken.MEDIUM,
    "LOW":    ConfidenceToken.LOW,
    "high":   ConfidenceToken.HIGH,
    "medium": ConfidenceToken.MEDIUM,
    "low":    ConfidenceToken.LOW,
}

_DECISION_STATUS_TOKEN: dict[str, DecisionStatusToken] = {
    "clear_winner":           DecisionStatusToken.WINNER,
    "equivalent_candidates":  DecisionStatusToken.EQUIVALENT,
    "no_meaningful_effect":   DecisionStatusToken.TENTATIVE,
}

_VERDICT_TOKEN: dict[str, VerdictToken] = {
    "improved":  VerdictToken.IMPROVED,
    "worsened":  VerdictToken.WORSENED,
    "unchanged": VerdictToken.NEUTRAL,
    "neutral":   VerdictToken.NEUTRAL,
    "baseline":  VerdictToken.BASELINE,
}

# KPI 단위 — 업스트림(compare _KPI_UNITS) 기준 + 보고서 자체 표시
_KPI_UNITS: dict[str, str] = {
    "q_time_min":           "min",
    "wait_ratio":           "ratio",
    "wip":                  "lots",
    "utilization_avg":      "ratio",
    "available_tool_ratio": "ratio",
    "max_util":             "ratio",
    "risk_score":           "score",
    "setup_ratio_avg":      "ratio",
}

# KPI 한글 라벨
_KPI_LABELS: dict[str, str] = {
    "risk_score":           "Risk Score",
    "q_time_min":           "평균 대기시간",
    "wait_ratio":           "Wait Ratio",
    "wip":                  "WIP",
    "utilization_avg":      "평균 가동률",
    "available_tool_ratio": "가용 호기 비율",
    "max_util":             "최대 가동률",
}

# 임계 상태 결정 함수 (도메인 규칙은 writer.py 시스템 프롬프트 L99-110과 일치)
def _risk_score_state(v: float | None) -> ThresholdStateToken | None:
    if v is None:
        return None
    if v >= 60: return ThresholdStateToken.DANGER
    if v >= 47: return ThresholdStateToken.CAUTION
    if v >= 25: return ThresholdStateToken.WARN
    return ThresholdStateToken.OK


def _wait_ratio_state(v: float | None) -> ThresholdStateToken | None:
    if v is None:
        return None
    if v > 1.0:   return ThresholdStateToken.DANGER
    if v >= 0.9:  return ThresholdStateToken.CAUTION
    if v >= 0.75: return ThresholdStateToken.WARN
    return ThresholdStateToken.OK


def _utilization_state(v: float | None) -> ThresholdStateToken | None:
    if v is None:
        return None
    if v >= 0.95: return ThresholdStateToken.DANGER
    if v >= 0.85: return ThresholdStateToken.CAUTION
    if v >= 0.6:  return ThresholdStateToken.WARN
    return ThresholdStateToken.OK


# 영향 공정 필터 임계 — 도메인적으로 '의미 있는 부하'
_HIGH_IMPACT_UTIL_THRESHOLD       = 0.8   # 80% 이상
_HIGH_IMPACT_WAIT_RATIO_THRESHOLD = 0.5   # 0.5 이상
_HIGH_IMPACT_WIP_THRESHOLD        = 50    # 50 lots 이상

# 데이터 품질 이상치 — wait_ratio는 정상적으로 0~수십, >10이면 의심
_WAIT_RATIO_ANOMALY_THRESHOLD = 10.0


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _round(v: Any, n: int = 4) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _pct_change(now: float | None, after: float | None) -> float | None:
    if now is None or after is None or now == 0:
        return None
    return round((after - now) / now * 100, 2)


# ── 블록별 빌더 ──────────────────────────────────────────────────────────────

def build_meta(
    tg: str,
    alert: BottleneckAlert,
    detected_at: str,
    snapshot_time: float | None = None,
    horizon_min: int | None = None,
    area_name: str | None = None,
) -> Meta:
    sev = alert.severity.value
    return Meta(
        toolgroup=tg,
        process_name=tg,
        area_name=area_name,
        severity=sev,
        severity_token=_SEVERITY_TOKEN.get(sev, SeverityToken.OK),
        severity_priority=_SEVERITY_PRIORITY.get(sev, 3),
        detected_at=detected_at,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        snapshot_time=snapshot_time,
        horizon_min=horizon_min,
    )


def build_approval(approval_info: dict) -> Approval | None:
    if not approval_info:
        return None
    status = approval_info.get("status") or "-"
    approver = approval_info.get("approved_by") or ""
    is_auto = approver.upper() == "AUTO"
    is_rejected = status == "반려"

    token = (
        ApprovalToken.REJECTED if is_rejected
        else ApprovalToken.AUTO_APPROVED if is_auto
        else ApprovalToken.APPROVED
    )
    return Approval(
        status=status,
        status_token=token,
        approver_name=approval_info.get("approved_by") or None,
        approver_role=approval_info.get("approved_role") or None,
        approved_at=approval_info.get("approved_at") or None,
        comment=approval_info.get("comment") or None,
        rejection_reason=approval_info.get("rejection_reason") or None,
        selected_label=approval_info.get("selected_label") or None,
    )


def build_risk(alert: BottleneckAlert) -> Risk:
    # risk_score는 composite_score * 100 (현재 _build_draft_item 규칙과 동일)
    score = _round(alert.composite_score * 100, 1) if alert.composite_score is not None else None
    return Risk(
        score=score,
        score_unit="score",
        score_threshold_state=_risk_score_state(score),
        composite_score=_round(alert.composite_score, 4),
        probability=_round(alert.probability, 4),
    )


def build_confidence(cause_report: CauseReport | None) -> Confidence:
    if cause_report is None:
        return Confidence()
    judgment = cause_report.judgment
    consensus = cause_report.consensus
    level = (judgment.primary_confidence if judgment else None) or consensus.confidence_level
    return Confidence(
        level=level,
        level_token=_CONFIDENCE_TOKEN.get(level) if level else None,
        needs_more_data=bool(judgment.needs_more_data) if judgment else False,
        g_star_probability=_round(consensus.g_star_proba, 4) if consensus else None,
    )


def build_if_no_action(current_state: dict) -> IfNoAction:
    """compare.current_state.natural_forecast_2h → IfNoAction."""
    nf = (current_state or {}).get("natural_forecast_2h") or {}
    if not nf:
        return IfNoAction(available=False)

    kpi_changes: list[KpiChange] = []
    for kpi_name, info in (nf.get("kpi") or {}).items():
        if not isinstance(info, dict):
            continue
        reliability = info.get("reliability")
        kpi_changes.append(KpiChange(
            kpi=kpi_name,
            unit=_KPI_UNITS.get(kpi_name, ""),
            now=_round(info.get("now"), 4),
            after=_round(info.get("after"), 4),
            delta=_round(info.get("delta"), 4),
            pct_change=_round(info.get("pct_change"), 2),
            reliability_token=ReliabilityToken(reliability) if reliability in {"HIGH", "MED", "LOW"} else None,
        ))

    return IfNoAction(
        available=True,
        will_get_worse=bool(nf.get("gets_worse")) if nf.get("gets_worse") is not None else None,
        horizon_min=120,  # compare는 항상 2h forecast
        kpi_changes=kpi_changes,
    )


def build_bottleneck_kpis(
    kpi: ToolGroupKPI | None,
    prev_kpi: ToolGroupKPI | None,
    alert: BottleneckAlert,
    trend_series: TrendSeries | None = None,
) -> list[KpiCard]:
    """anchor TG의 현재 KPI 카드. prev_kpi가 있으면 delta 포함."""
    cards: list[KpiCard] = []

    # risk_score 카드 (alert 기반)
    risk_value = _round(alert.composite_score * 100, 1) if alert.composite_score is not None else None
    cards.append(KpiCard(
        key="risk_score",
        label=_KPI_LABELS["risk_score"],
        value=risk_value,
        unit="score",
        threshold_state=_risk_score_state(risk_value),
    ))

    if kpi is None:
        return cards

    def _add(key: str, value: Any, threshold_fn=None, label: str | None = None, unit: str | None = None):
        prev = getattr(prev_kpi, key, None) if prev_kpi else None
        v = _round(value, 4)
        p = _round(prev, 4) if prev is not None else None
        delta = _round((v - p), 4) if (v is not None and p is not None) else None
        cards.append(KpiCard(
            key=key,
            label=label or _KPI_LABELS.get(key, key),
            value=v,
            unit=unit if unit is not None else _KPI_UNITS.get(key, ""),
            threshold_state=threshold_fn(v) if threshold_fn else None,
            prev_value=p,
            delta=delta,
            pct_change=_pct_change(p, v),
        ))

    _add("q_time_min",           kpi.q_time_min)
    _add("wait_ratio",           kpi.wait_ratio,           _wait_ratio_state)
    _add("utilization_avg",      kpi.utilization_avg,      _utilization_state)
    _add("wip",                  _safe_int(kpi.wip),       unit="lots")
    _add("available_tool_ratio", kpi.available_tool_ratio)
    _add("max_util",             kpi.max_util,             _utilization_state)

    return cards


def build_bottleneck_trend(cause_report: CauseReport | None) -> list[TrendPoint]:
    """cause_report.trend_top[*].values (5점 시계열) → TrendPoint list.

    각 trend_top의 values는 같은 길이로 정렬되어 있다고 가정.
    가장 긴 길이를 기준으로 시간축 만든다.
    """
    if cause_report is None or not cause_report.trend_top:
        return []

    n = max((len(t.values) for t in cause_report.trend_top), default=0)
    if n == 0:
        return []

    # T-0이 마지막 인덱스라 가정 (writer.py:218~221 로직과 일치)
    points: list[TrendPoint] = []
    for i in range(n):
        offset_min = -(n - 1 - i) * 60
        time_label = f"T-{(n - 1 - i) * 60}분" if i < n - 1 else "T-0분"
        values: dict[str, float | None] = {}
        for t in cause_report.trend_top:
            if i < len(t.values):
                values[t.feature] = _round(t.values[i], 4)
            else:
                values[t.feature] = None
        points.append(TrendPoint(
            time_label=time_label,
            offset_min=offset_min,
            values=values,
        ))
    return points


def build_trend_series(cause_report: CauseReport | None) -> TrendSeries | None:
    """cause.trend_series (slope·R²·significant) — 차트 데이터."""
    if cause_report is None or not cause_report.trend_top:
        return None

    n = max((len(t.values) for t in cause_report.trend_top), default=0)
    if n == 0:
        return None

    time_labels = [f"T-{(n - 1 - i) * 60}분" for i in range(n)]
    time_labels[-1] = "T-0분"
    time_offsets = [-(n - 1 - i) * 60 for i in range(n)]

    features: dict[str, TrendSeriesFeature] = {}
    for t in cause_report.trend_top:
        features[t.feature] = TrendSeriesFeature(
            values=[_round(v, 4) for v in t.values],
            slope_per_hour=_round(t.slope_per_hour, 4),
            r2=_round(t.r2, 4),
            significant=bool(t.significant),
        )
    return TrendSeries(
        time_labels=time_labels,
        time_offsets_min=time_offsets,
        features=features,
    )


# ── Diffusion ────────────────────────────────────────────────────────────────

def _classify_impact(
    util_pct: float | None,
    wait_ratio: float | None,
    wip: int | None,
) -> bool:
    """high-impact 여부. 임계 중 하나라도 넘으면 high."""
    if util_pct is not None and util_pct >= _HIGH_IMPACT_UTIL_THRESHOLD * 100:
        return True
    if wait_ratio is not None and wait_ratio >= _HIGH_IMPACT_WAIT_RATIO_THRESHOLD:
        return True
    if wip is not None and wip >= _HIGH_IMPACT_WIP_THRESHOLD:
        return True
    return False


def _dq_flags(wait_ratio: float | None) -> list[str]:
    flags: list[str] = []
    if wait_ratio is not None and wait_ratio > _WAIT_RATIO_ANOMALY_THRESHOLD:
        flags.append("wait_ratio_anomaly")
    return flags


def build_diffusion(
    tg: str,
    alert: BottleneckAlert,
    kpi_map: dict[str, ToolGroupKPI],
    cause_report: CauseReport | None,
) -> Diffusion:
    affected_tgs = alert.impact.affected_tgs or []

    high: list[AffectedProcess] = []
    low: list[AffectedProcess] = []
    for t in affected_tgs:
        tg_kpi = kpi_map.get(t)
        util_pct: float | None = None
        wait_ratio_v: float | None = None
        wip_v: int | None = None
        impact_score: float | None = None
        if tg_kpi is not None:
            util_pct = _round(tg_kpi.utilization_avg * 100, 1)
            wait_ratio_v = _round(tg_kpi.wait_ratio, 4)
            wip_v = _safe_int(tg_kpi.wip)
            # 간이 impact_score = 가동률·대기·WIP를 0~1 정규화 후 합산 / 3
            comps = []
            if tg_kpi.utilization_avg is not None: comps.append(min(tg_kpi.utilization_avg, 1.0))
            if tg_kpi.wait_ratio is not None:      comps.append(min(tg_kpi.wait_ratio / 2.0, 1.0))
            if tg_kpi.wip is not None:             comps.append(min(tg_kpi.wip / 200.0, 1.0))
            if comps:
                impact_score = _round(sum(comps) / len(comps), 4)

        item = AffectedProcess(
            toolgroup=t,
            utilization_pct=util_pct,
            wait_ratio=wait_ratio_v,
            wip=wip_v,
            impact_score=impact_score,
            data_quality_flags=_dq_flags(wait_ratio_v),
        )
        if tg_kpi is None:
            # KPI 없는 공정은 "low impact"로 분류 (정보 없음)
            low.append(item)
        elif _classify_impact(util_pct, wait_ratio_v, wip_v):
            high.append(item)
        else:
            low.append(item)

    # high_impact를 impact_score 기준 내림차순 정렬, 상위 8개만
    high.sort(key=lambda p: p.impact_score or 0.0, reverse=True)
    high_top = high[:8]

    # forward_simulation (anchor TG 단일 결과 — cause_report.sim_forecast)
    fwd = _build_forward_sim(tg, cause_report)

    diffusion_path = [tg] + [p.toolgroup for p in high_top[:3]]
    sev = alert.severity.value

    return Diffusion(
        bottleneck_location=tg,
        diffusion_path=diffusion_path,
        line_stop_expected_min=_round(alert.impact.ct_increase_min, 1),
        risk_level=sev,
        risk_level_token=_SEVERITY_TOKEN.get(sev, SeverityToken.OK),
        at_risk_lots=_round(alert.impact.at_risk_lots, 1),
        impact_pct=_round(alert.impact.impact_score * 100, 1),
        affected_toolgroups=list(affected_tgs),
        high_impact_processes=high_top,
        low_impact_processes=high[8:] + low,
        forward_simulation=fwd,
    )


def _build_forward_sim(tg: str, cause_report: CauseReport | None) -> ForwardSimulation:
    if cause_report is None or cause_report.sim_forecast is None:
        return ForwardSimulation(available=False)
    sf = cause_report.sim_forecast
    kd = sf.kpi_delta
    result = ForwardSimResult(
        toolgroup=tg,
        q_time_min_future=_round(kd["q_time_min"].future, 4) if "q_time_min" in kd else None,
        wait_ratio_future=_round(kd["wait_ratio"].future, 4) if "wait_ratio" in kd else None,
        wip_future=_safe_int(kd["wip"].future) if "wip" in kd else None,
        is_bottleneck_predicted=bool(sf.gets_worse),
    )
    return ForwardSimulation(
        available=True,
        horizon_min=int(sf.t_future - sf.t0),
        results=[result],
    )


# ── Cause ────────────────────────────────────────────────────────────────────

def build_cause(cause_report: CauseReport | None) -> Cause | None:
    if cause_report is None:
        return None

    judgment = cause_report.judgment
    primary: CausePrimary | None = None
    needs_more_data = False
    dismissed_items: list[CauseDismissed] = []
    secondary: list[str] = []

    if judgment is not None:
        conf_token = _CONFIDENCE_TOKEN.get(judgment.primary_confidence, ConfidenceToken.LOW)
        primary = CausePrimary(
            category=judgment.primary_category,
            feature=judgment.primary_cause,
            confidence=judgment.primary_confidence,
            confidence_token=conf_token,
            reasoning=judgment.primary_reasoning,
        )
        needs_more_data = bool(judgment.needs_more_data)
        # dismissed는 카테고리 이름 리스트 + 단일 dismissed_reason 문자열
        for name in (judgment.dismissed or []):
            dismissed_items.append(CauseDismissed(name=name, reason=judgment.dismissed_reason or ""))
        secondary = list(judgment.secondary_causes or [])

    consensus = cause_report.consensus
    g_star: CauseGStar | None = None
    consensus_axes: ConsensusAxes | None = None
    if consensus is not None:
        # Monte Carlo
        mc: GStarMonteCarlo | None = None
        if consensus.g_star_n_total:
            ratio = round(consensus.g_star_n_alarm / consensus.g_star_n_total * 100, 1) \
                if consensus.g_star_n_total > 0 else None
            mc = GStarMonteCarlo(
                n_total=consensus.g_star_n_total,
                n_alarm=consensus.g_star_n_alarm,
                alarm_ratio_pct=ratio,
            )
        sig_kpis = [
            GStarSignificantKpi(
                kpi=k.kpi,
                delta_mean=_round(k.delta_mean, 4) or 0.0,
                t_p_adj=_round(k.t_p_adj, 4) or 0.0,
                significant=bool(k.significant),
            )
            for k in (consensus.g_star_sig_kpis or [])
        ]
        g_star = CauseGStar(
            confirmed=bool(consensus.g_star_confirmed),
            probability=_round(consensus.g_star_proba, 4),
            monte_carlo=mc,
            upstream_confirmed_toolgroups=list(consensus.g_star_upstream_confirmed or []),
            significant_kpis=sig_kpis,
        )

        # 4축 합의: SHAP은 항상 존재 = True, 트렌드는 사용 trend_top 있고 significant 하나라도, 업스트림·G*는 컨센서스가 줌
        shap_supports = bool(cause_report.shap_top)
        trend_supports = any(t.significant for t in (cause_report.trend_top or []))
        upstream_supports = bool(consensus.upstream_aligns)
        g_star_supports = bool(consensus.g_star_confirmed)
        consensus_axes = ConsensusAxes(
            shap_supports=shap_supports,
            trend_supports=trend_supports,
            upstream_supports=upstream_supports,
            g_star_supports=g_star_supports,
            axes_agreed_count=sum([shap_supports, trend_supports, upstream_supports, g_star_supports]),
        )

    # categories
    categories = [
        CauseCategoryItem(
            name=c.name,
            features=list(c.features or []),
            shap_share_pct=_round(c.shap_share_pct, 2) or 0.0,
            n_trend_significant=int(c.n_trend_significant),
            upstream_match=bool(c.upstream_match),
            g_star_confirmed=bool(c.g_star_confirmed),
            total_score=_round(c.total_score, 4) or 0.0,
            confidence=c.confidence,
            confidence_token=_CONFIDENCE_TOKEN.get(c.confidence, ConfidenceToken.LOW),
        )
        for c in (cause_report.cause_categories or [])
    ]

    # shap_top
    total_abs = sum(abs(f.shap_value) for f in cause_report.shap_top) or 1.0
    shap_top = [
        ShapTopItem(
            rank=i + 1,
            feature=f.feature,
            value=_round(f.kpi_value, 4),
            shap=_round(f.shap_value, 4),
            contribution_pct=_round(abs(f.shap_value) / total_abs * 100, 1),
            direction_token=DirectionToken.BOTTLENECK_POSITIVE if f.shap_value > 0
                            else DirectionToken.BOTTLENECK_NEGATIVE,
        )
        for i, f in enumerate(cause_report.shap_top)
    ]

    # evidence_matrix
    evidence = [
        EvidenceMatrixItem(
            feature=e.feature,
            votes=int(e.votes),
            score=_round(e.score, 4) or 0.0,
            shap_rank=e.shap_rank,
            shap_value=_round(e.shap_value, 4),
            trend_slope=_round(e.trend_slope, 4),
            trend_r2=_round(e.trend_r2, 4),
            trend_significant=bool(e.trend_significant),
            upstream_match=bool(e.upstream_match),
            g_star_p_value=_round(e.g_star_p_value, 4),
            g_star_significant=bool(e.g_star_significant),
            confidence=e.confidence,
            confidence_token=_CONFIDENCE_TOKEN.get(e.confidence, ConfidenceToken.LOW),
        )
        for e in (cause_report.evidence_bundle or [])
    ]

    return Cause(
        summary=cause_report.cause_summary or "",
        primary=primary,
        secondary_categories=secondary,
        dismissed=dismissed_items,
        needs_more_data=needs_more_data,
        g_star=g_star,
        consensus_axes=consensus_axes,
        categories=categories,
        shap_top=shap_top,
        evidence_matrix=evidence,
        trend_series=build_trend_series(cause_report),
        upstream_suspects=list(cause_report.upstream_suspects or []),
    )


# ── Actions ──────────────────────────────────────────────────────────────────

def _split_release_interval(params: dict) -> ReleaseIntervalParam | None:
    """compare의 params dict → ReleaseIntervalParam 의미 분해."""
    if not params:
        return None
    cur = params.get("current_interval_minutes")
    tgt = params.get("release_interval_minutes")
    delta = params.get("release_interval_delta_min")
    if cur is None and tgt is None and delta is None:
        return None
    return ReleaseIntervalParam(
        current=_round(cur, 4),
        target=_round(tgt, 4),
        delta=_round(delta, 4),
        unit="min",
    )


def _split_params(params: dict) -> ActionParams | None:
    if not params:
        return None
    return ActionParams(
        release_interval=_split_release_interval(params),
        lot_priority_rule=params.get("lot_priority_rule"),
        superhotlot_enable=params.get("superhotlot_enable"),
        lot_adjustments=list(params.get("lot_adjustments") or []),
    )


def _build_kpi_impact(opt: dict) -> list[KpiImpactItem]:
    """option.kpi_impact (dict[kpi]=dict) → list[KpiImpactItem]."""
    kpi_impact = opt.get("kpi_impact") or {}
    items: list[KpiImpactItem] = []
    for kpi_name, info in kpi_impact.items():
        if not isinstance(info, dict):
            continue
        verdict = info.get("verdict")
        ci_w = info.get("ci_width")
        ci_low = ci_high = None
        # CI low/high는 보통 업스트림에 없고 ci_width만 옴. 가능하면 양쪽 계산.
        delta = info.get("delta")
        if ci_w is not None and delta is not None:
            half = _round(ci_w, 4)
            if half is not None:
                ci_low = _round(delta - half / 2, 4)
                ci_high = _round(delta + half / 2, 4)
        items.append(KpiImpactItem(
            kpi=kpi_name,
            unit=_KPI_UNITS.get(kpi_name, ""),
            now=_round(info.get("now"), 4),
            after=_round(info.get("after"), 4),
            delta=_round(delta, 4),
            pct_change=_round(info.get("pct_change"), 2),
            verdict=verdict,
            verdict_token=_VERDICT_TOKEN.get(verdict) if verdict else None,
            confidence=_round(info.get("confidence"), 4),
            ci_low=ci_low,
            ci_high=ci_high,
            ci_width=_round(ci_w, 4),
            paired_t_p=None,         # option-level이 아니라 simulation-level
            significant=None,
        ))
    return items


def _build_candidate(opt: dict, approved_label: str | None) -> ActionCandidate:
    label = str(opt.get("label", ""))
    is_baseline = bool(opt.get("is_baseline"))
    is_approved = (not is_baseline) and bool(approved_label) and label == approved_label

    sim = opt.get("simulation") or {}
    op = opt.get("operational") or {}
    tradeoffs = []
    for tradeoff in opt.get("tradeoffs") or []:
        if isinstance(tradeoff, str):
            tradeoffs.append(tradeoff)
            continue
        if isinstance(tradeoff, dict):
            kpi = tradeoff.get("kpi", "KPI")
            delta = _round(tradeoff.get("mean_delta"), 4)
            severity = tradeoff.get("severity", "")
            tradeoffs.append(
                f"{kpi} 악화"
                + (f" (Δ{delta:+g})" if delta is not None else "")
                + (f", {severity}" if severity else "")
            )

    return ActionCandidate(
        label=label,
        kind=str(opt.get("kind") or "UNKNOWN"),
        is_baseline=is_baseline,
        is_approved=is_approved,
        description=str(opt.get("description") or ""),
        target_toolgroups=list(opt.get("target_toolgroups") or []),
        params=_split_params(opt.get("params") or {}),
        kpi_impact=_build_kpi_impact(opt),
        operational=OperationalAttrs(
            effort=_safe_int(op.get("effort")),
            scope=op.get("scope"),
            reversibility=op.get("reversibility"),
        ) if op else None,
        simulation=SimulationStats(
            paired_n=_safe_int(sim.get("paired_n")),
            confidence=_round(sim.get("simulation_confidence"), 4),
            verdict=sim.get("verdict"),
        ) if sim else None,
        composite_score=_round(opt.get("composite_score"), 4),
        tradeoffs=tradeoffs,
        comparison_basis=str(opt.get("comparison_basis") or ""),
        per_tg_forecasts=dict(opt.get("per_tg_forecasts") or {}),
    )


def _build_recommendation_block(
    rec: dict,
    candidates: list[ActionCandidate] | None = None,
    approved_label: str | None = None,
    approval: dict | None = None,
) -> Recommendation | None:
    if not rec:
        return None
    why_recommended = rec.get("why_recommended") or {}
    approved = None
    for candidate in candidates or []:
        if approved_label and candidate.label == approved_label:
            approved = candidate
            break
    if approved is None:
        approved = next((candidate for candidate in candidates or [] if not candidate.is_baseline), None)

    primary_reason = str(rec.get("primary_reason") or rec.get("reason") or "")
    tradeoffs = list(rec.get("tradeoffs") or [])
    effect_parts = [text for text in [primary_reason] if text]
    if tradeoffs:
        effect_parts.append("리스크/트레이드오프: " + " · ".join(str(t) for t in tradeoffs))

    return Recommendation(
        headline=str(rec.get("headline") or ""),
        primary_reason=primary_reason,
        plan_description=str(rec.get("plan_description") or (approved.description if approved else "") or "") or None,
        effect_and_risk=str(rec.get("effect_and_risk") or " ".join(effect_parts) or "") or None,
        approval_reason=str(
            rec.get("approval_reason")
            or (approval or {}).get("comment")
            or (approval or {}).get("rejection_reason")
            or primary_reason
            or ""
        ) or None,
        confidence_level=rec.get("confidence_level"),
        confidence_token=_CONFIDENCE_TOKEN.get(rec.get("confidence_level")),
        tradeoffs=tradeoffs,
        why_not_others=[
            WhyNotOther(label=str(k), reason=str(v))
            for k, v in (rec.get("why_not_others") or {}).items()
        ],
        caveats=list(rec.get("caveats") or []),
        tiebreaker_chain=list(why_recommended.get("tiebreaker_chain") or []),
        selected_by=why_recommended.get("selected_by"),
    )


def _build_playbook(rec: dict) -> Playbook:
    """recommendation.immediate_actions/monitoring_kpis/rollback_condition → Playbook.

    업스트림이 비우면 available=False. 없는 데이터를 추측하지 않음.
    """
    if not rec:
        return Playbook(available=False)

    immediate = rec.get("immediate_actions") or []
    monitoring = rec.get("monitoring_kpis") or []
    rollback = rec.get("rollback_condition") or ""

    has_any = bool(immediate) or bool(monitoring) or bool(rollback.strip())
    if not has_any:
        return Playbook(available=False)

    actions: list[ImmediateAction] = []
    for i, step in enumerate(immediate, 1):
        if isinstance(step, dict):
            actions.append(ImmediateAction(order=int(step.get("order", i)), text=str(step.get("text", ""))))
        else:
            actions.append(ImmediateAction(order=i, text=str(step)))

    checks: list[MonitoringCheck] = []
    for m in monitoring:
        if not isinstance(m, dict):
            continue
        kpi_name = str(m.get("kpi", ""))
        raw_target = m.get("target")
        target = (
            _round(raw_target, 4)
            if isinstance(raw_target, (int, float))
            else str(raw_target or "")
        )
        checks.append(MonitoringCheck(
            kpi=kpi_name,
            target=target,
            unit=(
                _KPI_UNITS.get(kpi_name, "")
                if isinstance(raw_target, (int, float))
                else ""
            ),
            check_after_min=_safe_int(m.get("check_after_min")),
        ))

    return Playbook(
        available=True,
        immediate_actions=actions,
        monitoring=checks,
        rollback_condition=rollback or None,
    )


def build_actions(normalized: dict) -> Actions:
    """normalized = adapter.normalize_compare_result() 결과."""
    options = normalized.get("action_options") or []
    rec = normalized.get("recommendation") or {}
    dm = normalized.get("decision_meta") or {}
    approval = normalized.get("approval_info") or {}

    approved_label = (
        approval.get("selected_label")
        or rec.get("recommended_label")
        or rec.get("action_label")
    )

    candidates = [_build_candidate(opt, approved_label) for opt in options if isinstance(opt, dict)]
    real_candidates = [c for c in candidates if not c.is_baseline]
    available = bool(real_candidates)

    decision_status = dm.get("decision_status")
    return Actions(
        available=available,
        decision_status=decision_status,
        decision_status_token=_DECISION_STATUS_TOKEN.get(decision_status) if decision_status else None,
        decision_caveat=str(dm.get("decision_caveat") or ""),
        tiebreaker_used=dm.get("tiebreaker_used"),
        equivalent_set=list(dm.get("equivalent_set") or []),
        approved_label=approved_label if available else None,
        candidates=candidates,
        recommendation=_build_recommendation_block(rec, candidates, approved_label, approval),
        playbook=_build_playbook(rec),
    )


# ── Data quality ─────────────────────────────────────────────────────────────

def build_data_quality(
    upstream_dq: dict,
    diffusion: Diffusion | None,
) -> DataQuality:
    """업스트림 compare.data_quality + diffusion의 sanity flag 통합."""
    warnings: list[DataQualityWarning] = []

    # 업스트림이 만든 경고
    for w in (upstream_dq.get("warnings") or []):
        if isinstance(w, dict):
            warnings.append(DataQualityWarning(
                code=str(w.get("code") or "unknown"),
                toolgroup=w.get("toolgroup"),
                field=w.get("field"),
                value=_round(w.get("value"), 4),
                message=str(w.get("message") or ""),
            ))
        else:
            warnings.append(DataQualityWarning(code="unknown", message=str(w)))

    # diffusion의 sanity flag → 경고로 승격
    if diffusion is not None:
        for p in (diffusion.high_impact_processes + diffusion.low_impact_processes):
            for flag in p.data_quality_flags:
                if flag == "wait_ratio_anomaly":
                    warnings.append(DataQualityWarning(
                        code="wait_ratio_anomaly",
                        toolgroup=p.toolgroup,
                        field="wait_ratio",
                        value=p.wait_ratio,
                        message=f"wait_ratio={p.wait_ratio} 가 임계({_WAIT_RATIO_ANOMALY_THRESHOLD})를 초과 — 단위/계산 확인 필요",
                    ))

    # 자체 감지 경고가 하나라도 있으면 status는 무조건 'warning' 이상.
    # (업스트림이 'ok'라고 했어도 우리가 감지한 이상치가 더 강한 신호)
    upstream_status = upstream_dq.get("status") or "ok"
    if warnings and upstream_status == "ok":
        status = "warning"
    else:
        status = upstream_status
    return DataQuality(status=status, warnings=warnings)


# ── Provenance ───────────────────────────────────────────────────────────────

def build_provenance(
    snapshot_time: float | None,
    meta: dict,
) -> Provenance:
    return Provenance(
        snapshot_time=_round(snapshot_time, 4),
        t0=_round(meta.get("t0"), 4),
        run_id=meta.get("run_id"),
        scenario_id=meta.get("scenario_id") or meta.get("anchor_scenario_id"),
        target_toolgroups=list(meta.get("target_toolgroups") or []),
    )


# ── 통합 ─────────────────────────────────────────────────────────────────────

def build_report_v2(
    *,
    tg: str,
    alert: BottleneckAlert,
    kpi: ToolGroupKPI | None,
    prev_kpi: ToolGroupKPI | None,
    cause_report: CauseReport | None,
    kpi_map: dict[str, ToolGroupKPI],
    normalized_compare: dict,
    detected_at: str,
    snapshot_time: float | None = None,
) -> ReportV2:
    """모든 블록을 빌드해서 ReportV2 객체로 합친다.

    이 함수는 LLM을 호출하지 않는다. sections / rendered.markdown 채우기는
    이후 단계(writer.py)에서 한다.
    """
    compare_meta = normalized_compare.get("meta") or {}
    horizon_min = _safe_int(compare_meta.get("horizon_min")) or 120

    meta = build_meta(
        tg,
        alert,
        detected_at,
        snapshot_time,
        horizon_min,
        area_name=compare_meta.get("area_name") or compare_meta.get("areaName"),
    )
    approval = build_approval(normalized_compare.get("approval_info") or {})
    risk = build_risk(alert)
    confidence = build_confidence(cause_report)
    if_no_action = build_if_no_action(normalized_compare.get("current_state") or {})
    bottleneck_kpis = build_bottleneck_kpis(kpi, prev_kpi, alert)
    bottleneck_trend = build_bottleneck_trend(cause_report)
    diffusion = build_diffusion(tg, alert, kpi_map, cause_report)
    cause = build_cause(cause_report)
    actions = build_actions(normalized_compare)
    data_quality = build_data_quality(
        normalized_compare.get("data_quality") or {},
        diffusion,
    )
    provenance = build_provenance(snapshot_time, compare_meta)

    return ReportV2(
        meta=meta,
        approval=approval,
        risk=risk,
        confidence=confidence,
        if_no_action=if_no_action,
        bottleneck_kpis=bottleneck_kpis,
        bottleneck_trend=bottleneck_trend,
        diffusion=diffusion,
        cause=cause,
        actions=actions,
        data_quality=data_quality,
        provenance=provenance,
    )
