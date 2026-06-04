"""
원인 분석 결과 → 시뮬레이션 파라미터 조정 규칙.

SHAP 상위 피처 + 트렌드 + 심각도 → SolutionCandidate 목록 생성.
"""

from __future__ import annotations

from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseReport
from agents.schemas.solution import SimParamDelta, SolutionCandidate

# 규칙 매핑: primary SHAP 피처 → 대응 방향
_RULE_TABLE: dict[str, dict] = {
    "wip": {
        "action": "release_throttle",
        "desc": "WIP 과부하",
        "target_kpi": "wip",
    },
    "wait_ratio": {
        "action": "release_throttle",
        "desc": "대기 lot 비율 과부하",
        "target_kpi": "wait_ratio",
    },
    "max_util": {
        "action": "dispatch_change",
        "desc": "장비 최대 가동률 포화",
        "target_kpi": "max_util",
    },
    "utilization_avg": {
        "action": "dispatch_change",
        "desc": "평균 가동률 포화",
        "target_kpi": "utilization_avg",
    },
    "available_tool_ratio": {
        "action": "superhotlot",
        "desc": "장비 가용률 하락",
        "target_kpi": "available_tool_ratio",
    },
    "q_time_min": {
        "action": "release_throttle",
        "desc": "대기 시간 급증",
        "target_kpi": "q_time_min",
    },
}


def _release_interval_pct(severity: SeverityLevel) -> float:
    """심각도에 따른 RELEASE_INTERVAL 조정 비율."""
    return {
        SeverityLevel.CRITICAL: 35.0,
        SeverityLevel.HIGH: 25.0,
        SeverityLevel.MEDIUM: 15.0,
        SeverityLevel.LOW: 10.0,
    }[severity]


_TREND_NORM = 3.0  # 시간당 이 속도 이상이면 trend_bonus 최대치


def _confidence(cause_report: CauseReport, action: str) -> float:
    """SHAP 값 크기 + 악화 트렌드 기울기 기반 confidence 계산."""
    if not cause_report.shap_top:
        return 0.5

    top_shap = abs(cause_report.shap_top[0].shap_value)
    primary_feature = cause_report.shap_top[0].feature

    # 기본: SHAP 기여도 (3.0 이상 → 0.8, 0.5 이하 → 0.5)
    base = min(0.5 + top_shap / 10, 0.85)

    # 트렌드 보너스: primary 피처의 악화 속도가 빠를수록 최대 +0.10
    trend_bonus = 0.0
    for insight in cause_report.trend_top:
        if insight.feature == primary_feature:
            # 악화 방향 판단 (available_tool_ratio은 낮아질수록 나쁨)
            slope = (
                -insight.slope_per_hour
                if primary_feature == "available_tool_ratio"
                else insight.slope_per_hour
            )
            trend_bonus = min(slope / _TREND_NORM, 1.0) * 0.10
            break

    # upstream 의심 공정 있으면 release 대응안 +0.05
    upstream_bonus = (
        0.05 if action == "release_throttle" and cause_report.upstream_suspects else 0.0
    )

    return round(min(base + trend_bonus + upstream_bonus, 0.95), 2)


def generate_candidates(
    cause_report: CauseReport,
    alert: BottleneckAlert,
) -> list[SolutionCandidate]:
    """원인 분석 결과로부터 대응안 후보 2~3개 생성."""
    primary = cause_report.shap_top[0].feature if cause_report.shap_top else "wip"
    rule = _RULE_TABLE.get(primary, _RULE_TABLE["wip"])
    pct = _release_interval_pct(alert.severity)
    candidates: list[SolutionCandidate] = []

    # ── 대응안 1: Lot Release 간격 조정 (1순위) ──────────────────────
    if rule["action"] in ("release_throttle", "dispatch_change"):
        candidates.append(
            SolutionCandidate(
                rank=1,
                name="Lot Release 간격 증가",
                params=SimParamDelta(
                    release_interval_delta_pct=pct,
                    lot_priority_rule="HIGH_WIP_FIRST",
                ),
                target_kpi="wip",
                expected_effect=(
                    f"RELEASE_INTERVAL을 {pct:.0f}% 늘려 {alert.toolgroup}로의 WIP 유입을 줄입니다. "
                    f"투입 우선순위를 HIGH_WIP_FIRST로 전환해 병목 공정의 lot을 우선 처리합니다."
                ),
                confidence=_confidence(cause_report, "release_throttle"),
                rationale=f"주요 원인: {rule['desc']} — SHAP {cause_report.shap_top[0].shap_value:+.3f}",
            )
        )

    # ── 대응안 2: Dispatching Rule 변경 (2순위) ──────────────────────
    dispatch = "SPT" if primary in ("max_util", "utilization_avg", "q_time_min") else "EDD"
    dispatch_effect = {
        "SPT": "최단 처리 시간 lot 우선으로 처리량을 최대화합니다.",
        "EDD": "납기 임박 lot 우선으로 SLA 위반 위험을 줄입니다.",
    }
    candidates.append(
        SolutionCandidate(
            rank=2,
            name=f"Dispatching {dispatch}로 전환",
            params=SimParamDelta(
                dispatch_rule=dispatch,
                dispatch_priority_weight=1.5,
            ),
            target_kpi=rule["target_kpi"],
            expected_effect=(
                f"{alert.toolgroup} 장비 배정 규칙을 {dispatch}로 변경합니다. "
                f"{dispatch_effect[dispatch]}"
            ),
            confidence=round(_confidence(cause_report, "dispatch_change") - 0.05, 2),
            rationale=f"장비 처리 효율 개선 목적 — 현재 max_util={cause_report.shap_top[1].kpi_value:.3f}"
            if len(cause_report.shap_top) > 1
            else "장비 처리 효율 개선 목적",
        )
    )

    # ── 대응안 3: SUPERHOTLOT 활성화 (3순위, Critical/High만) ────────
    if alert.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
        n_upstream = len(cause_report.upstream_suspects)
        candidates.append(
            SolutionCandidate(
                rank=3,
                name="SUPERHOTLOT 우선 처리 활성화",
                params=SimParamDelta(
                    superhotlot_enable=True,
                    lot_priority_rule="SUPERHOTLOT_FIRST",
                ),
                target_kpi="wait_ratio",
                expected_effect=(
                    f"긴급 lot(SUPERHOTLOT) 플래그를 활성화하여 {alert.toolgroup} 대기 열에서 "
                    f"우선 처리합니다. 납기 위험 lot의 흐름 시간을 단축합니다."
                    + (f" 업스트림 {n_upstream}개 공정 연동 필요." if n_upstream else "")
                ),
                confidence=0.65,
                rationale=f"심각도 {alert.severity.value} — 즉각 대응 필요, wait_ratio={cause_report.shap_top[0].kpi_value:.1f}",
            )
        )

    return candidates
