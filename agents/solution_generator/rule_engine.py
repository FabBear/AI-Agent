"""
Critical alerts 전체 → Lot Release 테이블 기반 글로벌 플랜 A / B 생성.

조정 대상: release_interval_minutes, lot_priority_rule, superhotlot_enable.
모든 수치는 실제 KPI 값과 cascade impact 데이터에서 역산한다.
플랜 A / B 는 release_interval_minutes 값만 다르며, 나머지는 동일하다.
"""

from __future__ import annotations

from agents import config
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseReport
from agents.schemas.solution import GlobalSolutionPlan

_DEFAULT_INTERVAL = 60.0  # current_release_interval 미설정 시 fallback (분)
_COMPARISON_STEP = 2.0    # 플랜 A ↔ B 인터벌 차이 (분)

# 피처별 정상 임계값 (config.py 기준)
_FEATURE_THRESHOLD: dict[str, float] = {
    "wip": config.WIP_THR,
    "wait_ratio": config.W_THR,
    "max_util": config.U_HI,
    "utilization_avg": config.U_HI,
    "available_tool_ratio": config.AVAIL_THR,
    "q_time_min": config.Q_THR,
}

# 주요 원인 피처 → 투입 우선순위 룰
_FEATURE_TO_PRIORITY: dict[str, str] = {
    "wip": "HIGH_WIP_FIRST",
    "wait_ratio": "LONGEST_WAIT_FIRST",
    "max_util": "HIGH_WIP_FIRST",
    "utilization_avg": "HIGH_WIP_FIRST",
    "available_tool_ratio": "PRIORITY_FIRST",
    "q_time_min": "CRITICAL_RATIO",
}

# 수퍼핫랏 활성화 기준: Critical TG 전체 위험 lot 수 합산
_SUPERHOTLOT_LOT_THRESHOLD = 5.0


def _excess_ratio(feature: str, kpi_value: float) -> float:
    """KPI가 임계값을 얼마나 초과했는지 비율로 반환 (0.0 ~).

    available_tool_ratio는 낮을수록 나쁘므로 역방향으로 계산한다.
    """
    thr = _FEATURE_THRESHOLD.get(feature, 0.0)
    if thr == 0.0:
        return 0.0
    if feature == "available_tool_ratio":
        return max(0.0, (thr - kpi_value) / thr)
    return max(0.0, (kpi_value - thr) / thr)


def _interval_delta(cause_reports: list[CauseReport], base: float) -> float:
    """Critical TG 중 가장 심각한 KPI 초과 비율로 인터벌 증가량(분)을 역산한다.

    공식: max_excess × 15% × 현재_인터벌 (최소 2분, 최대 30% 제한)
    예) wait_ratio=2.5, 임계값=1.0 → 초과율=150% → 22.5% 증가 → 60분 기준 13.5분 증가
    """
    max_excess = 0.0
    for r in cause_reports:
        if not r.shap_top:
            continue
        top = r.shap_top[0]
        max_excess = max(max_excess, _excess_ratio(top.feature, top.kpi_value))

    pct = min(max_excess * 0.15, 0.30)
    return max(2.0, round(base * pct, 1))


def _priority_rule(cause_reports: list[CauseReport]) -> str:
    """Critical TG들의 SHAP 1위 피처 빈도로 FAB 전체 투입 우선순위 룰을 결정한다."""
    freq: dict[str, int] = {}
    for r in cause_reports:
        if r.shap_top:
            feat = r.shap_top[0].feature
            freq[feat] = freq.get(feat, 0) + 1
    if not freq:
        return "HIGH_WIP_FIRST"
    dominant = max(freq, key=lambda f: freq[f])
    return _FEATURE_TO_PRIORITY.get(dominant, "HIGH_WIP_FIRST")


def _superhotlot_enable(alerts: list[BottleneckAlert]) -> tuple[bool, float]:
    """위험 lot 총합이 임계값 이상이면 수퍼핫랏 활성화. (enable, total_at_risk_lots) 반환."""
    total = sum(a.impact.at_risk_lots for a in alerts)
    return total >= _SUPERHOTLOT_LOT_THRESHOLD, total


def _confidence(cause_reports: list[CauseReport]) -> float:
    if not cause_reports:
        return 0.5
    avg_shap = sum(
        abs(r.shap_top[0].shap_value) for r in cause_reports if r.shap_top
    ) / max(len(cause_reports), 1)
    return round(min(0.5 + avg_shap / 10, 0.90), 2)


def generate_global_plans(
    alerts: list[BottleneckAlert],
    cause_map: dict[str, CauseReport],
    current_interval: float | None = None,
) -> list[GlobalSolutionPlan]:
    """Critical 알림 전체를 받아 실데이터 기반 Lot Release 글로벌 플랜 A / B 를 생성한다."""
    base = current_interval if current_interval is not None else _DEFAULT_INTERVAL
    critical_alerts = [
        a for a in alerts
        if a.severity == SeverityLevel.CRITICAL and a.toolgroup in cause_map
    ]
    if not critical_alerts:
        return []
    critical_causes = [cause_map[a.toolgroup] for a in critical_alerts]
    target_tgs = [a.toolgroup for a in critical_alerts]

    delta = _interval_delta(critical_causes, base)
    priority = _priority_rule(critical_causes)
    superhotlot, at_risk_lots = _superhotlot_enable(critical_alerts)
    conf = _confidence(critical_causes)

    plans: list[GlobalSolutionPlan] = []
    for plan_id, interval in [("A", base + delta), ("B", base + delta + _COMPARISON_STEP)]:
        interval = round(interval, 1)
        plans.append(
            GlobalSolutionPlan(
                plan_id=plan_id,
                target_toolgroups=target_tgs,
                current_interval_minutes=base,
                release_interval_minutes=interval,
                lot_priority_rule=priority,
                superhotlot_enable=superhotlot,
                description=(
                    f"[플랜 {plan_id}] "
                    f"Release Interval {base:.1f}분 → {interval:.1f}분 (+{interval - base:.1f}분) | "
                    f"투입 우선순위 {priority} | "
                    f"SUPERHOTLOT {'활성화' if superhotlot else '비활성화'}"
                    f" (위험 lot {at_risk_lots:.0f}개) | "
                    f"대상 TG: {', '.join(target_tgs) if target_tgs else '없음'}"
                ),
                expected_effect="",
                confidence=conf,
            )
        )
    return plans
