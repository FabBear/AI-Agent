"""TG 단위 Lot Release 파라미터 조합 생성 (보수/표준/강화).

judgment.primary_category 기반 카테고리 규칙 테이블로 파라미터를 결정한다.
judgment가 None이면 generate_candidates()가 None을 반환한다.
"""

from __future__ import annotations

from agents import config
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseJudgment, CauseReport
from agents.schemas.solution import GlobalSolutionPlan

_DEFAULT_INTERVAL = 60.0   # generate_global_plans fallback — 레거시
_COMPARISON_STEP = 2.0     # generate_global_plans A↔B 차이 — 레거시

_SEVERITY_PCT: dict[SeverityLevel, dict[str, float]] = {
    SeverityLevel.LOW:      {"conservative": 5.0,  "standard": 8.0,  "aggressive": 12.0},
    SeverityLevel.MEDIUM:   {"conservative": 8.0,  "standard": 12.0, "aggressive": 18.0},
    SeverityLevel.HIGH:     {"conservative": 12.0, "standard": 18.0, "aggressive": 23.0},
    SeverityLevel.CRITICAL: {"conservative": 15.0, "standard": 22.0, "aggressive": 28.0},
}
_RELEASE_PCT = _SEVERITY_PCT[SeverityLevel.CRITICAL]

_SEVERITY_MAX_PCT: dict[SeverityLevel, float] = {
    SeverityLevel.LOW: 12.0,
    SeverityLevel.MEDIUM: 18.0,
    SeverityLevel.HIGH: 23.0,
    SeverityLevel.CRITICAL: 28.0,
}
_ABSOLUTE_MAX_PCT = 30.0

_CATEGORY_RULES: dict[str, dict] = {
    "WIP_누적": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": "DOWN", "aggressive": "DOWN"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "wip",
    },
    "대기_누적": {
        "pct_level": "down",
        "priority": {"conservative": "UP", "standard": "UP", "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": True, "aggressive": True},
        "target_kpi": "wait_ratio",
    },
    "설비_포화": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": None, "aggressive": None},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": False},
        "target_kpi": "utilization_avg",
    },
    "공급_부족": {
        "pct_level": "up",
        "priority": {"conservative": None, "standard": None, "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": True, "aggressive": True},
        "target_kpi": "available_tool_ratio",
    },
}

_FALLBACK_CATEGORY = "WIP_누적"
_PLAN_LEVELS = ("conservative", "standard", "aggressive")

_FEATURE_TO_CATEGORY: dict[str, str] = {
    "max_util":                  "설비_포화",
    "max_util_delta_120":        "설비_포화",
    "utilization_avg":           "설비_포화",
    "utilization_avg_delta_120": "설비_포화",
    "wait_ratio":                "대기_누적",
    "wait_ratio_delta_120":      "대기_누적",
    "q_time_min":                "대기_누적",
    "q_time_min_delta_120":      "대기_누적",
    "wip":                       "WIP_누적",
    "wip_delta_120":             "WIP_누적",
    "available_tool_ratio":      "공급_부족",
}


def _to_category(name: str) -> str:
    """피처명 또는 카테고리명을 카테고리명으로 정규화."""
    return _FEATURE_TO_CATEGORY.get(name, name)


def _resolve_pct(pct_level: str, severity: SeverityLevel, plan_level: str) -> float:
    """pct_level 지시에 따라 실제 조정 수치(%) 결정."""
    sev_pct = _SEVERITY_PCT[severity]
    if pct_level == "same":
        return sev_pct[plan_level]
    if pct_level == "down":
        if plan_level == "conservative":
            return 0.0
        if plan_level == "standard":
            return sev_pct["conservative"]
        return sev_pct["standard"]
    if plan_level == "conservative":
        return sev_pct["standard"]
    return sev_pct["aggressive"]


def _check_superhotlot(
    alert: BottleneckAlert,
    eligible: bool,
    severity: SeverityLevel,
    plan_level: str,
) -> bool:
    """SUPERHOTLOT 활성화 여부."""
    if not eligible:
        return False
    if plan_level == "conservative" and severity != SeverityLevel.CRITICAL:
        return False
    if plan_level == "standard" and severity not in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}:
        return False
    if plan_level == "aggressive" and severity == SeverityLevel.LOW:
        return False
    return alert.impact.at_risk_lots > 0


def _detect_scenario(judgment: CauseJudgment) -> str | None:
    """복합 원인 시나리오 감지."""
    if not judgment.secondary_causes:
        return None

    all_cats = {judgment.primary_category} | {_to_category(s) for s in judgment.secondary_causes}

    if len(all_cats) >= 3:
        return "S4"
    if {"WIP_누적", "공급_부족"}.issubset(all_cats):
        return "S1"
    if {"대기_누적", "WIP_누적"}.issubset(all_cats):
        return "S2"
    if {"설비_포화", "공급_부족"}.issubset(all_cats):
        return "S5"
    if "설비_포화" in all_cats:
        return "S3"
    return None


def _apply_scenario(
    params: dict[str, dict],
    scenario: str | None,
    severity_pct: dict[str, float],
) -> tuple[dict[str, dict], bool, str | None]:
    """시나리오별 파라미터 보정. (params, hitl_flag, escalation_reason) 반환."""
    if scenario is None:
        return params, False, None

    agg_pct = severity_pct["aggressive"]

    if scenario == "S1":
        for lv in _PLAN_LEVELS:
            params[lv]["release_interval_delta_pct"] = agg_pct
            params[lv]["superhotlot_enable"] = True
        return params, False, None

    if scenario == "S2":
        for lv in _PLAN_LEVELS:
            params[lv]["priority_direction"] = "UP"
            params[lv]["superhotlot_enable"] = True
        params["conservative"]["release_interval_delta_pct"] = max(
            params["conservative"]["release_interval_delta_pct"],
            severity_pct["conservative"],
        )
        return params, False, None

    if scenario == "S3":
        params["aggressive"]["superhotlot_enable"] = True
        reason = "설비 포화 복합 원인 — 장비 상태 또는 PM 일정 검토 권장"
        return params, True, reason

    if scenario == "S4":
        for lv in _PLAN_LEVELS:
            params[lv]["release_interval_delta_pct"] = agg_pct
            params[lv]["priority_direction"] = "UP"
            params[lv]["superhotlot_enable"] = True
        reason = "3개 이상 원인 동시 악화 — HITL 에스컬레이션 권장"
        return params, True, reason

    if scenario == "S5":
        for lv in _PLAN_LEVELS:
            params[lv]["superhotlot_enable"] = True
        return params, False, None

    return params, False, None


def clip_interval_pct(proposed_pct: float, severity: SeverityLevel | None = None) -> float:
    """release_interval_delta_pct를 severity 상한 및 절대 상한으로 클리핑."""
    if severity is None:
        severity = SeverityLevel.CRITICAL
    return min(proposed_pct, _SEVERITY_MAX_PCT.get(severity, _ABSOLUTE_MAX_PCT), _ABSOLUTE_MAX_PCT)


def generate_candidates(
    alert: BottleneckAlert,
    cause_report: CauseReport,
) -> dict | None:
    """TG 1개에 대한 보수/표준/강화 파라미터 dict 생성."""
    if cause_report.judgment is None:
        return None

    severity = alert.severity
    severity_pct = _SEVERITY_PCT[severity]
    judgment = cause_report.judgment

    primary = judgment.primary_category
    rule = _CATEGORY_RULES.get(primary) or _CATEGORY_RULES[_FALLBACK_CATEGORY]

    params: dict[str, dict] = {}
    for lv in _PLAN_LEVELS:
        pct = _resolve_pct(rule["pct_level"], severity, lv)
        superhotlot = _check_superhotlot(
            alert,
            rule["superhotlot_eligible"][lv],
            severity,
            lv,
        )
        params[lv] = {
            "release_interval_delta_pct": pct,
            "priority_direction": rule["priority"][lv],
            "superhotlot_enable": superhotlot,
            "target_kpi": rule["target_kpi"],
        }

    scenario = _detect_scenario(judgment)
    params, hitl, escalation_reason = _apply_scenario(params, scenario, severity_pct)

    return {
        "conservative": params["conservative"],
        "standard": params["standard"],
        "aggressive": params["aggressive"],
        "hitl_escalation_recommended": hitl,
        "escalation_reason": escalation_reason,
    }


def _excess_ratio(feature: str, kpi_value: float) -> float:
    thr_map = {
        "wip": config.WIP_THR,
        "wait_ratio": config.W_THR,
        "max_util": config.U_HI,
        "utilization_avg": config.U_HI,
        "available_tool_ratio": config.AVAIL_THR,
        "q_time_min": config.Q_THR,
    }
    thr = thr_map.get(feature, 0.0)
    if thr == 0.0:
        return 0.0
    if feature == "available_tool_ratio":
        return max(0.0, (thr - kpi_value) / thr)
    return max(0.0, (kpi_value - thr) / thr)


def _superhotlot_enable(alerts: list[BottleneckAlert]) -> tuple[bool, float]:
    total = sum(a.impact.at_risk_lots for a in alerts)
    return total >= 5.0, total


def generate_global_plans(
    alerts: list[BottleneckAlert],
    cause_map: dict[str, CauseReport],
    current_interval: float | None = None,
) -> list[GlobalSolutionPlan]:
    """레거시: node.py Step 4 재작성 전까지 유지."""
    base = current_interval if current_interval is not None else _DEFAULT_INTERVAL
    critical_alerts = [
        a for a in alerts
        if a.severity in {SeverityLevel.CRITICAL, SeverityLevel.HIGH} and a.toolgroup in cause_map
    ]
    if not critical_alerts:
        return []
    critical_causes = [cause_map[a.toolgroup] for a in critical_alerts]
    target_tgs = [a.toolgroup for a in critical_alerts]

    max_excess = 0.0
    for r in critical_causes:
        if r.shap_top:
            top = r.shap_top[0]
            max_excess = max(max_excess, _excess_ratio(top.feature, top.kpi_value))
    delta = max(2.0, round(base * min(max_excess * 0.15, 0.30), 1))

    superhotlot, at_risk_lots = _superhotlot_enable(critical_alerts)

    plans: list[GlobalSolutionPlan] = []
    for plan_id, interval in [("A", base + delta), ("B", base + delta + _COMPARISON_STEP)]:
        interval = round(interval, 1)
        plans.append(
            GlobalSolutionPlan(
                plan_id=plan_id,
                target_toolgroups=target_tgs,
                current_interval_minutes=base,
                release_interval_minutes=interval,
                superhotlot_enable=superhotlot,
                description=(
                    f"[플랜 {plan_id}] "
                    f"Release Interval {base:.1f}분 → {interval:.1f}분 (+{interval - base:.1f}분) | "
                    f"SUPERHOTLOT {'활성화' if superhotlot else '비활성화'}"
                    f" (위험 lot {at_risk_lots:.0f}개) | "
                    f"대상 TG: {', '.join(target_tgs) if target_tgs else '없음'}"
                ),
                expected_effect="",
            )
        )
    return plans
