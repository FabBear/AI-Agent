"""
TG 단위 Lot Release 파라미터 조합 생성 (보수/표준/강화).

kpi_value는 raw값(비정규화)으로 가정한다.
cause_analyzer 확인 후 정규화값이면 _FEATURE_THRESHOLD와 SUPERHOTLOT 조건
임계값을 정규화 기준으로 교체한다 (A-11).
"""

from __future__ import annotations

from agents import config
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseReport
from agents.schemas.solution import GlobalSolutionPlan

_DEFAULT_INTERVAL = 60.0  # current_release_interval 미설정 시 fallback (분)
_COMPARISON_STEP = 2.0    # generate_global_plans A↔B 차이 — Step 4에서 제거 예정

# ── KPI 임계값 (excess_ratio 계산용, raw값 기준) ──────────────────────────────
_FEATURE_THRESHOLD: dict[str, float] = {
    "wip": config.WIP_THR,
    "wait_ratio": config.W_THR,
    "max_util": config.U_HI,
    "utilization_avg": config.U_HI,
    "available_tool_ratio": config.AVAIL_THR,
    "q_time_min": config.Q_THR,
}

# ── severity별 RELEASE_INTERVAL 조정 수치 (%) ────────────────────────────────
_SEVERITY_PCT: dict[SeverityLevel, dict[str, float]] = {
    SeverityLevel.LOW:      {"conservative": 5.0,  "standard": 8.0,  "aggressive": 12.0},
    SeverityLevel.MEDIUM:   {"conservative": 8.0,  "standard": 12.0, "aggressive": 18.0},
    SeverityLevel.HIGH:     {"conservative": 12.0, "standard": 18.0, "aggressive": 23.0},
    SeverityLevel.CRITICAL: {"conservative": 15.0, "standard": 22.0, "aggressive": 28.0},
}

# ── post-processing clip 상한 ─────────────────────────────────────────────────
_SEVERITY_MAX_PCT: dict[SeverityLevel, float] = {
    SeverityLevel.LOW: 12.0,
    SeverityLevel.MEDIUM: 18.0,
    SeverityLevel.HIGH: 23.0,
    SeverityLevel.CRITICAL: 28.0,
}
_ABSOLUTE_MAX_PCT = 30.0

# ── 피처별 규칙 테이블 ───────────────────────────────────────────────────────
# pct_level:
#   "same" — plan level 그대로  (conservative→C, standard→S, aggressive→A)
#   "down" — 한 단계 낮춤       (conservative→0%, standard→C, aggressive→S)
#   "up"   — 한 단계 높임       (conservative→S, standard→A, aggressive→A)
_FEATURE_RULES: dict[str, dict] = {
    "wip": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": "DOWN", "aggressive": "DOWN"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "wip",
    },
    "wait_ratio": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": "UP", "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "wait_ratio",
    },
    "q_time_min": {
        "pct_level": "down",
        "priority": {"conservative": "UP", "standard": "UP", "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "q_time_min",
    },
    "max_util": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": None, "aggressive": None},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "max_util",
    },
    "utilization_avg": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": "DOWN", "aggressive": "DOWN"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": False},
        "target_kpi": "utilization_avg",
    },
    "available_tool_ratio": {
        "pct_level": "up",
        "priority": {"conservative": None, "standard": None, "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": True, "aggressive": True},
        "target_kpi": "available_tool_ratio",
    },
}

_PLAN_LEVELS = ("conservative", "standard", "aggressive")


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _excess_ratio(feature: str, kpi_value: float) -> float:
    """KPI가 임계값을 얼마나 초과했는지 비율 반환 (0.0 ~).
    available_tool_ratio는 낮을수록 나쁘므로 역방향 계산.
    """
    thr = _FEATURE_THRESHOLD.get(feature, 0.0)
    if thr == 0.0:
        return 0.0
    if feature == "available_tool_ratio":
        return max(0.0, (thr - kpi_value) / thr)
    return max(0.0, (kpi_value - thr) / thr)


def _dominant_feature(cause_report: CauseReport) -> str | None:
    """SHAP 절대값 기준 dominant feature 반환. 알 수 없는 피처는 skip."""
    for sf in sorted(cause_report.shap_top, key=lambda x: abs(x.shap_value), reverse=True):
        if sf.feature in _FEATURE_RULES:
            return sf.feature
    return None


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
    # "up"
    if plan_level == "conservative":
        return sev_pct["standard"]
    return sev_pct["aggressive"]


def _check_superhotlot(
    cause_report: CauseReport,
    alert: BottleneckAlert,
    plan_level: str,
    severity: SeverityLevel,
    eligible: bool,
) -> bool:
    """SUPERHOTLOT 활성화 여부 판정.

    eligible=False 이면 즉시 False.
    severity 게이트 통과 후 피처 기반 조건 1~4 중 plan_level별 조건 충족 시 True.
    """
    if not eligible:
        return False

    # severity 게이트
    if plan_level == "conservative" and severity != SeverityLevel.CRITICAL:
        return False
    if plan_level == "standard" and severity not in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}:
        return False
    if plan_level == "aggressive" and severity == SeverityLevel.LOW:
        return False

    kpi = {f.feature: f.kpi_value for f in cause_report.shap_top}
    q_time = kpi.get("q_time_min", 0.0)
    avail = kpi.get("available_tool_ratio", 1.0)
    wait = kpi.get("wait_ratio", 0.0)

    qtc_map: dict[str, float] = getattr(config, "QTC_THRESHOLD_MIN", {})
    qtc = qtc_map.get(alert.toolgroup)
    cond1 = qtc is not None and q_time > qtc * 0.80
    cond2 = avail < 0.40
    cond3 = wait > 0.70 and avail < 0.50
    cond4 = alert.impact.at_risk_lots > 0

    if plan_level == "conservative":
        return cond1 or cond4
    if plan_level == "standard":
        return cond1 or cond3 or cond4
    return cond1 or cond2 or cond3 or cond4


def _detect_scenario(cause_report: CauseReport) -> str | None:
    """복수 피처 악화 시 상호작용 시나리오(S1~S5) 감지. 우선순위 순으로 반환."""
    positive = {
        sf.feature for sf in cause_report.shap_top
        if sf.shap_value > 0 and sf.feature in _FEATURE_RULES
    }
    kpi = {f.feature: f.kpi_value for f in cause_report.shap_top}
    avail = kpi.get("available_tool_ratio", 1.0)

    # S4: 3중 악화 — 최우선
    if {"wip", "wait_ratio", "q_time_min"}.issubset(positive):
        return "S4"
    # S1: WIP 과부하 + 가용 장비 급감
    if "wip" in positive and "available_tool_ratio" in positive and avail < 0.5:
        return "S1"
    # S2: 대기 비율 + q-time 급증
    if {"wait_ratio", "q_time_min"}.issubset(positive):
        return "S2"
    # S5: 가동률 포화 + 가용 장비 하락
    if "utilization_avg" in positive and "available_tool_ratio" in positive and avail < 0.5:
        return "S5"
    # S3: 최대 + 평균 가동률 동시 포화
    if {"max_util", "utilization_avg"}.issubset(positive):
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
        params["conservative"]["release_interval_delta_pct"] = severity_pct["conservative"]
        return params, False, None

    if scenario == "S3":
        params["aggressive"]["superhotlot_enable"] = True
        reason = "max_util + utilization_avg 동시 포화 — 장비 증설 또는 PM 일정 검토 권장"
        return params, True, reason

    if scenario == "S4":
        for lv in _PLAN_LEVELS:
            params[lv]["release_interval_delta_pct"] = agg_pct
            params[lv]["priority_direction"] = "UP"
            params[lv]["superhotlot_enable"] = True
        reason = "3중 악화(wip + wait_ratio + q_time_min) — HITL 에스컬레이션 강력 권장"
        return params, True, reason

    if scenario == "S5":
        for lv in _PLAN_LEVELS:
            params[lv]["superhotlot_enable"] = True
        return params, False, None

    return params, False, None


def clip_interval_pct(proposed_pct: float, severity: SeverityLevel) -> float:
    """release_interval_delta_pct를 severity 상한 및 절대 상한으로 클리핑."""
    return min(proposed_pct, _SEVERITY_MAX_PCT.get(severity, _ABSOLUTE_MAX_PCT), _ABSOLUTE_MAX_PCT)


# ── 메인 API ─────────────────────────────────────────────────────────────────

def generate_candidates(
    alert: BottleneckAlert,
    cause_report: CauseReport,
) -> dict:
    """TG 1개에 대한 보수/표준/강화 파라미터 dict 생성.

    반환 구조 (5-3절):
    {
        "conservative": {"release_interval_delta_pct", "priority_direction",
                         "superhotlot_enable", "target_kpi"},
        "standard":     {...},
        "aggressive":   {...},
        "hitl_escalation_recommended": bool,
        "escalation_reason": str | None,
    }
    """
    severity = alert.severity
    severity_pct = _SEVERITY_PCT[severity]

    dominant = _dominant_feature(cause_report) or "wip"
    rule = _FEATURE_RULES[dominant]

    params: dict[str, dict] = {}
    for lv in _PLAN_LEVELS:
        pct = _resolve_pct(rule["pct_level"], severity, lv)
        superhotlot = _check_superhotlot(
            cause_report, alert, lv, severity, rule["superhotlot_eligible"][lv]
        )
        params[lv] = {
            "release_interval_delta_pct": pct,
            "priority_direction": rule["priority"][lv],
            "superhotlot_enable": superhotlot,
            "target_kpi": rule["target_kpi"],
        }

    scenario = _detect_scenario(cause_report)
    params, hitl, escalation_reason = _apply_scenario(params, scenario, severity_pct)

    return {
        "conservative": params["conservative"],
        "standard": params["standard"],
        "aggressive": params["aggressive"],
        "hitl_escalation_recommended": hitl,
        "escalation_reason": escalation_reason,
    }


# ── 레거시 (Step 4에서 제거 예정) ─────────────────────────────────────────────

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
