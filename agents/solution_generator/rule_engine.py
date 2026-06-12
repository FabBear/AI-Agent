"""TG 단위 Lot Release 파라미터 조합 생성 (보수/표준/강화).

judgment.primary_category 기반 카테고리 규칙 테이블로 파라미터 결정.
judgment가 None이면 generate_candidates()가 None을 반환한다.
"""

from __future__ import annotations

from agents import config
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.cause import CauseJudgment, CauseReport
from agents.schemas.solution import GlobalSolutionPlan

_DEFAULT_INTERVAL = 60.0   # generate_global_plans fallback — 레거시
_COMPARISON_STEP = 2.0     # generate_global_plans A↔B 차이 — 레거시

# ── CRITICAL severity RELEASE_INTERVAL 조정 수치 (%) ─────────────────────────
_RELEASE_PCT: dict[str, float] = {"conservative": 15.0, "standard": 22.0, "aggressive": 28.0}
_MAX_PCT = 28.0
_ABSOLUTE_MAX_PCT = 30.0

# ── 카테고리별 규칙 테이블 ────────────────────────────────────────────────────
# pct_level:
#   "same" — severity 수치 그대로  (C→C, S→S, A→A)
#   "down" — 한 단계 낮춤          (C→0%, S→C, A→S)
#   "up"   — 한 단계 높임          (C→S, S→A, A→A)
_CATEGORY_RULES: dict[str, dict] = {
    "WIP_누적": {
        "pct_level": "same",
        "priority": {"conservative": None, "standard": "DOWN", "aggressive": "DOWN"},
        "superhotlot_eligible": {"conservative": False, "standard": False, "aggressive": True},
        "target_kpi": "wip",
    },
    "대기_누적": {
        "pct_level": "down",   # interval 억제보다 priority 조정이 핵심
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
        "pct_level": "up",     # 장비 급감 → 더 강하게 투입 억제
        "priority": {"conservative": None, "standard": None, "aggressive": "UP"},
        "superhotlot_eligible": {"conservative": False, "standard": True, "aggressive": True},
        "target_kpi": "available_tool_ratio",
    },
}

_FALLBACK_CATEGORY = "WIP_누적"  # 미등록 카테고리 fallback
_PLAN_LEVELS = ("conservative", "standard", "aggressive")

# ── 피처명 → 카테고리명 매핑 ──────────────────────────────────────────────────
# _delta_120 suffix는 동일 피처의 120분 변화량으로 같은 카테고리로 취급한다.
# secondary_causes가 카테고리명 대신 피처명을 담아올 경우 이 테이블로 정규화한다.
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
    """피처명 또는 카테고리명을 카테고리명으로 정규화.

    이미 카테고리명이면 그대로 반환한다.
    """
    return _FEATURE_TO_CATEGORY.get(name, name)


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _resolve_pct(pct_level: str, plan_level: str) -> float:
    """pct_level 지시에 따라 실제 조정 수치(%) 결정."""
    if pct_level == "same":
        return _RELEASE_PCT[plan_level]
    if pct_level == "down":
        if plan_level == "conservative":
            return 0.0
        if plan_level == "standard":
            return _RELEASE_PCT["conservative"]
        return _RELEASE_PCT["standard"]
    # "up"
    if plan_level == "conservative":
        return _RELEASE_PCT["standard"]
    return _RELEASE_PCT["aggressive"]


def _check_superhotlot(alert: BottleneckAlert, eligible: bool) -> bool:
    """SUPERHOTLOT 활성화 여부.

    eligible=False이면 즉시 False.
    eligible=True이고 at_risk_lots > 0이면 True.
    """
    if not eligible:
        return False
    return alert.impact.at_risk_lots > 0


def _detect_scenario(judgment: CauseJudgment) -> str | None:
    """복합 원인 시나리오 감지. secondary_causes가 없으면 None.

    secondary_causes는 카테고리명 또는 피처명(_delta_120 포함)을 담을 수 있다.
    _to_category()로 정규화 후 비교한다.
    """
    if not judgment.secondary_causes:
        return None

    all_cats = {judgment.primary_category} | {_to_category(s) for s in judgment.secondary_causes}

    # S4: 3개 이상 동시 악화 — 최우선
    if len(all_cats) >= 3:
        return "S4"
    # S1: WIP 과부하 + 공급 부족
    if {"WIP_누적", "공급_부족"}.issubset(all_cats):
        return "S1"
    # S2: 대기 누적 + WIP 누적
    if {"대기_누적", "WIP_누적"}.issubset(all_cats):
        return "S2"
    # S5: 설비 포화 + 공급 부족
    if {"설비_포화", "공급_부족"}.issubset(all_cats):
        return "S5"
    # S3: 설비 포화 포함 복합
    if "설비_포화" in all_cats:
        return "S3"
    return None


def _apply_scenario(params: dict[str, dict], scenario: str | None) -> dict[str, dict]:
    """시나리오별 파라미터 보정."""
    if scenario is None:
        return params

    if scenario == "S1":
        for lv in _PLAN_LEVELS:
            params[lv]["release_interval_delta_pct"] = _RELEASE_PCT["aggressive"]
            params[lv]["superhotlot_enable"] = True
        return params

    if scenario == "S2":
        for lv in _PLAN_LEVELS:
            params[lv]["priority_direction"] = "UP"
            params[lv]["superhotlot_enable"] = True
        params["conservative"]["release_interval_delta_pct"] = max(
            params["conservative"]["release_interval_delta_pct"],
            _RELEASE_PCT["conservative"],
        )
        return params

    if scenario == "S3":
        params["aggressive"]["superhotlot_enable"] = True
        return params

    if scenario == "S4":
        for lv in _PLAN_LEVELS:
            params[lv]["release_interval_delta_pct"] = _RELEASE_PCT["aggressive"]
            params[lv]["priority_direction"] = "UP"
            params[lv]["superhotlot_enable"] = True
        return params

    if scenario == "S5":
        for lv in _PLAN_LEVELS:
            params[lv]["superhotlot_enable"] = True
        return params

    return params


def clip_interval_pct(proposed_pct: float) -> float:
    """release_interval_delta_pct를 상한으로 클리핑."""
    return min(proposed_pct, _MAX_PCT, _ABSOLUTE_MAX_PCT)


# ── 메인 API ─────────────────────────────────────────────────────────────────

def generate_candidates(
    alert: BottleneckAlert,
    cause_report: CauseReport,
) -> dict | None:
    """TG 1개에 대한 보수/표준/강화 파라미터 dict 생성.

    cause_report.judgment가 None이면 None 반환 (node.py에서 skip 처리).

    반환 구조:
    {
        "conservative": {"release_interval_delta_pct", "priority_direction",
                         "superhotlot_enable", "target_kpi"},
        "standard":     {...},
        "aggressive":   {...},
    }
    """
    if cause_report.judgment is None:
        return None

    judgment = cause_report.judgment

    primary = judgment.primary_category
    rule = _CATEGORY_RULES.get(primary) or _CATEGORY_RULES[_FALLBACK_CATEGORY]

    params: dict[str, dict] = {}
    for lv in _PLAN_LEVELS:
        pct = _resolve_pct(rule["pct_level"], lv)
        superhotlot = _check_superhotlot(alert, rule["superhotlot_eligible"][lv])
        params[lv] = {
            "release_interval_delta_pct": pct,
            "priority_direction": rule["priority"][lv],
            "superhotlot_enable": superhotlot,
            "target_kpi": rule["target_kpi"],
        }

    scenario = _detect_scenario(judgment)
    params = _apply_scenario(params, scenario)

    return {
        "conservative": params["conservative"],
        "standard": params["standard"],
        "aggressive": params["aggressive"],
    }


# ── 레거시 (Step 4에서 제거 예정) ─────────────────────────────────────────────

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
