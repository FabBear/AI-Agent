from typing import Literal

from pydantic import BaseModel, Field


# ── 신규: 글로벌 복합 대응안 스키마 ──────────────────────────────────────────────

class LotAdjustment(BaseModel):
    """FAB 투입 전 lot 1건의 우선순위/superhotlot 조정 정보."""

    lot_plan_id: int                                 # mes_lot_release_plan.id (표시용 식별자)
    lot_type: str                                    # mes_lot_release_plan.lot_type (UPDATE 매칭 키)
    product_name: str
    release_time: float                              # 원래 계획 투입 시간 (분)
    whatif_release_time: float                       # WHATIF 시나리오 기준 투입 시간 (multiplier 적용 후)
    action_kind: Literal["SET_SUPER_HOT", "LOT_PRIORITY"]
    priority: int                                    # 30 (위험/경고상단) | 20 (경고하단)
    time_to_due: float                               # 계산된 time_to_due (분)
    zone: Literal["danger", "warn_upper", "warn_lower"]


class GlobalCompositeCandidate(BaseModel):
    """Critical TG 전체를 묶은 글로벌 복합 대응안 (보수/표준/강화 중 1개)."""

    plan_id: Literal["conservative", "standard", "aggressive"]
    target_toolgroups: list[str]
    release_interval_delta_pct: float               # FAB 전체 투입 간격 조정 (%)
    lot_adjustments: list[LotAdjustment] = []       # FAB 투입 전 lot별 우선순위 조정
    cause_complexity: Literal["single", "mixed", "complex"] = "single"
    hitl_escalation_recommended: bool = False
    escalation_reason: str = ""


# ── 레거시 스키마 (하위 호환) ──────────────────────────────────────────────────────

class TGAction(BaseModel):
    """TG별 dispatch rule 변경 액션."""

    toolgroup: str
    dispatch_rule: str | None = None  # "superhotlot setupavoidance" | "setupavoidance" | "EDD" | None


class SolutionPlan(BaseModel):
    """원인 카테고리 기반 대응 플랜 (Plan A / B / C)."""

    plan_id: str                            # "A" | "B" | "C"
    global_interval_delta_pct: float        # 미래 lot 투입 간격 증가율 (%, FAB 전체)
    per_tg_actions: list[TGAction]          # TG별 dispatch rule (원인 기반)
    hitl_escalation: bool = False           # 설비_고장 TG 있으면 True
    description: str = ""
    expected_effect: str = ""


class SimParamDelta(BaseModel):
    """WHATIF 시뮬레이션에 적용할 파라미터 변화량."""

    release_interval_delta_pct: float | None = None
    priority_direction: Literal["UP", "DOWN"] | None = None
    lot_priority_rule: str | None = None
    dispatch_rule: str | None = None
    superhotlot_enable: bool = False


class SolutionCandidate(BaseModel):
    """툴그룹별 대응안 후보 (verification_agent 입력)."""

    rank: int = 1
    name: str = ""
    target_kpi: str = "q_time_min"
    params: SimParamDelta = Field(default_factory=SimParamDelta)
    expected_effect: str = ""
    rationale: str = ""


class GlobalSolutionPlan(BaseModel):
    """전체 Critical TG를 한번에 커버하는 글로벌 대응 플랜 (시뮬레이션 입력값).

    Lot Release 테이블 파라미터(인터벌·우선순위·수퍼핫랏)만 조정한다.
    플랜 A / B 는 release_interval_minutes 값만 다르며, 나머지는 동일하다.
    """

    plan_id: str  # "A" | "B"
    target_toolgroups: list[str]  # 이번 플랜의 대상 Critical TG 목록
    current_interval_minutes: float  # 현재 인터벌 (참고용)
    release_interval_minutes: float  # 목표 Lot Release Interval (분) — 시뮬 입력 절댓값
    lot_priority_rule: str | None = None  # 투입 우선순위 룰 (HIGH_WIP_FIRST 등)
    superhotlot_enable: bool  # 긴급 lot 플래그 활성화
    description: str
    expected_effect: str
