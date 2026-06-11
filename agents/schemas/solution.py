from typing import Literal

from pydantic import BaseModel, Field


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
