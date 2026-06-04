from pydantic import BaseModel, Field


class SimParamDelta(BaseModel):
    """시뮬레이션 입력 파라미터 변경안."""

    # Lot Release 조정
    release_interval_delta_pct: float | None = None  # RELEASE_INTERVAL 변경률 (%, 양수=증가)
    lot_priority_rule: str | None = None  # 투입 우선순위 (HIGH_WIP_FIRST, DUE_DATE 등)
    superhotlot_enable: bool | None = None  # 긴급 lot 플래그 활성화 여부

    # Dispatching Rule 조정
    dispatch_rule: str | None = None  # SPT / EDD / FIFO / PRIORITY
    dispatch_priority_weight: float | None = None  # 우선순위 가중치 조정


class SolutionCandidate(BaseModel):
    rank: int
    name: str
    params: SimParamDelta
    target_kpi: str  # 주로 개선 목표로 하는 KPI
    expected_effect: str  # 기대 효과 텍스트
    confidence: float = Field(ge=0.0, le=1.0)  # 신뢰도
    rationale: str  # 선택 근거
