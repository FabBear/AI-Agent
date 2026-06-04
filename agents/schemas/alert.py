from enum import Enum

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class PotentialBottleneck(BaseModel):
    """XGBoost 1차 출력 — 심각도 미결정."""

    toolgroup: str
    snapshot_time: float
    probability: float = Field(ge=0.0, le=1.0)


class CascadeImpact(BaseModel):
    capacity_stress_score: float = Field(ge=0.0, le=1.0)  # 후속 공정 용량 포화도
    ct_increase_min: float = Field(ge=0.0)  # CT 증가 추정(분)
    at_risk_lots: float = Field(ge=0.0)  # 위험 lot 수
    affected_tgs: list[str]  # 영향받는 후속 TG 목록
    impact_score: float = Field(ge=0.0, le=1.0)  # 세 지표 가중합 [0,1]


class BottleneckAlert(BaseModel):
    """cascade 분석 후 최종 알림 — 심각도 결정됨."""

    toolgroup: str
    severity: SeverityLevel
    composite_score: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)
    impact: CascadeImpact
    snapshot_time: float
