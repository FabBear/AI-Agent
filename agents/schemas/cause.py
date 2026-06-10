from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    kpi_value: float


class TrendInsight(BaseModel):
    feature: str
    slope_per_hour: float
    r2: float = 0.0
    significant: bool = False
    values: list[float]


class KpiComparison(BaseModel):
    now: float
    future: float
    delta: float
    pct_change: float
    reliability: str = "HIGH"


class SimForecast(BaseModel):
    t0: float
    t_future: float
    kpi_delta: dict[str, KpiComparison]
    gets_worse: bool


class GStarKpiResult(BaseModel):
    kpi: str
    delta_mean: float
    t_p_adj: float
    significant: bool


class FeatureEvidence(BaseModel):
    """4개 분석에서 하나의 피처에 대한 모든 증거를 집계한 번들."""

    feature: str
    votes: int  # 0~4: 몇 개 분석이 이 피처를 병목 원인으로 지목했나

    # SHAP
    shap_rank: int | None = None
    shap_value: float | None = None

    # 트렌드
    trend_slope: float | None = None
    trend_r2: float | None = None
    trend_significant: bool = False

    # 업스트림 (capacity/flow 피처에만 적용)
    upstream_match: bool = False

    # G* t-test
    g_star_p_value: float | None = None
    g_star_significant: bool = False

    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"


class CauseJudgment(BaseModel):
    """LLM 판정 에이전트의 최종 원인 판정 결과."""

    primary_cause: str
    primary_confidence: Literal["HIGH", "MEDIUM", "LOW"]
    primary_reasoning: str

    secondary_causes: list[str] = []
    dismissed: list[str] = []
    dismissed_reason: str = ""

    needs_more_data: bool = False  # True면 더 긴 window로 재시도

    cause_summary: str  # [주요 원인] / [악화 추세] / [업스트림] / [2시간 전망]


class ConsensusResult(BaseModel):
    agreed_features: list[str]
    conflicted_features: list[str]
    upstream_aligns: bool
    sim_aligns: bool
    g_star_confirmed: bool = False
    g_star_upstream_confirmed: list[str] = []
    g_star_sig_kpis: list[GStarKpiResult] = []
    g_star_proba: float = 0.0
    g_star_n_total: int = 0
    g_star_n_alarm: int = 0
    g_star_toolgroups_all: list[str] = []
    confidence_level: str = "LOW"
    summary: str = ""


class CauseReport(BaseModel):
    toolgroup: str
    snapshot_time: float
    shap_top: list[SHAPFeature]
    trend_top: list[TrendInsight]
    upstream_suspects: list[str]
    sim_forecast: SimForecast | None
    consensus: ConsensusResult
    evidence_bundle: list[FeatureEvidence] = []
    judgment: CauseJudgment | None = None
    cause_summary: str
