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
    """4개 분석에서 하나의 피처에 대한 증거 번들."""

    feature: str
    votes: int = 0
    score: float = 0.0  # 가중 종합 점수 (SHAP비중 + 트렌드R² + G* + 업스트림)

    shap_rank: int | None = None
    shap_value: float | None = None

    trend_slope: float | None = None
    trend_r2: float | None = None
    trend_significant: bool = False

    upstream_match: bool = False

    g_star_p_value: float | None = None
    g_star_significant: bool = False

    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"


class CauseCategory(BaseModel):
    """여러 관련 피처를 묶어 하나의 원인 카테고리로 집계한 결과."""

    name: str                       # "설비_포화", "대기_누적", "WIP_누적", "공급_부족"
    features: list[str]             # 포함된 피처 목록
    shap_share_pct: float           # 전체 양수 SHAP 중 이 카테고리 비율 (%)
    n_trend_significant: int        # 유의미 트렌드 피처 수
    upstream_match: bool
    g_star_confirmed: bool
    total_score: float              # 종합 점수 (랭킹 기준)
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


class CauseJudgment(BaseModel):
    """LLM 판정 에이전트의 최종 원인 판정."""

    primary_category: str = ""     # 주요 원인 카테고리 ("설비_포화" 등)
    primary_cause: str             # 카테고리 내 대표 피처
    primary_confidence: Literal["HIGH", "MEDIUM", "LOW"]
    primary_reasoning: str

    secondary_causes: list[str] = []   # 보조 카테고리 또는 피처
    dismissed: list[str] = []          # 기각된 카테고리
    dismissed_reason: str = ""

    needs_more_data: bool = False
    cause_summary: str


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
    cause_categories: list[CauseCategory] = []
    judgment: CauseJudgment | None = None
    cause_summary: str
