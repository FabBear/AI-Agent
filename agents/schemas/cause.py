from typing import Literal

from pydantic import BaseModel


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    kpi_value: float


class TrendInsight(BaseModel):
    feature: str
    slope_per_hour: float
    values: list[float]


class KpiComparison(BaseModel):
    now: float
    future: float
    delta: float
    pct_change: float
    reliability: str = "HIGH"  # HIGH / MED / LOW


class SimForecast(BaseModel):
    """Forward 시뮬레이션 2h 예측 결과."""

    t0: float
    t_future: float
    kpi_delta: dict[str, KpiComparison]  # {kpi_name: KpiComparison}
    gets_worse: bool  # 주요 지표가 악화되는지 여부


class GStarKpiResult(BaseModel):
    kpi: str
    delta_mean: float
    t_p_adj: float
    significant: bool


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
    cause_summary: str
