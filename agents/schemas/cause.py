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


class CauseReport(BaseModel):
    toolgroup: str
    snapshot_time: float
    shap_top: list[SHAPFeature]
    trend_top: list[TrendInsight]
    upstream_suspects: list[str]
    sim_forecast: SimForecast | None  # Forward 시뮬 결과 (없을 수도 있음)
    cause_summary: str
