"""최근 N 스냅샷의 KPI 변화율을 분석해 가장 빠르게 악화 중인 지표를 찾는다."""

import numpy as np

from agents.schemas.cause import TrendInsight
from agents.schemas.kpi import ToolGroupKPI

_KPI_FIELDS = [
    "q_time_min",
    "wait_ratio",
    "wip",
    "available_tool_ratio",
    "utilization_avg",
    "max_util",
]
_INVERSE_KPI = {"available_tool_ratio"}

# |slope_per_hour| / mean_value > 이 비율이면 유의미한 추세로 간주
_SLOPE_SIGNIFICANCE_RATIO = 0.05
_R2_MIN = 0.5


def get_trend_top(
    window: dict[float, list[ToolGroupKPI]],
    toolgroup: str,
    top_n: int = 3,
    priority_features: list[str] | None = None,
) -> list[TrendInsight]:
    times = sorted(window.keys())
    if len(times) < 2:
        return []

    series: dict[str, list[float]] = {f: [] for f in _KPI_FIELDS}
    for t in times:
        kpi_map = {k.toolgroup: k for k in window[t]}
        kpi = kpi_map.get(toolgroup)
        for field in _KPI_FIELDS:
            series[field].append(getattr(kpi, field, 0.0) if kpi else 0.0)

    interval_h = (times[1] - times[0]) / 60.0 if len(times) >= 2 else 1.0

    insights: list[TrendInsight] = []
    for field, values in series.items():
        if all(v == 0 for v in values):
            continue

        arr = np.array(values, dtype=float)
        x = np.arange(len(arr), dtype=float)
        coeffs = np.polyfit(x, arr, 1)
        slope_per_step = float(coeffs[0])
        slope_per_hour = slope_per_step / max(interval_h, 1e-6)

        predicted = np.polyval(coeffs, x)
        ss_res = float(np.sum((arr - predicted) ** 2))
        ss_tot = float(np.sum((arr - arr.mean()) ** 2))
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        mean_val = max(abs(arr.mean()), 1e-6)
        normalized_slope = abs(slope_per_hour) / mean_val
        significant = normalized_slope > _SLOPE_SIGNIFICANCE_RATIO and r2 > _R2_MIN

        insights.append(
            TrendInsight(
                feature=field,
                slope_per_hour=round(slope_per_hour, 4),
                r2=round(r2, 4),
                significant=significant,
                values=[round(v, 4) for v in values],
            )
        )

    priority_set = list(priority_features or [])

    def _sort_key(ins: TrendInsight) -> tuple:
        priority_rank = next(
            (i for i, f in enumerate(priority_set) if f == ins.feature), len(priority_set)
        )
        raw = ins.slope_per_hour
        deterioration = -raw if ins.feature in _INVERSE_KPI else raw
        return (priority_rank, -deterioration)

    insights.sort(key=_sort_key)
    return insights[:top_n]
