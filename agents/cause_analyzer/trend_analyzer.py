"""최근 N 스냅샷의 KPI 변화율을 분석해 가장 빠르게 악화 중인 지표를 찾는다."""

import numpy as np

from agents.schemas.cause import TrendInsight
from agents.schemas.kpi import ToolGroupKPI

_KPI_FIELDS = [
    "q_time_min",
    "wait_ratio",
    "wip",
    "available_tool_ratio",  # 낮아질수록 나쁨 → slope 반전
    "utilization_avg",
    "max_util",
    "max_avg_q_time",
]
# 값이 낮아질수록 나쁜 KPI (slope 부호 반전해서 "악화율"로 통일)
_INVERSE_KPI = {"available_tool_ratio"}


def get_trend_top(
    window: dict[float, list[ToolGroupKPI]],
    toolgroup: str,
    top_n: int = 3,
) -> list[TrendInsight]:
    """
    window: {snapshot_time: [ToolGroupKPI, ...]} — 오래된 순
    악화 속도가 빠른 KPI 상위 top_n 반환.
    """
    times = sorted(window.keys())
    if len(times) < 2:
        return []

    # toolgroup별 시계열 추출
    series: dict[str, list[float]] = {f: [] for f in _KPI_FIELDS}
    for t in times:
        kpi_map = {k.toolgroup: k for k in window[t]}
        kpi = kpi_map.get(toolgroup)
        for field in _KPI_FIELDS:
            series[field].append(getattr(kpi, field, 0.0) if kpi else 0.0)

    # 시간 간격 추정 (분 → 시간)
    interval_h = (times[1] - times[0]) / 60.0 if len(times) >= 2 else 1.0

    insights: list[TrendInsight] = []
    for field, values in series.items():
        if all(v == 0 for v in values):
            continue
        x = np.arange(len(values), dtype=float)
        slope_per_step = float(np.polyfit(x, values, 1)[0])
        slope_per_hour = slope_per_step / max(interval_h, 1e-6)

        insights.append(
            TrendInsight(
                feature=field,
                slope_per_hour=round(slope_per_hour, 4),
                values=[round(v, 4) for v in values],
            )
        )

    # 악화 속도 내림차순 정렬
    def _deterioration(ins: TrendInsight) -> float:
        raw = ins.slope_per_hour
        return -raw if ins.feature in _INVERSE_KPI else raw

    insights.sort(key=_deterioration, reverse=True)
    return insights[:top_n]
