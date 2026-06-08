from typing import TypedDict

from agents.schemas.alert import BottleneckAlert, PotentialBottleneck
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI


class PipelineState(TypedDict):
    kpi_snapshot: list[ToolGroupKPI]
    prev_kpi_snapshot: list[ToolGroupKPI]  # t-120분 스냅샷 (delta 피처용)
    potential_bottlenecks: list[PotentialBottleneck]
    alerts: list[BottleneckAlert]
    cause_reports: list[CauseReport]
    cascade_report: str | None
    current_release_interval: float | None  # 현재 Lot Release Interval (분), 없으면 None
    solution_candidates: list[dict]
    hitl_approved: bool | None
