from typing import TypedDict

from agents.schemas.alert import BottleneckAlert, PotentialBottleneck
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI


class PipelineState(TypedDict):
    kpi_snapshot: list[ToolGroupKPI]
    potential_bottlenecks: list[PotentialBottleneck]
    alerts: list[BottleneckAlert]
    cause_reports: list[CauseReport]
    cascade_report: str | None
    solution_candidates: list[dict]
    hitl_approved: bool | None
