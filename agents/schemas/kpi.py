from pydantic import BaseModel, Field


class ToolGroupKPI(BaseModel):
    toolgroup: str
    snapshot_time: float
    available_tool_ratio: float = Field(ge=0.0, le=1.0)
    q_time_min: float = Field(ge=0.0)
    wait_ratio: float = Field(ge=0.0)  # lot/machine 비율 — 1 초과 가능
    wip: float = Field(ge=0.0)
    setup_ratio_avg: float = Field(ge=0.0, le=1.0)
    utilization_avg: float = Field(ge=0.0, le=1.0)
    max_util: float = Field(ge=0.0, le=1.0)
