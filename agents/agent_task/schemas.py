"""사용자 호출형 Agent Task의 공용 요청/응답 스키마(/api/agent/tasks 계약).

여러 화면-호출 에이전트(fab_briefing_agent, period_report_agent)가 동일한
AgentTask* 입출력 계약을 공유한다."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AgentModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class AgentTaskProgressStep(AgentModel):
    step_name: str = Field(alias="stepName")
    status: str
    message: str | None = None
    occurred_at: datetime = Field(alias="occurredAt")


class EvidenceItem(AgentModel):
    label: str
    value: str
    description: str
    severity: str = "info"


class Propagation(AgentModel):
    summary: str
    affected_processes: list[str] = Field(default_factory=list, alias="affectedProcesses")
    affected_tool_groups: list[str] = Field(default_factory=list, alias="affectedToolGroups")
    horizon: str = "현재 스냅샷 기준"


class ResponseDirection(AgentModel):
    title: str
    description: str
    owner: str = "Fab 운영"
    caution: str | None = None


class AgentReferences(AgentModel):
    case_ids: list[str] = Field(default_factory=list, alias="caseIds")
    tg_ids: list[str] = Field(default_factory=list, alias="tgIds")
    report_ids: list[str] = Field(default_factory=list, alias="reportIds")
    doc_ids: list[str] = Field(default_factory=list, alias="docIds")


class AgentArtifact(AgentModel):
    type: str
    title: str
    description: str
    ref_id: str | None = Field(default=None, alias="refId")


class WatchToolGroup(AgentModel):
    """현장 작업자가 지금 살펴보면 좋은 TG와 그 근거(현황 브리핑용 칩)."""

    tg_name: str = Field(alias="tgName")
    area_name: str | None = Field(default=None, alias="areaName")
    reason: str
    severity: str = "info"


class AgentTaskResult(AgentModel):
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    propagation: Propagation
    response_directions: list[ResponseDirection] = Field(
        default_factory=list,
        alias="responseDirections",
    )
    references: AgentReferences = Field(default_factory=AgentReferences)
    follow_up_prompts: list[str] = Field(default_factory=list, alias="followUpPrompts")
    artifacts: list[AgentArtifact] = Field(default_factory=list)
    watch_tool_groups: list[WatchToolGroup] = Field(default_factory=list, alias="watchToolGroups")


class AgentTaskAgentRequest(AgentModel):
    task_id: UUID = Field(alias="taskId")
    fab_id: UUID = Field(alias="fabId")
    user_id: UUID = Field(alias="userId")
    task_type: str = Field(alias="taskType")
    source_page: str = Field(alias="sourcePage")
    context: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class AgentTaskAgentResponse(AgentModel):
    status: str
    progress: list[AgentTaskProgressStep] = Field(default_factory=list)
    result: AgentTaskResult | None = None
    error_message: str | None = Field(default=None, alias="errorMessage")
