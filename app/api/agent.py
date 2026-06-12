from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import Field, model_validator

from agents.agent_task import AgentTaskAgentRequest, AgentTaskAgentResponse
from agents.fab_briefing_agent import build_fab_briefing_response
from agents.period_report_agent import build_period_report_response
from app.api.deps import InternalUser, get_db, get_internal_user, verify_internal_token
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success
from app.repositories.agent_step_repository import AgentStepRepository
from app.services.agent_service import run_pipeline_with_timeout, run_post_hitl

router = APIRouter()
_USER_TASKS: dict[UUID, AgentTaskAgentResponse] = {}


class RiskGrade(str, Enum):
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HitlDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class AgentRunRequest(ApiModel):
    case_id: UUID = Field(alias="caseId")
    tg_id: UUID = Field(alias="tgId")
    tg_code: str = Field(alias="tgCode", min_length=1)
    snapshot_time: float = Field(alias="snapshotTime", ge=0)
    bottleneck_prob: float = Field(alias="bottleneckProb", ge=0, le=1)
    risk_grade: RiskGrade = Field(alias="riskGrade")


class AgentRunResult(ApiModel):
    case_id: UUID = Field(alias="caseId")
    status: Literal["STARTED"] = "STARTED"


class HitlResultRequest(ApiModel):
    case_id: UUID = Field(alias="caseId")
    decision_id: UUID = Field(alias="decisionId")
    decision: HitlDecision
    selected_plan_id: UUID | None = Field(default=None, alias="selectedPlanId")
    comment: str | None = None
    decided_by: UUID = Field(alias="decidedBy")
    decided_at: datetime = Field(alias="decidedAt")

    @model_validator(mode="after")
    def validate_selected_plan(self) -> "HitlResultRequest":
        if self.decision == HitlDecision.APPROVED and self.selected_plan_id is None:
            raise ValueError("APPROVED 결정에는 selectedPlanId가 필요합니다.")
        return self


class HitlResult(ApiModel):
    case_id: UUID = Field(alias="caseId")
    next_step: Literal["REPORT_GENERATION", "CLOSED"] = Field(alias="nextStep")


class ProgressStep(ApiModel):
    step_order: int = Field(alias="stepOrder")
    step_name: str = Field(alias="stepName")
    status: Literal["PENDING", "IN_PROGRESS", "DONE", "FAILED"] = "PENDING"
    started_at: datetime | None = Field(default=None, alias="startedAt")
    completed_at: datetime | None = Field(default=None, alias="completedAt")
    output_summary: str | None = Field(default=None, alias="outputSummary")
    attempt_no: int = Field(default=1, alias="attemptNo")


class ProgressResult(ApiModel):
    case_id: UUID = Field(alias="caseId")
    steps: list[ProgressStep]


@router.post("/run", status_code=202, response_model=ApiResponse[AgentRunResult])
async def run_agent(
    request: AgentRunRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(verify_internal_token)],
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> ApiResponse[AgentRunResult]:
    background_tasks.add_task(
        run_pipeline_with_timeout,
        request.case_id,
        request.tg_id,
        request.tg_code,
        request.snapshot_time,
        request.bottleneck_prob,
        request.risk_grade.value,
        pool,
    )
    return success(AgentRunResult(case_id=request.case_id))


@router.post("/tasks", response_model=AgentTaskAgentResponse)
async def create_agent_task(
    request: AgentTaskAgentRequest,
    _: Annotated[None, Depends(verify_internal_token)],
) -> AgentTaskAgentResponse:
    if request.task_type == "REPORT_PERIOD_SUMMARY":
        response = await build_period_report_response(request)
    else:
        response = await build_fab_briefing_response(request)
    _USER_TASKS[request.task_id] = response
    return response


@router.get("/tasks/{task_id}", response_model=AgentTaskAgentResponse)
async def get_agent_task(
    task_id: UUID,
    _: Annotated[None, Depends(verify_internal_token)],
) -> AgentTaskAgentResponse:
    response = _USER_TASKS.get(task_id)
    if response is None:
        return AgentTaskAgentResponse(
            status="FAILED",
            progress=[],
            errorMessage="해당 taskId의 Agent 작업을 찾을 수 없습니다.",
        )
    return response


@router.post("/hitl-result", response_model=ApiResponse[HitlResult])
async def receive_hitl_result(
    request: HitlResultRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(verify_internal_token)],
    __: Annotated[InternalUser, Depends(get_internal_user)],
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> ApiResponse[HitlResult]:
    background_tasks.add_task(
        run_post_hitl,
        request.case_id,
        request.decision.value,
        request.selected_plan_id,
        request.decided_by,
        request.decided_at,
        request.comment,
        pool,
    )
    next_step = "REPORT_GENERATION" if request.decision == HitlDecision.APPROVED else "CLOSED"
    return success(HitlResult(case_id=request.case_id, next_step=next_step))


@router.get("/cases/{case_id}/progress", response_model=ApiResponse[ProgressResult])
async def get_case_progress(
    case_id: UUID,
    _: Annotated[InternalUser, Depends(get_internal_user)],
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
) -> ApiResponse[ProgressResult]:
    rows = await AgentStepRepository(pool).find_by_case(case_id)
    steps = [ProgressStep.model_validate(row) for row in rows]
    return success(ProgressResult(case_id=case_id, steps=steps))
