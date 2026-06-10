"""Agent API stubs."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import Field, model_validator

from app.api.deps import InternalUser, get_internal_user, verify_internal_token
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success

router = APIRouter()


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


STEP_NAMES = (
    "DIFFUSION_ANALYSIS",
    "CAUSE_ANALYSIS",
    "ACTION_PLAN_GEN",
    "ACTION_PLAN_COMPARE",
    "HITL_WAITING",
    "REPORT_GEN",
)


@router.post("/run", status_code=202, response_model=ApiResponse[AgentRunResult])
async def run_agent(
    request: AgentRunRequest,
    _: Annotated[None, Depends(verify_internal_token)],
) -> ApiResponse[AgentRunResult]:
    return success(AgentRunResult(case_id=request.case_id))


@router.post("/hitl-result", response_model=ApiResponse[HitlResult])
async def receive_hitl_result(
    request: HitlResultRequest,
    _: Annotated[None, Depends(verify_internal_token)],
    __: Annotated[InternalUser, Depends(get_internal_user)],
) -> ApiResponse[HitlResult]:
    next_step = "REPORT_GENERATION" if request.decision == HitlDecision.APPROVED else "CLOSED"
    return success(HitlResult(case_id=request.case_id, next_step=next_step))


@router.get("/cases/{case_id}/progress", response_model=ApiResponse[ProgressResult])
async def get_case_progress(
    case_id: UUID,
    _: Annotated[InternalUser, Depends(get_internal_user)],
) -> ApiResponse[ProgressResult]:
    steps = [
        ProgressStep(step_order=index, step_name=step_name)
        for index, step_name in enumerate(STEP_NAMES, start=1)
    ]
    return success(ProgressResult(case_id=case_id, steps=steps))
