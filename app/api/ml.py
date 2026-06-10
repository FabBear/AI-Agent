"""ML API stubs."""

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.deps import verify_internal_token
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success

router = APIRouter()


class PredictRequest(ApiModel):
    fab_id: UUID = Field(alias="fabId")
    snapshot_time: datetime | None = Field(default=None, alias="snapshotTime")


class ShapFeature(ApiModel):
    feature: str
    importance: float


class Prediction(ApiModel):
    tg_id: UUID = Field(alias="tgId")
    tg_code: str = Field(alias="tgCode")
    bottleneck_prob: float = Field(alias="bottleneckProb")
    risk_grade: str = Field(alias="riskGrade")
    shap_top: list[ShapFeature] = Field(alias="shapTop")


class PredictResult(ApiModel):
    snapshot_time: datetime = Field(alias="snapshotTime")
    predictions: list[Prediction]
    high_critical_count: int = Field(alias="highCriticalCount")


@router.post("/predict", response_model=ApiResponse[PredictResult])
async def predict(
    request: PredictRequest,
    _: Annotated[None, Depends(verify_internal_token)],
) -> ApiResponse[PredictResult]:
    snapshot_time = request.snapshot_time or datetime.now(timezone.utc)
    return success(
        PredictResult(
            snapshot_time=snapshot_time,
            predictions=[],
            high_critical_count=0,
        )
    )
