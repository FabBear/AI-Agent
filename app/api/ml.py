import asyncio
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.deps import get_db, get_predict_service, verify_internal_token
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success
from app.config import Settings, get_settings
from app.repositories.tg_metrics_repository import TgMetricsRepository
from app.services.predict_service import PredictService
from app.services.spring_client import get_spring_client

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
    pool: Annotated[asyncpg.Pool, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[PredictService, Depends(get_predict_service)],
) -> ApiResponse[PredictResult]:
    records = await TgMetricsRepository(pool).find_latest_records_by_fab(
        request.fab_id,
        request.snapshot_time,
    )
    details = await asyncio.to_thread(
        service.predict_all,
        [record.kpi for record in records],
    )

    # [MLOps] 전체 ToolGroup 예측을 백엔드에 위임 적재 (Drift 평가용 데이터 축적).
    # 추론 응답을 지연시키지 않도록 fire-and-forget (실패는 save_ml_predictions 내부에서 로깅).
    persist_payload = [
        {
            "runId": settings.live_run_id,
            "snapshotTime": detail.snapshot_time,
            "tgName": detail.toolgroup,
            "predProb": detail.probability,
            "isBottleneckPred": detail.probability >= settings.alarm_proba_threshold,
        }
        for detail in details
    ]
    asyncio.create_task(get_spring_client().save_ml_predictions(persist_payload))

    records_by_code = {record.kpi.toolgroup: record for record in records}
    predictions = [
        Prediction(
            tg_id=records_by_code[detail.toolgroup].tg_id,
            tg_code=detail.toolgroup,
            bottleneck_prob=detail.probability,
            risk_grade=_risk_grade(detail.probability, settings),
            shap_top=[
                ShapFeature(feature=item.feature, importance=item.importance)
                for item in detail.shap_top
            ],
        )
        for detail in details
    ]
    snapshot_time = max(
        (record.measured_at for record in records),
        default=request.snapshot_time,
    )
    if snapshot_time is None:
        snapshot_time = datetime.now(UTC)
    return success(
        PredictResult(
            snapshot_time=snapshot_time,
            predictions=predictions,
            high_critical_count=sum(
                prediction.risk_grade in {"HIGH", "CRITICAL"} for prediction in predictions
            ),
        )
    )


def _risk_grade(probability: float, settings: Settings) -> str:
    if probability >= settings.risk_critical_threshold:
        return "CRITICAL"
    if probability >= settings.risk_high_threshold:
        return "HIGH"
    if probability >= settings.risk_medium_threshold:
        return "MEDIUM"
    return "LOW"
