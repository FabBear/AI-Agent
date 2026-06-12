"""XGBoost 추론만 담당 — 심각도 판단은 cascade analyzer에서.

전체 예측의 DB 적재(MLOps)는 운영 예측 경로(/api/ml/predict)가 단일 책임지므로
여기서는 G*(임계값 초과 ToolGroup) 추출만 수행한다.
"""

import xgboost as xgb
from pathlib import Path

from agents import config
from agents.bottleneck_detector.feature_engineering import build_feature_matrix
from agents.schemas.alert import PotentialBottleneck
from agents.schemas.kpi import ToolGroupKPI

_ROOT = Path(__file__).parent.parent.parent
_booster: xgb.Booster | None = None


def _load_model() -> xgb.Booster:
    global _booster
    if _booster is None:
        _booster = xgb.Booster()
        _booster.load_model(str(_ROOT / config.MODEL_DIR / config.MODEL_FILENAME))
    return _booster


def detect(
    kpi_list: list[ToolGroupKPI],
    prev_kpi_list: list[ToolGroupKPI] | None = None,
) -> list[PotentialBottleneck]:
    """XGBoost로 병목 확률을 계산하고 임계값 초과 TG(G*)만 반환한다.

    전체 예측의 DB 적재는 운영 예측 경로(/api/ml/predict)가 단일 책임진다.
    """
    if not kpi_list:
        return []
    booster = _load_model()
    features = build_feature_matrix(kpi_list, prev_kpi_list=prev_kpi_list)
    probabilities = booster.predict(xgb.DMatrix(features))

    threshold = getattr(config, "ALARM_PROBA_THRESHOLD", 0.7)
    return [
        PotentialBottleneck(
            toolgroup=kpi.toolgroup,
            snapshot_time=kpi.snapshot_time,
            probability=round(float(prob), 4),
        )
        for kpi, prob in zip(kpi_list, probabilities)
        if round(float(prob), 4) >= threshold
    ]
