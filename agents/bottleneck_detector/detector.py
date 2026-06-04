"""XGBoost 추론만 담당 — 심각도 판단은 cascade analyzer에서."""

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


def detect(kpi_list: list[ToolGroupKPI]) -> list[PotentialBottleneck]:
    """XGBoost로 병목 확률을 계산하고 0.5 초과 TG를 반환한다."""
    if not kpi_list:
        return []
    booster = _load_model()
    features = build_feature_matrix(kpi_list)
    probabilities = booster.predict(xgb.DMatrix(features))

    return [
        PotentialBottleneck(
            toolgroup=kpi.toolgroup,
            snapshot_time=kpi.snapshot_time,
            probability=round(float(prob), 4),
        )
        for kpi, prob in zip(kpi_list, probabilities)
        if prob > 0.5
    ]
