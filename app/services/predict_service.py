from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb

from agents.bottleneck_detector.feature_engineering import build_feature_matrix
from agents.schemas.alert import PotentialBottleneck
from agents.schemas.kpi import ToolGroupKPI
from app.config import Settings, get_settings


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    importance: float


@dataclass(frozen=True)
class PredictionDetail:
    toolgroup: str
    snapshot_time: float
    probability: float
    shap_top: list[FeatureContribution]


class PredictService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._booster: xgb.Booster | None = None

    def predict(
        self,
        kpi_list: list[ToolGroupKPI],
        prev_kpi_list: list[ToolGroupKPI] | None = None,
    ) -> list[PotentialBottleneck]:
        return [
            PotentialBottleneck(
                toolgroup=result.toolgroup,
                snapshot_time=result.snapshot_time,
                probability=result.probability,
            )
            for result in self.predict_all(kpi_list, prev_kpi_list)
            if result.probability > self._settings.bottleneck_threshold
        ]

    def predict_all(
        self,
        kpi_list: list[ToolGroupKPI],
        prev_kpi_list: list[ToolGroupKPI] | None = None,
    ) -> list[PredictionDetail]:
        if not kpi_list:
            return []

        features = build_feature_matrix(kpi_list, prev_kpi_list=prev_kpi_list)
        matrix = xgb.DMatrix(features)
        booster = self._load_model()
        probabilities = booster.predict(matrix)
        contributions = booster.predict(matrix, pred_contribs=True)

        results: list[PredictionDetail] = []
        for index, (kpi, probability) in enumerate(zip(kpi_list, probabilities)):
            ranked = sorted(
                zip(features.columns, contributions[index][:-1]),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:3]
            results.append(
                PredictionDetail(
                    toolgroup=kpi.toolgroup,
                    snapshot_time=kpi.snapshot_time,
                    probability=round(float(probability), 4),
                    shap_top=[
                        FeatureContribution(feature=name, importance=round(float(value), 4))
                        for name, value in ranked
                    ],
                )
            )
        return results

    def _load_model(self) -> xgb.Booster:
        if self._booster is None:
            model_path = Path(self._settings.model_dir) / "bottleneck_xgb.ubj"
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[2] / model_path
            self._booster = xgb.Booster()
            self._booster.load_model(model_path)
        return self._booster
