import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.tracking import MlflowClient

from agents.bottleneck_detector.feature_engineering import build_feature_matrix
from agents.schemas.alert import PotentialBottleneck
from agents.schemas.kpi import ToolGroupKPI
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


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
        self._model_lock = Lock()
        self._last_refresh = 0.0           # 마지막 refresh 시각(monotonic)
        self._loaded_version: str | None = None  # 현재 로드된 MLflow Production 버전(local이면 None)

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
        """MLflow Production 모델을 우선 사용하되, 주기적으로 새 버전 승격을 감지해 교체한다.

        - refresh_interval 이내면 캐시 그대로 사용(매 예측마다 Registry 조회 안 함).
        - MLflow 서버 미응답/조회 실패 시 현재 캐시 모델 유지(graceful degradation).
        - 최초 로드 시 MLflow 실패하면 로컬 .ubj 폴백.
        """
        now = time.monotonic()
        with self._model_lock:
            cached = self._booster is not None
            fresh = (now - self._last_refresh) < self._settings.model_refresh_interval_sec
            if cached and fresh:
                return self._booster

            self._last_refresh = now
            target_version = self._current_production_version()

            # 서버 미응답(target None)이거나 버전 동일 → 현행 모델 유지
            if cached and (target_version is None or target_version == self._loaded_version):
                return self._booster

            booster, version = self._resolve_booster(target_version)
            self._booster = booster
            self._loaded_version = version
            return self._booster

    def _current_production_version(self) -> str | None:
        """Registry에서 현재 Production 스테이지 버전 번호를 조회(실패 시 None)."""
        try:
            client = MlflowClient(tracking_uri=self._settings.mlflow_tracking_uri)
            versions = client.get_latest_versions(
                self._settings.mlflow_model_name,
                stages=[self._settings.mlflow_model_stage],
            )
            return versions[0].version if versions else None
        except Exception as exc:  # 서버 다운/모델 없음 등
            logger.warning("MLflow Production 버전 조회 실패(캐시/로컬 유지): %s", exc)
            return None

    def _resolve_booster(self, version: str | None) -> tuple[xgb.Booster, str | None]:
        """MLflow Production 모델 로드 시도 → 실패/부재 시 로컬 .ubj 폴백. (booster, version) 반환."""
        if version is not None:
            try:
                mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
                loaded = mlflow.xgboost.load_model(
                    f"models:/{self._settings.mlflow_model_name}/{self._settings.mlflow_model_stage}"
                )
                # XGBClassifier(sklearn)면 내부 Booster 추출 → predict(DMatrix, pred_contribs) 인터페이스 통일
                booster = loaded.get_booster() if hasattr(loaded, "get_booster") else loaded
                logger.info(
                    "MLflow Production 모델 로드: %s v%s",
                    self._settings.mlflow_model_name,
                    version,
                )
                return booster, version
            except Exception as exc:
                logger.warning("MLflow 모델 로드 실패, 로컬 폴백: %s", exc)

        model_path = Path(self._settings.model_dir) / "bottleneck_xgb.ubj"
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        logger.info("로컬 모델 로드: %s", model_path)
        return booster, None
