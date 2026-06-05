"""SHAP TreeExplainer로 병목 TG의 KPI 기여도를 분석한다."""

from pathlib import Path

import shap
import xgboost as xgb

from agents import config
from agents.bottleneck_detector.feature_engineering import build_feature_matrix
from agents.schemas.cause import SHAPFeature
from agents.schemas.kpi import ToolGroupKPI

_ROOT = Path(__file__).parent.parent.parent
_explainer: shap.TreeExplainer | None = None


def _load_explainer() -> shap.TreeExplainer:
    global _explainer
    if _explainer is None:
        booster = xgb.Booster()
        booster.load_model(str(_ROOT / config.MODEL_DIR / config.MODEL_FILENAME))
        _explainer = shap.TreeExplainer(booster)
    return _explainer


def get_shap_top(kpi: ToolGroupKPI, top_n: int = 5) -> list[SHAPFeature]:
    """해당 TG에 대한 SHAP 상위 피처를 반환한다."""
    explainer = _load_explainer()
    feature_df = build_feature_matrix([kpi])
    shap_vals = explainer.shap_values(feature_df)[0]  # (n_features,)

    pairs = sorted(
        zip(feature_df.columns, shap_vals, feature_df.iloc[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return [
        SHAPFeature(
            feature=feat,
            shap_value=round(float(sv), 4),
            kpi_value=round(float(kv), 4),
        )
        for feat, sv, kv in pairs
        if feat != "toolgroup_enc"
    ][:top_n]
