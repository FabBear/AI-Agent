"""Build the feature matrix matching training in MLOps (feature_cols_kpi.json)."""

import json
import pickle
from pathlib import Path

import pandas as pd

from agents import config
from agents.schemas.kpi import ToolGroupKPI

_ROOT = Path(__file__).parent.parent.parent  # AI-Agent/
_le_tg = None
_feature_cols: list[str] | None = None


def _load_artifacts() -> tuple:
    global _le_tg, _feature_cols
    if _le_tg is None:
        with open(_ROOT / config.MODEL_DIR / config.LABEL_ENCODER_FILENAME, "rb") as f:
            _le_tg = pickle.load(f)
    if _feature_cols is None:
        with open(_ROOT / config.MODEL_DIR / config.FEATURE_COLS_FILENAME) as f:
            _feature_cols = json.load(f)
    return _le_tg, _feature_cols


def build_feature_matrix(kpi_list: list[ToolGroupKPI]) -> pd.DataFrame:
    """Convert ToolGroupKPI list → DataFrame with exact training column order."""
    le_tg, feature_cols = _load_artifacts()

    rows = [kpi.model_dump() for kpi in kpi_list]
    df = pd.DataFrame(rows)

    # Unseen toolgroups fall back to first known class
    known = set(le_tg.classes_)
    df["toolgroup"] = df["toolgroup"].where(df["toolgroup"].isin(known), other=le_tg.classes_[0])
    df["toolgroup_enc"] = le_tg.transform(df["toolgroup"].astype(str))

    return df[feature_cols].fillna(0.0)
