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

_DELTA_COLS = ["q_time_min", "wait_ratio", "wip", "max_util", "utilization_avg"]


def _load_artifacts() -> tuple:
    global _le_tg, _feature_cols
    if _le_tg is None:
        with open(_ROOT / config.MODEL_DIR / config.LABEL_ENCODER_FILENAME, "rb") as f:
            _le_tg = pickle.load(f)
    if _feature_cols is None:
        with open(_ROOT / config.MODEL_DIR / config.FEATURE_COLS_FILENAME) as f:
            _feature_cols = json.load(f)
    return _le_tg, _feature_cols


def build_feature_matrix(
    kpi_list: list[ToolGroupKPI],
    prev_kpi_list: list[ToolGroupKPI] | None = None,
) -> pd.DataFrame:
    """Convert ToolGroupKPI list → DataFrame with exact training column order.

    prev_kpi_list: t-120분 스냅샷. 제공 시 delta 피처 계산, 없으면 0.0으로 채움.
    """
    le_tg, feature_cols = _load_artifacts()

    rows = [kpi.model_dump() for kpi in kpi_list]
    df = pd.DataFrame(rows)

    # Unseen toolgroups fall back to first known class
    known = set(le_tg.classes_)
    df["toolgroup"] = df["toolgroup"].where(df["toolgroup"].isin(known), other=le_tg.classes_[0])
    df["toolgroup_enc"] = le_tg.transform(df["toolgroup"].astype(str))

    # delta 피처 계산
    delta_feature_names = [f"{c}_delta_120" for c in _DELTA_COLS]
    needs_delta = any(c in feature_cols for c in delta_feature_names)

    if needs_delta:
        if prev_kpi_list:
            prev_map = {kpi.toolgroup: kpi for kpi in prev_kpi_list}
            for col in _DELTA_COLS:
                delta_col = f"{col}_delta_120"
                if delta_col in feature_cols:
                    prev_vals = df["toolgroup"].map(
                        {tg: getattr(kpi, col, 0.0) for tg, kpi in prev_map.items()}
                    ).fillna(0.0)
                    df[delta_col] = df[col] - prev_vals
        else:
            for col in _DELTA_COLS:
                delta_col = f"{col}_delta_120"
                if delta_col in feature_cols:
                    df[delta_col] = 0.0

    return df[feature_cols].fillna(0.0)
