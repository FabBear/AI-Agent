"""Build the feature matrix matching Simulation/ML/data_labeling.ipynb §8 feature set."""

import json
from pathlib import Path

import pandas as pd

from agents import config
from agents.schemas.kpi import ToolGroupKPI

_ROOT = Path(__file__).parent.parent.parent  # AI-Agent/
_feature_cols: list[str] | None = None

_DELTA_COLS = ["q_time_min", "wait_ratio", "wip", "max_util", "utilization_avg"]


def _load_feature_cols() -> list[str]:
    global _feature_cols
    if _feature_cols is None:
        with open(_ROOT / config.MODEL_DIR / config.FEATURE_COLS_FILENAME) as f:
            _feature_cols = json.load(f)
    return _feature_cols


def build_feature_matrix(
    kpi_list: list[ToolGroupKPI],
    prev_kpi_list: list[ToolGroupKPI] | None = None,
) -> pd.DataFrame:
    """Convert ToolGroupKPI list → DataFrame with exact training column order.

    prev_kpi_list: t-120분 스냅샷. 제공 시 delta 피처 계산, 없으면 0.0으로 채움.
    """
    feature_cols = _load_feature_cols()

    rows = [kpi.model_dump() for kpi in kpi_list]
    df = pd.DataFrame(rows)

    # delta 피처 계산
    delta_feature_names = [f"{c}_delta_120" for c in _DELTA_COLS]
    needs_delta = any(c in feature_cols for c in delta_feature_names)

    if needs_delta:
        if prev_kpi_list:
            prev_df = pd.DataFrame([kpi.model_dump() for kpi in prev_kpi_list]).set_index("toolgroup")
            for col in _DELTA_COLS:
                delta_col = f"{col}_delta_120"
                if delta_col in feature_cols and col in prev_df.columns:
                    prev_vals = df["toolgroup"].map(prev_df[col]).fillna(0.0)
                    df[delta_col] = df[col] - prev_vals
        else:
            for col in _DELTA_COLS:
                delta_col = f"{col}_delta_120"
                if delta_col in feature_cols:
                    df[delta_col] = 0.0

    return df[feature_cols].fillna(0.0)
