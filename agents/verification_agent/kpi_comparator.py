"""baseline vs whatif KPI 30쌍 paired 비교 → D_i 계산.

whatif_effect.py 동일 방식:
  D_i = whatif_i − baseline_i  (각 seed별 paired 차이)
  결과를 kpi_name별 list[float]로 반환 → paired t-test 입력으로 사용.

kpi_toolgroup.csv 컬럼: run_id, snapshot_time, scope, kpi_name, value, ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PAIRING_KPIS = (
    "q_time_min",
    "wip",
    "wait_ratio",
    "utilization_avg",
    "available_tool_ratio",
)


def _read_tg_kpis(csv_dir: Path, toolgroup: str) -> dict[str, float]:
    """kpi_toolgroup.csv 마지막 스냅샷에서 toolgroup KPI 추출."""
    path = csv_dir / "kpi_toolgroup.csv"
    if not path.is_file():
        return {}

    df = pd.read_csv(path, usecols=["snapshot_time", "scope", "kpi_name", "value"])
    df["snapshot_time"] = df["snapshot_time"].astype(float)

    tg_df = df[df["scope"] == toolgroup]
    if tg_df.empty:
        return {}

    last_t = tg_df["snapshot_time"].max()
    snap = tg_df[tg_df["snapshot_time"] == last_t]

    result: dict[str, float] = {}
    for _, row in snap.iterrows():
        if row["kpi_name"] in _PAIRING_KPIS:
            try:
                result[row["kpi_name"]] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return result


def compute_paired_deltas(
    pairs: list[dict],
    toolgroup: str,
) -> dict[str, list[float]]:
    """
    D_i = whatif_i − baseline_i, 30쌍 계산.

    Args:
        pairs:     [{baseline_csv_dir, whatif_csv_dir, ...}, ...]  (N=30)
        toolgroup: 비교 대상 toolgroup scope 이름

    Returns:
        {kpi_name: [D_1, D_2, ..., D_N]}
        두 런 모두 해당 KPI 값이 있는 쌍만 포함.
    """
    per_kpi: dict[str, list[float]] = {k: [] for k in _PAIRING_KPIS}

    for pair in pairs:
        b = _read_tg_kpis(Path(pair["baseline_csv_dir"]), toolgroup)
        w = _read_tg_kpis(Path(pair["whatif_csv_dir"]), toolgroup)

        for kpi in _PAIRING_KPIS:
            bv = b.get(kpi)
            wv = w.get(kpi)
            if bv is not None and wv is not None:
                per_kpi[kpi].append(wv - bv)

    return per_kpi
