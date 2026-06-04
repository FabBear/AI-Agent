"""Read KPI snapshots from Simulation sim_csv_out CSVs."""

from pathlib import Path

import pandas as pd

from agents.schemas.kpi import ToolGroupKPI

# kpi_toolgroup.csv wide pivot 캐시 (트렌드 분석에서 재사용)
_tg_wide_cache: dict[Path, pd.DataFrame] = {}

_TG_INSTANT_KPIS = ("q_time_min", "wait_ratio", "wip", "available_tool_ratio")
_TG_WINDOW_KPIS = ("utilization_avg", "setup_ratio_avg")
_TOOL_KPIS = {"utilization": "max_util", "avg_q_time": "max_avg_q_time"}
_TOOL_CHUNK = 2_000_000


def _tool_id_to_toolgroup(tool_id: str) -> str:
    return tool_id.rsplit("#", 1)[0] if "#" in tool_id else tool_id


def load_kpi_snapshot(
    csv_dir: str | Path, snapshot_time: float | None = None
) -> list[ToolGroupKPI]:
    """
    Load one snapshot from sim_csv_out CSVs and return ToolGroupKPI list.

    snapshot_time: specific sim minute to load; if None, uses the latest snapshot.
    """
    csv_dir = Path(csv_dir)
    tg_path = csv_dir / "kpi_toolgroup.csv"
    tool_path = csv_dir / "kpi_tool.csv"

    tg_long = pd.read_csv(
        tg_path,
        usecols=["snapshot_time", "scope", "kpi_name", "value", "window_minutes"],
    ).rename(columns={"scope": "toolgroup"})
    tg_long["snapshot_time"] = tg_long["snapshot_time"].astype(float)

    if snapshot_time is None:
        snapshot_time = float(tg_long["snapshot_time"].max())

    tg_snap = tg_long[tg_long["snapshot_time"] == snapshot_time]

    instant = tg_snap[tg_snap["kpi_name"].isin(_TG_INSTANT_KPIS) & tg_snap["window_minutes"].isna()]
    tg_wide = instant.pivot_table(
        index=["snapshot_time", "toolgroup"],
        columns="kpi_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    window = tg_snap[tg_snap["kpi_name"].isin(_TG_WINDOW_KPIS)]
    tg_wide_util = window.pivot_table(
        index=["snapshot_time", "toolgroup"],
        columns="kpi_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide = tg_wide.merge(tg_wide_util, on=["snapshot_time", "toolgroup"], how="outer")

    # Aggregate max tool-level KPIs for the snapshot
    if tool_path.exists():
        parts = []
        reader = pd.read_csv(
            tool_path,
            chunksize=_TOOL_CHUNK,
            usecols=["snapshot_time", "scope", "kpi_name", "value"],
        )
        for chunk in reader:
            chunk = chunk[
                (chunk["snapshot_time"].astype(float) == snapshot_time)
                & chunk["kpi_name"].isin(_TOOL_KPIS)
            ]
            if chunk.empty:
                continue
            chunk["toolgroup"] = chunk["scope"].map(_tool_id_to_toolgroup)
            chunk["snapshot_time"] = chunk["snapshot_time"].astype(float)
            parts.append(
                chunk.groupby(["snapshot_time", "toolgroup", "kpi_name"], as_index=False)[
                    "value"
                ].max()
            )

        if parts:
            tool_agg = (
                pd.concat(parts)
                .pivot(index=["snapshot_time", "toolgroup"], columns="kpi_name", values="value")
                .reset_index()
                .rename(columns=_TOOL_KPIS)
            )
            wide = wide.merge(tool_agg, on=["snapshot_time", "toolgroup"], how="left")

    for col in ("max_util", "max_avg_q_time"):
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide.fillna(0.0)

    return [
        ToolGroupKPI(
            toolgroup=row["toolgroup"],
            snapshot_time=row["snapshot_time"],
            available_tool_ratio=row.get("available_tool_ratio", 0.0),
            q_time_min=row.get("q_time_min", 0.0),
            wait_ratio=row.get("wait_ratio", 0.0),
            wip=row.get("wip", 0.0),
            setup_ratio_avg=row.get("setup_ratio_avg", 0.0),
            utilization_avg=row.get("utilization_avg", 0.0),
            max_avg_q_time=row.get("max_avg_q_time", 0.0),
            max_util=row.get("max_util", 0.0),
        )
        for _, row in wide.iterrows()
    ]


def _load_tg_wide(csv_dir: Path) -> pd.DataFrame:
    """kpi_toolgroup.csv를 전체 wide 형태로 로드 (캐시)."""
    if csv_dir not in _tg_wide_cache:
        tg_path = csv_dir / "kpi_toolgroup.csv"
        tg_long = pd.read_csv(
            tg_path,
            usecols=["snapshot_time", "scope", "kpi_name", "value", "window_minutes"],
        ).rename(columns={"scope": "toolgroup"})
        tg_long["snapshot_time"] = tg_long["snapshot_time"].astype(float)

        instant = tg_long[
            tg_long["kpi_name"].isin(_TG_INSTANT_KPIS) & tg_long["window_minutes"].isna()
        ]
        wide = instant.pivot_table(
            index=["snapshot_time", "toolgroup"],
            columns="kpi_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        window = tg_long[tg_long["kpi_name"].isin(_TG_WINDOW_KPIS)]
        util_wide = window.pivot_table(
            index=["snapshot_time", "toolgroup"],
            columns="kpi_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        wide = wide.merge(util_wide, on=["snapshot_time", "toolgroup"], how="outer").fillna(0.0)
        _tg_wide_cache[csv_dir] = wide

    return _tg_wide_cache[csv_dir]


def load_kpi_window(
    csv_dir: str | Path,
    snapshot_time: float,
    n_snapshots: int = 6,
    toolgroups: list[str] | None = None,
) -> dict[float, list[ToolGroupKPI]]:
    """
    snapshot_time 포함 직전 n_snapshots개 스냅샷의 KPI를 반환한다.
    Returns: {snapshot_time: [ToolGroupKPI, ...]} (시간 오름차순)
    """
    csv_dir = Path(csv_dir)
    wide = _load_tg_wide(csv_dir)

    all_times = sorted(wide["snapshot_time"].unique())
    idx = next((i for i, t in enumerate(all_times) if t >= snapshot_time), len(all_times) - 1)
    window_times = all_times[max(0, idx - n_snapshots + 1) : idx + 1]

    result: dict[float, list[ToolGroupKPI]] = {}
    for t in window_times:
        rows = wide[wide["snapshot_time"] == t]
        if toolgroups:
            rows = rows[rows["toolgroup"].isin(toolgroups)]
        result[t] = [
            ToolGroupKPI(
                toolgroup=row["toolgroup"],
                snapshot_time=row["snapshot_time"],
                available_tool_ratio=row.get("available_tool_ratio", 0.0),
                q_time_min=row.get("q_time_min", 0.0),
                wait_ratio=row.get("wait_ratio", 0.0),
                wip=row.get("wip", 0.0),
                setup_ratio_avg=row.get("setup_ratio_avg", 0.0),
                utilization_avg=row.get("utilization_avg", 0.0),
                max_avg_q_time=row.get("max_avg_q_time", 0.0),
                max_util=row.get("max_util", 0.0),
            )
            for _, row in rows.iterrows()
        ]
    return result
