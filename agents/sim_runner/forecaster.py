"""
Forward 시뮬 출력 KPI를 읽어서 T0 vs T0+horizon 비교 결과를 반환한다.

신뢰도 분류:
  HIGH  — wip, wait_ratio, available_tool_ratio : 순간 카운트, 스냅샷 품질에 의존
  MED   — utilization_avg : forward sim 내부 계산값 (윈도우 기간 내)
  LOW   — q_time_min : 윈도우 평균이라 forward sim 시작 시 리셋됨
  SKIP  — max_util, max_avg_q_time : kpi_tool.csv 미로드로 항상 0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.schemas.kpi import ToolGroupKPI

_TG_INSTANT = ("q_time_min", "wait_ratio", "wip", "available_tool_ratio")
_TG_WINDOW = ("utilization_avg", "setup_ratio_avg")

# 신뢰 가능한 KPI만 비교 표시 (LOW/SKIP 제외)
# q_time_min: forward sim 시작 시 윈도우 리셋 → 제외
# max_util, max_avg_q_time: kpi_tool.csv 미로드 → 제외
_COMPARE_FIELDS: list[tuple[str, str]] = [
    ("wip", "HIGH"),
    ("wait_ratio", "HIGH"),
    ("available_tool_ratio", "HIGH"),
    ("utilization_avg", "MED"),
]


def load_forward_kpis(csv_dir: Path) -> dict[str, ToolGroupKPI]:
    """Forward 시뮬 출력 CSV에서 마지막 스냅샷의 TG KPI를 반환. {toolgroup: ToolGroupKPI}"""
    tg_path = csv_dir / "kpi_toolgroup.csv"
    if not tg_path.exists():
        return {}

    tg_long = pd.read_csv(
        tg_path,
        usecols=["snapshot_time", "scope", "kpi_name", "value", "window_minutes"],
    ).rename(columns={"scope": "toolgroup"})
    tg_long["snapshot_time"] = tg_long["snapshot_time"].astype(float)

    last_t = tg_long["snapshot_time"].max()
    snap = tg_long[tg_long["snapshot_time"] == last_t]

    instant = snap[snap["kpi_name"].isin(_TG_INSTANT) & snap["window_minutes"].isna()]
    wide = instant.pivot_table(
        index=["snapshot_time", "toolgroup"], columns="kpi_name", values="value", aggfunc="first"
    ).reset_index()

    window = snap[snap["kpi_name"].isin(_TG_WINDOW)]
    if not window.empty:
        util_wide = window.pivot_table(
            index=["snapshot_time", "toolgroup"],
            columns="kpi_name",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide = wide.merge(util_wide, on=["snapshot_time", "toolgroup"], how="outer")

    wide = wide.fillna(0.0)
    result: dict[str, ToolGroupKPI] = {}
    for _, row in wide.iterrows():
        result[row["toolgroup"]] = ToolGroupKPI(
            toolgroup=row["toolgroup"],
            snapshot_time=row["snapshot_time"],
            available_tool_ratio=row.get("available_tool_ratio", 0.0),
            q_time_min=row.get("q_time_min", 0.0),
            wait_ratio=row.get("wait_ratio", 0.0),
            wip=row.get("wip", 0.0),
            setup_ratio_avg=row.get("setup_ratio_avg", 0.0),
            utilization_avg=row.get("utilization_avg", 0.0),
            max_avg_q_time=0.0,
            max_util=0.0,
        )
    return result


def compare_kpis(
    now: ToolGroupKPI,
    future: ToolGroupKPI,
) -> dict[str, dict]:
    """
    신뢰 가능한 KPI만 비교. {kpi_name: {now, future, delta, pct_change, reliability}}
    max_util / max_avg_q_time 는 forward sim에서 항상 0이므로 제외.
    """
    result = {}
    for field, reliability in _COMPARE_FIELDS:
        now_v = getattr(now, field, 0.0)
        fut_v = getattr(future, field, 0.0)
        delta = fut_v - now_v
        pct = (delta / now_v * 100) if now_v != 0 else 0.0
        result[field] = {
            "now": round(now_v, 3),
            "future": round(fut_v, 3),
            "delta": round(delta, 3),
            "pct_change": round(pct, 1),
            "reliability": reliability,
        }
    return result
