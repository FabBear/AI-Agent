"""Read KPI snapshots from Backend DB (fabbear · ps_tg_metrics).

Prerequisites:
  - Backend DB V9 migration 적용 완료 (ps_tg_metrics.max_util 컬럼 존재)
  - BACKEND_DATABASE_URL 환경변수 또는 기본값 사용
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from agents.schemas.kpi import ToolGroupKPI

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_BACKEND_DB_URL = os.getenv(
    "BACKEND_DATABASE_URL",
    "postgresql+psycopg://fabbear_user@localhost:5432/fabbear",
)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_BACKEND_DB_URL, pool_pre_ping=True)
    return _engine


_SNAPSHOT_SQL = text("""
SELECT
    tg.tg_code                                        AS toolgroup,
    m.time_step::float                                AS snapshot_time,
    COALESCE(m.utilization_rate,      0)::float       AS utilization_avg,
    COALESCE(m.wip_count,             0)::float       AS wip,
    COALESCE(m.available_tool_ratio,  0)::float       AS available_tool_ratio,
    COALESCE(m.avg_qtime_min,         0)::float       AS q_time_min,
    COALESCE(m.setup_ratio,           0)::float       AS setup_ratio_avg,
    COALESCE(m.wait_ratio,            0)::float       AS wait_ratio,
    COALESCE(tool_max.max_util,       0)::float       AS max_util
FROM ps_tg_metrics m
JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
LEFT JOIN (
    SELECT t.tg_id, MAX(tm.utilization_rate)::float AS max_util
    FROM ps_tool_metrics tm
    JOIN tm_tool t ON t.tool_id = tm.tool_id
    WHERE tm.measured_at = :measured_at
    GROUP BY t.tg_id
) tool_max ON tool_max.tg_id = m.tg_id
WHERE m.measured_at = :measured_at
ORDER BY tg.tg_code
""")


def _rows_to_kpi_list(rows) -> list[ToolGroupKPI]:
    return [
        ToolGroupKPI(
            toolgroup=r.toolgroup,
            snapshot_time=float(r.snapshot_time),
            utilization_avg=float(r.utilization_avg),
            wip=float(r.wip),
            available_tool_ratio=float(r.available_tool_ratio),
            q_time_min=float(r.q_time_min),
            setup_ratio_avg=float(r.setup_ratio_avg),
            wait_ratio=float(r.wait_ratio),
            max_util=float(r.max_util),
        )
        for r in rows
    ]


def load_kpi_snapshot(snapshot_time: float | None = None) -> list[ToolGroupKPI]:
    """최신 (또는 지정한 epoch-minute에 가장 가까운) 스냅샷의 TG KPI를 반환한다."""
    engine = _get_engine()
    with engine.connect() as conn:
        if snapshot_time is None:
            measured_at = conn.execute(
                text("SELECT MAX(measured_at) FROM ps_tg_metrics")
            ).scalar()
        else:
            # snapshot_time은 시뮬 tick(time_step); 숫자 비교로 가장 가까운 measured_at 선택
            measured_at = conn.execute(text("""
                SELECT measured_at
                FROM ps_tg_metrics
                ORDER BY ABS(time_step - :st)
                LIMIT 1
            """), {"st": int(snapshot_time)}).scalar()

        if measured_at is None:
            return []

        rows = conn.execute(_SNAPSHOT_SQL, {"measured_at": measured_at}).fetchall()

    return _rows_to_kpi_list(rows)


def load_kpi_window(
    snapshot_time: float,
    n_snapshots: int = 6,
    toolgroups: list[str] | None = None,
) -> dict[float, list[ToolGroupKPI]]:
    """snapshot_time 포함 직전 n_snapshots개 스냅샷의 KPI를 반환한다.

    Returns: {snapshot_time(epoch-min): [ToolGroupKPI, ...]} (시간 오름차순)
    """
    engine = _get_engine()
    with engine.connect() as conn:
        times_rows = conn.execute(text("""
            SELECT DISTINCT measured_at
            FROM ps_tg_metrics
            WHERE time_step <= :st
            ORDER BY measured_at DESC
            LIMIT :n
        """), {"st": int(snapshot_time), "n": n_snapshots}).fetchall()

        if not times_rows:
            return {}

        result: dict[float, list[ToolGroupKPI]] = {}
        for tr in reversed(times_rows):
            rows = conn.execute(_SNAPSHOT_SQL, {"measured_at": tr[0]}).fetchall()
            if toolgroups:
                rows = [r for r in rows if r.toolgroup in toolgroups]
            if not rows:
                continue
            epoch_min = float(rows[0].snapshot_time)
            result[epoch_min] = _rows_to_kpi_list(rows)

    return result
