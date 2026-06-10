from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import asyncpg

from agents.schemas.kpi import ToolGroupKPI


@dataclass(frozen=True)
class TgMetricSnapshot:
    tg_id: UUID
    measured_at: datetime
    kpi: ToolGroupKPI


class TgMetricsRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def find_latest_by_fab(
        self,
        fab_id: UUID,
        snapshot_time: datetime | None,
    ) -> list[ToolGroupKPI]:
        records = await self.find_latest_records_by_fab(fab_id, snapshot_time)
        return [record.kpi for record in records]

    async def find_latest_records_by_fab(
        self,
        fab_id: UUID,
        snapshot_time: datetime | None,
    ) -> list[TgMetricSnapshot]:
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT ON (m.tg_id)
                m.tg_id,
                tg.tg_code,
                m.measured_at,
                m.utilization_rate,
                m.wip_count,
                m.available_tool_ratio,
                m.avg_qtime_min,
                m.setup_ratio,
                m.wait_ratio
            FROM ps_tg_metrics m
            JOIN tm_tool_group tg ON tg.tg_id = m.tg_id
            JOIN tm_area a ON a.area_id = tg.area_id
            WHERE a.fab_id = $1
              AND tg.is_active = TRUE
              AND ($2::timestamptz IS NULL OR m.measured_at <= $2)
            ORDER BY m.tg_id, m.measured_at DESC
            """,
            fab_id,
            snapshot_time,
        )
        return [self._to_snapshot(row) for row in rows]

    @staticmethod
    def _to_snapshot(row: asyncpg.Record) -> TgMetricSnapshot:
        measured_at = row["measured_at"]
        utilization = float(row["utilization_rate"] or 0)
        q_time = float(row["avg_qtime_min"] or 0)
        return TgMetricSnapshot(
            tg_id=row["tg_id"],
            measured_at=measured_at,
            kpi=ToolGroupKPI(
                toolgroup=row["tg_code"],
                snapshot_time=measured_at.timestamp() / 60,
                available_tool_ratio=float(row["available_tool_ratio"] or 0),
                q_time_min=q_time,
                wait_ratio=float(row["wait_ratio"] or 0),
                wip=float(row["wip_count"] or 0),
                setup_ratio_avg=float(row["setup_ratio"] or 0),
                utilization_avg=utilization,
                max_avg_q_time=q_time,
                max_util=utilization,
            ),
        )
