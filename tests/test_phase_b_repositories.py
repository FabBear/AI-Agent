from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.repositories.agent_step_repository import AgentStepRepository
from app.repositories.tg_metrics_repository import TgMetricsRepository


@pytest.mark.asyncio
async def test_tg_metrics_repository_maps_database_row() -> None:
    pool = AsyncMock()
    tg_id = uuid4()
    measured_at = datetime(2026, 6, 10, tzinfo=UTC)
    pool.fetch.return_value = [
        {
            "tg_id": tg_id,
            "tg_code": "DIFFUSION_FE_127",
            "measured_at": measured_at,
            "utilization_rate": 0.8,
            "wip_count": 12,
            "available_tool_ratio": 0.9,
            "avg_qtime_min": 15.5,
            "setup_ratio": 0.1,
            "wait_ratio": 0.7,
        }
    ]

    records = await TgMetricsRepository(pool).find_latest_records_by_fab(uuid4(), None)

    assert records[0].tg_id == tg_id
    assert records[0].kpi.toolgroup == "DIFFUSION_FE_127"
    assert records[0].kpi.max_util == 0.8
    assert records[0].kpi.max_avg_q_time == 15.5


@pytest.mark.asyncio
async def test_agent_step_repository_uses_ddl_step_name() -> None:
    pool = AsyncMock()

    await AgentStepRepository(pool).mark_done(uuid4(), "cause", "원인 분석 완료")

    args = pool.execute.await_args.args
    assert args[2] == "CAUSE_ANALYSIS"
    assert args[3] == "원인 분석 완료"
