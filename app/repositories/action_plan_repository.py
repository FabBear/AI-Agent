import json
import math
from uuid import UUID

import asyncpg


def _sanitize_nan(obj):
    """NaN/Inf float를 None으로 치환 — PostgreSQL jsonb 직렬화 전처리."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


class ActionPlanRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_compare_json(self, case_id: UUID, result_v2: dict) -> None:
        await self._pool.execute(
            """
            UPDATE tt_bottleneck_case
            SET compare_json = $2::jsonb,
                updated_at = NOW()
            WHERE case_id = $1
            """,
            case_id,
            json.dumps(_sanitize_nan(result_v2), ensure_ascii=False),
        )

    async def bulk_insert(self, case_id: UUID, candidates: list[dict]) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                for sequence, candidate in enumerate(candidates[:5], start=1):
                    kpi = candidate.get("kpi_stats") or {}
                    await connection.execute(
                        """
                        INSERT INTO td_action_plan (
                            case_id,
                            plan_seq,
                            plan_type,
                            plan_title,
                            plan_detail,
                            simulation_basis,
                            est_util_delta,
                            est_q_time_delta,
                            est_wip_delta,
                            est_wait_ratio_delta,
                            sim_paired_n,
                            sim_paired_p_value
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (case_id, plan_seq) DO UPDATE SET
                            plan_type = EXCLUDED.plan_type,
                            plan_title = EXCLUDED.plan_title,
                            plan_detail = EXCLUDED.plan_detail,
                            simulation_basis = EXCLUDED.simulation_basis,
                            est_util_delta = EXCLUDED.est_util_delta,
                            est_q_time_delta = EXCLUDED.est_q_time_delta,
                            est_wip_delta = EXCLUDED.est_wip_delta,
                            est_wait_ratio_delta = EXCLUDED.est_wait_ratio_delta,
                            sim_paired_n = EXCLUDED.sim_paired_n,
                            sim_paired_p_value = EXCLUDED.sim_paired_p_value
                        """,
                        case_id,
                        sequence,
                        self._plan_type(candidate),
                        self._plan_title(candidate, sequence),
                        self._plan_detail(candidate),
                        json.dumps(_sanitize_nan(candidate), ensure_ascii=False),
                        kpi.get("utilization_avg", {}).get("mean_delta"),
                        kpi.get("q_time_min", {}).get("mean_delta"),
                        kpi.get("wip", {}).get("mean_delta"),
                        kpi.get("wait_ratio", {}).get("mean_delta"),
                        kpi.get("q_time_min", {}).get("paired_n"),
                        kpi.get("q_time_min", {}).get("paired_t_p"),
                    )

    async def find_by_id(self, case_id: UUID, plan_id: UUID) -> dict | None:
        row = await self._pool.fetchrow(
            """
            SELECT plan_id, plan_seq, plan_type, plan_title, plan_detail, simulation_basis
            FROM td_action_plan
            WHERE case_id = $1 AND plan_id = $2
            """,
            case_id,
            plan_id,
        )
        return dict(row) if row else None

    @staticmethod
    def _plan_type(candidate: dict) -> str:
        if "release_interval_minutes" in candidate or "release_interval_delta_pct" in candidate:
            return "LOT_RELEASE_INTERVAL"
        params = candidate.get("params") or {}
        if params.get("dispatch_rule"):
            return "DISPATCH_RULE_OVERRIDE"
        if params.get("lot_priority_rule"):
            return "LOT_PRIORITY_RULE"
        return candidate.get("plan_type") or "ACTION_PLAN"

    @staticmethod
    def _plan_title(candidate: dict, sequence: int) -> str:
        _PLAN_ID_TITLE = {
            "conservative": "보수적 조정안",
            "standard": "표준 조정안",
            "aggressive": "강화 조정안",
        }
        plan_id = candidate.get("plan_id", "")
        return str(
            candidate.get("name")
            or candidate.get("description")
            or _PLAN_ID_TITLE.get(plan_id)
            or plan_id
            or f"대응안 {sequence}"
        )[:200]

    @staticmethod
    def _plan_detail(candidate: dict) -> str:
        delta = candidate.get("release_interval_delta_pct")
        if delta is not None:
            return f"Release Interval Δ{delta}%"
        return str(
            candidate.get("expected_effect")
            or candidate.get("description")
            or "대응안"
        )
