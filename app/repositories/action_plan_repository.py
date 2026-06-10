import json
from uuid import UUID

import asyncpg


class ActionPlanRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def bulk_insert(self, case_id: UUID, candidates: list[dict]) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                for sequence, candidate in enumerate(candidates[:5], start=1):
                    await connection.execute(
                        """
                        INSERT INTO td_action_plan (
                            case_id,
                            plan_seq,
                            plan_type,
                            plan_title,
                            plan_detail,
                            simulation_basis
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (case_id, plan_seq) DO UPDATE SET
                            plan_type = EXCLUDED.plan_type,
                            plan_title = EXCLUDED.plan_title,
                            plan_detail = EXCLUDED.plan_detail,
                            simulation_basis = EXCLUDED.simulation_basis
                        """,
                        case_id,
                        sequence,
                        self._plan_type(candidate),
                        self._plan_title(candidate, sequence),
                        self._plan_detail(candidate),
                        json.dumps(candidate, ensure_ascii=False),
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
        if "release_interval_minutes" in candidate:
            return "LOT_RELEASE_INTERVAL"
        params = candidate.get("params") or {}
        if params.get("dispatch_rule"):
            return "DISPATCH_RULE_OVERRIDE"
        if params.get("lot_priority_rule"):
            return "LOT_PRIORITY_RULE"
        return candidate.get("plan_type") or "ACTION_PLAN"

    @staticmethod
    def _plan_title(candidate: dict, sequence: int) -> str:
        return str(
            candidate.get("name")
            or candidate.get("description")
            or candidate.get("plan_id")
            or f"대응안 {sequence}"
        )[:200]

    @staticmethod
    def _plan_detail(candidate: dict) -> str:
        return str(
            candidate.get("expected_effect")
            or candidate.get("description")
            or candidate
        )
