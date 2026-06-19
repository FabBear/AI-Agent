from uuid import UUID

import asyncpg

STEP_NAME_MAP = {
    "cascade": "DIFFUSION_ANALYSIS",
    "cause": "CAUSE_ANALYSIS",
    "solution": "ACTION_PLAN_GEN",
    "compare": "ACTION_PLAN_COMPARE",
    "hitl": "HITL_WAITING",
    "report": "REPORT_GEN",
    "cascade_close": "CASCADE_CLOSED",
}

STEP_ORDER_MAP = {
    "DIFFUSION_ANALYSIS": 1,
    "CAUSE_ANALYSIS": 2,
    "ACTION_PLAN_GEN": 3,
    "ACTION_PLAN_COMPARE": 4,
    "HITL_WAITING": 5,
    "REPORT_GEN": 6,
}


class AgentStepRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def init_steps(self, case_id: UUID) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                for step_name, step_order in STEP_ORDER_MAP.items():
                    await connection.execute(
                        """
                        INSERT INTO th_agent_step_log (
                            case_id, step_name, step_order, attempt_no, status
                        )
                        VALUES ($1, $2, $3, 1, 'PENDING')
                        ON CONFLICT (case_id, step_order, attempt_no) DO NOTHING
                        """,
                        case_id,
                        step_name,
                        step_order,
                    )

    async def mark_in_progress(self, case_id: UUID, node_name: str) -> None:
        step_name = self._step_name(node_name)
        await self._pool.execute(
            """
            UPDATE th_agent_step_log
            SET status = 'IN_PROGRESS',
                started_at = COALESCE(started_at, NOW()),
                completed_at = NULL,
                error_msg = NULL,
                updated_at = NOW()
            WHERE case_id = $1 AND step_name = $2 AND attempt_no = 1
            """,
            case_id,
            step_name,
        )

    async def mark_done(
        self,
        case_id: UUID,
        node_name: str,
        output_summary: str | None = None,
        llm_model_name: str | None = None,
        token_count: int | None = None,
        model_version_id: UUID | None = None,
    ) -> None:
        step_name = self._step_name(node_name)
        await self._pool.execute(
            """
            UPDATE th_agent_step_log
            SET status = 'DONE',
                started_at = COALESCE(started_at, NOW()),
                completed_at = NOW(),
                output_summary = $3,
                llm_model_name = COALESCE($4, llm_model_name),
                token_count = COALESCE($5, token_count),
                model_version_id = COALESCE($6, model_version_id),
                error_msg = NULL,
                updated_at = NOW()
            WHERE case_id = $1 AND step_name = $2 AND attempt_no = 1
            """,
            case_id,
            step_name,
            output_summary,
            llm_model_name,
            token_count,
            model_version_id,
        )

    async def mark_failed(self, case_id: UUID, node_name: str, error_msg: str) -> None:
        step_name = self._step_name(node_name)
        await self._pool.execute(
            """
            UPDATE th_agent_step_log
            SET status = 'FAILED',
                started_at = COALESCE(started_at, NOW()),
                completed_at = NOW(),
                error_msg = $3,
                updated_at = NOW()
            WHERE case_id = $1 AND step_name = $2 AND attempt_no = 1
            """,
            case_id,
            step_name,
            error_msg[:2000],
        )

    async def find_by_case(self, case_id: UUID) -> list[dict]:
        rows = await self._pool.fetch(
            """
            SELECT
                step_order,
                step_name,
                status,
                started_at,
                completed_at,
                output_summary,
                attempt_no
            FROM th_agent_step_log
            WHERE case_id = $1
            ORDER BY step_order ASC, attempt_no ASC, created_at ASC
            """,
            case_id,
        )
        return [dict(row) for row in rows]

    async def mark_active_failed(self, case_id: UUID, error_msg: str) -> str | None:
        row = await self._pool.fetchrow(
            """
            UPDATE th_agent_step_log
            SET status = 'FAILED',
                completed_at = NOW(),
                error_msg = $2,
                updated_at = NOW()
            WHERE step_log_id = (
                SELECT step_log_id
                FROM th_agent_step_log
                WHERE case_id = $1 AND status = 'IN_PROGRESS'
                ORDER BY step_order DESC
                LIMIT 1
            )
            RETURNING step_name
            """,
            case_id,
            error_msg[:2000],
        )
        return row["step_name"] if row else None

    @staticmethod
    def _step_name(node_name: str) -> str:
        if node_name in STEP_ORDER_MAP:
            return node_name
        try:
            return STEP_NAME_MAP[node_name]
        except KeyError as exc:
            raise ValueError(f"지원하지 않는 Agent step: {node_name}") from exc
