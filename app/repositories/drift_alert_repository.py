from uuid import UUID

import json

import asyncpg


class DriftAlertRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(
        self,
        model_version_id: UUID,
        trigger_type: str,
        psi_score: float | None = None,
        f1_at_detection: float | None = None,
        detail: dict | None = None,
    ) -> UUID:
        row = await self._pool.fetchrow(
            """
            INSERT INTO th_drift_alert (
                model_version_id,
                detected_at,
                trigger_type,
                psi_score,
                f1_at_detection,
                detail
            )
            VALUES ($1, NOW(), $2, $3, $4, $5::jsonb)
            RETURNING drift_id
            """,
            model_version_id,
            trigger_type,
            psi_score,
            f1_at_detection,
            json.dumps(detail) if detail is not None else None,
        )
        return row["drift_id"]
