import asyncpg


class MlModelRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def find_active(self) -> dict | None:
        row = await self._pool.fetchrow(
            """
            SELECT
                model_version_id,
                mlflow_run_id,
                mlflow_version,
                model_name,
                trained_at,
                accuracy,
                f1_score,
                auc_roc,
                train_row_count,
                feature_list,
                status,
                registered_at
            FROM th_ml_model_version
            WHERE status = 'ACTIVE'
            ORDER BY registered_at DESC
            LIMIT 1
            """
        )
        return dict(row) if row else None
