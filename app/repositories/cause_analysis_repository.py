import json
from uuid import UUID

import asyncpg

from agents.schemas.cause import CauseReport


class CauseAnalysisRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert(
        self,
        case_id: UUID,
        report: CauseReport,
        model_accuracy: float | None = None,
        model_f1: float | None = None,
    ) -> None:
        affected_ids = await self._find_tg_ids(report.upstream_suspects)
        shap_features = [
            {
                "feature": feature.feature,
                "importance": abs(feature.shap_value),
                "rank": index,
            }
            for index, feature in enumerate(report.shap_top, start=1)
        ]
        cause_type = report.shap_top[0].feature if report.shap_top else None
        await self._pool.execute(
            """
            INSERT INTO td_cause_analysis (
                case_id,
                shap_features,
                bottleneck_cause_type,
                diffusion_affected_tg_ids,
                rag_referenced_case_ids,
                model_accuracy,
                model_f1,
                feature_count
            )
            VALUES ($1, $2::jsonb, $3, $4::jsonb, $5::jsonb, $6, $7, $8)
            ON CONFLICT (case_id) DO UPDATE SET
                shap_features = EXCLUDED.shap_features,
                bottleneck_cause_type = EXCLUDED.bottleneck_cause_type,
                diffusion_affected_tg_ids = EXCLUDED.diffusion_affected_tg_ids,
                rag_referenced_case_ids = EXCLUDED.rag_referenced_case_ids,
                model_accuracy = EXCLUDED.model_accuracy,
                model_f1 = EXCLUDED.model_f1,
                feature_count = EXCLUDED.feature_count
            """,
            case_id,
            json.dumps(shap_features),
            cause_type,
            json.dumps([str(tg_id) for tg_id in affected_ids]),
            json.dumps([]),
            model_accuracy,
            model_f1,
            len(shap_features),
        )

    async def _find_tg_ids(self, tg_codes: list[str]) -> list[UUID]:
        if not tg_codes:
            return []
        rows = await self._pool.fetch(
            "SELECT tg_id FROM tm_tool_group WHERE tg_code = ANY($1::text[])",
            tg_codes,
        )
        return [row["tg_id"] for row in rows]
