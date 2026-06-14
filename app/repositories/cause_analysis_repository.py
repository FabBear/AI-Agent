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
        affected_tgs: list[str] | None = None,
        model_accuracy: float | None = None,
        model_f1: float | None = None,
    ) -> None:
        affected_ids = await self._find_tg_ids(affected_tgs or [])
        shap_features = [
            {
                "feature": feature.feature,
                "importance": feature.shap_value,
                "rank": index,
            }
            for index, feature in enumerate(report.shap_top, start=1)
        ]
        cause_type = report.judgment.primary_category if report.judgment else None
        primary_cause_feature = report.judgment.primary_cause if report.judgment else None
        cause_judgment_json = json.dumps(report.judgment.model_dump()) if report.judgment else None
        consensus_json = json.dumps(report.consensus.model_dump()) if report.consensus else None
        trend_json = json.dumps([t.model_dump() for t in report.trend_top]) if report.trend_top else None
        await self._pool.execute(
            """
            INSERT INTO td_cause_analysis (
                case_id,
                shap_features,
                bottleneck_cause_type,
                primary_cause_feature,
                cause_judgment_json,
                consensus_json,
                trend_json,
                diffusion_affected_tg_ids,
                rag_referenced_case_ids,
                model_accuracy,
                model_f1,
                feature_count
            )
            VALUES ($1, $2::jsonb, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb, $10, $11, $12)
            ON CONFLICT (case_id) DO UPDATE SET
                shap_features = EXCLUDED.shap_features,
                bottleneck_cause_type = EXCLUDED.bottleneck_cause_type,
                primary_cause_feature = EXCLUDED.primary_cause_feature,
                cause_judgment_json = EXCLUDED.cause_judgment_json,
                consensus_json = EXCLUDED.consensus_json,
                trend_json = EXCLUDED.trend_json,
                diffusion_affected_tg_ids = EXCLUDED.diffusion_affected_tg_ids,
                rag_referenced_case_ids = EXCLUDED.rag_referenced_case_ids,
                model_accuracy = EXCLUDED.model_accuracy,
                model_f1 = EXCLUDED.model_f1,
                feature_count = EXCLUDED.feature_count
            """,
            case_id,
            json.dumps(shap_features),
            cause_type,
            primary_cause_feature,
            cause_judgment_json,
            consensus_json,
            trend_json,
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
