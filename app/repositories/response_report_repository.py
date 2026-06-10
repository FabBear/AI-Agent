import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import asyncpg


class ResponseReportRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(self, case_id: UUID, report_result: dict) -> None:
        report_html = self._report_text(report_result)
        meta = report_result.get("meta") or {}
        summary = str(meta.get("summary") or report_html[:500])
        await self._pool.execute(
            """
            INSERT INTO td_response_report (
                case_id,
                report_html,
                summary,
                timeline_json,
                root_cause_text,
                action_comparison_text,
                pdf_path,
                generated_at
            )
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8)
            ON CONFLICT (case_id) DO UPDATE SET
                report_html = EXCLUDED.report_html,
                summary = EXCLUDED.summary,
                timeline_json = EXCLUDED.timeline_json,
                root_cause_text = EXCLUDED.root_cause_text,
                action_comparison_text = EXCLUDED.action_comparison_text,
                pdf_path = EXCLUDED.pdf_path,
                generated_at = EXCLUDED.generated_at,
                regenerated_count = td_response_report.regenerated_count + 1,
                qdrant_indexed = FALSE,
                qdrant_indexed_at = NULL,
                qdrant_prev_doc_id = td_response_report.qdrant_doc_id,
                qdrant_doc_id = NULL,
                updated_at = NOW()
            """,
            case_id,
            report_html,
            summary,
            json.dumps(report_result, ensure_ascii=False),
            report_result.get("root_cause_text"),
            report_result.get("action_comparison_text"),
            report_result.get("pdf_path"),
            datetime.now(UTC),
        )

    async def mark_qdrant_indexed(self, case_id: UUID, qdrant_doc_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE td_response_report
            SET qdrant_indexed = TRUE,
                qdrant_indexed_at = NOW(),
                qdrant_doc_id = $2,
                updated_at = NOW()
            WHERE case_id = $1
            """,
            case_id,
            qdrant_doc_id,
        )

    @staticmethod
    def _report_text(report_result: dict) -> str:
        direct = report_result.get("final_report") or report_result.get("full_markdown")
        if direct:
            return str(direct)
        output_path = report_result.get("output_path")
        if output_path and Path(output_path).is_file():
            return Path(output_path).read_text(encoding="utf-8")
        return json.dumps(report_result, ensure_ascii=False)
