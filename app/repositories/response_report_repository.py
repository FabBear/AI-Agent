import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import asyncpg


class ResponseReportRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(self, case_id: UUID, report_result: dict) -> None:
        rendered_markdown = self._rendered_markdown(report_result)
        report_json_obj = self._report_json(report_result)
        root_cause_text = self._root_cause_text(report_json_obj)
        action_comparison_text = self._action_comparison_text(report_json_obj)
        summary = str((report_json_obj or {}).get("summary") or rendered_markdown[:500])
        await self._pool.execute(
            """
            INSERT INTO td_response_report (
                case_id,
                rendered_markdown,
                report_json,
                report_schema_version,
                summary,
                root_cause_text,
                action_comparison_text,
                pdf_path,
                generated_at
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (case_id) DO UPDATE SET
                rendered_markdown = EXCLUDED.rendered_markdown,
                report_json = EXCLUDED.report_json,
                report_schema_version = EXCLUDED.report_schema_version,
                summary = EXCLUDED.summary,
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
            rendered_markdown,
            json.dumps(report_json_obj, ensure_ascii=False) if report_json_obj is not None else None,
            "report/1.0",
            summary,
            root_cause_text,
            action_comparison_text,
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
    def _rendered_markdown(report_result: dict) -> str:
        direct = report_result.get("final_report") or report_result.get("full_markdown")
        if direct:
            return str(direct)
        output_path = report_result.get("output_path")
        if output_path and Path(output_path).is_file():
            return Path(output_path).read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _report_json(report_result: dict) -> dict | None:
        json_output_path = report_result.get("json_output_path")
        if json_output_path and Path(json_output_path).is_file():
            try:
                return json.loads(Path(json_output_path).read_text(encoding="utf-8"))
            except Exception:
                pass
        return report_result.get("report_json")

    @staticmethod
    def _root_cause_text(report_json: dict | None) -> str | None:
        if not report_json:
            return None
        cause = report_json.get("cause") or {}
        return cause.get("summary")

    @staticmethod
    def _action_comparison_text(report_json: dict | None) -> str | None:
        if not report_json:
            return None
        recommendation = (report_json.get("actions") or {}).get("recommendation") or {}
        headline = recommendation.get("headline") or ""
        primary_reason = recommendation.get("primary_reason") or ""
        if not headline and not primary_reason:
            return None
        return f"{headline}: {primary_reason}" if headline and primary_reason else headline or primary_reason
