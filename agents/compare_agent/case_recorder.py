"""case_recorder 노드 — 보고서 생성 완료 후 Qdrant에 이번 케이스를 인덱싱한다.

report_save 노드가 report_results[].output_path 에 MD 파일 경로를 기록한다.
이 노드는 그 MD를 읽어 bottleneck_cases 컬렉션에 upsert한다.

- 실패해도 파이프라인을 중단하지 않는다 (예외 캐치 후 warning 로그).
- case_id = MD 파일명 stem → 재실행 시 동일 ID로 덮어씀 (idempotent).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.state import PipelineState

logger = logging.getLogger(__name__)


def _parse_meta_from_md(text: str) -> dict:
    """시스템 생성 보고서 MD에서 메타(툴그룹, 심각도, 원인)를 추출한다."""
    meta: dict = {}

    m = re.search(r"\|\s*공정명\s*\|\s*`?([^`|\n]+)`?\s*\|", text)
    if m:
        meta["tg_code"] = m.group(1).strip()

    m = re.search(
        r"\|\s*심각도\s*\|[^|]*?(Critical|High|Medium|Low)",
        text,
        re.IGNORECASE,
    )
    if m:
        meta["risk_grade"] = m.group(1).capitalize()

    m = re.search(r"\|\s*탐지시각\s*\|\s*([0-9\- :]+)\s*\|", text)
    if m:
        meta["detected_at"] = m.group(1).strip()

    for pattern, label in (
        (r"설비_포화|설비 포화", "설비_포화"),
        (r"WIP_누적|WIP 누적", "WIP_누적"),
        (r"대기_누적|대기 누적", "대기_누적"),
        (r"공급_부족|공급 부족", "공급_부족"),
    ):
        if re.search(pattern, text):
            meta["bottleneck_cause_type"] = label
            break

    # cause_summary: 요약 섹션 첫 인용문
    m = re.search(r"> \*\*(.+?)\*\*", text)
    if m:
        meta["cause_summary"] = m.group(1)[:200]

    return meta


def case_recorder(state: "PipelineState") -> dict:
    """보고서 MD를 읽어 Qdrant bottleneck_cases 컬렉션에 인덱싱."""
    report_results = state.get("report_results") or []
    if not report_results:
        return {}

    try:
        from app.services.qdrant_case_service import QdrantCaseService
        service = QdrantCaseService()
    except Exception:
        logger.warning("case_recorder: QdrantCaseService 초기화 실패 — 인덱싱 건너뜀")
        return {}

    for report in report_results:
        md_path_str = report.get("output_path") or report.get("md_path")
        if not md_path_str:
            continue
        md_path = Path(md_path_str)
        if not md_path.exists():
            logger.warning("case_recorder: MD 파일 없음 — %s", md_path)
            continue
        try:
            narrative = md_path.read_text(encoding="utf-8")
            meta = _parse_meta_from_md(narrative)
            service.index_case(
                case_id=md_path.stem,
                tg_code=meta.get("tg_code", ""),
                area_name="",
                detected_at=meta.get("detected_at", ""),
                bottleneck_cause_type=meta.get("bottleneck_cause_type", ""),
                risk_grade=meta.get("risk_grade", ""),
                cause_summary=meta.get("cause_summary", ""),
                report_title=report.get("title") or md_path.stem,
                source_path=str(md_path.resolve()),
                report_url=report.get("report_url") or report.get("url") or "",
                narrative=narrative,
            )
            logger.info("case_recorder: 인덱싱 완료 — %s", md_path.name)
        except Exception:
            logger.exception("case_recorder: 인덱싱 실패 — %s", md_path.name)

    return {}
