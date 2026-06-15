"""Snapshot 3780 demo environment bootstrap."""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from app.services.qdrant_case_service import QdrantCaseService

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAG_CASES_DIR = Path(__file__).resolve().parent / "rag_cases"
DEMO_COLLECTION = "bottleneck_cases_demo_3780"


def configure_demo_environment(snapshot_time: float) -> None:
    """Enable mock verification and select the matching RAG collection."""
    os.environ["DEMO_MOCK_VERIFICATION"] = "1"
    os.environ["G_STAR_N_RUNS"] = "10"
    if int(snapshot_time) in {3780, 3800}:
        os.environ["QDRANT_COLLECTION_CASES"] = DEMO_COLLECTION


def ensure_demo_rag_cases(force: bool = False, clean: bool = False) -> int:
    """Seed the static demo reports when the demo collection is empty."""
    load_dotenv(ROOT / ".env")
    os.environ["QDRANT_COLLECTION_CASES"] = DEMO_COLLECTION

    service = QdrantCaseService()
    if clean:
        service.delete_collection()
        service = QdrantCaseService()  # 컬렉션 재생성

    existing_count = service.count_cases()
    if not force and not clean and existing_count:
        logger.info(
            "RAG demo collection already prepared: %s (%d cases)",
            DEMO_COLLECTION,
            existing_count,
        )
        return existing_count

    for path in sorted(RAG_CASES_DIR.glob("*.md")):
        _index_demo_case(path, service)

    seeded_count = service.count_cases()
    if seeded_count is None:
        logger.warning(
            "RAG demo cases could not be verified. Check Qdrant and OPENAI_API_KEY."
        )
        return 0
    logger.info(
        "RAG demo collection prepared: %s (%d cases)",
        DEMO_COLLECTION,
        seeded_count,
    )
    return seeded_count


def _index_demo_case(path: Path, service: QdrantCaseService) -> None:
    markdown = path.read_text(encoding="utf-8")
    frontmatter = _parse_frontmatter(markdown)
    service.index_case(
        case_id=path.stem,
        tg_code=frontmatter.get("toolgroup", ""),
        bottleneck_cause_type=frontmatter.get("cause_category", ""),
        risk_grade=frontmatter.get("risk_grade", ""),
        cause_summary=_extract_summary(markdown),
        report_title=frontmatter.get("title", path.stem),
        source_path=str(path.resolve()),
        report_url=frontmatter.get("report_url", ""),
        narrative=markdown,
    )


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---", markdown, re.DOTALL)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def _extract_summary(markdown: str) -> str:
    quotes = re.findall(r"^>\s*\*\*(.+?)\*\*$", markdown, re.MULTILINE)
    return " ".join(quotes[:3])[:500]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare snapshot 3780 demo RAG cases")
    parser.add_argument("--force", action="store_true", help="Upsert all demo cases again")
    parser.add_argument("--clean", action="store_true", help="Delete and recreate the collection before seeding")
    args = parser.parse_args()
    ensure_demo_rag_cases(force=args.force or args.clean, clean=args.clean)


if __name__ == "__main__":
    main()
