"""병목 케이스 RAG — Qdrant 검색 및 인덱싱 서비스.

app/chatbot/rag.py 의 httpx + OpenAI 임베딩 패턴을 따름.
컬렉션: bottleneck_cases (QDRANT_COLLECTION_CASES 환경변수로 override 가능)
임베딩 모델: text-embedding-3-small (1536차원)
"""

from __future__ import annotations

import logging
import os
import uuid

logger = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_VECTOR_SIZE = 1536


class QdrantCaseService:
    """병목 사례 벡터 DB — 검색·인덱싱 담당."""

    def __init__(self) -> None:
        self._url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
        self._collection = os.getenv("QDRANT_COLLECTION_CASES", "bottleneck_cases")
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.70,
        cause_type_filter: str | None = None,
        tg_filter: str | None = None,
    ) -> list[dict]:
        """쿼리와 유사한 과거 사례 검색.

        Args:
            cause_type_filter: 원인 유형 필터 (설비_포화 / WIP_누적 / 대기_누적 / 공급_부족).
            tg_filter: 툴그룹 필터. 지정하면 같은 TG 케이스만 검색.

        Returns:
            [{case_id, score, tg_code, area_name, bottleneck_cause_type,
              risk_grade, cause_summary, report_title, source_path,
              report_url, text}, ...]
        """
        if not query.strip():
            return []
        try:
            embedding = self._embed(query[:4000])
            import httpx

            body: dict = {"vector": embedding, "limit": top_k, "with_payload": True}
            must_clauses = []
            if cause_type_filter:
                must_clauses.append({"key": "bottleneck_cause_type", "match": {"value": cause_type_filter}})
            if tg_filter:
                must_clauses.append({"key": "tg_code", "match": {"value": tg_filter}})
            if must_clauses:
                body["filter"] = {"must": must_clauses}

            resp = httpx.post(
                f"{self._url}/collections/{self._collection}/points/search",
                json=body,
                timeout=15,
            )
            resp.raise_for_status()
            out: list[dict] = []
            for hit in resp.json().get("result", []):
                score = hit.get("score")
                if score is None or score < min_score:
                    continue
                payload = hit.get("payload") or {}
                out.append(
                    {
                        "case_id": payload.get("case_id", hit.get("id")),
                        "score": score,
                        "tg_code": payload.get("tg_code", ""),
                        "area_name": payload.get("area_name", ""),
                        "bottleneck_cause_type": payload.get("bottleneck_cause_type", ""),
                        "risk_grade": payload.get("risk_grade", ""),
                        "cause_summary": payload.get("cause_summary", ""),
                        "report_title": payload.get("report_title", ""),
                        "source_path": payload.get("source_path", ""),
                        "report_url": payload.get("report_url", ""),
                        "text": payload.get("text", ""),
                    }
                )
            return out
        except Exception:
            logger.exception("QdrantCaseService.search_similar 실패")
            return []

    def index_case(
        self,
        case_id: str,
        tg_code: str,
        area_name: str = "",
        detected_at: str = "",
        bottleneck_cause_type: str = "",
        risk_grade: str = "",
        cause_summary: str = "",
        report_title: str = "",
        source_path: str = "",
        report_url: str = "",
        narrative: str = "",
    ) -> str | None:
        """케이스를 임베딩하여 Qdrant에 upsert. 성공 시 point_id, 실패/스킵 시 None 반환.

        Args:
            case_id: 결정론적 ID (파일명 stem 등). 동일 ID 재실행 시 덮어씀.
            narrative: 임베딩 소스 — 원인+행동+결과 전체 텍스트.
            cause_summary: 단문 요약 (payload 저장용, 임베딩 소스 아님).
            source_path: 원문 보고서 파일 경로.
            report_url: UI에서 원문 보고서를 열 수 있는 URL.
        """
        if not narrative.strip():
            logger.warning("index_case: narrative 없음, 건너뜀 (case_id=%s)", case_id)
            return None
        try:
            embedding = self._embed(narrative[:8000])
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, case_id))
            import httpx

            resp = httpx.put(
                f"{self._url}/collections/{self._collection}/points",
                json={
                    "points": [
                        {
                            "id": point_id,
                            "vector": embedding,
                            "payload": {
                                "case_id": case_id,
                                "tg_code": tg_code,
                                "area_name": area_name,
                                "detected_at": detected_at,
                                "bottleneck_cause_type": bottleneck_cause_type,
                                "risk_grade": risk_grade,
                                "cause_summary": cause_summary,
                                "report_title": report_title or case_id,
                                "source_path": source_path,
                                "report_url": report_url,
                                "text": narrative,
                            },
                        }
                    ]
                },
                timeout=30,
            )
            resp.raise_for_status()
            logger.debug("index_case 완료: %s → point_id %s", case_id, point_id)
            return point_id
        except Exception:
            logger.exception("QdrantCaseService.index_case 실패 (case_id=%s)", case_id)
            return None

    def count_cases(self) -> int | None:
        """Return indexed case count, or None when Qdrant is unavailable."""
        try:
            import httpx

            resp = httpx.post(
                f"{self._url}/collections/{self._collection}/points/count",
                json={"exact": True},
                timeout=10,
            )
            resp.raise_for_status()
            return int((resp.json().get("result") or {}).get("count", 0))
        except Exception:
            logger.warning(
                "QdrantCaseService.count_cases 실패 (collection=%s)",
                self._collection,
            )
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        from openai import OpenAI

        return (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            .embeddings.create(model=_EMBED_MODEL, input=text)
            .data[0]
            .embedding
        )

    def delete_collection(self) -> None:
        """컬렉션 삭제 — clean reseed 용."""
        try:
            import httpx

            resp = httpx.delete(
                f"{self._url}/collections/{self._collection}",
                timeout=10,
            )
            if resp.status_code in (200, 404):
                logger.info("Qdrant 컬렉션 '%s' 삭제됨", self._collection)
            else:
                logger.warning("컬렉션 삭제 응답: %s", resp.status_code)
        except Exception:
            logger.warning("delete_collection 실패 — Qdrant 미기동일 수 있음")

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성 (idempotent)."""
        try:
            import httpx

            check = httpx.get(
                f"{self._url}/collections/{self._collection}",
                timeout=5,
            )
            if check.status_code == 200:
                return
            # 404 → 생성
            create = httpx.put(
                f"{self._url}/collections/{self._collection}",
                json={
                    "vectors": {
                        "size": _VECTOR_SIZE,
                        "distance": "Cosine",
                    }
                },
                timeout=10,
            )
            create.raise_for_status()
            logger.info("Qdrant 컬렉션 '%s' 생성 완료", self._collection)
        except Exception:
            logger.warning(
                "QdrantCaseService._ensure_collection 실패 — Qdrant 미기동일 수 있음"
            )
