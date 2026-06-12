"""사내 지식 RAG — Qdrant 검색, LLM 컨텍스트 블록 조립, 화면 출처 카드 dedupe."""

import logging
import os

from app.chatbot.config import (
    EMBED_MODEL,
    RAG_CHUNK_CHARS,
    RAG_MIN_SCORE,
    RAG_TOP_K,
    SOURCE_MAX,
    SOURCE_MIN_SCORE,
    SOURCE_TOP_GAP,
    chat_enabled,
)

logger = logging.getLogger(__name__)


def rag_search(query: str) -> list[dict]:
    """질문과 관련된 사내 반도체 지식을 Qdrant에서 검색. 실패/무관 시 빈 리스트."""
    if not chat_enabled() or not (query or "").strip():
        return []
    try:
        import httpx
        from openai import OpenAI

        embedding = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            .embeddings.create(model=EMBED_MODEL, input=query[:2000])
            .data[0]
            .embedding
        )
        url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
        collection = os.getenv("QDRANT_COLLECTION", "fabbear_rag_documents")
        resp = httpx.post(
            f"{url}/collections/{collection}/points/search",
            json={"vector": embedding, "limit": RAG_TOP_K, "with_payload": True},
            timeout=10,
        )
        resp.raise_for_status()
        out: list[dict] = []
        for hit in resp.json().get("result", []):
            score = hit.get("score")
            if score is not None and score < RAG_MIN_SCORE:
                continue
            payload = hit.get("payload") or {}
            out.append({
                "title": payload.get("title") or "지식",
                "source": payload.get("source_path"),
                "category": payload.get("category"),
                "text": (payload.get("chunk_text") or "").strip(),
                "score": score,
            })
        return out
    except Exception:  # noqa: BLE001
        logger.exception("RAG 검색 실패")
        return []


def knowledge_block(hits: list[dict]) -> str | None:
    if not hits:
        return None
    parts = []
    for hit in hits:
        text = hit["text"][:RAG_CHUNK_CHARS]
        parts.append(f"### {hit['title']} (출처: {hit.get('source') or '사내 지식'})\n{text}")
    return (
        "[참고 지식 — 사내 반도체 지식베이스에서 검색됨]\n"
        "질문과 관련 있을 때만 활용하고, 활용하면 어떤 개념을 참고했는지 자연스럽게 밝혀라.\n\n"
        + "\n\n".join(parts)
    )


def dedupe_sources(hits: list[dict]) -> list[dict]:
    """화면에 표시할 출처: 최상위와 근접하고 충분히 관련 있는 것만(노이즈 제거), 문서 단위 중복 제거."""
    if not hits:
        return []
    top = hits[0].get("score") or 0.0
    seen: set[str] = set()
    sources: list[dict] = []
    for hit in hits:
        score = hit.get("score") or 0.0
        if score < SOURCE_MIN_SCORE or (top - score) > SOURCE_TOP_GAP:
            continue
        key = hit.get("source") or hit.get("title") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": hit.get("title") or "지식",
            "sourcePath": hit.get("source"),
            "category": hit.get("category"),
        })
        if len(sources) >= SOURCE_MAX:
            break
    return sources
