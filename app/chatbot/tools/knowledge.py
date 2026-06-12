"""사내 지식 RAG 도구."""

from app.chatbot.context import ChatContext
from app.chatbot.rag import knowledge_block, rag_search


def search_knowledge(ctx: ChatContext, query: str) -> str:
    """반도체 FAB 도메인 지식(개념·정의·원리·SOP·용어·설비/공정)을 사내 지식베이스에서 검색한다."""
    hits = rag_search(query)
    ctx.used_hits.extend(hits)
    return knowledge_block(hits) or "관련 사내 지식을 찾지 못했습니다."
