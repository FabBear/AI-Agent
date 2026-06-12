"""실시간 FAB 현황 도구 — Spring이 넘긴 live_status 텍스트 기반(DB 미조회)."""

from app.chatbot.context import ChatContext
from app.chatbot.ui_cards import _fab_status_lookup, _status_card


def get_fab_status(ctx: ChatContext, area: str = "") -> str:
    """현재 FAB 실시간 현황(전체 가동률·WIP·설비상태, 구역별 WIP·가동률·가용률)을 조회한다."""
    ctx.set_card("status", _status_card(ctx.live_status))
    return _fab_status_lookup(ctx.live_status, area)
