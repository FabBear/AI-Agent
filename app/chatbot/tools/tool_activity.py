"""특정 툴(설비) 1대의 최근 활동 추이 도구(ps_tool_metrics)."""

import logging

from app.chatbot.context import ChatContext
from app.chatbot.tools.params import safe_int, safe_str
from app.chatbot.ui_cards import _fmt_tool_activity, _tool_activity_card

logger = logging.getLogger(__name__)


async def get_tool_activity(ctx: ChatContext, tool: str = "", hours: int = 6) -> str:
    """특정 툴(설비) 1대의 최근 활동 추이(가동률·대기 Lot·Q-time·다운율 시계열)를 조회한다."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 설비 활동을 조회할 수 없습니다."
    hours = safe_int(hours, default=6, min_value=1, max_value=72)
    bucket = 5 if hours <= 2 else (15 if hours <= 8 else 60)
    try:
        rows = await repo.tool_activity(ctx.fab_id, safe_str(tool), hours, bucket)
    except Exception:  # noqa: BLE001
        logger.exception("get_tool_activity 실패")
        return "설비 활동 조회 중 오류가 발생했습니다."
    # 여러 설비가 매칭되면 정확일치 우선으로 1대만 선택(차트는 단일 설비 기준).
    if rows:
        codes = {r["tool_code"] for r in rows}
        if len(codes) > 1:
            target = safe_str(tool).lower()
            pick = next((c for c in codes if c.lower() == target), rows[0]["tool_code"])
            rows = [r for r in rows if r["tool_code"] == pick]
    ctx.set_card("trend", _tool_activity_card(rows, hours))
    return _fmt_tool_activity(rows, tool)
