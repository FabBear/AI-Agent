"""WIP·대기 적체(get_lot_status), TG 순위(get_top_toolgroups), 개별 툴 현황(get_tool_status) 도구."""

import logging

from app.chatbot.context import ChatContext
from app.chatbot.tools.params import safe_int, safe_str
from app.chatbot.ui_cards import (
    _TG_METRIC_LABELS,
    _fmt_lot,
    _fmt_tools,
    _fmt_top_tgs,
    _lot_card,
    _tool_card,
    _top_tg_card,
)

logger = logging.getLogger(__name__)


async def get_lot_status(ctx: ChatContext, area: str = "") -> str:
    """구역별 WIP·대기·Q-time 현황을 조회한다(개별 Lot 추적 데이터는 없어 구역 집계)."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 WIP 현황을 조회할 수 없습니다."
    try:
        rows = await repo.lot_status(ctx.fab_id, safe_str(area))
    except Exception:  # noqa: BLE001
        logger.exception("get_lot_status 실패")
        return "WIP 현황 조회 중 오류가 발생했습니다."
    ctx.set_card("lot", _lot_card(rows, area))
    return _fmt_lot(rows, area)


async def get_top_toolgroups(
    ctx: ChatContext, area: str = "", metric: str = "util", order: str = "desc", limit: int = 5,
) -> str:
    """툴그룹(TG) 단위 현황·순위를 조회한다. '가동률 가장 높은/낮은 툴그룹' 등 순위 질문용."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 툴그룹 현황을 조회할 수 없습니다."
    metric = safe_str(metric)
    metric = metric if metric in _TG_METRIC_LABELS else "util"
    order = "asc" if str(order).lower() == "asc" else "desc"
    limit = safe_int(limit, default=5, min_value=1, max_value=15)
    try:
        rows = await repo.top_toolgroups(ctx.fab_id, safe_str(area), metric, order, limit)
    except Exception:  # noqa: BLE001
        logger.exception("get_top_toolgroups 실패")
        return "툴그룹 현황 조회 중 오류가 발생했습니다."
    ctx.set_card("lot", _top_tg_card(rows, area, metric, order, limit))
    return _fmt_top_tgs(rows, area, metric, order)


async def get_tool_status(ctx: ChatContext, tg: str = "", metric: str = "util") -> str:
    """특정 툴그룹에 속한 개별 툴(설비)들의 현황(가동률·OEE·대기 Lot·셋업/다운)을 조회한다."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 툴 현황을 조회할 수 없습니다."
    try:
        rows = await repo.tool_status(ctx.fab_id, safe_str(tg))
    except Exception:  # noqa: BLE001
        logger.exception("get_tool_status 실패")
        return "툴 현황 조회 중 오류가 발생했습니다."
    metric = safe_str(metric).lower() or "util"
    if metric in ("queue", "wait", "wip", "lot", "대기"):
        sort_key = "queue_lot_count"
    elif metric in ("down", "다운"):
        sort_key = "down_ratio"
    elif metric in ("qtime", "q-time", "대기시간"):
        sort_key = "avg_qtime_min"
    else:
        sort_key = "utilization_rate"
    rows = sorted(rows, key=lambda r: float(r[sort_key] or 0), reverse=True)[:15]
    ctx.set_card("lot", _tool_card(rows, tg))
    return _fmt_tools(rows, tg)
