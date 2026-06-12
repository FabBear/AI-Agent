"""KPI 시계열 추세·기간 비교 도구(ps_tg_metrics)."""

import logging

from app.chatbot.context import ChatContext
from app.chatbot.tools.params import safe_int, safe_str
from app.chatbot.ui_cards import _fmt_trend, _trend_card

logger = logging.getLogger(__name__)


async def get_kpi_trend(ctx: ChatContext, area: str = "", hours: int = 6, group_by: str = "area") -> str:
    """구역/TG의 KPI(WIP·가동률·Q-time) 시계열 추세를 조회한다. group_by='tg'면 툴그룹 단위."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 추세를 조회할 수 없습니다."
    hours = safe_int(hours, default=6, min_value=1, max_value=72)
    bucket = 5 if hours <= 2 else (15 if hours <= 8 else 60)
    group_by = group_by if group_by in ("area", "tg") else "area"
    try:
        rows = await repo.kpi_trend(ctx.fab_id, safe_str(area), hours, bucket, group_by)
    except Exception:  # noqa: BLE001
        logger.exception("get_kpi_trend 실패")
        return "추세 조회 중 오류가 발생했습니다."
    ctx.set_card("trend", _trend_card(rows, area, hours, group_by))
    return _fmt_trend(rows, area)


async def compare_periods(ctx: ChatContext, area: str = "", hours: int = 4) -> str:
    """최근 N시간 vs 직전 N시간을 비교(WIP·가동률·Q-time 델타). '어제/아까/전 교대 대비' 질문에 사용."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 기간 비교를 할 수 없습니다."
    hours = safe_int(hours, default=4, min_value=1, max_value=24)
    try:
        rows = await repo.compare_periods(ctx.fab_id, safe_str(area), hours)
    except Exception:  # noqa: BLE001
        logger.exception("compare_periods 실패")
        return "기간 비교 중 오류가 발생했습니다."
    if not rows:
        return f"'{area or '전체'}' 비교 데이터가 없습니다."
    by_area: dict[str, dict] = {}
    for r in rows:
        by_area.setdefault(r["area_name"], {})[r["period"]] = r
    lines = [f"[최근 {hours}시간 vs 직전 {hours}시간]"]
    for name, p in by_area.items():
        rec, prev = p.get("recent"), p.get("previous")
        if not rec or not prev:
            continue
        d_wip = float(rec["wip"] or 0) - float(prev["wip"] or 0)
        d_util = (float(rec["util"] or 0) - float(prev["util"] or 0)) * 100
        d_q = float(rec["qtime"] or 0) - float(prev["qtime"] or 0)
        lines.append(
            f"- {name}: WIP {float(prev['wip'] or 0):.0f}→{float(rec['wip'] or 0):.0f}({d_wip:+.0f}), "
            f"가동률 {d_util:+.1f}%p, Q-time {d_q:+.0f}분"
        )
    return "\n".join(lines) if len(lines) > 1 else "비교 가능한 기간 데이터가 부족합니다."
