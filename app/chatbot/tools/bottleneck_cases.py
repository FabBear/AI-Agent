"""병목 케이스 이력 목록(search_bottleneck_cases)과 케이스 1건 상세(get_case_detail) 도구."""

import logging

from app.chatbot.context import ChatContext
from app.chatbot.ui_cards import _cases_card, _fmt_case_detail, _fmt_cases

logger = logging.getLogger(__name__)


async def search_bottleneck_cases(ctx: ChatContext, area: str = "", status: str = "") -> str:
    """최근 병목 케이스(위험등급·감지확률·상태)를 조회한다. status: DETECTED/ANALYZING/AWAITING_HITL/RESOLVED."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 병목 케이스를 조회할 수 없습니다."
    try:
        rows = await repo.bottleneck_cases(ctx.fab_id, (area or "").strip(), (status or "").strip().upper(), 10)
    except Exception:  # noqa: BLE001
        logger.exception("search_bottleneck_cases 실패")
        return "병목 케이스 조회 중 오류가 발생했습니다."
    if rows:
        ctx.set_card("cases", _cases_card(rows, area))
    return _fmt_cases(rows, area)


async def get_case_detail(ctx: ChatContext, case_ref: str = "") -> str:
    """병목 케이스 1건의 상세(원인 SHAP·대응안·HITL 결정·리포트 요약)를 조회한다."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 케이스 상세를 조회할 수 없습니다."
    try:
        row = await repo.case_detail(ctx.fab_id, (case_ref or "").strip())
        if row is None:
            return f"'{case_ref or '최근'}' 병목 케이스를 찾지 못했습니다."
        plans = await repo.case_plans(row["case_id"])
        hitl = await repo.case_hitl(row["case_id"])
    except Exception:  # noqa: BLE001
        logger.exception("get_case_detail 실패")
        return "케이스 상세 조회 중 오류가 발생했습니다."
    return _fmt_case_detail(row, plans, hitl)
