"""Lot 투입계획 조회 도구."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.chatbot.context import ChatContext
from app.chatbot.tools.params import safe_int, safe_str

logger = logging.getLogger(__name__)

_MIN_PER_DAY = 24 * 60
_RANGES = {
    "24h": (24 * 60, 4 * 60, "향후 24시간"),
    "48h": (48 * 60, 4 * 60, "향후 48시간"),
    "7d": (7 * _MIN_PER_DAY, _MIN_PER_DAY, "향후 7일"),
    "30d": (30 * _MIN_PER_DAY, _MIN_PER_DAY, "향후 30일"),
    "all": (None, 7 * _MIN_PER_DAY, "전체 예정 기간"),
}


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _minutes_label(value: Any) -> str:
    minutes = _num(value)
    if minutes <= 0:
        return "즉시"
    if minutes < 60:
        return f"{minutes:.0f}분 후"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}시간 후"
    return f"D+{hours / 24:.1f}"


def _range_value(raw: str = "", days: int = 0) -> str:
    text = safe_str(raw).lower()
    compact = re.sub(r"[\s_-]+", "", text)
    if days:
        if days <= 1:
            return "24h"
        if days <= 2:
            return "48h"
        if days <= 7:
            return "7d"
        return "30d"
    if any(term in compact for term in ("7d", "7일", "일주일", "이번주", "이번주간", "week")):
        return "7d"
    if any(term in compact for term in ("30d", "30일", "한달", "이번달", "month")):
        return "30d"
    if any(term in compact for term in ("48h", "48시간", "이틀", "2일")):
        return "48h"
    if any(term in compact for term in ("all", "전체")):
        return "all"
    return "24h"


def _bucket_label(bucket_index: int, bucket_min: int) -> str:
    from_hour = bucket_index * bucket_min / 60
    to_hour = (bucket_index + 1) * bucket_min / 60
    if bucket_min >= _MIN_PER_DAY:
        return f"D+{bucket_index}"
    return f"{from_hour:.0f}-{to_hour:.0f}h"


def _fmt_release_plan(data: dict, range_value: str, range_label: str, bucket_min: int) -> str:
    summary = data["summary"]
    planned = int(_num(summary["planned_lots"]))
    if planned <= 0:
        return f"{range_label} Lot 투입 예정 데이터가 없습니다."

    days = max(1.0, (_RANGES[range_value][0] or _MIN_PER_DAY) / _MIN_PER_DAY)
    lots_per_day = planned / days
    priority = int(_num(summary["priority_lots"]))
    super_hot = int(_num(summary["super_hot_lots"]))
    due = int(_num(summary["due_lots"]))
    wafers = int(_num(summary["planned_wafers"]))
    avg_slack = _num(summary["avg_due_slack_min"])
    source = summary["source_run_id"] or "-"

    bucket_lines = []
    for row in data["buckets"][:10]:
        label = _bucket_label(int(row["bucket_index"]), bucket_min)
        bucket_lines.append(
            f"- {label}: 투입 {int(_num(row['total_lots']))} Lot, Priority {int(_num(row['priority_lots']))}건"
        )

    product_lines = [
        f"{row['name']} {int(_num(row['lots']))} Lot"
        for row in data["product_mix"][:5]
    ]
    hot_lines = []
    for row in data["hot_lots"][:6]:
        grade = "SuperHot" if row["is_super_hot"] else "Priority"
        hot_lines.append(
            f"- {grade} {row['lot_id']}: {row['product_name'] or '-'}, "
            f"투입 {_minutes_label(row['release_in_min'])}, 납기 {_minutes_label(row['due_in_min'])}"
        )

    lines = [
        f"[Lot 투입계획 — {range_label}]",
        (
            f"계산 요약: 총 투입 {planned} Lot, 일평균 {lots_per_day:.1f} Lot/일, "
            f"웨이퍼 {wafers}장, Priority {priority}건, SuperHot {super_hot}건, "
            f"기간 내 납기 {due}건, 다음 투입 {_minutes_label(summary['next_release_in_min'])}, "
            f"평균 납기 여유 {avg_slack / 60:.1f}시간, source_run_id={source}."
        ),
    ]
    if bucket_lines:
        lines.append("기간별 투입:")
        lines.extend(bucket_lines)
    if product_lines:
        lines.append("제품 구성 상위: " + ", ".join(product_lines))
    if hot_lines:
        lines.append("우선 확인 Lot:")
        lines.extend(hot_lines)
    return "\n".join(lines)


def _release_plan_card(data: dict, range_label: str, bucket_min: int) -> dict | None:
    buckets = data.get("buckets") or []
    if not buckets:
        return None
    return {
        "type": "lot",
        "props": {
            "title": f"Lot 투입 계획 · {range_label}",
            "labelHeader": "기간",
            "columns": [
                {"key": "lots", "label": "투입 Lot"},
                {"key": "priority", "label": "Priority"},
                {"key": "from", "label": "시작"},
                {"key": "to", "label": "종료"},
            ],
            "rows": [
                {
                    "label": _bucket_label(int(row["bucket_index"]), bucket_min),
                    "lots": int(_num(row["total_lots"])),
                    "priority": int(_num(row["priority_lots"])),
                    "from": _minutes_label(row["from_min"]),
                    "to": _minutes_label(row["to_min"]),
                }
                for row in buckets[:10]
            ],
        },
    }


async def get_lot_release_plan(ctx: ChatContext, range: str = "24h", days: int = 0, limit: int = 6) -> str:
    """향후 Lot 투입계획/스케줄을 조회한다."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 Lot 투입계획을 조회할 수 없습니다."
    range_value = _range_value(range, safe_int(days, default=0, min_value=0, max_value=30))
    window_min, bucket_min, range_label = _RANGES[range_value]
    limit = safe_int(limit, default=6, min_value=1, max_value=20)
    try:
        data = await repo.lot_release_plan(ctx.fab_id, window_min, bucket_min, limit)
    except Exception:  # noqa: BLE001
        logger.exception("get_lot_release_plan 실패")
        return "Lot 투입계획 조회 중 오류가 발생했습니다."
    if not data:
        return f"{range_label} Lot 투입계획 데이터가 없습니다."
    ctx.set_card("lot", _release_plan_card(data, range_label, bucket_min))
    return _fmt_release_plan(data, range_value, range_label, bucket_min)
