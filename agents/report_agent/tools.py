"""report_agent Level 3 도구 — DB에서 과거 이력 조회.

ReportV2에 없는 두 가지 정보를 DB에서 직접 가져온다:
  1. 이 TG의 반복 병목 횟수 (최근 N일)
  2. 과거 승인 조치의 실측 효과 (예상 vs 실측 KPI 변화)

agent_service.run_post_hitl()이 파이프라인 실행 전에 호출해서
PipelineState["historical_context"]에 주입한다.
"""

from __future__ import annotations

import asyncpg


async def fetch_repeat_count(
    pool: asyncpg.Pool, tg_code: str, days: int = 30
) -> str:
    """최근 days일간 이 TG가 병목으로 감지된 횟수를 반환한다."""
    row = await pool.fetchrow(
        """
        SELECT COUNT(*) AS cnt
        FROM tt_bottleneck_case c
        JOIN tm_tool_group tg ON tg.tg_id = c.tg_id
        WHERE tg.tg_code = $1
          AND c.detected_at >= NOW() - make_interval(days => $2)
        """,
        tg_code, days,
    )
    count = int(row["cnt"]) if row else 0
    if count <= 1:
        return f"{tg_code}: 최근 {days}일간 반복 병목 이력 없음 (이번이 처음 또는 1회)."
    return f"{tg_code}: 최근 {days}일간 {count}회 병목 감지됨 — 반복 병목 패턴."


async def fetch_past_action_effectiveness(
    pool: asyncpg.Pool, tg_code: str
) -> str:
    """이 TG의 최근 케이스 중 실측 검증된 조치 효과(예상 vs 실측)를 반환한다.

    실측값(actual_avg_wait_delta)이 있는 케이스를 우선 탐색하고,
    없으면 가장 최근 케이스의 예상 효과만 반환한다.
    """
    case_rows = await pool.fetch(
        """
        SELECT c.case_id
        FROM tt_bottleneck_case c
        JOIN tm_tool_group tg ON tg.tg_id = c.tg_id
        WHERE tg.tg_code = $1
        ORDER BY c.detected_at DESC
        LIMIT 5
        """,
        tg_code,
    )
    if not case_rows:
        return f"{tg_code}: 과거 케이스 이력 없음."

    for case_row in case_rows:
        case_id = case_row["case_id"]
        plans = await pool.fetch(
            """
            SELECT p.plan_type, p.plan_title,
                   p.est_q_time_delta AS est_avg_wait_delta,
                   p.actual_avg_wait_delta, p.validated_at,
                   COALESCE(p.plan_id = h.selected_plan_id, FALSE) AS selected
            FROM td_action_plan p
            LEFT JOIN LATERAL (
                SELECT selected_plan_id FROM th_hitl_decision d
                WHERE d.case_id = p.case_id
                ORDER BY d.re_decision_seq DESC LIMIT 1
            ) h ON TRUE
            WHERE p.case_id = $1
            ORDER BY p.plan_seq
            """,
            case_id,
        )
        selected = next((p for p in plans if p["selected"]), None)
        if selected is None:
            continue

        est = selected["est_avg_wait_delta"]
        actual = selected["actual_avg_wait_delta"]
        kind = selected["plan_type"] or "조치"

        if actual is not None:
            return (
                f"{tg_code} 최근 승인 조치 실측: {kind} — "
                f"예상 대기시간 {float(est):+.1f}분, 실측 {float(actual):+.1f}분 "
                f"(검증: {selected['validated_at']})."
            )
        if est is not None:
            return (
                f"{tg_code} 최근 승인 조치(미검증): {kind} — "
                f"예상 대기시간 {float(est):+.1f}분 (실측 미수집)."
            )

    return f"{tg_code}: 실측 검증된 과거 조치 효과 없음."
