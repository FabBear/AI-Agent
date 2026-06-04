"""
T0 스냅샷 구성: lot_event_log + tool_state_log → mes_wip_snapshot, mes_tool_snapshot.

개선사항:
  - FINISH 이벤트 lot → process_step 테이블로 다음 스텝 조회 → QUEUE 상태로 정확히 복원
  - LOADING/BATCH와 FINISH가 동시 발생 시 LOADING 우선 (DISTINCT ON + 정렬)
  - ARRIVAL만 있는 lot (첫 스텝 대기) 포함
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from agents.sim_runner.db_connector import get_session, query_df


@dataclass
class WipRow:
    lot_id: str
    route_id: str
    current_step_seq: int
    status: str  # PROCESSING or QUEUE
    tool_group: str | None
    tool_id: str | None
    queue_position: int
    due_date_sim: float | None
    priority: int
    rem_steps: int | None
    processing_remaining_min: float | None
    wafers_per_lot: int
    product: str
    is_super_hot: bool


@dataclass
class ToolRow:
    tool_id: str
    tool_group: str
    op_state: str
    current_setup: str | None
    held_lot_id: str | None


def build_t0_snapshot(t0: float) -> tuple[list[WipRow], list[ToolRow]]:
    wip_rows = _build_wip_snapshot(t0)
    tool_rows = _build_tool_snapshot(t0)
    return wip_rows, tool_rows


def _load_route_map() -> dict[str, list[tuple[int, str]]]:
    """process_step 테이블에서 route별 정렬된 (step_seq, tool_group) 목록 반환."""
    df = query_df("""
        SELECT route_id, step_seq, target_tool_group
        FROM process_step
        ORDER BY route_id, step_seq
    """)
    route_map: dict[str, list[tuple[int, str]]] = {}
    for _, row in df.iterrows():
        rid = str(row["route_id"])
        if rid not in route_map:
            route_map[rid] = []
        if row["target_tool_group"]:
            route_map[rid].append((int(row["step_seq"]), str(row["target_tool_group"])))
    return route_map


def _build_next_step_lookup(
    route_map: dict[str, list[tuple[int, str]]],
) -> dict[tuple[str, int], tuple[int, str]]:
    """(route_id, current_step_seq) → (next_step_seq, next_tool_group)"""
    lookup: dict[tuple[str, int], tuple[int, str]] = {}
    for rid, steps in route_map.items():
        for i, (seq, _tg) in enumerate(steps):
            if i + 1 < len(steps):
                lookup[(rid, seq)] = steps[i + 1]
    return lookup


def _build_wip_snapshot(t0: float) -> list[WipRow]:
    """
    각 lot의 T0 상태 재구성.
    - LOADING/BATCH → PROCESSING at current step
    - FINISH → QUEUE at NEXT step (process_step 테이블 참조)
    - ARRIVAL only → QUEUE at first step
    """
    # LOADING과 FINISH가 같은 시각이면 LOADING 우선 (CASE 정렬)
    df = query_df(
        """
        SELECT DISTINCT ON (lot_id)
            lot_id, route_id, step_seq, tool_group, tool_id, event_type, event_time
        FROM lot_event_log
        WHERE event_time <= :t0
        ORDER BY lot_id,
                 event_time DESC,
                 CASE event_type
                     WHEN 'LOADING'            THEN 0
                     WHEN 'BATCH_START'        THEN 0
                     WHEN 'BATCH_MEMBER_START' THEN 0
                     ELSE 1
                 END ASC
    """,
        {"t0": t0},
    )

    if df.empty:
        return []

    # lot_release 마스터 메타
    lot_meta_df = query_df("""
        SELECT product_name, wafers_per_lot, priority,
               (is_super_hot_lot = 'yes') AS is_super_hot
        FROM lot_release ORDER BY priority
    """)
    meta_map: dict[str, dict] = {}
    for _, row in lot_meta_df.iterrows():
        pname = str(row["product_name"])
        if pname not in meta_map:
            w = row.get("wafers_per_lot")
            p = row.get("priority")
            s = row.get("is_super_hot")
            meta_map[pname] = {
                "wafers_per_lot": int(w) if pd.notna(w) else 25,
                "priority": int(p) if pd.notna(p) else 10,
                "is_super_hot": bool(s) if pd.notna(s) else False,
            }

    route_map = _load_route_map()
    next_step_lookup = _build_next_step_lookup(route_map)

    rows: list[WipRow] = []
    for _, r in df.iterrows():
        parts = str(r["lot_id"]).split("_")
        product = "_".join(parts[1:-1]) if len(parts) >= 3 else "Product_3"
        meta = meta_map.get(product, {"wafers_per_lot": 25, "priority": 10, "is_super_hot": False})
        event = str(r["event_type"]).upper()
        route_id = str(r["route_id"])
        import math

        raw_seq = r["step_seq"]
        step_seq = (
            int(float(raw_seq))
            if (raw_seq is not None and not (isinstance(raw_seq, float) and math.isnan(raw_seq)))
            else 0
        )

        if event in ("LOADING", "BATCH_START", "BATCH_MEMBER_START"):
            # PROCESSING at current step
            rows.append(
                WipRow(
                    lot_id=r["lot_id"],
                    route_id=route_id,
                    current_step_seq=step_seq,
                    status="PROCESSING",
                    tool_group=r["tool_group"],
                    tool_id=r["tool_id"],
                    queue_position=0,
                    due_date_sim=None,
                    priority=meta["priority"],
                    rem_steps=None,
                    processing_remaining_min=5.0,
                    wafers_per_lot=meta["wafers_per_lot"],
                    product=product,
                    is_super_hot=meta["is_super_hot"],
                )
            )

        elif event == "FINISH":
            # 다음 스텝 조회 → QUEUE
            next_info = next_step_lookup.get((route_id, step_seq))
            if next_info:
                next_seq, next_tg = next_info
                rows.append(
                    WipRow(
                        lot_id=r["lot_id"],
                        route_id=route_id,
                        current_step_seq=next_seq,
                        status="QUEUE",
                        tool_group=next_tg,
                        tool_id=None,
                        queue_position=0,
                        due_date_sim=None,
                        priority=meta["priority"],
                        rem_steps=None,
                        processing_remaining_min=None,
                        wafers_per_lot=meta["wafers_per_lot"],
                        product=product,
                        is_super_hot=meta["is_super_hot"],
                    )
                )
            # 마지막 스텝 FINISH → 공정 완료, 스냅샷에 포함 안 함

        elif event == "ARRIVAL":
            # 첫 스텝 대기
            first = route_map.get(route_id, [(1, None)])[0]
            first_seq, first_tg = first
            if first_tg:
                rows.append(
                    WipRow(
                        lot_id=r["lot_id"],
                        route_id=route_id,
                        current_step_seq=first_seq,
                        status="QUEUE",
                        tool_group=first_tg,
                        tool_id=None,
                        queue_position=0,
                        due_date_sim=None,
                        priority=meta["priority"],
                        rem_steps=None,
                        processing_remaining_min=None,
                        wafers_per_lot=meta["wafers_per_lot"],
                        product=product,
                        is_super_hot=meta["is_super_hot"],
                    )
                )

    return rows


def _build_tool_snapshot(t0: float) -> list[ToolRow]:
    df = query_df(
        """
        SELECT DISTINCT ON (tool_id)
            tool_id, tool_group, state, lot_id, setup_name
        FROM tool_state_log
        WHERE state_change_time <= :t0
          AND tool_id IS NOT NULL AND tool_id != ''
        ORDER BY tool_id, state_change_time DESC
    """,
        {"t0": t0},
    )

    if df.empty:
        return []

    rows: list[ToolRow] = []
    for _, r in df.iterrows():
        state = str(r["state"] or "IDLE").upper()
        if state not in ("IDLE", "RUN", "DOWN_PM", "DOWN_BM", "SETUP"):
            state = "IDLE"
        rows.append(
            ToolRow(
                tool_id=r["tool_id"],
                tool_group=r["tool_group"],
                op_state=state,
                current_setup=r.get("setup_name") or None,
                held_lot_id=r.get("lot_id") if state == "RUN" else None,
            )
        )
    return rows


def insert_t0_snapshot(scenario_id: str, t0: float) -> tuple[int, int]:
    """DB에 T0 스냅샷 삽입. (wip_count, tool_count) 반환."""
    wip_rows, tool_rows = build_t0_snapshot(t0)

    session = get_session()
    try:
        if wip_rows:
            session.execute(
                text("""
                INSERT INTO mes_wip_snapshot
                    (scenario_id, snapshot_time, lot_id, route_id, current_step_seq,
                     status, tool_group, tool_id, queue_position, due_date_sim,
                     priority, rem_steps, processing_remaining_min,
                     wafers_per_lot, product, is_super_hot)
                VALUES (:sid, :t0, :lot_id, :route_id, :step_seq,
                        :status, :tg, :tool_id, :qpos, :due,
                        :prio, :rem, :proc_rem, :wafers, :product, :super)
            """),
                [
                    {
                        "sid": scenario_id,
                        "t0": t0,
                        "lot_id": w.lot_id,
                        "route_id": w.route_id,
                        "step_seq": w.current_step_seq,
                        "status": w.status,
                        "tg": w.tool_group,
                        "tool_id": w.tool_id,
                        "qpos": w.queue_position,
                        "due": w.due_date_sim,
                        "prio": w.priority,
                        "rem": w.rem_steps,
                        "proc_rem": w.processing_remaining_min,
                        "wafers": w.wafers_per_lot,
                        "product": w.product,
                        "super": w.is_super_hot,
                    }
                    for w in wip_rows
                ],
            )

        if tool_rows:
            session.execute(
                text("""
                INSERT INTO mes_tool_snapshot
                    (scenario_id, tool_id, tool_group, op_state, current_setup, held_lot_id)
                VALUES (:sid, :tid, :tg, :op, :setup, :held)
            """),
                [
                    {
                        "sid": scenario_id,
                        "tid": t.tool_id,
                        "tg": t.tool_group,
                        "op": t.op_state,
                        "setup": t.current_setup,
                        "held": t.held_lot_id,
                    }
                    for t in tool_rows
                ],
            )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return len(wip_rows), len(tool_rows)
