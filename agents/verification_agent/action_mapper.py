"""SolutionCandidate / GlobalSolutionPlan → mes_whatif_action rows 변환.

FabEnv가 지원하는 action_kind:
  DISPATCH_RULE_OVERRIDE  — 툴그룹 디스패치 규칙 변경
  SET_SUPER_HOT           — lot super_hot 플래그 설정
  LOT_HOLD                — lot 보류

release_interval은 action row가 아닌 lot_release_plan 복사 시
배수(multiplier)로 처리.
"""

from __future__ import annotations

import json

from agents.schemas.alert import BottleneckAlert
from agents.schemas.solution import GlobalSolutionPlan, SolutionCandidate

# lot_priority_rule → DISPATCH_RULE_OVERRIDE 규칙 매핑
_PRIORITY_RULE_TO_DISPATCH: dict[str, str] = {
    "HIGH_WIP_FIRST": "SPT",
    "DUE_DATE": "EDD",
    "SUPERHOTLOT_FIRST": "superhotlot setupavoidance",
    "FIFO": "FIFO",
}


def candidate_to_action_rows(
    candidate: SolutionCandidate,
    alert: BottleneckAlert,
    t0: float,
) -> tuple[list[dict], float]:
    """SolutionCandidate → (action_rows, release_interval_multiplier)."""
    pct = candidate.params.release_interval_delta_pct
    release_multiplier = 1.0 + (pct / 100.0) if pct is not None else 1.0

    tg = alert.toolgroup
    rows: list[dict] = []
    seq = 0

    dispatch_rule: str | None = None
    if candidate.params.lot_priority_rule is not None:
        dispatch_rule = _PRIORITY_RULE_TO_DISPATCH.get(
            candidate.params.lot_priority_rule, candidate.params.lot_priority_rule
        )
    elif candidate.params.dispatch_rule is not None:
        dispatch_rule = candidate.params.dispatch_rule

    if dispatch_rule is not None:
        rows.append({
            "seq": seq,
            "action_kind": "DISPATCH_RULE_OVERRIDE",
            "effective_time": t0,
            "lot_id": None,
            "route_id": None,
            "step_seq": None,
            "tool_group": tg,
            "tool_id": None,
            "payload_json": json.dumps({"tool_group": tg, "dispatch_rule": dispatch_rule}),
            "source": "AGENT",
        })
        seq += 1

    if candidate.params.superhotlot_enable:
        rows.append({
            "seq": seq,
            "action_kind": "DISPATCH_RULE_OVERRIDE",
            "effective_time": t0,
            "lot_id": None,
            "route_id": None,
            "step_seq": None,
            "tool_group": tg,
            "tool_id": None,
            "payload_json": json.dumps({
                "tool_group": tg,
                "dispatch_rule": "superhotlot setupavoidance",
            }),
            "source": "AGENT",
        })

    return rows, release_multiplier


def plan_to_action_rows(
    plan: GlobalSolutionPlan,
    t0: float,
) -> tuple[list[dict], float]:
    """
    GlobalSolutionPlan → (action_rows, release_interval_multiplier)

    target_toolgroups 전체에 대해 action_rows를 한번에 생성한다.
    release_interval_multiplier = release_interval_minutes / current_interval_minutes
    """
    current = plan.current_interval_minutes or 1.0
    release_multiplier = plan.release_interval_minutes / current

    rows: list[dict] = []
    seq = 0

    for tg in plan.target_toolgroups:
        if plan.lot_priority_rule is not None:
            dispatch = _PRIORITY_RULE_TO_DISPATCH.get(
                plan.lot_priority_rule, plan.lot_priority_rule
            )
            rows.append({
                "seq": seq,
                "action_kind": "DISPATCH_RULE_OVERRIDE",
                "effective_time": t0,
                "lot_id": None,
                "route_id": None,
                "step_seq": None,
                "tool_group": tg,
                "tool_id": None,
                "payload_json": json.dumps({"tool_group": tg, "dispatch_rule": dispatch}),
                "source": "AGENT",
            })
            seq += 1

        if plan.superhotlot_enable:
            rows.append({
                "seq": seq,
                "action_kind": "DISPATCH_RULE_OVERRIDE",
                "effective_time": t0,
                "lot_id": None,
                "route_id": None,
                "step_seq": None,
                "tool_group": tg,
                "tool_id": None,
                "payload_json": json.dumps({
                    "tool_group": tg,
                    "dispatch_rule": "superhotlot setupavoidance",
                }),
                "source": "AGENT",
            })
            seq += 1

    return rows, release_multiplier
