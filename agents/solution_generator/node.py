"""LangGraph 노드: alerts + cause_reports → per-TG SolutionCandidate 3개 생성."""

from __future__ import annotations

from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.solution import SimParamDelta, SolutionCandidate
from agents.solution_generator.rule_engine import (
    clip_interval_pct,
    generate_candidates,
)
from agents.state import PipelineState

_RANK_META = {
    "conservative": (1, "보수적 조정안"),
    "standard":     (2, "표준 조정안"),
    "aggressive":   (3, "강화 조정안"),
}


def _build_candidate(level: str, params_dict: dict) -> SolutionCandidate:
    rank, name = _RANK_META[level]
    clipped_pct = (
        clip_interval_pct(params_dict["release_interval_delta_pct"])
        if params_dict["release_interval_delta_pct"] is not None
        else None
    )
    return SolutionCandidate(
        rank=rank,
        name=name,
        target_kpi=params_dict["target_kpi"],
        params=SimParamDelta(
            release_interval_delta_pct=clipped_pct,
            priority_direction=params_dict["priority_direction"],
            superhotlot_enable=params_dict["superhotlot_enable"],
        ),
    )


def generate_solutions(state: PipelineState) -> PipelineState:
    alerts: list[BottleneckAlert] = state["alerts"]
    cause_map = {r.toolgroup: r for r in state["cause_reports"]}

    target_alerts = [
        a for a in alerts
        if a.severity == SeverityLevel.CRITICAL
        and a.toolgroup in cause_map
    ]

    solution_candidates = []
    for alert in target_alerts:
        cause_report = cause_map[alert.toolgroup]

        result = generate_candidates(alert, cause_report)
        if result is None:  # judgment=None → 대응안 생성 불가
            continue

        candidates = [
            _build_candidate(lv, result[lv])
            for lv in ("conservative", "standard", "aggressive")
        ]

        solution_candidates.append({
            "toolgroup": alert.toolgroup,
            "candidates": [c.model_dump() for c in candidates],
        })

    return {**state, "solution_candidates": solution_candidates}
