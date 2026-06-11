"""LangGraph 노드: alerts + cause_reports → per-TG SolutionCandidate 3개 생성."""

from __future__ import annotations

from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.solution import SimParamDelta, SolutionCandidate
from agents.solution_generator.llm_generator import generate_texts
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


def _build_candidate(
    level: str,
    params_dict: dict,
    texts: dict[str, dict[str, str]],
    severity: SeverityLevel,
) -> SolutionCandidate:
    rank, name = _RANK_META[level]
    clipped_pct = (
        clip_interval_pct(params_dict["release_interval_delta_pct"], severity)
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
        expected_effect=texts[level].get("expected_effect", ""),
        rationale=texts[level].get("rationale", ""),
    )


def generate_solutions(state: PipelineState) -> PipelineState:
    alerts: list[BottleneckAlert] = state["alerts"]
    cause_map = {r.toolgroup: r for r in state["cause_reports"]}

    target_alerts = [
        a for a in alerts
        if a.severity in {SeverityLevel.CRITICAL, SeverityLevel.HIGH}
        and a.toolgroup in cause_map
    ]

    solution_candidates = []
    for alert in target_alerts:
        cause_report = cause_map[alert.toolgroup]

        # 1. 규칙 엔진: 보수/표준/강화 파라미터 확정
        result = generate_candidates(alert, cause_report)

        # 2. LLM: 텍스트 생성 (expected_effect + rationale)
        texts = generate_texts(alert, cause_report, result)

        # 3. post-processing clip + SolutionCandidate 조립
        candidates = [
            _build_candidate(lv, result[lv], texts, alert.severity)
            for lv in ("conservative", "standard", "aggressive")
        ]

        # HITL 에스컬레이션 플래그는 rationale에 포함
        if result.get("hitl_escalation_recommended"):
            reason = result.get("escalation_reason", "")
            for c in candidates:
                if reason and "[HITL]" not in c.rationale:
                    c.rationale = f"[HITL 에스컬레이션 권고] {reason} | {c.rationale}"

        solution_candidates.append({
            "toolgroup": alert.toolgroup,
            "candidates": [c.model_dump() for c in candidates],
        })

    return {**state, "solution_candidates": solution_candidates}
