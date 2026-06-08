"""LangGraph 노드: alerts + cause_reports → 시뮬 입력용 글로벌 플랜 A / B."""

from agents.schemas.alert import SeverityLevel
from agents.solution_generator.llm_generator import refine_candidates
from agents.solution_generator.rule_engine import generate_candidates
from agents.solution_generator.llm_generator import refine_plan
from agents.solution_generator.rule_engine import generate_global_plans
from agents.state import PipelineState


def generate_solutions(state: PipelineState) -> PipelineState:
    alerts = state["alerts"]
    cause_map = {r.toolgroup: r for r in state["cause_reports"]}
    current_interval = state.get("current_release_interval")

    plans = generate_global_plans(alerts, cause_map, current_interval)
    plans = [refine_plan(plan, alerts, cause_map) for plan in plans]

    for alert in alerts:
        if alert.severity not in {SeverityLevel.CRITICAL, SeverityLevel.HIGH}:
            continue
        cause = cause_map.get(alert.toolgroup)
        if cause is None:
            continue

        candidates = generate_candidates(cause, alert)
        candidates = refine_candidates(alert, cause, candidates)

        all_candidates.append(
            {
                "toolgroup": alert.toolgroup,
                "severity": alert.severity.value,
                "composite_score": alert.composite_score,
                "candidates": [c.model_dump() for c in candidates],
            }
        )

    return {**state, "solution_candidates": all_candidates}
