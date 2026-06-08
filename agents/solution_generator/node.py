"""LangGraph 노드: alerts + cause_reports → 시뮬 입력용 글로벌 플랜 A / B."""

from agents.solution_generator.llm_generator import refine_plan
from agents.solution_generator.rule_engine import generate_global_plans
from agents.state import PipelineState


def generate_solutions(state: PipelineState) -> PipelineState:
    alerts = state["alerts"]
    cause_map = {r.toolgroup: r for r in state["cause_reports"]}
    current_interval = state.get("current_release_interval")

    plans = generate_global_plans(alerts, cause_map, current_interval)
    plans = [refine_plan(plan, alerts, cause_map) for plan in plans]

    return {**state, "solution_candidates": [p.model_dump() for p in plans]}
