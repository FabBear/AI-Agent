"""AG-001~004 파이프라인을 LangGraph StateGraph로 연결한다."""

from __future__ import annotations

import functools
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.bottleneck_detector.node import detect_bottlenecks
from agents.cascade_analyzer.node import analyze_cascade
from agents.cause_analyzer.node import analyze_cause
from agents.solution_generator.node import generate_solutions
from agents.state import PipelineState


def _no_alerts(state: PipelineState) -> str:
    return END if not state["alerts"] else "cause"


def build_pipeline(
    csv_dir: str | Path,
    run_sim: bool = True,
):
    """파이프라인 그래프를 빌드하고 컴파일된 그래프를 반환한다."""
    g = StateGraph(PipelineState)

    g.add_node("detect", detect_bottlenecks)
    g.add_node("cascade", functools.partial(analyze_cascade, csv_dir=csv_dir))
    g.add_node("cause", functools.partial(analyze_cause, csv_dir=csv_dir, run_sim=run_sim))
    g.add_node("solution", generate_solutions)

    g.set_entry_point("detect")
    g.add_edge("detect", "cascade")
    g.add_conditional_edges("cascade", _no_alerts, {"cause": "cause", END: END})
    g.add_edge("cause", "solution")
    g.add_edge("solution", END)

    return g.compile()
