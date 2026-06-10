"""LangGraph 노드: potential_bottlenecks + kpi_snapshot → alerts (심각도 결정)."""

from agents import config
from agents.cascade_analyzer.dag_builder import build_dag, get_downstream_tgs
from agents.cascade_analyzer.impact_calculator import compute_impact
from agents.cascade_analyzer.scorer import build_alert
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.state import PipelineState

def analyze_cascade(state: PipelineState) -> PipelineState:
    potential = state["potential_bottlenecks"]
    kpi_map = {k.toolgroup: k for k in state["kpi_snapshot"]}

    G = build_dag()

    alerts: list[BottleneckAlert] = []
    for pb in potential:
        downstream = get_downstream_tgs(G, pb.toolgroup, config.MAX_CASCADE_HOPS)
        impact = compute_impact(kpi_map[pb.toolgroup], downstream, kpi_map)
        alerts.append(build_alert(pb, impact))

    _SEVERITY_ORDER = [s.value for s in SeverityLevel]
    alerts.sort(key=lambda a: (_SEVERITY_ORDER.index(a.severity.value), -a.composite_score))

    return {**state, "alerts": alerts}
