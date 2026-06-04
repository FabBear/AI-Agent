from agents.bottleneck_detector.detector import detect
from agents.state import PipelineState


def detect_bottlenecks(state: PipelineState) -> PipelineState:
    return {**state, "potential_bottlenecks": detect(state["kpi_snapshot"])}
