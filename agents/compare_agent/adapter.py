"""Agent 4(verification_results) → Agent 5(CompareState) 형식 변환."""

from __future__ import annotations

from agents.schemas.alert import BottleneckAlert
from agents.schemas.kpi import ToolGroupKPI
from agents.verification_agent.sim_executor import HORIZON_MIN


def _build_bottleneck_info(
    toolgroup: str,
    alert: BottleneckAlert,
    kpi_map: dict[str, ToolGroupKPI],
) -> dict:
    kpi = kpi_map.get(toolgroup)
    info: dict = {
        "tool_group": toolgroup,
        "risk_score": round(float(alert.composite_score), 4),
    }
    if kpi:
        info.update({
            "wip_count": int(getattr(kpi, "wip", 0) or 0),
            "avg_queue_time_min": round(float(getattr(kpi, "q_time_min", 0.0) or 0.0), 2),
            "wait_ratio": round(float(getattr(kpi, "wait_ratio", 0.0) or 0.0), 4),
            "available_tool_ratio": round(float(getattr(kpi, "available_tool_ratio", 0.0) or 0.0), 4),
            "utilization_avg": round(float(getattr(kpi, "utilization_avg", 0.0) or 0.0), 4),
        })
    return info


def _candidate_to_action_candidate(vc: dict) -> dict:
    """verified_candidate → CompareState action_candidate 형식 변환."""
    # action_kind: action_rows 첫 번째 항목에서 추출
    action_rows = vc.get("action_rows", [])
    action_kind = action_rows[0]["action_kind"] if action_rows else "UNKNOWN"

    # kpi_delta: kpi_stats에서 변환
    kpi_stats = vc.get("kpi_stats", {})
    kpi_delta = {
        "avg_queue_time_min": kpi_stats.get("q_time_min", {}).get("mean_delta", 0.0),
        "wip_count": int(round(kpi_stats.get("wip", {}).get("mean_delta", 0.0))),
        "throughput_delta": 0,  # kpi_toolgroup.csv 기반 paired 분석에 throughput 미포함
    }

    # simulation_confidence: 1 - p_value (p값이 낮을수록 신뢰도 높음)
    p_val = vc.get("paired_t_p")
    if p_val is not None:
        confidence = round(max(0.0, min(1.0, 1.0 - float(p_val))), 4)
    else:
        confidence = 0.5  # p값 없으면 중립

    return {
        "label": vc["label"],
        "action_kind": action_kind,
        "description": vc.get("name", ""),
        "simulation_confidence": confidence,
        "kpi_delta": kpi_delta,
        "baseline_metrics": {},   # paired 분석에서 baseline 절대값 미보유
        "whatif_metrics": {},     # paired 분석에서 whatif 절대값 미보유
        "kpi_stats": kpi_stats,   # paired t-test 상세 (Agent 5 참고용)
        "verdict": vc.get("verdict", "unknown"),
        "paired_n": vc.get("paired_n", 0),
        "paired_t_p": p_val,
    }


def build_compare_input(
    result_group: dict,
    alert: BottleneckAlert,
    kpi_map: dict[str, ToolGroupKPI],
    t0: float,
) -> dict:
    """
    verification_results 그룹 하나 → CompareState 입력 dict 변환.

    Args:
        result_group: verification_results의 툴그룹 단위 항목
        alert:        해당 툴그룹의 BottleneckAlert
        kpi_map:      toolgroup → ToolGroupKPI 매핑
        t0:           시뮬레이션 기준 시각
    """
    toolgroup = result_group["toolgroup"]

    action_candidates = [
        _candidate_to_action_candidate(vc)
        for vc in result_group.get("verified_candidates", [])
    ]

    return {
        "process_name": toolgroup,
        "severity": result_group.get("severity", ""),
        "snapshot_time": float(result_group.get("snapshot_time", t0)),
        "t0": float(t0),
        "horizon_min": HORIZON_MIN,
        "bottleneck_info": _build_bottleneck_info(toolgroup, alert, kpi_map),
        "tool_status": [],            # WIP 상세 미포함 → 빈 list
        "affected_lots_detail": [],   # Lot 상세 미포함 → 빈 list
        "action_candidates": action_candidates,
    }
