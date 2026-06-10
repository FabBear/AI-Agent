from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agents.pipeline import build_pipeline
from agents.schemas.kpi import ToolGroupKPI
from app.services.agent_service import (
    _extract_summary,
    _initial_state,
    _reconstruct_phase2_state,
)


def test_extract_summary_counts_alerts() -> None:
    class Severity:
        value = "Critical"

    class Alert:
        severity = Severity()

    assert _extract_summary("cascade", {"alerts": [Alert()]}) == (
        "병목 알림 1건, CRITICAL 1건"
    )


def test_reconstruct_phase2_state_applies_selected_plan() -> None:
    pending = {
        "hitl_token": "token",
        "compare_formatted": [
            {
                "toolgroup": "Diffusion_FE_127",
                "recommendation": {"reason": "기존 추천"},
                "action_effects": [],
            }
        ],
        "alerts": [],
        "kpi_snapshot": [],
        "prev_kpi_snapshot": [],
        "cause_reports": [],
        "solution_candidates": [],
    }
    selected_plan = {"plan_seq": 2}

    state = _reconstruct_phase2_state(
        pending,
        selected_plan,
        uuid4(),
        datetime(2026, 6, 10, tzinfo=UTC),
        "승인",
    )

    assert state["compare_results"][0]["recommendation"]["action_label"] == "B"
    assert state["hitl_approved"] is True


def test_initial_state_uses_requested_bottleneck_without_redetection() -> None:
    kpi = ToolGroupKPI(
        toolgroup="Diffusion_FE_127",
        snapshot_time=120.0,
        available_tool_ratio=1.0,
        q_time_min=20.0,
        wait_ratio=0.5,
        wip=10.0,
        setup_ratio_avg=0.1,
        utilization_avg=0.8,
        max_avg_q_time=30.0,
        max_util=0.9,
    )

    state = _initial_state([kpi], [], "Diffusion_FE_127", 120.0, 0.91)

    assert len(state["potential_bottlenecks"]) == 1
    assert state["potential_bottlenecks"][0].toolgroup == "Diffusion_FE_127"
    assert state["potential_bottlenecks"][0].probability == 0.91


def test_initial_state_rejects_unknown_target_tg() -> None:
    with pytest.raises(ValueError, match="요청한 TG의 KPI"):
        _initial_state([], [], "Unknown_TG", 120.0, 0.91)


def test_pipeline_can_start_from_cascade() -> None:
    pipeline = build_pipeline(
        csv_dir="../Simulation/simulation/sim_csv_out",
        run_sim=False,
        run_g_star=False,
        phase1_only=True,
        run_detection=False,
    )

    start_edges = [
        edge for edge in pipeline.get_graph().edges if edge.source == "__start__"
    ]

    assert [edge.target for edge in start_edges] == ["cascade"]
