"""AG-001~004 파이프라인을 LangGraph StateGraph로 연결한다."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.bottleneck_detector.node import detect_bottlenecks
from agents.cascade_analyzer.node import analyze_cascade
from agents.cause_analyzer.node import analyze_cause
from agents.logger import get_logger
from agents.solution_generator.node import generate_solutions
from agents.state import PipelineState

_log = get_logger(__name__)

_SIM_ROOT = Path(__file__).parent.parent.parent / "Simulation" / "simulation"
_SIM_CSV = _SIM_ROOT / "sim_csv_out"
_SIM_PY = _SIM_ROOT / ".venv" / "bin" / "python"
_ML_G_STAR = _SIM_ROOT / "tools" / "ml_g_star_at_t0.py"
_TRIGGER_FWD = _SIM_ROOT / "tools" / "trigger_forward_pipeline.py"
_G_STAR_OUT = _SIM_ROOT / "out" / "ml_g_star_e2e"


def _no_alerts(state: PipelineState) -> str:
    return END if not state["alerts"] else "g_star"


def _run_g_star(state: PipelineState) -> PipelineState:
    """Critical/High 알림 발생 시 G* 파이프라인을 실행한다."""
    if not state["kpi_snapshot"]:
        _log.warning("[G*] kpi_snapshot이 비어 있어 G*를 실행할 수 없습니다.")
        return state
    snapshot_time = state["kpi_snapshot"][0].snapshot_time
    alerts = state["alerts"]
    anchor = max(alerts, key=lambda a: a.composite_score).toolgroup if alerts else ""
    scenario_id = f"FWD_G_STAR_T{int(snapshot_time)}"
    g_star_file = _G_STAR_OUT / f"g_star_T{int(snapshot_time)}.json"

    _log.info(f"[G*] 실행 중... t0={snapshot_time}, anchor={anchor}")

    # Step 1: ML G* at T0
    try:
        r = subprocess.run(
            [str(_SIM_PY), str(_ML_G_STAR),
             "--train-csv-dir", str(_SIM_CSV),
             "--inference-csv-dir", str(_SIM_CSV),
             "--t0", str(int(snapshot_time)),
             "--out-dir", str(_G_STAR_OUT),
             "--alarm-threshold", "0.7",
             "--snapshot-stride", "10",
             "--shap-top-k", "0"],
            capture_output=True, text=True,
            timeout=120, cwd=str(_SIM_ROOT),
        )
        if r.returncode == 0:
            _log.info(f"[G*] Step1 완료 — {g_star_file.name}")
        else:
            _log.warning(f"[G*] Step1 실패: {r.stderr[-200:]}")
            return state
    except Exception as e:
        _log.warning(f"[G*] Step1 스킵: {e}")
        return state

    # Step 2: trigger_forward_pipeline (Monte Carlo + t-test)
    try:
        r = subprocess.run(
            [str(_SIM_PY), str(_TRIGGER_FWD),
             "--sim-csv-dir", str(_SIM_CSV),
             "--run-id", scenario_id,
             "--t0", str(int(snapshot_time)),
             "--horizon", "120",
             "--scenario-id", scenario_id,
             "--g-star-file", str(g_star_file),
             "--baseline-csv-dir", str(_SIM_CSV),
             "--anchor-tg", anchor,
             "--n-runs", "5",
             "--parallel", "4",
             "--out-dir", str(_G_STAR_OUT),
             "--skip-sim-if-manifest-exists"],
            capture_output=True, text=True,
            timeout=600, cwd=str(_SIM_ROOT),
        )
        if r.returncode == 0:
            _log.info("[G*] Step2 완료 — agent_handoff_g_star_analysis.json 생성")
        else:
            _log.warning(f"[G*] Step2 실패: {r.stdout[-300:]}\n{r.stderr[-300:]}")
    except Exception as e:
        _log.warning(f"[G*] Step2 스킵: {e}")

    return state


def build_pipeline(
    csv_dir: str | Path,
    run_sim: bool = True,
    run_g_star: bool = True,
):
    """파이프라인 그래프를 빌드하고 컴파일된 그래프를 반환한다."""
    g = StateGraph(PipelineState)

    g.add_node("detect", detect_bottlenecks)
    g.add_node("cascade", functools.partial(analyze_cascade, csv_dir=csv_dir))
    g.add_node("g_star", _run_g_star if run_g_star else lambda s: s)
    g.add_node("cause", functools.partial(analyze_cause, csv_dir=csv_dir, run_sim=run_sim))
    g.add_node("solution", generate_solutions)

    g.set_entry_point("detect")
    g.add_edge("detect", "cascade")
    g.add_conditional_edges("cascade", _no_alerts, {"g_star": "g_star", END: END})
    g.add_edge("g_star", "cause")
    g.add_edge("cause", "solution")
    g.add_edge("solution", END)

    return g.compile()
