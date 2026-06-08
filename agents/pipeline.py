"""Agent 1~6 파이프라인 — 단일 LangGraph StateGraph."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.bottleneck_detector.node import detect_bottlenecks
from agents.cascade_analyzer.node import analyze_cascade
from agents.cause_analyzer.node import analyze_cause
from agents.solution_generator.node import generate_solutions
from agents.verification_agent.node import verify_solutions
from agents.compare_agent.node import compare_rank, compare_llm, compare_hitl
from agents.report_agent.node import (
    report_prepare,
    report_summary,
    report_diffusion,
    report_cause,
    report_actions,
    report_save,
)
from agents.state import PipelineState
from agents.logger import get_logger

_log = get_logger(__name__)

_SIM_ROOT = Path(__file__).parent.parent.parent / "Simulation" / "simulation"
_SIM_CSV = _SIM_ROOT / "sim_csv_out"
_SIM_PY = _SIM_ROOT / ".venv" / "bin" / "python"
_ML_G_STAR = _SIM_ROOT / "tools" / "ml_g_star_at_t0.py"
_TRIGGER_FWD = _SIM_ROOT / "tools" / "trigger_forward_pipeline.py"
_G_STAR_OUT = _SIM_ROOT / "out" / "ml_g_star_e2e"


def _read_run_id(csv_dir: Path) -> str:
    import csv
    path = csv_dir / "lot_events.csv"
    if not path.exists():
        return ""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if run_id:
                return run_id
    return ""


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

    run_id = _read_run_id(_SIM_CSV)
    if not run_id:
        _log.warning("[G*] sim_csv_out/lot_events.csv에서 run_id를 읽을 수 없어 G*를 스킵합니다.")
        return state

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
             "--run-id", run_id,
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
    phase1_only: bool | None = None,
):
    """파이프라인 그래프를 빌드하고 컴파일된 그래프를 반환한다.

    phase1_only=True  : detect → … → compare_hitl → END  (Webhook 모드 Phase 1)
    phase1_only=False : detect → … → compare_hitl → report_* → END  (일체형)
    phase1_only=None  : WEBHOOK_MODE 환경변수로 자동 결정
    """
    import os as _os
    if phase1_only is None:
        phase1_only = _os.environ.get("WEBHOOK_MODE", "").lower() in ("1", "true", "yes")

    g = StateGraph(PipelineState)

    # Agent 1: 병목 감지
    g.add_node("detect", detect_bottlenecks)
    # Agent 2: 확산 영향
    g.add_node("cascade", functools.partial(analyze_cascade, csv_dir=csv_dir))
    # G* 분석 (cascade 이후, cause 이전)
    g.add_node("g_star", _run_g_star if run_g_star else lambda s: s)
    # Agent 3: 원인 분석 + 대응안 생성
    g.add_node("cause",    functools.partial(analyze_cause, csv_dir=csv_dir, run_sim=run_sim))
    g.add_node("solution", generate_solutions)
    # Agent 4: 대응안 효과 검증
    g.add_node("verify", verify_solutions)
    # Agent 5: 대응안 비교분석
    g.add_node("compare_rank", compare_rank)
    g.add_node("compare_llm",  compare_llm)
    g.add_node("compare_hitl", compare_hitl)

    g.set_entry_point("detect")
    g.add_edge("detect",   "cascade")
    g.add_conditional_edges("cascade", _no_alerts, {"g_star": "g_star", END: END})
    g.add_edge("g_star",   "cause")
    g.add_edge("cause",    "solution")
    g.add_edge("solution", "verify")
    g.add_edge("verify",       "compare_rank")
    g.add_edge("compare_rank", "compare_llm")
    g.add_edge("compare_llm",  "compare_hitl")

    if phase1_only:
        # Phase 1 종료: compare_hitl → END
        g.add_edge("compare_hitl", END)
    else:
        # 일체형: compare_hitl → 보고서 생성
        g.add_node("report_prepare",   report_prepare)
        g.add_node("report_summary",   report_summary)
        g.add_node("report_diffusion", report_diffusion)
        g.add_node("report_cause",     report_cause)
        g.add_node("report_actions",   report_actions)
        g.add_node("report_save",      report_save)

        g.add_edge("compare_hitl",     "report_prepare")
        g.add_edge("report_prepare",   "report_summary")
        g.add_edge("report_summary",   "report_diffusion")
        g.add_edge("report_diffusion", "report_cause")
        g.add_edge("report_cause",     "report_actions")
        g.add_edge("report_actions",   "report_save")
        g.add_edge("report_save",      END)

    return g.compile()


def build_phase2_pipeline():
    """Phase 2 파이프라인: PipelineState를 재구성한 뒤 보고서 노드만 실행."""
    g = StateGraph(PipelineState)

    g.add_node("report_prepare",   report_prepare)
    g.add_node("report_summary",   report_summary)
    g.add_node("report_diffusion", report_diffusion)
    g.add_node("report_cause",     report_cause)
    g.add_node("report_actions",   report_actions)
    g.add_node("report_save",      report_save)

    g.set_entry_point("report_prepare")
    g.add_edge("report_prepare",   "report_summary")
    g.add_edge("report_summary",   "report_diffusion")
    g.add_edge("report_diffusion", "report_cause")
    g.add_edge("report_cause",     "report_actions")
    g.add_edge("report_actions",   "report_save")
    g.add_edge("report_save",      END)

    return g.compile()
