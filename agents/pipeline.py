"""Agent 1~6 파이프라인 — 단일 LangGraph StateGraph."""

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
from agents.stage_writer import emit_stage1, emit_stage2
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

_log = get_logger(__name__)

_SIM_ROOT = Path(__file__).parent.parent.parent / "Simulation" / "simulation"
_SIM_CSV = _SIM_ROOT / "sim_csv_out"
_SIM_PY = _SIM_ROOT / ".venv" / "bin" / "python"
_TRIGGER_FWD = _SIM_ROOT / "tools" / "trigger_forward_pipeline.py"


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
    return END if not state["alerts"] else "emit1"


def _run_g_star(state: PipelineState) -> PipelineState:
    """병목 감지 에이전트의 CRITICAL/HIGH TG를 기반으로 G* 통계 검정 파이프라인을 실행한다.

    ML 임계값 대신 이미 감지된 병목 TG를 직접 사용하므로,
    원인 분석 에이전트가 항상 G* 통계 근거를 가진 상태로 진입할 수 있다.
    """
    import json as _json
    from agents.schemas.alert import SeverityLevel

    kpi_snapshot = state["kpi_snapshot"]
    if not kpi_snapshot:
        _log.warning("[G*] kpi_snapshot 없음 — G*를 스킵합니다.")
        return state

    snapshot_time = kpi_snapshot[0].snapshot_time
    alerts = state["alerts"]

    # CRITICAL TG만 G* 분석 대상 (composite_score 내림차순)
    target_alerts = sorted(
        [a for a in alerts if a.severity == SeverityLevel.CRITICAL],
        key=lambda a: -a.composite_score,
    )
    if not target_alerts:
        _log.info("[G*] CRITICAL 알림 없음 — G*를 스킵합니다.")
        return state

    anchor = target_alerts[0].toolgroup
    target_tgs = [a.toolgroup for a in target_alerts]
    scenario_id = f"FWD_G_STAR_T{int(snapshot_time)}"

    fwd_base_dir = _SIM_CSV / f"fwd_base_t{int(snapshot_time)}"
    fwd_base_dir.mkdir(parents=True, exist_ok=True)
    g_star_file = fwd_base_dir / f"g_star_T{int(snapshot_time)}.json"

    run_id = _read_run_id(_SIM_CSV)
    if not run_id:
        _log.warning("[G*] run_id 없음 — G*를 스킵합니다.")
        return state

    # Step 1: 병목 감지 결과로 g_star JSON 직접 생성 (ML 임계값 대신)
    g_star_data = {
        "anchor_tg": anchor,
        "toolgroups": target_tgs,
        "n_g_star": len(target_tgs),
        "t0_sim_minute": snapshot_time,
        "source": "bottleneck_detector",
    }
    g_star_file.write_text(
        _json.dumps(g_star_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log.info(
        f"[G*] Step1 완료 — {len(target_tgs)}개 TG (CRITICAL/HIGH): {target_tgs}"
    )

    # Step 2: trigger_forward_pipeline (Monte Carlo 30회 + t-test)
    try:
        r = subprocess.run(
            [str(_SIM_PY), str(_TRIGGER_FWD),
             "--source", "db",
             "--run-id", run_id,
             "--t0", str(int(snapshot_time)),
             "--horizon", "120",
             "--scenario-id", scenario_id,
             "--g-star-file", str(g_star_file),
             "--anchor-tg", anchor,
             "--n-runs", "30",
             "--parallel", "8",
             "--out-dir", str(fwd_base_dir),
             "--baseline-csv-dir", str(_SIM_CSV),
             "--skip-sim-if-manifest-exists"],
            capture_output=True, text=True,
            timeout=600, cwd=str(_SIM_ROOT),
        )
        if r.returncode == 0:
            _log.info(f"[G*] Step2 완료 — {fwd_base_dir.name}/agent_handoff_g_star_analysis.json")
        else:
            _log.warning(f"[G*] Step2 실패: {r.stdout[-300:]}\n{r.stderr[-300:]}")
    except Exception as e:
        _log.warning(f"[G*] Step2 스킵: {e}")

    return state


def build_pipeline(
    run_sim: bool = True,
    run_g_star: bool = True,
    phase1_only: bool | None = None,
    run_detection: bool = True,
):
    """파이프라인 그래프를 빌드하고 컴파일된 그래프를 반환한다.

    phase1_only=True  : detect → … → compare_hitl → END  (Webhook 모드 Phase 1)
    phase1_only=False : detect → … → compare_hitl → report_* → END  (일체형)
    phase1_only=None  : WEBHOOK_MODE 환경변수로 자동 결정
    run_detection=False: 입력 state의 potential_bottlenecks를 사용해 cascade부터 실행
    """
    import os as _os
    if phase1_only is None:
        phase1_only = _os.environ.get("WEBHOOK_MODE", "").lower() in ("1", "true", "yes")

    g = StateGraph(PipelineState)

    # Agent 1: 병목 감지
    g.add_node("detect", detect_bottlenecks)
    # Agent 2: 확산 영향
    g.add_node("cascade", analyze_cascade)
    g.add_node("emit1", emit_stage1)          # Stage 1 JSON 저장
    # G* 분석 (cascade 이후, cause 이전)
    g.add_node("g_star", _run_g_star if run_g_star else lambda s: s)
    # Agent 3: 원인 분석 + 대응안 생성
    g.add_node("cause", functools.partial(analyze_cause, run_sim=run_sim))
    g.add_node("emit2", emit_stage2)          # Stage 2 JSON 저장
    g.add_node("solution", generate_solutions)
    # Agent 4: 대응안 효과 검증
    g.add_node("verify", verify_solutions)
    # Agent 5: 대응안 비교분석
    g.add_node("compare_rank", compare_rank)
    g.add_node("compare_llm",  compare_llm)
    g.add_node("compare_hitl", compare_hitl)

    if run_detection:
        g.set_entry_point("detect")
        g.add_edge("detect", "cascade")
    else:
        g.set_entry_point("cascade")
    g.add_conditional_edges("cascade", _no_alerts, {"emit1": "emit1", END: END})
    g.add_edge("emit1",    "g_star")
    g.add_edge("g_star",   "cause")
    g.add_edge("cause",    "emit2")
    g.add_edge("emit2",    "solution")
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
