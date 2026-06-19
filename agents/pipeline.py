"""Agent 1~6 파이프라인 — 단일 LangGraph StateGraph."""

from __future__ import annotations

import asyncio
import functools
import os
import subprocess
import time
from pathlib import Path

from langgraph.graph import END, StateGraph

from agents.bottleneck_detector.node import detect_bottlenecks
from agents.cascade_analyzer.node import analyze_cascade
from agents.cause_analyzer.node import analyze_cause
from agents.logger import get_logger
from agents.solution_generator.node import generate_solutions
from agents.stage_writer import emit_stage1, emit_stage2, emit_stage3
from agents.verification_agent.node import verify_solutions
from agents.compare_agent.node import (
    compare_hitl,
    compare_llm,
    compare_rag,
    compare_rank,
)
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
_ML_G_STAR = _SIM_ROOT / "tools" / "ml_g_star_at_t0.py"
_TRIGGER_FWD = _SIM_ROOT / "tools" / "trigger_forward_pipeline.py"
_G_STAR_OUT = _SIM_ROOT / "out" / "ml_g_star_e2e"


def _read_run_id(csv_dir: Path) -> str:
    import csv
    path = csv_dir / "lot_events.csv"
    if not path.exists():
        _poll_interval = int(os.environ.get("G_STAR_LOT_POLL_SEC", "10"))
        _poll_max = int(os.environ.get("G_STAR_LOT_TIMEOUT_SEC", "60"))
        _waited = 0
        while not path.exists() and _waited < _poll_max:
            _log.info(f"[G*] lot_events.csv 대기 중 ({_waited}/{_poll_max}s)...")
            time.sleep(_poll_interval)
            _waited += _poll_interval
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
    alerts = state.get("alerts", [])
    if not alerts:
        return END
    if not any(a.severity.value.upper() == "CRITICAL" for a in alerts):
        return END
    return "g_star"


async def _run_g_star(state: PipelineState) -> PipelineState:
    """Critical/High 알림 발생 시 G* 파이프라인을 실행한다."""
    # T0 = kpi_snapshot의 time_step (시뮬 tick 기준, epoch-minutes 아님)
    kpi_snapshot = state["kpi_snapshot"]
    if not kpi_snapshot:
        _log.warning("[G*] kpi_snapshot 없음 — G*를 스킵합니다.")
        return state
    snapshot_time = kpi_snapshot[0].snapshot_time
    alerts = state["alerts"]
    anchor = max(alerts, key=lambda a: a.composite_score).toolgroup if alerts else ""
    scenario_id = f"FWD_G_STAR_T{int(snapshot_time)}"

    # T0 기반 동적 경로: G* 결과 + WHATIF baseline manifest 모두 여기에 저장
    fwd_base_dir = _SIM_CSV / f"fwd_base_t{int(snapshot_time)}"
    fwd_base_dir.mkdir(parents=True, exist_ok=True)
    g_star_file = fwd_base_dir / f"g_star_T{int(snapshot_time)}.json"

    run_id = await asyncio.to_thread(_read_run_id, _SIM_CSV)
    if not run_id:
        _log.warning("[G*] sim_csv_out/lot_events.csv에서 run_id를 읽을 수 없어 G*를 스킵합니다.")
        return state

    # 동일 t0에서 G*가 이미 완료됐으면 재사용.
    # 다른 프로세스가 실행 중이면 lock이 해제될 때까지 대기(동시 N개 중복 실행 방지).
    lock_file = fwd_base_dir / f".g_star_t{int(snapshot_time)}.lock"
    if not g_star_file.exists():
        if lock_file.exists():
            # Step1 다른 파이프라인이 실행 중 — 완료될 때까지 폴링
            _step1_wait_max = int(os.environ.get("G_STAR_STEP1_WAIT_SEC", "360"))
            _waited = 0
            _log.info(f"[G*] t0={snapshot_time} Step1 다른 프로세스 실행 중 — 완료 대기 (max {_step1_wait_max}s)")
            while lock_file.exists() and not g_star_file.exists() and _waited < _step1_wait_max:
                await asyncio.sleep(15)
                _waited += 15
            if not g_star_file.exists():
                _log.warning(f"[G*] Step1 대기 타임아웃 ({_step1_wait_max}s) — G* 스킵")
                return state
            _log.info(f"[G*] Step1 완료 확인 — Step2 진행")
        else:
            # 이 프로세스가 Step1 담당 — lock 생성
            lock_file.touch()
            _log.info(f"[G*] 실행 중... t0={snapshot_time}, anchor={anchor}, out={fwd_base_dir.name}")
            try:
                r = await asyncio.to_thread(
                    subprocess.run,
                    [str(_SIM_PY), str(_ML_G_STAR),
                     "--train-csv-dir", str(_SIM_CSV),
                     "--inference-csv-dir", str(_SIM_CSV),
                     "--t0", str(int(snapshot_time)),
                     "--out-dir", str(fwd_base_dir),
                     "--alarm-threshold", "0.7",
                     "--snapshot-stride", "10",
                     "--shap-top-k", "0"],
                    capture_output=True, text=True,
                    timeout=300, cwd=str(_SIM_ROOT),
                )
                if r.returncode == 0:
                    _log.info(f"[G*] Step1 완료 — {g_star_file.name}")
                else:
                    _log.warning(f"[G*] Step1 실패: {r.stderr[-200:]}")
                    return state
            except Exception as e:
                _log.warning(f"[G*] Step1 스킵: {e}")
                return state
            finally:
                lock_file.unlink(missing_ok=True)
    else:
        _log.info(f"[G*] 캐시 히트 — {g_star_file.name} 재사용, Step2로 바로 진행")

    # Step 2: trigger_forward_pipeline (Monte Carlo + t-test) — DB 모드
    # runs_manifest.csv + agent_handoff_g_star_analysis.json 모두 fwd_base_dir에 저장
    step2_lock = fwd_base_dir / f".g_star_step2_t{int(snapshot_time)}.lock"
    manifest_path = fwd_base_dir / "runs_manifest.csv"

    if manifest_path.exists():
        _log.info(f"[G*] Step2 이미 완료 — runs_manifest.csv 재사용")
    elif step2_lock.exists():
        # Step2 다른 파이프라인이 실행 중 — lock 해제될 때까지 폴링 (성공/실패 무관)
        _step2_wait_max = int(os.environ.get("G_STAR_STEP2_WAIT_SEC", "600"))
        _waited = 0
        _log.info(f"[G*] Step2 다른 프로세스 실행 중 — lock 해제 대기 (max {_step2_wait_max}s)")
        while step2_lock.exists() and _waited < _step2_wait_max:
            await asyncio.sleep(15)
            _waited += 15
        if manifest_path.exists():
            _log.info(f"[G*] Step2 완료 확인 — runs_manifest.csv 사용")
        else:
            _log.warning(f"[G*] Step2 실패 또는 타임아웃 ({_step2_wait_max}s) — 검증 단계에서 fallback")
    else:
        step2_lock.touch()
        n_runs = os.environ.get("G_STAR_N_RUNS", "30")
        # 베이스라인 forward sim 동시 실행 수. 케이스 순차(max_concurrent_pipelines=1) + max_connections=200
        # 환경에선 한 번에 G* Step2 하나만 도므로 8-parallel이 안전하다. 3으로 낮추면 30런이 600s를
        # 못 끝내고 타임아웃→manifest 미생성→cause fallback/solution 빈 release/verify 재생성으로 줄줄이 degrade.
        g_star_parallel = os.environ.get("G_STAR_PARALLEL", "8")
        try:
            r = await asyncio.to_thread(
                subprocess.run,
                [str(_SIM_PY), str(_TRIGGER_FWD),
                 "--source", "db",
                 "--run-id", run_id,
                 "--t0", str(int(snapshot_time)),
                 "--horizon", "120",
                 "--scenario-id", scenario_id,
                 "--g-star-file", str(g_star_file),
                 "--anchor-tg", anchor,
                 "--n-runs", n_runs,
                 "--parallel", g_star_parallel,
                 "--out-dir", str(fwd_base_dir),
                 "--baseline-csv-dir", str(_SIM_CSV),
                 "--skip-sim-if-manifest-exists"],
                capture_output=True, text=True,
                timeout=int(os.environ.get("G_STAR_STEP2_TIMEOUT_SEC", "900")), cwd=str(_SIM_ROOT),
            )
            if r.returncode == 0:
                _log.info(f"[G*] Step2 완료 — {fwd_base_dir.name}/agent_handoff_g_star_analysis.json")
            else:
                _log.warning(f"[G*] Step2 실패: {r.stdout[-300:]}\n{r.stderr[-300:]}")
        except Exception as e:
            _log.warning(f"[G*] Step2 스킵: {e}")
        finally:
            step2_lock.unlink(missing_ok=True)

    return state


def build_pipeline(
    run_sim: bool = True,
    run_g_star: bool = True,
    stop_at_hitl: bool | None = None,
    run_detection: bool = True,
    run_cascade: bool = True,
):
    """파이프라인 그래프를 빌드하고 컴파일된 그래프를 반환한다.

    stop_at_hitl=True  : detect → … → compare_hitl → END  (Webhook 모드 — HITL 승인 대기)
    stop_at_hitl=False : detect → … → compare_hitl → report_* → END  (일체형)
    stop_at_hitl=None  : WEBHOOK_MODE 환경변수로 자동 결정
    run_detection=False: 입력 state의 potential_bottlenecks를 사용해 cascade부터 실행
    run_cascade=False : cascade를 외부에서 이미 실행한 경우 g_star부터 시작
    """
    import os as _os
    if stop_at_hitl is None:
        stop_at_hitl = _os.environ.get("WEBHOOK_MODE", "").lower() in ("1", "true", "yes")

    g = StateGraph(PipelineState)

    if run_cascade:
        # Agent 1·2: 감지·확산 영향 (run_cascade=False이면 Phase 1에서 이미 완료)
        g.add_node("detect", detect_bottlenecks)
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
    g.add_node("compare_rag",  compare_rag)
    g.add_node("compare_llm",  compare_llm)
    g.add_node("compare_hitl", compare_hitl)

    if not run_cascade:
        # cascade가 Phase 1에서 이미 완료됨 — g_star부터 시작
        g.set_entry_point("g_star")
    elif run_detection:
        g.set_entry_point("detect")
        g.add_edge("detect", "cascade")
    else:
        g.set_entry_point("cascade")
        g.add_conditional_edges("cascade", _no_alerts, {"g_star": "g_star", END: END})
    g.add_edge("g_star",   "emit1")
    g.add_edge("emit1",    "cause")
    g.add_edge("cause",    "emit2")
    g.add_edge("emit2",    "solution")
    g.add_edge("solution", "verify")
    g.add_edge("verify",       "compare_rank")
    g.add_edge("compare_rank", "compare_rag")
    g.add_edge("compare_rag",  "compare_llm")
    g.add_edge("compare_llm",  "compare_hitl")

    g.add_node("emit3", emit_stage3)           # Stage 3 JSON 저장

    if stop_at_hitl:
        # Webhook 모드: compare_hitl → emit3 → END (보고서는 HITL 승인 후 별도 실행)
        g.add_edge("compare_hitl", "emit3")
        g.add_edge("emit3", END)
    else:
        # 일체형: compare_hitl → 보고서 생성
        g.add_node("report_prepare",   report_prepare)
        g.add_node("report_summary",   report_summary)
        g.add_node("report_diffusion", report_diffusion)
        g.add_node("report_cause",     report_cause)
        g.add_node("report_actions",   report_actions)
        g.add_node("report_save",      report_save)

        g.add_edge("compare_hitl",     "emit3")
        g.add_edge("emit3",            "report_prepare")
        g.add_edge("report_prepare",   "report_summary")
        g.add_edge("report_summary",   "report_diffusion")
        g.add_edge("report_diffusion", "report_cause")
        g.add_edge("report_cause",     "report_actions")
        g.add_edge("report_actions",   "report_save")
        g.add_edge("report_save",      END)

    return g.compile()


def build_report_pipeline():
    """보고서 생성 파이프라인: HITL 승인 후 PipelineState를 재구성한 뒤 보고서 노드만 실행."""
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
