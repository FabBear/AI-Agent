import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import asyncpg

from agents.cascade_analyzer.node import analyze_cascade
from agents.data.kpi_loader import load_kpi_snapshot, load_kpi_window
from agents.pipeline import build_pipeline, build_report_pipeline
from agents.schemas.alert import BottleneckAlert, PotentialBottleneck
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI
from app.config import get_settings
from app.repositories.action_plan_repository import ActionPlanRepository
from app.repositories.agent_step_repository import AgentStepRepository
from app.repositories.cause_analysis_repository import CauseAnalysisRepository
from app.repositories.ml_model_repository import MlModelRepository
from app.repositories.response_report_repository import ResponseReportRepository
from app.services.spring_client import SpringClient, get_spring_client

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_HITL_PENDING_DIR = _ROOT / "hitl_pending"
_SIM_EPOCH_UTC = datetime(2019, 12, 31, 15, 0, tzinfo=UTC)

_pipeline_semaphore: asyncio.Semaphore | None = None
# HITL 승인 후 보고서 생성은 시뮬레이션 파이프라인과 스레드 풀을 공유하지 않도록 전용 executor 사용
_hitl_report_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hitl-report")


async def run_pipeline_with_timeout(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    snapshot_time: float,
    bottleneck_prob: float,
    risk_grade: str,
    pool: asyncpg.Pool,
) -> None:
    global _pipeline_semaphore
    settings = get_settings()
    if _pipeline_semaphore is None:
        _pipeline_semaphore = asyncio.Semaphore(settings.max_concurrent_pipelines)

    # Phase 1: cascade 실행 (세마포어 없이 — 모든 케이스가 즉시 확산영향분석을 완료한다)
    # CRITICAL이면 cascade state를 반환하고 Phase 2로 진행, 아니면 cascade-only로 종료
    cascade_state = await _run_cascade_phase(
        case_id, tg_id, tg_code, snapshot_time, bottleneck_prob, risk_grade, pool
    )
    if cascade_state is None:
        return

    # Phase 2: 원인/대응 분석 (세마포어 게이트 — 동시 실행 수 제한)
    # Phase 1에서 CAUSE_ANALYSIS가 IN_PROGRESS로 마킹됐으므로 화면에 "원인분석 대기중"이 표시된다
    async with _pipeline_semaphore:
        try:
            await asyncio.wait_for(
                run_pipeline(case_id, tg_id, tg_code, bottleneck_prob, risk_grade, cascade_state, pool),
                timeout=settings.pipeline_timeout_sec,
            )
        except TimeoutError:
            step_repo = AgentStepRepository(pool)
            failed_step = await step_repo.mark_active_failed(case_id, "파이프라인 타임아웃")
            await get_spring_client(settings).notify_agent_step(
                case_id,
                failed_step or "CAUSE_ANALYSIS",
                "파이프라인 타임아웃",
                status="FAILED",
            )
            logger.exception("Agent pipeline timeout: case_id=%s", case_id)
        except Exception:
            logger.exception("Agent pipeline failed: case_id=%s", case_id)


async def _run_cascade_phase(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    snapshot_time: float,
    bottleneck_prob: float,
    risk_grade: str,
    pool: asyncpg.Pool,
) -> dict | None:
    """Phase 1: init_steps + cascade 실행 (세마포어 없이).

    - 모든 케이스가 즉시 확산영향분석을 완료한다.
    - CRITICAL 확정 시 cascade state를 반환 → Phase 2(세마포어 게이트)로 전달.
    - non-CRITICAL 또는 오류 시 None 반환 → Phase 2 건너뜀.
    """
    step_repo = AgentStepRepository(pool)
    spring_client = get_spring_client(get_settings())

    await step_repo.init_steps(case_id)
    await step_repo.mark_in_progress(case_id, "cascade")

    try:
        kpi_list = await asyncio.to_thread(load_kpi_snapshot, snapshot_time)
        window = await asyncio.to_thread(load_kpi_window, snapshot_time, 2)
        window_times = sorted(window)
        prev_kpi = window[window_times[0]] if len(window_times) >= 2 else []
        initial_state = _initial_state(kpi_list, prev_kpi, tg_code, snapshot_time, bottleneck_prob)
    except Exception as exc:
        await step_repo.mark_failed(case_id, "cascade", str(exc))
        await spring_client.notify_agent_step(case_id, "cascade", str(exc)[:200], status="FAILED")
        logger.exception("Phase1 KPI 로딩 실패: case_id=%s", case_id)
        return None

    # ML=HIGH: cascade-only (원인분석 불필요)
    if risk_grade != "CRITICAL":
        try:
            await _run_cascade_only(
                case_id, tg_id, tg_code, bottleneck_prob, risk_grade, initial_state, step_repo, spring_client
            )
        except Exception as exc:
            await step_repo.mark_failed(case_id, "cascade", str(exc))
            await spring_client.notify_agent_step(case_id, "cascade", str(exc)[:200], status="FAILED")
            logger.exception("cascade-only 실패: case_id=%s", case_id)
        return None

    # ML=CRITICAL: cascade 직접 실행 후 CRITICAL 여부 판단
    try:
        cascade_output = await asyncio.to_thread(analyze_cascade, initial_state)
    except Exception as exc:
        await step_repo.mark_failed(case_id, "cascade", str(exc))
        await spring_client.notify_agent_step(case_id, "cascade", str(exc)[:200], status="FAILED")
        logger.exception("cascade 분석 실패: case_id=%s", case_id)
        return None

    # cascade 노드 출력과 initial_state를 합친 전체 state
    cascade_state = {**initial_state, **(cascade_output if isinstance(cascade_output, dict) else {})}
    alerts = cascade_state.get("alerts", [])

    active_model = await MlModelRepository(pool).find_active()
    model_version_id: UUID | None = UUID(str(active_model["model_version_id"])) if active_model else None
    summary = _extract_summary("cascade", cascade_state)

    is_critical = bool(alerts) and any(a.severity.value.upper() == "CRITICAL" for a in alerts)

    if not is_critical:
        # cascade 강등: CRITICAL ML이지만 composite_score 미달
        await _complete_step(case_id, "cascade", summary, step_repo, spring_client, model_version_id)
        for step in ("cause", "solution", "compare", "hitl", "report"):
            await step_repo.mark_done(case_id, step, "cascade 강등(composite 점수 미달, 원인분석 생략)")
        await _request_snapshot(case_id, tg_id, tg_code, bottleneck_prob, risk_grade, cascade_state, spring_client)
        await spring_client.notify_agent_step(case_id, "cascade_close", "cascade 강등(composite 점수 미달)")
        return None

    # CRITICAL 확정: cascade 완료 후 원인분석 대기 상태로 전환
    await _complete_step(case_id, "cascade", summary, step_repo, spring_client, model_version_id)
    await _request_snapshot(case_id, tg_id, tg_code, bottleneck_prob, risk_grade, cascade_state, spring_client)
    await step_repo.mark_in_progress(case_id, "cause")
    asyncio.create_task(spring_client.notify_agent_step(
        case_id, "cause", "G* 분석 후 원인분석 진행 예정", status="IN_PROGRESS"
    ))
    return cascade_state


async def run_pipeline(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    bottleneck_prob: float,
    risk_grade: str,
    cascade_state: dict,
    pool: asyncpg.Pool,
) -> None:
    """Phase 2: g_star → cause → hitl (cascade는 Phase 1에서 완료됨)."""
    settings = get_settings()
    step_repo = AgentStepRepository(pool)
    cause_repo = CauseAnalysisRepository(pool)
    plan_repo = ActionPlanRepository(pool)
    spring_client = get_spring_client(settings)
    current_step = "cause"

    active_model = await MlModelRepository(pool).find_active()
    model_version_id: UUID | None = (
        UUID(str(active_model["model_version_id"])) if active_model else None
    )

    pipeline = build_pipeline(
        run_sim=True,
        stop_at_hitl=True,
        run_detection=False,
        run_cascade=False,
    )
    final_state = dict(cascade_state)

    try:
        async for event in pipeline.astream(cascade_state):
            for node_name, output in event.items():
                if isinstance(output, dict):
                    final_state.update(output)
                current_step = await _handle_pipeline_event(
                    case_id,
                    node_name,
                    final_state,
                    step_repo,
                    cause_repo,
                    plan_repo,
                    spring_client,
                    tg_code,
                    tg_id,
                    bottleneck_prob,
                    risk_grade,
                    model_version_id,
                )

        await _store_hitl_case_mapping(case_id, final_state.get("hitl_token"))
        await _request_snapshot(
            case_id,
            tg_id,
            tg_code,
            bottleneck_prob,
            risk_grade,
            final_state,
            spring_client,
        )
    except Exception as exc:
        await step_repo.mark_failed(case_id, current_step, str(exc))
        await spring_client.notify_agent_step(
            case_id,
            current_step,
            str(exc)[:200],
            status="FAILED",
        )
        raise


async def run_post_hitl(
    case_id: UUID,
    decision: str,
    selected_plan_id: UUID | None,
    decided_by: UUID,
    decided_at: datetime,
    comment: str | None,
    pool: asyncpg.Pool,
) -> None:
    step_repo = AgentStepRepository(pool)
    report_repo = ResponseReportRepository(pool)
    spring_client = get_spring_client()
    is_rejected = decision == "REJECTED"

    if not is_rejected and selected_plan_id is None:
        raise ValueError("승인 결정에는 selected_plan_id가 필요합니다.")

    await step_repo.mark_done(case_id, "hitl", "관리자 반려" if is_rejected else "관리자 승인")
    await step_repo.mark_in_progress(case_id, "report")
    try:
        pending = await asyncio.to_thread(_load_pending_state, case_id)
        decided_by_name = await _fetch_user_name(pool, decided_by)
        selected_plan = None
        if not is_rejected:
            selected_plan = await ActionPlanRepository(pool).find_by_id(case_id, selected_plan_id)
            if selected_plan is None:
                raise ValueError("선택한 대응안을 찾을 수 없습니다.")
        state = _reconstruct_report_state(
            pending,
            selected_plan,
            decided_by,
            decided_at,
            comment,
            decision=decision,
            decided_by_name=decided_by_name,
        )
        tg_code = str((pending.get("compare_formatted") or [{}])[0].get("toolgroup", ""))
        state["historical_context"] = await _fetch_historical_context(pool, tg_code)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_hitl_report_executor, build_report_pipeline().invoke, state)
        report_results = result.get("report_results", [])
        tg_code = str((pending.get("compare_formatted") or [{}])[0].get("toolgroup", ""))
        for report_result in report_results:
            await report_repo.insert(case_id, report_result)
            await _index_report_to_qdrant(pool, case_id, tg_code, report_result)
        summary = _report_summary(report_results)
        await step_repo.mark_done(case_id, "report", summary)
        await spring_client.notify_agent_step(case_id, "report", summary)
    except Exception as exc:
        await step_repo.mark_failed(case_id, "report", str(exc))
        await spring_client.notify_agent_step(
            case_id,
            "report",
            str(exc)[:200],
            status="FAILED",
        )
        logger.exception("Post-HITL pipeline failed: case_id=%s", case_id)


async def _handle_pipeline_event(
    case_id: UUID,
    node_name: str,
    state: dict,
    step_repo: AgentStepRepository,
    cause_repo: CauseAnalysisRepository,
    plan_repo: ActionPlanRepository,
    spring_client: SpringClient,
    tg_code: str,
    tg_id: UUID,
    input_bottleneck_prob: float,
    input_risk_grade: str,
    model_version_id: UUID | None = None,
) -> str:
    if node_name == "detect":
        return "cascade"
    if node_name == "cascade":
        summary = _extract_summary("cascade", state)
        await _complete_step(case_id, "cascade", summary, step_repo, spring_client, model_version_id)
        alerts = state.get("alerts", [])
        if not alerts:
            for step in ("cause", "solution", "compare", "hitl", "report"):
                await step_repo.mark_done(case_id, step, "병목 없음")
            await spring_client.notify_agent_step(case_id, "report", "병목 없음(cascade 미확인)")
            return "cascade"
        if not any(a.severity.value.upper() == "CRITICAL" for a in alerts):
            for step in ("cause", "solution", "compare", "hitl", "report"):
                await step_repo.mark_done(case_id, step, "cascade 강등(composite 점수 미달, 원인분석 생략)")
            # cascade 직후 즉시 스냅샷 전송 — 파이프라인 이후 실패 시에도 composite_score 보존
            await _request_snapshot(
                case_id, tg_id, tg_code, input_bottleneck_prob, input_risk_grade, state, spring_client
            )
            await spring_client.notify_agent_step(case_id, "cascade_close", "cascade 강등(composite 점수 미달)")
            return "cascade"
        # CRITICAL 확정: cascade 직후 즉시 스냅샷 전송 + 원인분석 중 상태 알림
        # 알림 카드 최초 표시 시점에 "원인 분석 중"이 같이 표시되어야 함
        await _request_snapshot(
            case_id, tg_id, tg_code, input_bottleneck_prob, input_risk_grade, state, spring_client
        )
        await step_repo.mark_in_progress(case_id, "cause")
        asyncio.create_task(spring_client.notify_agent_step(
            case_id, "cause", "G* 분석 후 원인분석 진행 예정", status="IN_PROGRESS"
        ))
        # cause가 IN_PROGRESS이므로 다음 실패는 cause로 보고 (g_star는 STEP_NAME_MAP에 없음).
        return "cause"
    if node_name == "emit1":
        return "cause"
    if node_name == "cause":
        reports = state.get("cause_reports", [])
        report = next((item for item in reports if item.toolgroup == tg_code), None)
        if report is None and reports:
            report = reports[0]
        if report is not None:
            alerts = state.get("alerts", [])
            alert = next((a for a in alerts if a.toolgroup == tg_code), None)
            affected_tgs = (
                alert.impact.affected_tgs
                if alert and getattr(alert, "impact", None)
                else None
            )
            await cause_repo.upsert(case_id, report, affected_tgs=affected_tgs)
        summary = _extract_summary("cause", state)
        await _complete_step(case_id, "cause", summary, step_repo, spring_client, model_version_id)
        await step_repo.mark_in_progress(case_id, "solution")
        return "solution"
    if node_name == "solution":
        summary = _extract_summary("solution", state)
        await _complete_step(case_id, "solution", summary, step_repo, spring_client)
        await step_repo.mark_in_progress(case_id, "compare")
        return "compare"
    if node_name == "compare_rank":
        candidates = list(state.get("solution_candidates", []))
        compare_inputs = state.get("compare_inputs", [])
        if compare_inputs:
            action_cands = [
                c for c in compare_inputs[0].get("action_candidates", [])
                if not c.get("is_baseline")
            ]
            for i, action in enumerate(action_cands):
                if i < len(candidates):
                    candidates[i] = {**candidates[i], "kpi_stats": action.get("kpi_stats")}
        await plan_repo.bulk_insert(case_id, candidates)
        return "compare"
    if node_name == "compare_llm":
        compare_formatted = state.get("compare_formatted") or []
        if compare_formatted:
            formatted = next(
                (
                    item
                    for item in compare_formatted
                    if item.get("toolgroup") == tg_code
                ),
                compare_formatted[0],
            )
            result_v2 = formatted.get("result_v2") or {}
            if result_v2:
                await plan_repo.upsert_compare_json(case_id, result_v2)
        summary = _extract_summary("compare", state)
        await _complete_step(case_id, "compare", summary, step_repo, spring_client)
        await step_repo.mark_in_progress(case_id, "hitl")
        return "hitl"
    if node_name == "compare_hitl":
        if not state.get("compare_formatted"):
            # verification 없어서 compare_formatted 비어있음 → hitl_pending 파일 미생성 상태
            # HITL 스킵하고 바로 RESOLVED 처리 (승인 대기 상태로 방치하면 FileNotFoundError)
            await step_repo.mark_done(case_id, "hitl", "compare 데이터 없음(스킵)")
            await step_repo.mark_done(case_id, "report", "compare 데이터 없음(스킵)")
            await spring_client.notify_agent_step(case_id, "report", "compare 데이터 없음")
            return "hitl"
        # HITL_WAITING은 IN_PROGRESS 유지 (mark_done 안 함 — 관리자 승인 후 run_post_hitl에서 처리).
        # 백엔드에만 DONE 알림 → AWAITING_HITL 상태 전환 + HITL_PENDING 알림 트리거.
        await spring_client.notify_agent_step(case_id, "hitl", "관리자 승인 대기")
        return "hitl"
    if node_name == "emit2":
        # current_step은 mark_failed로 들어가므로 STEP_NAME_MAP 유효 스텝이어야 한다.
        # emit2 시점의 IN_PROGRESS 스텝은 solution(cause 핸들러에서 마킹됨).
        return "solution"
    if node_name == "g_star":
        # g_star 동안 IN_PROGRESS 스텝은 cause(Phase1에서 마킹됨).
        return "cause"
    if node_name == "compare_rag":
        return "compare"
    return "compare" if node_name == "verify" else "cascade"


async def _complete_step(
    case_id: UUID,
    node_name: str,
    summary: str | None,
    step_repo: AgentStepRepository,
    spring_client: SpringClient,
    model_version_id: UUID | None = None,
) -> None:
    await step_repo.mark_done(case_id, node_name, summary, model_version_id=model_version_id)
    asyncio.create_task(spring_client.notify_agent_step(case_id, node_name, summary))


def _initial_state(
    kpi_list: list[ToolGroupKPI],
    prev_kpi: list[ToolGroupKPI],
    tg_code: str,
    snapshot_time: float,
    bottleneck_prob: float,
) -> dict:
    if not any(kpi.toolgroup == tg_code for kpi in kpi_list):
        raise ValueError(f"요청한 TG의 KPI를 찾을 수 없습니다: {tg_code}")
    return {
        "kpi_snapshot": kpi_list,
        "prev_kpi_snapshot": prev_kpi,
        "potential_bottlenecks": [
            PotentialBottleneck(
                toolgroup=tg_code,
                snapshot_time=snapshot_time,
                probability=bottleneck_prob,
            )
        ],
        "alerts": [],
        "cause_reports": [],
        "cascade_report": None,
        "current_release_interval": None,
        "solution_candidates": [],
        "hitl_approved": None,
        "hitl_token": None,
        "verification_results": [],
        "compare_inputs": [],
        "compare_formatted": [],
        "compare_results": [],
        "report_draft": [],
        "report_results": [],
        "historical_context": None,
    }


def _extract_summary(node_name: str, state: dict) -> str | None:
    if node_name == "cascade":
        alerts = state.get("alerts", [])
        if not alerts:
            return "확산 영향 없음"
        critical = [a for a in alerts if a.severity.value.upper() == "CRITICAL"]
        primary = critical[0] if critical else alerts[0]
        path_parts = [primary.toolgroup]
        affected = list(getattr(primary.impact, "affected_tgs", []))
        if affected:
            path_parts.extend(affected[:2])
        path = " → ".join(path_parts) + " 확산 경로 확인"
        at_risk = getattr(primary.impact, "at_risk_lots", None)
        ct = getattr(primary.impact, "ct_increase_min", None)
        metrics = []
        if at_risk is not None:
            metrics.append(f"위험 Lot {int(at_risk)}건")
        if ct is not None:
            metrics.append(f"CT +{int(ct)}분 예측")
        suffix = f", {', '.join(metrics)}" if metrics else ""
        return f"{path}{suffix}, 병목 알림 {len(alerts)}건 CRITICAL {len(critical)}건"
    if node_name == "cause":
        reports = state.get("cause_reports", [])
        return reports[0].cause_summary[:500] if reports else None
    if node_name == "solution":
        candidates = state.get("solution_candidates", [])
        if not candidates:
            return "대응안 후보 없음"
        names = []
        for c in candidates[:3]:
            pid = getattr(c, "plan_id", None)
            nm = getattr(c, "name", None)
            if pid:
                names.append(str(pid))
            elif nm:
                names.append(str(nm))
        name_str = f": {', '.join(names)}" if names else ""
        return f"대응안 {len(candidates)}건 생성{name_str}"
    if node_name == "compare":
        formatted = state.get("compare_formatted", [])
        if not formatted:
            return None
        rec = (formatted[0].get("result_v2") or {}).get("recommendation") or {}
        return str(rec.get("reason") or rec.get("primary_reason") or "")[:500]
    return None


async def _run_cascade_only(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    bottleneck_prob: float,
    risk_grade: str,
    initial_state: dict,
    step_repo: AgentStepRepository,
    spring_client: SpringClient,
) -> None:
    """비-CRITICAL(HIGH): 확산영향(cascade)만 계산해 composite 위험 점수를 스냅샷에 저장하고 종료한다.
    무거운 원인분석/대응안/리포트는 돌리지 않는다 → 맵에는 점수가 뜨지만 알림은 없다(알림은 백엔드가 CRITICAL만)."""
    cascade_state = await asyncio.to_thread(analyze_cascade, initial_state)
    target = next((a for a in cascade_state.get("alerts", []) if a.toolgroup == tg_code), None)
    composite = target.composite_score if target else None
    impact = target.impact if target else None
    # 맵에 composite score 기반 등급 표시. cascade에서 alert 없으면 ML 등급 fallback.
    composite_grade = target.severity.value.upper() if target else risk_grade
    snapshot_time = _snapshot_time_from_state(cascade_state) or _snapshot_time_from_state(initial_state)
    try:
        await spring_client.request_snapshot(
            case_id,
            tg_id,
            bottleneck_prob,
            composite_grade,
            _detected_at_from_snapshot_time(snapshot_time),
            simulation_tick=_simulation_tick(snapshot_time),
            composite_score=composite,
            impact_score=impact.impact_score if impact else None,
            affected_count=len(impact.affected_tgs) if impact else None,
            ct_increase_min=impact.ct_increase_min if impact else None,
            at_risk_lots=impact.at_risk_lots if impact else None,
        )
    except Exception as exc:
        logger.error("cascade-only 스냅샷 요청 실패: case_id=%s error=%s", case_id, exc)
    await step_repo.mark_done(case_id, "cascade", "위험 점수 산정 완료(cascade-only)")
    for step in ("cause", "solution", "compare", "hitl", "report"):
        await step_repo.mark_done(case_id, step, "cascade 강등(composite 점수 미달, 원인분석 생략)")
    await spring_client.notify_agent_step(case_id, "cascade", "위험 점수 산정 완료")
    await spring_client.notify_agent_step(case_id, "cascade_close", "cascade 강등(composite 점수 미달)")


async def _request_snapshot(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    input_probability: float,
    input_risk_grade: str,
    state: dict,
    spring_client: SpringClient,
) -> None:
    alerts = state.get("alerts", [])
    target = next((alert for alert in alerts if alert.toolgroup == tg_code), None)
    probability = target.probability if target else input_probability
    risk_grade = target.severity.value.upper() if target else input_risk_grade
    composite_score = target.composite_score if target else None
    impact = target.impact if target else None
    snapshot_time = _snapshot_time_from_state(state)
    try:
        await spring_client.request_snapshot(
            case_id,
            tg_id,
            probability,
            risk_grade,
            _detected_at_from_snapshot_time(snapshot_time),
            simulation_tick=_simulation_tick(snapshot_time),
            composite_score=composite_score,
            impact_score=impact.impact_score if impact else None,
            affected_count=len(impact.affected_tgs) if impact else None,
            ct_increase_min=impact.ct_increase_min if impact else None,
            at_risk_lots=impact.at_risk_lots if impact else None,
        )
    except Exception as exc:
        logger.error("Snapshot 요청 실패: case_id=%s error=%s", case_id, exc)
    if risk_grade not in {"HIGH", "CRITICAL"}:
        # cascade_close가 이미 NOT_ACTIONABLE 처리 — 추가 notify 불필요
        return


def _snapshot_time_from_state(state: dict) -> float | None:
    kpi_snapshot = state.get("kpi_snapshot") or []
    if kpi_snapshot:
        return _as_float(getattr(kpi_snapshot[0], "snapshot_time", None))

    potential_bottlenecks = state.get("potential_bottlenecks") or []
    if potential_bottlenecks:
        return _as_float(getattr(potential_bottlenecks[0], "snapshot_time", None))

    return None


def _detected_at_from_snapshot_time(snapshot_time: float | None) -> datetime:
    if snapshot_time is None:
        return datetime.now(UTC)
    return _SIM_EPOCH_UTC + timedelta(minutes=snapshot_time)


def _simulation_tick(snapshot_time: float | None) -> int | None:
    if snapshot_time is None:
        return None
    return int(round(snapshot_time))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _store_hitl_case_mapping(case_id: UUID, hitl_token: str | None) -> None:
    if not hitl_token:
        return
    _HITL_PENDING_DIR.mkdir(exist_ok=True)
    mapping_path = _HITL_PENDING_DIR / f"case_{case_id}.json"
    await asyncio.to_thread(
        mapping_path.write_text,
        json.dumps({"hitl_token": hitl_token}),
        "utf-8",
    )


def _load_pending_state(case_id: UUID) -> dict:
    mapping_path = _HITL_PENDING_DIR / f"case_{case_id}.json"
    if not mapping_path.is_file():
        raise FileNotFoundError("HITL 대기 상태 매핑을 찾을 수 없습니다.")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    pending_path = _HITL_PENDING_DIR / f"{mapping['hitl_token']}.json"
    if not pending_path.is_file():
        raise FileNotFoundError("HITL 대기 상태를 찾을 수 없습니다.")
    return json.loads(pending_path.read_text(encoding="utf-8"))


def _reconstruct_report_state(
    pending: dict,
    selected_plan: dict | None,
    decided_by: UUID,
    decided_at: datetime,
    comment: str | None,
    decision: str = "APPROVED",
    decided_by_name: str | None = None,
) -> dict:
    is_rejected = decision == "REJECTED"
    selected_label = (
        None
        if selected_plan is None
        else chr(ord("A") + int(selected_plan["plan_seq"]) - 1)
    )
    approval_info = {
        "status": "반려" if is_rejected else "승인",
        "approved_by": decided_by_name or str(decided_by),
        "approved_role": "ROLE_ADMIN",
        "approved_at": decided_at.isoformat(),
        "comment": comment or ("반려" if is_rejected else "승인"),
        "rejection_reason": (comment or "관리자 반려") if is_rejected else None,
        "selected_label": selected_label,
    }
    compare_results = []
    for formatted in pending.get("compare_formatted", []):
        recommendation = dict(formatted.get("recommendation") or {})
        if selected_label:
            recommendation["action_label"] = selected_label
            recommendation["reason"] = (
                recommendation.get("reason")
                or f"{selected_label} 대응안 관리자 승인"
            )
        else:
            recommendation["reason"] = (
                recommendation.get("reason")
                or "관리자 반려로 승인된 대응안 없음"
            )
        compare_results.append(
            {
                "toolgroup": formatted["toolgroup"],
                "result_v2": formatted.get("result_v2"),
                "recommendation": recommendation,
                "approval_info": approval_info,
                "action_effects": formatted.get("action_effects", []),
                "json_output_path": "",
            }
        )
    return {
        "kpi_snapshot": [
            ToolGroupKPI.model_validate(item) for item in pending.get("kpi_snapshot", [])
        ],
        "prev_kpi_snapshot": [
            ToolGroupKPI.model_validate(item)
            for item in pending.get("prev_kpi_snapshot", [])
        ],
        "potential_bottlenecks": [],
        "alerts": [
            BottleneckAlert.model_validate(item) for item in pending.get("alerts", [])
        ],
        "cause_reports": [
            CauseReport.model_validate(item) for item in pending.get("cause_reports", [])
        ],
        "cascade_report": pending.get("cascade_report"),
        "current_release_interval": None,
        "solution_candidates": pending.get("solution_candidates", []),
        "hitl_approved": not is_rejected,
        "hitl_token": pending.get("hitl_token"),
        "verification_results": [],
        "compare_inputs": [],
        "compare_formatted": pending.get("compare_formatted", []),
        "compare_results": compare_results,
        "report_draft": [],
        "report_results": [],
        "historical_context": None,
    }


async def _fetch_user_name(pool: asyncpg.Pool, user_id: UUID) -> str | None:
    try:
        row = await pool.fetchrow(
            "SELECT user_name FROM tm_app_user WHERE user_id = $1", user_id
        )
        return row["user_name"] if row else None
    except Exception:
        return None


async def _fetch_historical_context(pool: asyncpg.Pool, tg_code: str) -> dict | None:
    """report_agent Level 3: DB에서 반복 이력·과거 조치 효과를 조회한다.
    실패 시 None 반환 — 보고서 생성은 historical 없이 계속 진행된다."""
    if not tg_code:
        return None
    try:
        from agents.report_agent.tools import (
            fetch_past_action_effectiveness,
            fetch_repeat_count,
        )
        repeat = await fetch_repeat_count(pool, tg_code)
        effectiveness = await fetch_past_action_effectiveness(pool, tg_code)
        return {"repeat_count": repeat, "past_effectiveness": effectiveness}
    except Exception as exc:
        logger.warning("historical_context 조회 실패(무시): tg=%s err=%s", tg_code, exc)
        return None


def _report_summary(report_results: list[dict]) -> str:
    if not report_results:
        return "보고서 생성 결과 없음"
    return f"보고서 {len(report_results)}건 생성"


async def _index_report_to_qdrant(
    pool: asyncpg.Pool, case_id: UUID, tg_code: str, report_result: dict
) -> None:
    try:
        from app.services.qdrant_case_service import QdrantCaseService

        report_text = ResponseReportRepository._rendered_markdown(report_result)
        if not report_text.strip():
            logger.warning("Qdrant 인덱싱 스킵: 보고서 본문 없음 case_id=%s", case_id)
            return
        meta = report_result.get("meta") or {}
        summary = str(meta.get("summary") or report_text[:500])

        qdrant = QdrantCaseService()
        # index_case는 동기 + 예외 자체 처리(성공 시 point_id, 실패 시 None) → 스레드에서 실행.
        qdrant_doc_id = await asyncio.to_thread(
            qdrant.index_case,
            case_id=str(case_id),
            tg_code=tg_code,
            area_name=str(meta.get("area_name") or ""),
            detected_at=str(meta.get("detected_at") or ""),
            risk_grade=str(meta.get("severity") or ""),
            cause_summary=summary,
            report_title=str(meta.get("process_name") or tg_code),
            narrative=report_text,
        )
        if not qdrant_doc_id:
            logger.warning("Qdrant 인덱싱 실패(업서트 미완료): case_id=%s", case_id)
            return
        await ResponseReportRepository(pool).mark_qdrant_indexed(case_id, qdrant_doc_id)
        logger.info("Qdrant 인덱싱 완료: case_id=%s doc_id=%s", case_id, qdrant_doc_id)
    except Exception:
        logger.warning("Qdrant 인덱싱 실패: case_id=%s", case_id, exc_info=True)
