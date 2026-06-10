import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import asyncpg

from agents.data.kpi_loader import load_kpi_snapshot, load_kpi_window
from agents.pipeline import build_phase2_pipeline, build_pipeline
from agents.schemas.alert import BottleneckAlert, PotentialBottleneck
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI
from app.config import get_settings
from app.repositories.action_plan_repository import ActionPlanRepository
from app.repositories.agent_step_repository import AgentStepRepository
from app.repositories.cause_analysis_repository import CauseAnalysisRepository
from app.repositories.response_report_repository import ResponseReportRepository
from app.services.spring_client import SpringClient, get_spring_client

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_HITL_PENDING_DIR = _ROOT / "hitl_pending"


async def run_pipeline_with_timeout(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    snapshot_time: float,
    bottleneck_prob: float,
    risk_grade: str,
    pool: asyncpg.Pool,
) -> None:
    settings = get_settings()
    try:
        await asyncio.wait_for(
            run_pipeline(
                case_id,
                tg_id,
                tg_code,
                snapshot_time,
                bottleneck_prob,
                risk_grade,
                pool,
            ),
            timeout=settings.pipeline_timeout_sec,
        )
    except TimeoutError:
        step_repo = AgentStepRepository(pool)
        failed_step = await step_repo.mark_active_failed(case_id, "파이프라인 타임아웃")
        await get_spring_client(settings).notify_agent_step(
            case_id,
            failed_step or "DIFFUSION_ANALYSIS",
            "파이프라인 타임아웃",
            status="FAILED",
        )
        logger.exception("Agent pipeline timeout: case_id=%s", case_id)
    except Exception:
        logger.exception("Agent pipeline failed: case_id=%s", case_id)


async def run_pipeline(
    case_id: UUID,
    tg_id: UUID,
    tg_code: str,
    snapshot_time: float,
    bottleneck_prob: float,
    risk_grade: str,
    pool: asyncpg.Pool,
) -> None:
    settings = get_settings()
    step_repo = AgentStepRepository(pool)
    cause_repo = CauseAnalysisRepository(pool)
    plan_repo = ActionPlanRepository(pool)
    spring_client = get_spring_client(settings)
    current_step = "cascade"

    await step_repo.init_steps(case_id)
    await step_repo.mark_in_progress(case_id, current_step)

    try:
        kpi_list = await asyncio.to_thread(
            load_kpi_snapshot,
            settings.agent_csv_dir,
            snapshot_time,
        )
        window = await asyncio.to_thread(
            load_kpi_window,
            settings.agent_csv_dir,
            snapshot_time,
            2,
        )
        window_times = sorted(window)
        prev_kpi = window[window_times[0]] if len(window_times) >= 2 else []
        initial_state = _initial_state(
            kpi_list,
            prev_kpi,
            tg_code,
            snapshot_time,
            bottleneck_prob,
        )
        pipeline = build_pipeline(
            csv_dir=settings.agent_csv_dir,
            run_sim=True,
            phase1_only=True,
            run_detection=False,
        )
        final_state = dict(initial_state)

        async for event in pipeline.astream(initial_state):
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

    if decision == "REJECTED":
        await step_repo.mark_done(case_id, "hitl", "관리자 반려")
        await step_repo.mark_done(case_id, "report", "반려로 인한 종료")
        await spring_client.notify_agent_step(case_id, "hitl", "관리자 반려")
        return

    if selected_plan_id is None:
        raise ValueError("승인 결정에는 selected_plan_id가 필요합니다.")

    await step_repo.mark_done(case_id, "hitl", "관리자 승인")
    await step_repo.mark_in_progress(case_id, "report")
    try:
        pending = await asyncio.to_thread(_load_pending_state, case_id)
        selected_plan = await ActionPlanRepository(pool).find_by_id(case_id, selected_plan_id)
        if selected_plan is None:
            raise ValueError("선택한 대응안을 찾을 수 없습니다.")
        state = _reconstruct_phase2_state(
            pending,
            selected_plan,
            decided_by,
            decided_at,
            comment,
        )
        result = await asyncio.to_thread(build_phase2_pipeline().invoke, state)
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
) -> str:
    if node_name == "detect":
        return "cascade"
    if node_name == "cascade":
        summary = _extract_summary("cascade", state)
        await _complete_step(case_id, "cascade", summary, step_repo, spring_client)
        if not state.get("alerts"):
            for step in ("cause", "solution", "compare", "hitl", "report"):
                await step_repo.mark_done(case_id, step, "병목 없음")
            return "cascade"
        await step_repo.mark_in_progress(case_id, "cause")
        return "cause"
    if node_name == "cause":
        reports = state.get("cause_reports", [])
        report = next((item for item in reports if item.toolgroup == tg_code), None)
        if report is None and reports:
            report = reports[0]
        if report is not None:
            await cause_repo.upsert(case_id, report)
        summary = _extract_summary("cause", state)
        await _complete_step(case_id, "cause", summary, step_repo, spring_client)
        await step_repo.mark_in_progress(case_id, "solution")
        return "solution"
    if node_name == "solution":
        await plan_repo.bulk_insert(case_id, state.get("solution_candidates", []))
        summary = _extract_summary("solution", state)
        await _complete_step(case_id, "solution", summary, step_repo, spring_client)
        await step_repo.mark_in_progress(case_id, "compare")
        return "compare"
    if node_name == "compare_rank":
        return "compare"
    if node_name == "compare_llm":
        summary = _extract_summary("compare", state)
        await _complete_step(case_id, "compare", summary, step_repo, spring_client)
        await step_repo.mark_in_progress(case_id, "hitl")
        return "hitl"
    if node_name == "compare_hitl":
        summary = "관리자 승인 대기"
        await _complete_step(case_id, "hitl", summary, step_repo, spring_client)
        return "hitl"
    return "compare" if node_name == "verify" else "cascade"


async def _complete_step(
    case_id: UUID,
    node_name: str,
    summary: str | None,
    step_repo: AgentStepRepository,
    spring_client: SpringClient,
) -> None:
    await step_repo.mark_done(case_id, node_name, summary)
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
    }


def _extract_summary(node_name: str, state: dict) -> str | None:
    if node_name == "cascade":
        alerts = state.get("alerts", [])
        critical = sum(alert.severity.value.upper() == "CRITICAL" for alert in alerts)
        return f"병목 알림 {len(alerts)}건, CRITICAL {critical}건"
    if node_name == "cause":
        reports = state.get("cause_reports", [])
        return reports[0].cause_summary[:500] if reports else None
    if node_name == "solution":
        return f"대응안 {len(state.get('solution_candidates', []))}건 생성"
    if node_name == "compare":
        formatted = state.get("compare_formatted", [])
        if not formatted:
            return None
        return str(formatted[0].get("recommendation", {}).get("reason") or "")[:500]
    return None


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
    if risk_grade not in {"HIGH", "CRITICAL"}:
        return
    try:
        await spring_client.request_snapshot(
            case_id,
            tg_id,
            probability,
            risk_grade,
            datetime.now(UTC),
        )
    except Exception as exc:
        logger.error("Snapshot 요청 실패: case_id=%s error=%s", case_id, exc)


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


def _reconstruct_phase2_state(
    pending: dict,
    selected_plan: dict,
    decided_by: UUID,
    decided_at: datetime,
    comment: str | None,
) -> dict:
    selected_label = chr(ord("A") + int(selected_plan["plan_seq"]) - 1)
    approval_info = {
        "status": "승인",
        "approved_by": str(decided_by),
        "approved_role": "ROLE_ADMIN",
        "approved_at": decided_at.isoformat(),
        "comment": comment or "승인",
        "rejection_reason": None,
    }
    compare_results = []
    for formatted in pending.get("compare_formatted", []):
        recommendation = dict(formatted.get("recommendation") or {})
        recommendation["action_label"] = selected_label
        recommendation["reason"] = (
            recommendation.get("reason")
            or f"{selected_label} 대응안 관리자 승인"
        )
        compare_results.append(
            {
                "toolgroup": formatted["toolgroup"],
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
        "hitl_approved": True,
        "hitl_token": pending.get("hitl_token"),
        "verification_results": [],
        "compare_inputs": [],
        "compare_formatted": pending.get("compare_formatted", []),
        "compare_results": compare_results,
        "report_draft": [],
        "report_results": [],
    }


def _report_summary(report_results: list[dict]) -> str:
    if not report_results:
        return "보고서 생성 결과 없음"
    return f"보고서 {len(report_results)}건 생성"


async def _index_report_to_qdrant(
    pool: asyncpg.Pool, case_id: UUID, tg_code: str, report_result: dict
) -> None:
    try:
        from app.services.qdrant_service import QdrantService

        report_text = ResponseReportRepository._report_text(report_result)
        meta = report_result.get("meta") or {}
        summary = str(meta.get("summary") or report_text[:500])
        qdrant = QdrantService(get_settings())
        qdrant_doc_id = await qdrant.index_report(
            case_id=str(case_id),
            tg_code=tg_code,
            summary=summary,
            report_text=report_text,
        )
        await ResponseReportRepository(pool).mark_qdrant_indexed(case_id, qdrant_doc_id)
        logger.info("Qdrant 인덱싱 완료: case_id=%s doc_id=%s", case_id, qdrant_doc_id)
    except Exception:
        logger.warning("Qdrant 인덱싱 실패: case_id=%s", case_id, exc_info=True)
