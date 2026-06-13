"""LangGraph 노드: alerts + cause_reports → GlobalCompositeCandidate 3개 생성."""

from __future__ import annotations

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.solution_generator.release_plan_loader import load_release_plan
from agents.solution_generator.rule_engine import generate_global_candidates
from agents.state import PipelineState

_log = get_logger(__name__)


def generate_solutions(state: PipelineState) -> PipelineState:
    alerts: list[BottleneckAlert] = state["alerts"]
    cause_map = {r.toolgroup: r for r in state["cause_reports"]}

    target_alerts = [
        a for a in alerts
        if a.severity == SeverityLevel.CRITICAL and a.toolgroup in cause_map
    ]

    if not target_alerts:
        _log.info("[Solution] CRITICAL 알림 없음 — 대응안 생성 스킵")
        return {**state, "solution_candidates": []}

    kpi_snapshot = state.get("kpi_snapshot", [])
    t0 = kpi_snapshot[0].snapshot_time if kpi_snapshot else 0.0

    release_plan_rows = load_release_plan(t0)
    if not release_plan_rows:
        _log.warning("[Solution] mes_lot_release_plan 조회 결과 없음 — lot_adjustments 빈 리스트로 진행")

    candidates = generate_global_candidates(target_alerts, cause_map, release_plan_rows, t0)

    if not candidates:
        _log.warning("[Solution] 유효한 judgment 없음 — 대응안 미생성")
        return {**state, "solution_candidates": []}

    if any(c.hitl_escalation_recommended for c in candidates):
        _log.warning(f"[Solution] HITL 에스컬레이션 권고: {candidates[0].escalation_reason}")

    _log.info(
        f"[Solution] 글로벌 복합 대응안 {len(candidates)}개 생성 "
        f"(TG={[c.target_toolgroups for c in candidates[:1]]}, "
        f"complexity={candidates[0].cause_complexity})"
    )

    return {**state, "solution_candidates": [c.model_dump() for c in candidates]}
