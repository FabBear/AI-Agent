"""LangGraph 노드: solution_candidates → verification_results.

GlobalSolutionPlan 또는 per-TG SolutionCandidate 각각에 대해 WHATIF 시뮬을 paired 실행한다.
"""

from __future__ import annotations

import os

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.solution import GlobalCompositeCandidate, GlobalSolutionPlan, SolutionCandidate
from agents.state import PipelineState
from agents.verification_agent.action_mapper import (
    candidate_to_action_rows,
    composite_to_action_rows,
    plan_to_action_rows,
)
from agents.verification_agent.confidence_scorer import compute_paired_stats
from agents.verification_agent.kpi_comparator import compute_paired_deltas
from agents.verification_agent.sim_executor import (
    HORIZON_MIN,
    find_baseline_scenario,
    run_whatif_paired,
)

_log = get_logger(__name__)

_RANK_TO_LABEL = {1: "A", 2: "B", 3: "C"}


def _verify_composite_candidates(
    solution_candidates: list[dict],
    alerts: list[BottleneckAlert],
    t0: float,
) -> list[dict]:
    """GlobalCompositeCandidate 3개(보수/표준/강화) 각각을 30회 paired 시뮬로 검증."""
    critical_alerts = [a for a in alerts if a.severity == SeverityLevel.CRITICAL]
    anchor_alert = max(critical_alerts, key=lambda a: a.composite_score) if critical_alerts else None
    if anchor_alert is None:
        _log.warning("[Verify] CRITICAL 알림 없음 — anchor TG 없이 스킵")
        return []

    results: list[dict] = []

    for cand_dict in solution_candidates:
        try:
            candidate = GlobalCompositeCandidate(**cand_dict)
        except Exception as e:
            _log.warning(f"[Verify] GlobalCompositeCandidate 파싱 실패: {e}")
            continue

        action_rows, release_multiplier = composite_to_action_rows(candidate, t0)
        lot_adjustments = [adj.model_dump() if hasattr(adj, "model_dump") else adj
                           for adj in candidate.lot_adjustments]

        _log.info(
            f"[Verify] 플랜 {candidate.plan_id} — "
            f"{len(candidate.target_toolgroups)}개 TG, "
            f"+{candidate.release_interval_delta_pct}%, "
            f"lot_adjustments={len(lot_adjustments)}건 "
            f"× 30 paired 시뮬 시작 (anchor={anchor_alert.toolgroup})"
        )

        try:
            group_id, pairs, baseline_scenario_id = run_whatif_paired(
                t0=t0,
                horizon_min=HORIZON_MIN,
                action_rows=action_rows,
                release_interval_multiplier=release_multiplier,
                label=f"COMPOSITE_{candidate.plan_id.upper()}",
                lot_adjustments=lot_adjustments,
            )
        except Exception as e:
            _log.error(f"[Verify] 플랜 {candidate.plan_id} 시뮬 실패: {e}")
            continue

        kpi_deltas = compute_paired_deltas(pairs, anchor_alert.toolgroup)
        kpi_stats: dict[str, dict] = {}
        for kpi_name, deltas in kpi_deltas.items():
            if len(deltas) >= 2:
                kpi_stats[kpi_name] = compute_paired_stats(deltas)

        target_stats = kpi_stats.get("q_time_min", {})

        plan_meta = {
            "plan_id": candidate.plan_id,
            "target_toolgroups": candidate.target_toolgroups,
            "release_interval_delta_pct": candidate.release_interval_delta_pct,
            "cause_complexity": candidate.cause_complexity,
            "lot_adjustments_count": len(lot_adjustments),
            "hitl_escalation_recommended": candidate.hitl_escalation_recommended,
            "escalation_reason": candidate.escalation_reason,
        }

        results.append({
            "plan_id": candidate.plan_id,
            "target_toolgroups": candidate.target_toolgroups,
            "snapshot_time": t0,
            "baseline_scenario_id": baseline_scenario_id,
            "verified_candidates": [{
                "label": candidate.plan_id,
                "name": {"conservative": "보수적 조정안", "standard": "표준 조정안", "aggressive": "강화 조정안"}.get(
                    candidate.plan_id, candidate.plan_id
                ),
                "target_kpi": "q_time_min",
                "action_rows": action_rows,
                "whatif_scenario_group_id": group_id,
                "baseline_scenario_id": baseline_scenario_id,
                "paired_n": len(pairs),
                "kpi_stats": kpi_stats,
                "target_kpi_stats": target_stats,
                "verdict": target_stats.get("verdict", "unknown"),
                "paired_t_p": target_stats.get("paired_t_p"),
                "plan_meta": plan_meta,
            }],
        })

    _log.info(f"[Verify] GlobalCompositeCandidate 검증 완료 — {len(results)}개 플랜")
    return results


def _verify_candidates(
    alert: BottleneckAlert,
    candidates_raw: list[dict],
    t0: float,
) -> list[dict]:
    """후보 목록 각각을 30회 paired 시뮬로 검증."""
    verified: list[dict] = []

    for cand_dict in candidates_raw:
        rank = cand_dict.get("rank", 1)
        label = _RANK_TO_LABEL.get(rank, str(rank))

        try:
            candidate = SolutionCandidate(**cand_dict)
        except Exception as e:
            _log.warning(f"[Verify] candidate 파싱 실패 (rank={rank}): {e}")
            continue

        action_rows, release_multiplier = candidate_to_action_rows(candidate, alert, t0)

        try:
            group_id, pairs, baseline_id = run_whatif_paired(
                t0=t0,
                horizon_min=HORIZON_MIN,
                action_rows=action_rows,
                release_interval_multiplier=release_multiplier,
                label=f"{alert.toolgroup}_{label}",
                lot_action_config={
                    "target_tg": alert.toolgroup,
                    "priority_direction": candidate.params.priority_direction,
                    "superhotlot_enable": candidate.params.superhotlot_enable,
                },
            )
        except Exception as e:
            _log.error(f"[Verify] {alert.toolgroup} {label} 시뮬 실패: {e}")
            continue

        kpi_deltas = compute_paired_deltas(pairs, alert.toolgroup)

        kpi_stats: dict[str, dict] = {}
        for kpi_name, deltas in kpi_deltas.items():
            if len(deltas) >= 2:
                kpi_stats[kpi_name] = compute_paired_stats(deltas)

        target_stats = kpi_stats.get(candidate.target_kpi, {})

        params_meta = {
            "target_toolgroups": [alert.toolgroup],
            "release_interval_delta_pct": candidate.params.release_interval_delta_pct,
            "priority_direction": candidate.params.priority_direction,
            "lot_priority_rule": candidate.params.lot_priority_rule,
            "dispatch_rule": candidate.params.dispatch_rule,
            "superhotlot_enable": candidate.params.superhotlot_enable,
            "expected_effect": candidate.expected_effect,
            "rationale": candidate.rationale,
        }

        verified.append({
            "rank": rank,
            "label": label,
            "name": candidate.name,
            "target_kpi": candidate.target_kpi,
            "action_rows": action_rows,
            "whatif_scenario_group_id": group_id,
            "baseline_scenario_id": baseline_id,
            "paired_n": len(pairs),
            "kpi_stats": kpi_stats,
            "target_kpi_stats": target_stats,
            "verdict": target_stats.get("verdict", "unknown"),
            "paired_t_p": target_stats.get("paired_t_p"),
            "plan_meta": params_meta,
        })

    return verified


def _verify_global_plans(
    solution_candidates: list[dict],
    alerts: list[BottleneckAlert],
    t0: float,
) -> list[dict]:
    """GlobalSolutionPlan A/B 각각을 30회 paired 시뮬로 검증."""
    critical_alerts = [a for a in alerts if a.severity == SeverityLevel.CRITICAL]
    anchor_alert = max(critical_alerts, key=lambda a: a.composite_score) if critical_alerts else None
    if anchor_alert is None:
        _log.warning("[Verify] CRITICAL 알림 없음 — anchor TG 없이 스킵")
        return []

    results: list[dict] = []

    for plan_dict in solution_candidates:
        try:
            plan = GlobalSolutionPlan(**plan_dict)
        except Exception as e:
            _log.warning(f"[Verify] GlobalSolutionPlan 파싱 실패: {e}")
            continue

        action_rows, release_multiplier = plan_to_action_rows(plan, t0)
        _log.info(
            f"[Verify] 플랜 {plan.plan_id} — {len(plan.target_toolgroups)}개 TG "
            f"× 30 paired 시뮬 시작 (anchor={anchor_alert.toolgroup})"
        )

        try:
            group_id, pairs, baseline_scenario_id = run_whatif_paired(
                t0=t0,
                horizon_min=HORIZON_MIN,
                action_rows=action_rows,
                release_interval_multiplier=release_multiplier,
                label=f"PLAN_{plan.plan_id}",
            )
        except Exception as e:
            _log.error(f"[Verify] 플랜 {plan.plan_id} 시뮬 실패: {e}")
            continue

        kpi_deltas = compute_paired_deltas(pairs, anchor_alert.toolgroup)
        kpi_stats: dict[str, dict] = {}
        for kpi_name, deltas in kpi_deltas.items():
            if len(deltas) >= 2:
                kpi_stats[kpi_name] = compute_paired_stats(deltas)

        target_stats = kpi_stats.get("q_time_min", {})

        plan_meta = {
            "target_toolgroups": list(plan.target_toolgroups),
            "release_interval_minutes": plan.release_interval_minutes,
            "current_interval_minutes": plan.current_interval_minutes,
            "release_interval_delta_min": round(
                plan.release_interval_minutes - plan.current_interval_minutes, 4
            ),
            "lot_priority_rule": plan.lot_priority_rule,
            "superhotlot_enable": plan.superhotlot_enable,
            "expected_effect": plan.expected_effect,
        }

        results.append({
            "plan_id": plan.plan_id,
            "target_toolgroups": plan.target_toolgroups,
            "snapshot_time": t0,
            "baseline_scenario_id": baseline_scenario_id,
            "verified_candidates": [{
                "label": plan.plan_id,
                "name": plan.description,
                "target_kpi": "q_time_min",
                "action_rows": action_rows,
                "whatif_scenario_group_id": group_id,
                "baseline_scenario_id": baseline_scenario_id,
                "paired_n": len(pairs),
                "kpi_stats": kpi_stats,
                "target_kpi_stats": target_stats,
                "verdict": target_stats.get("verdict", "unknown"),
                "paired_t_p": target_stats.get("paired_t_p"),
                "plan_meta": plan_meta,
            }],
        })

    _log.info(f"[Verify] GlobalSolutionPlan 검증 완료 — {len(results)}개 플랜")
    return results


def verify_solutions(state: PipelineState) -> PipelineState:
    """solution_candidates 각 후보를 30회 paired WHATIF 시뮬로 검증."""
    solution_candidates = state.get("solution_candidates", [])
    if not solution_candidates:
        _log.info("[Verify] solution_candidates 없음 — 스킵")
        return {**state, "verification_results": []}

    alerts = state["alerts"]
    kpi_snapshot = state["kpi_snapshot"]
    t0 = kpi_snapshot[0].snapshot_time if kpi_snapshot else 0.0

    if os.environ.get("DEMO_MOCK_VERIFICATION", "").lower() in (
        "1", "true", "yes",
    ):
        from demo.verification_data import (
            build_demo_verification_results,
        )

        _log.info(
            "[Verify] DEMO 목 통계 사용"
        )
        return {
            **state,
            "verification_results": build_demo_verification_results(
                solution_candidates,
                t0,
            ),
        }

    baseline_id = find_baseline_scenario(t0)
    if baseline_id is None:
        _log.error("[Verify] baseline 시나리오 없음 — 전체 스킵")
        return {**state, "verification_results": []}

    first = solution_candidates[0]
    if first.get("plan_id") in ("conservative", "standard", "aggressive"):
        # 신규: GlobalCompositeCandidate 포맷
        return {**state, "verification_results": _verify_composite_candidates(
            solution_candidates, alerts, t0,
        )}
    if "plan_id" in first:
        # 레거시: GlobalSolutionPlan (plan_id="A"/"B") 포맷
        return {**state, "verification_results": _verify_global_plans(
            solution_candidates, alerts, t0,
        )}

    alert_map = {a.toolgroup: a for a in alerts}
    results: list[dict] = []

    for group in solution_candidates:
        tg = group["toolgroup"]
        alert = alert_map.get(tg)
        if alert is None:
            continue

        candidates_raw = group.get("candidates", [])
        _log.info(f"[Verify] {tg}: {len(candidates_raw)}개 후보 × 30 paired 시뮬 시작")

        verified = _verify_candidates(alert, candidates_raw, t0)
        if not verified:
            continue

        results.append({
            "toolgroup": tg,
            "severity": alert.severity.value,
            "snapshot_time": t0,
            "baseline_scenario_id": baseline_id,
            "verified_candidates": verified,
        })

    _log.info(f"[Verify] 완료 — {len(results)}개 공정 검증")
    return {**state, "verification_results": results}
