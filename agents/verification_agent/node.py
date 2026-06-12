"""LangGraph 노드: solution_candidates → verification_results.

GlobalSolutionPlan A/B 각각에 대해 WHATIF 시뮬 30회 paired 실행:
  - seed = baseline 런과 동일 (runs_manifest.csv 기준)
  - D_i = whatif_i − baseline_i per KPI per pair
  - paired t-test → p-value, 95% CI, verdict
"""

from __future__ import annotations

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.solution import GlobalSolutionPlan
from agents.state import PipelineState
from agents.verification_agent.action_mapper import plan_to_action_rows
from agents.verification_agent.confidence_scorer import compute_paired_stats
from agents.verification_agent.kpi_comparator import compute_paired_deltas
from agents.verification_agent.sim_executor import (
    HORIZON_MIN,
    find_baseline_scenario,
    run_whatif_paired,
)

_log = get_logger(__name__)


def _verify_global_plans(
    solution_candidates: list[dict],
    alerts: list[BottleneckAlert],
    t0: float,
) -> list[dict]:
    """GlobalSolutionPlan A/B 각각을 30회 paired 시뮬로 검증."""
    # anchor TG: CRITICAL 중 composite_score 최고인 툴그룹
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

        # anchor TG 기준으로 KPI delta 산출
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

    baseline_id = find_baseline_scenario(t0)
    if baseline_id is None:
        _log.error("[Verify] baseline 시나리오 없음 — 전체 스킵")
        return {**state, "verification_results": []}

    return {**state, "verification_results": _verify_global_plans(
        solution_candidates, alerts, t0,
    )}
