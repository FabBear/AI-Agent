"""LangGraph 노드: solution_candidates → verification_results.

각 대응안 후보(rank 1~3)에 대해 WHATIF 시뮬 30회 paired 실행:
  - seed = baseline 런과 동일 (runs_manifest.csv 기준)
  - D_i = whatif_i − baseline_i per KPI per pair
  - paired t-test → p-value, 95% CI, verdict
"""

from __future__ import annotations

from agents.logger import get_logger
from agents.schemas.alert import BottleneckAlert, SeverityLevel
from agents.schemas.solution import GlobalSolutionPlan, SolutionCandidate
from agents.state import PipelineState
from agents.verification_agent.action_mapper import candidate_to_action_rows, plan_to_action_rows
from agents.verification_agent.confidence_scorer import compute_paired_stats
from agents.verification_agent.kpi_comparator import compute_paired_deltas
from agents.verification_agent.sim_executor import (
    HORIZON_MIN,
    find_baseline_scenario,
    run_whatif_paired,
)

_log = get_logger(__name__)

_RANK_TO_LABEL = {1: "A", 2: "B", 3: "C"}


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
            )
        except Exception as e:
            _log.error(f"[Verify] {alert.toolgroup} {label} 시뮬 실패: {e}")
            continue

        # D_i = whatif_i − baseline_i, KPI별 30쌍
        kpi_deltas = compute_paired_deltas(pairs, alert.toolgroup)

        # KPI별 paired t-test 통계
        kpi_stats: dict[str, dict] = {}
        for kpi_name, deltas in kpi_deltas.items():
            if len(deltas) >= 2:
                kpi_stats[kpi_name] = compute_paired_stats(deltas)

        # target KPI 기준 주요 통계
        target_stats = kpi_stats.get(candidate.target_kpi, {})

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
        })

    return verified


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

    # GlobalSolutionPlan 포맷 (plan_id 키 존재) → 플랜별 통합 시뮬
    if "plan_id" in solution_candidates[0]:
        return {**state, "verification_results": _verify_global_plans(
            solution_candidates, alerts, t0,
        )}

    # 기존 per-TG 포맷
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
