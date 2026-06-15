"""Snapshot별 데모용 120분 paired 검증 결과."""

from __future__ import annotations

from agents.schemas.solution import GlobalCompositeCandidate


DEMO_PAIRED_N = 10

_DEMO_STATS_3180: dict[str, dict[str, dict]] = {
    "conservative": {
        "q_time_min": {
            "mean_delta": -0.03,
            "ci_lo": -0.10,
            "ci_hi": 0.04,
            "paired_t_p": 0.32,
            "verdict": "unchanged",
        },
        "wip": {
            "mean_delta": -0.20,
            "ci_lo": -0.70,
            "ci_hi": 0.30,
            "paired_t_p": 0.28,
            "verdict": "unchanged",
        },
        "wait_ratio": {
            "mean_delta": -0.01,
            "ci_lo": -0.04,
            "ci_hi": 0.02,
            "paired_t_p": 0.35,
            "verdict": "unchanged",
        },
        "utilization_avg": {
            "mean_delta": -0.005,
            "ci_lo": -0.02,
            "ci_hi": 0.01,
            "paired_t_p": 0.44,
            "verdict": "unchanged",
        },
        "available_tool_ratio": {
            "mean_delta": 0.0,
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "paired_t_p": 0.80,
            "verdict": "unchanged",
        },
    },
    "standard": {
        "q_time_min": {
            "mean_delta": -0.28,
            "ci_lo": -0.40,
            "ci_hi": -0.16,
            "paired_t_p": 0.008,
            "verdict": "improved",
        },
        "wip": {
            "mean_delta": -1.40,
            "ci_lo": -2.00,
            "ci_hi": -0.80,
            "paired_t_p": 0.012,
            "verdict": "improved",
        },
        "wait_ratio": {
            "mean_delta": -0.06,
            "ci_lo": -0.09,
            "ci_hi": -0.03,
            "paired_t_p": 0.015,
            "verdict": "improved",
        },
        "utilization_avg": {
            "mean_delta": -0.02,
            "ci_lo": -0.04,
            "ci_hi": 0.0,
            "paired_t_p": 0.09,
            "verdict": "unchanged",
        },
        "available_tool_ratio": {
            "mean_delta": 0.0,
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "paired_t_p": 0.75,
            "verdict": "unchanged",
        },
    },
    "aggressive": {
        "q_time_min": {
            "mean_delta": -0.36,
            "ci_lo": -0.50,
            "ci_hi": -0.22,
            "paired_t_p": 0.004,
            "verdict": "improved",
        },
        "wip": {
            "mean_delta": 0.80,
            "ci_lo": 0.20,
            "ci_hi": 1.40,
            "paired_t_p": 0.025,
            "verdict": "worsened",
        },
        "wait_ratio": {
            "mean_delta": -0.08,
            "ci_lo": -0.11,
            "ci_hi": -0.05,
            "paired_t_p": 0.009,
            "verdict": "improved",
        },
        "utilization_avg": {
            "mean_delta": -0.03,
            "ci_lo": -0.05,
            "ci_hi": -0.01,
            "paired_t_p": 0.04,
            "verdict": "improved",
        },
        "available_tool_ratio": {
            "mean_delta": -0.07,
            "ci_lo": -0.10,
            "ci_hi": -0.04,
            "paired_t_p": 0.018,
            "verdict": "worsened",
        },
    },
}

_DEMO_FORECASTS_3780 = {
    "conservative": {
        "DE_FE_1": {
            "current": {
                "q_time_min": 62.96,
                "wip": 10.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.9943,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 96.39,
                "wip": 12.0,
                "wait_ratio": 0.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 82.0,
                "wip": 11.0,
                "wait_ratio": 0.42,
                "utilization_avg": 0.99,
                "available_tool_ratio": 1.0,
            },
        },
        "Diffusion_FE_125": {
            "current": {
                "q_time_min": 35.98,
                "wip": 5.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.8153,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 46.26,
                "wip": 18.0,
                "wait_ratio": 3.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 41.0,
                "wip": 14.0,
                "wait_ratio": 2.60,
                "utilization_avg": 0.96,
                "available_tool_ratio": 1.0,
            },
        },
    },
    "standard": {
        "DE_FE_1": {
            "current": {
                "q_time_min": 62.96,
                "wip": 10.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.9943,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 96.39,
                "wip": 12.0,
                "wait_ratio": 0.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 58.0,
                "wip": 8.0,
                "wait_ratio": 0.18,
                "utilization_avg": 0.88,
                "available_tool_ratio": 1.0,
            },
        },
        "Diffusion_FE_125": {
            "current": {
                "q_time_min": 35.98,
                "wip": 5.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.8153,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 46.26,
                "wip": 18.0,
                "wait_ratio": 3.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 30.0,
                "wip": 8.0,
                "wait_ratio": 0.65,
                "utilization_avg": 0.82,
                "available_tool_ratio": 1.0,
            },
        },
    },
    "aggressive": {
        "DE_FE_1": {
            "current": {
                "q_time_min": 62.96,
                "wip": 10.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.9943,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 96.39,
                "wip": 12.0,
                "wait_ratio": 0.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 48.0,
                "wip": 6.0,
                "wait_ratio": 0.10,
                "utilization_avg": 0.80,
                "available_tool_ratio": 0.90,
            },
        },
        "Diffusion_FE_125": {
            "current": {
                "q_time_min": 35.98,
                "wip": 5.0,
                "wait_ratio": 0.25,
                "utilization_avg": 0.8153,
                "available_tool_ratio": 1.0,
            },
            "no_action": {
                "q_time_min": 46.26,
                "wip": 18.0,
                "wait_ratio": 3.50,
                "utilization_avg": 1.0,
                "available_tool_ratio": 1.0,
            },
            "action": {
                "q_time_min": 24.0,
                "wip": 20.0,
                "wait_ratio": 0.45,
                "utilization_avg": 0.74,
                "available_tool_ratio": 0.86,
            },
        },
    },
}

_DEMO_STATS_3780 = {
    "conservative": {
        "q_time_min": {
            "mean_delta": -5.26,
            "ci_lo": -11.30,
            "ci_hi": 0.78,
            "paired_t_p": 0.12,
            "verdict": "unchanged",
        },
        "wip": {
            "mean_delta": -1.0,
            "ci_lo": -2.20,
            "ci_hi": 0.20,
            "paired_t_p": 0.18,
            "verdict": "unchanged",
        },
        "wait_ratio": {
            "mean_delta": -0.08,
            "ci_lo": -0.20,
            "ci_hi": 0.04,
            "paired_t_p": 0.22,
            "verdict": "unchanged",
        },
        "utilization_avg": {
            "mean_delta": -0.01,
            "ci_lo": -0.04,
            "ci_hi": 0.02,
            "paired_t_p": 0.28,
            "verdict": "unchanged",
        },
        "available_tool_ratio": {
            "mean_delta": 0.0,
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "paired_t_p": 0.80,
            "verdict": "unchanged",
        },
    },
    "standard": {
        "q_time_min": {
            "mean_delta": -16.26,
            "ci_lo": -21.40,
            "ci_hi": -11.12,
            "paired_t_p": 0.006,
            "verdict": "improved",
        },
        "wip": {
            "mean_delta": -4.0,
            "ci_lo": -5.80,
            "ci_hi": -2.20,
            "paired_t_p": 0.008,
            "verdict": "improved",
        },
        "wait_ratio": {
            "mean_delta": -0.32,
            "ci_lo": -0.48,
            "ci_hi": -0.16,
            "paired_t_p": 0.012,
            "verdict": "improved",
        },
        "utilization_avg": {
            "mean_delta": -0.12,
            "ci_lo": -0.18,
            "ci_hi": -0.06,
            "paired_t_p": 0.018,
            "verdict": "improved",
        },
        "available_tool_ratio": {
            "mean_delta": 0.0,
            "ci_lo": -0.01,
            "ci_hi": 0.01,
            "paired_t_p": 0.75,
            "verdict": "unchanged",
        },
    },
    "aggressive": {
        "q_time_min": {
            "mean_delta": -22.26,
            "ci_lo": -28.10,
            "ci_hi": -16.42,
            "paired_t_p": 0.004,
            "verdict": "improved",
        },
        "wip": {
            "mean_delta": 2.0,
            "ci_lo": 0.60,
            "ci_hi": 3.40,
            "paired_t_p": 0.018,
            "verdict": "worsened",
        },
        "wait_ratio": {
            "mean_delta": -0.40,
            "ci_lo": -0.55,
            "ci_hi": -0.25,
            "paired_t_p": 0.006,
            "verdict": "improved",
        },
        "utilization_avg": {
            "mean_delta": -0.20,
            "ci_lo": -0.27,
            "ci_hi": -0.13,
            "paired_t_p": 0.010,
            "verdict": "improved",
        },
        "available_tool_ratio": {
            "mean_delta": -0.14,
            "ci_lo": -0.18,
            "ci_hi": -0.10,
            "paired_t_p": 0.009,
            "verdict": "worsened",
        },
    },
}


def _scenario_for_t0(t0: float) -> tuple[dict, dict | None, str]:
    if int(t0) == 3780:
        return _DEMO_STATS_3780, _DEMO_FORECASTS_3780, "DEMO_3780_MULTI_TG"
    return _DEMO_STATS_3180, None, "DEMO_3180_SINGLE_TG"


def _with_sample_size(stats: dict[str, dict]) -> dict[str, dict]:
    return {
        kpi: {**values, "paired_n": DEMO_PAIRED_N}
        for kpi, values in stats.items()
    }


def _components(candidate: GlobalCompositeCandidate) -> dict:
    priority_products = sorted({
        adjustment.product_name
        for adjustment in candidate.lot_adjustments
        if adjustment.action_kind == "LOT_PRIORITY"
    })
    superhotlot_products = sorted({
        adjustment.product_name
        for adjustment in candidate.lot_adjustments
        if adjustment.action_kind == "SET_SUPER_HOT"
    })
    return {
        "has_priority_bump": bool(priority_products),
        "has_superhotlot": bool(superhotlot_products),
        "priority_products": priority_products,
        "superhotlot_products": superhotlot_products,
    }


def build_demo_verification_results(
    solution_candidates: list[dict],
    t0: float,
) -> list[dict]:
    """실제 생성된 대응안 메타데이터에 데모 통계만 결합한다."""
    scenario_stats, scenario_forecasts, scenario_id = _scenario_for_t0(t0)
    results: list[dict] = []
    for raw_candidate in solution_candidates:
        candidate = GlobalCompositeCandidate(**raw_candidate)
        stats = _with_sample_size(scenario_stats[candidate.plan_id])
        target_stats = stats["q_time_min"]
        action_kind = (
            "INTERVAL"
            if not candidate.lot_adjustments
            else "DISPATCH_RULE_OVERRIDE"
        )
        results.append({
            "plan_id": candidate.plan_id,
            "target_toolgroups": list(candidate.target_toolgroups),
            "snapshot_time": t0,
            "baseline_scenario_id": f"DEMO_BASE_T{int(t0)}",
            "demo_mock": True,
            "demo_scenario_id": scenario_id,
            "verified_candidates": [{
                "label": candidate.plan_id,
                "name": {
                    "conservative": "보수적 조정안",
                    "standard": "표준 조정안",
                    "aggressive": "강화 조정안",
                }[candidate.plan_id],
                "target_kpi": "q_time_min",
                "action_rows": [{"action_kind": action_kind}],
                "whatif_scenario_group_id": f"DEMO_{candidate.plan_id.upper()}",
                "baseline_scenario_id": f"DEMO_BASE_T{int(t0)}",
                "paired_n": DEMO_PAIRED_N,
                "kpi_stats": stats,
                "target_kpi_stats": target_stats,
                "verdict": target_stats["verdict"],
                "paired_t_p": target_stats["paired_t_p"],
                "demo_mock": True,
                "demo_scenario_id": scenario_id,
                "per_tg_forecasts": (
                    scenario_forecasts.get(candidate.plan_id, {})
                    if scenario_forecasts
                    else {}
                ),
                "aggregation_rule": (
                    "복합 TG별 대응안-무대응 delta 중 가장 불리한 값을 점수에 사용"
                    if scenario_forecasts
                    else "단일 TG 대응안 delta"
                ),
                "plan_meta": {
                    "plan_id": candidate.plan_id,
                    "target_toolgroups": list(candidate.target_toolgroups),
                    "release_interval_delta_pct": (
                        candidate.release_interval_delta_pct
                    ),
                    "release_interval_delta_min": round(
                        candidate.release_interval_delta_pct * 0.6,
                        2,
                    ),
                    "cause_complexity": candidate.cause_complexity,
                    "components": _components(candidate),
                },
            }],
        })
    return results
