"""프론트엔드 UI 작업용 샘플 compare/2.0 결과 JSON 생성.

새 9블록 스키마(meta·current_state·cause·cascade·action_options·recommendation·decision_meta·data_quality·approval_info)로
실제 LLM 호출을 통해 시나리오별 샘플 JSON을 생성한다.

사용:
    .venv/bin/python scripts/gen_mock_compare_output.py [scenario]
    scenario: no_effect (default) | clear_winner | equivalent
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

load_dotenv(ROOT / ".env")

from agents.compare_agent.node import (  # noqa: E402
    SCHEMA_VERSION,
    _build_action_option,
    _build_cascade_block,
    _build_cause_block,
    _build_current_state_block,
    _build_current_state_option,
    _build_data_quality,
    _build_decision_meta,
    _build_hitl_prompt,
    _build_meta,
    _build_recommendation_block,
    _rank_candidates,
)
from agents.compare_agent.recommendation import generate_recommendation  # noqa: E402


# ── 시나리오 정의 ────────────────────────────────────────────────────────────

BASE_CI = {
    "toolgroup": "DefMEt_FE_118",
    "process_name": "글로벌 플랜 A/B",
    "scenario_type": "global_plan",
    "scenario_name": "글로벌 플랜 A/B",
    "anchor_toolgroup": "DefMEt_FE_118",
    "target_toolgroups": [
        "DefMEt_FE_118", "Diffusion_FE_120", "Delay_32",
        "DE_FE_86", "LithoTrack_FE_115", "Diffusion_FE_127", "Diffusion_FE_44",
    ],
    "severity": "Critical",
    "snapshot_time": 4740.0,
    "t0": 4740.0,
    "horizon_min": 240,
    "bottleneck_info": {
        "tool_group": "DefMEt_FE_118",
        "risk_score": 0.85,
        "wip_count": 820,
        "avg_queue_time_min": 145.3,
        "wait_ratio": 1.42,
        "available_tool_ratio": 0.68,
        "utilization_avg": 0.91,
    },
    "cause_context": {
        "cause_summary": "DefMEt_FE_118의 평균 큐 대기시간이 임계치를 초과하며 WIP 누적이 가속되고 있습니다. 업스트림 LithoTrack_FE_115에서의 투입 변동이 주요 원인으로 지목됩니다.",
        "shap_top": [
            {"feature": "q_time_min", "shap_value": 0.412, "kpi_value": 145.3},
            {"feature": "wip", "shap_value": 0.287, "kpi_value": 820.0},
            {"feature": "wait_ratio", "shap_value": 0.156, "kpi_value": 1.42},
        ],
        "trend_top": [
            {"feature": "wip", "slope_per_hour": 12.5},
            {"feature": "q_time_min", "slope_per_hour": 8.2},
        ],
        "upstream_suspects": ["LithoTrack_FE_115", "Diffusion_FE_44"],
        "consensus_summary": "SHAP·트렌드·업스트림이 모두 WIP 증가를 지목하며, G* 검증에서도 유의 확인됨.",
        "consensus_confidence": "HIGH",
        "sim_forecast": {
            "t0": 4740.0,
            "t_future": 4860.0,
            "gets_worse": True,
            "kpi_delta": {
                "q_time_min": {"now": 145.3, "future": 178.5, "delta": 33.2, "pct_change": 22.8, "reliability": "HIGH"},
                "wip": {"now": 820, "future": 905, "delta": 85, "pct_change": 10.4, "reliability": "HIGH"},
                "wait_ratio": {"now": 1.42, "future": 1.58, "delta": 0.16, "pct_change": 11.3, "reliability": "MED"},
            },
        },
    },
    "cascade_impact": {
        "capacity_stress_score": 0.78,
        "ct_increase_min": 45.2,
        "at_risk_lots": 820,
        "affected_tgs": ["DE_FE_86", "Diffusion_FE_127", "LithoTrack_FE_115"],
        "impact_score": 0.81,
    },
}


def _plan_meta(release_interval: float) -> dict:
    return {
        "target_toolgroups": [
            "DefMEt_FE_118", "Diffusion_FE_120", "Delay_32",
            "DE_FE_86", "LithoTrack_FE_115", "Diffusion_FE_127", "Diffusion_FE_44",
        ],
        "release_interval_minutes": release_interval,
        "current_interval_minutes": 60.0,
        "release_interval_delta_min": round(release_interval - 60.0, 4),
        "lot_priority_rule": "HIGH_WIP_FIRST",
        "superhotlot_enable": True,
        "expected_effect": f"Release Interval {release_interval}분으로 조정 시 WIP 안정화 기대",
        "confidence": 0.7,
    }


def _make_kpi_stats(scenario: str, label: str) -> dict:
    """시나리오·라벨별 kpi_stats — paired t-test 통계 mock."""
    if scenario == "no_effect":
        # 모든 KPI verdict=unchanged, mean_delta=0
        return {
            "q_time_min":           {"mean_delta": 0.0, "paired_t_p": 0.8, "verdict": "unchanged", "ci_lo": -0.8, "ci_hi": 0.8},
            "wip":                  {"mean_delta": 0.0, "paired_t_p": 0.8, "verdict": "unchanged", "ci_lo": -0.5, "ci_hi": 0.5},
            "wait_ratio":           {"mean_delta": 0.0, "paired_t_p": 0.8, "verdict": "unchanged", "ci_lo": -0.01, "ci_hi": 0.01},
            "utilization_avg":      {"mean_delta": 0.0, "paired_t_p": 0.8, "verdict": "unchanged", "ci_lo": -0.005, "ci_hi": 0.005},
            "available_tool_ratio": {"mean_delta": 0.0, "paired_t_p": 0.8, "verdict": "unchanged", "ci_lo": -0.005, "ci_hi": 0.005},
        }
    if scenario == "clear_winner":
        # A가 명확한 1위 — 큰 개선, B는 작은 개선
        if label == "A":
            return {
                "q_time_min":           {"mean_delta": -45.2, "paired_t_p": 0.012, "verdict": "improved", "ci_lo": -55.0, "ci_hi": -35.0},
                "wip":                  {"mean_delta": -55.0, "paired_t_p": 0.015, "verdict": "improved", "ci_lo": -70.0, "ci_hi": -40.0},
                "wait_ratio":           {"mean_delta": -0.12, "paired_t_p": 0.020, "verdict": "improved", "ci_lo": -0.18, "ci_hi": -0.06},
                "utilization_avg":      {"mean_delta": -0.02, "paired_t_p": 0.30,  "verdict": "unchanged", "ci_lo": -0.05, "ci_hi": 0.01},
                "available_tool_ratio": {"mean_delta": 0.04,  "paired_t_p": 0.18,  "verdict": "unchanged", "ci_lo": -0.01, "ci_hi": 0.09},
            }
        return {
            "q_time_min":           {"mean_delta": -8.0,  "paired_t_p": 0.20, "verdict": "unchanged", "ci_lo": -18.0, "ci_hi": 2.0},
            "wip":                  {"mean_delta": -10.0, "paired_t_p": 0.25, "verdict": "unchanged", "ci_lo": -22.0, "ci_hi": 2.0},
            "wait_ratio":           {"mean_delta": -0.03, "paired_t_p": 0.30, "verdict": "unchanged", "ci_lo": -0.08, "ci_hi": 0.02},
            "utilization_avg":      {"mean_delta": -0.01, "paired_t_p": 0.40, "verdict": "unchanged", "ci_lo": -0.04, "ci_hi": 0.02},
            "available_tool_ratio": {"mean_delta": 0.02,  "paired_t_p": 0.35, "verdict": "unchanged", "ci_lo": -0.02, "ci_hi": 0.06},
        }
    # equivalent — A, B 둘 다 유사한 개선 (composite_score 차이 ε 이내)
    return {
        "q_time_min":           {"mean_delta": -22.0, "paired_t_p": 0.04, "verdict": "improved", "ci_lo": -32.0, "ci_hi": -12.0},
        "wip":                  {"mean_delta": -25.0, "paired_t_p": 0.05, "verdict": "improved", "ci_lo": -38.0, "ci_hi": -12.0},
        "wait_ratio":           {"mean_delta": -0.06, "paired_t_p": 0.08, "verdict": "improved", "ci_lo": -0.10, "ci_hi": -0.02},
        "utilization_avg":      {"mean_delta": -0.01, "paired_t_p": 0.50, "verdict": "unchanged", "ci_lo": -0.04, "ci_hi": 0.02},
        "available_tool_ratio": {"mean_delta": 0.02,  "paired_t_p": 0.40, "verdict": "unchanged", "ci_lo": -0.02, "ci_hi": 0.06},
    }


def _candidate(label: str, release_interval: float, scenario: str) -> dict:
    pm = _plan_meta(release_interval)
    kpi_stats = _make_kpi_stats(scenario, label)
    p_val = kpi_stats.get("q_time_min", {}).get("paired_t_p")
    conf = round(max(0.0, min(1.0, 1.0 - float(p_val))), 4) if p_val is not None else 0.5
    desc = (
        f"[플랜 {label}] Release Interval 60.0분 → {release_interval}분 "
        f"(+{release_interval-60.0:.1f}분) | 투입 우선순위 HIGH_WIP_FIRST | "
        f"SUPERHOTLOT 활성화 (위험 lot 820개)"
    )
    return {
        "label": label,
        "action_kind": "DISPATCH_RULE_OVERRIDE",
        "description": desc,
        "simulation_confidence": conf,
        "kpi_stats": kpi_stats,
        "verdict": kpi_stats.get("q_time_min", {}).get("verdict", "unknown"),
        "paired_n": 30,
        "paired_t_p": p_val,
        "plan_meta": pm,
    }


SCENARIOS = {
    "no_effect": {
        "candidates": [_candidate("A", 62.2, "no_effect"), _candidate("B", 64.2, "no_effect")],
    },
    "clear_winner": {
        "candidates": [_candidate("A", 62.2, "clear_winner"), _candidate("B", 64.2, "clear_winner")],
    },
    "equivalent": {
        "candidates": [_candidate("A", 62.2, "equivalent"), _candidate("B", 64.2, "equivalent")],
    },
}


# ── Pipeline 실행 ────────────────────────────────────────────────────────────

def run_scenario(scenario: str) -> Path:
    print(f"\n{'=' * 60}\n시나리오: {scenario}\n{'=' * 60}")

    candidates = SCENARIOS[scenario]["candidates"]

    # 1) rank 적용 (scorer)
    print("[1/4] scorer 적용 (tie-breaker chain)...")
    candidates, _scored, decision_info = _rank_candidates(candidates)
    print(f"      decision_status: {decision_info['decision_status']}")
    print(f"      top_label: {decision_info['top_label']}")
    print(f"      tiebreaker_used: {decision_info.get('tiebreaker_used')}")
    for step in decision_info.get("tiebreaker_chain_evaluated", []):
        print(f"      chain: {step}")

    # 2) data_quality 계산
    data_quality = _build_data_quality(candidates)
    print(f"[2/4] data_quality.status: {data_quality['status']}")

    # 3) LLM 호출
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 .env에 없습니다.")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.2, max_tokens=1500)

    top_label = decision_info["top_label"]
    top_candidate = next((c for c in candidates if c["label"] == top_label), candidates[0])

    ci_with_dq_hint = {**BASE_CI, "data_quality_hint": data_quality}

    print("[3/4] LLM 호출 중...")
    rec = generate_recommendation(
        ci=ci_with_dq_hint,
        candidates=candidates,
        decision_info=decision_info,
        top_candidate=top_candidate,
        llm=llm,
    )
    print(f"      headline: {rec.headline}")
    print(f"      confidence_level: {rec.confidence_level}")
    print(f"      why_recommended.selected_by: {rec.why_recommended.selected_by}")
    print(f"      why_recommended.tiebreaker_chain: {rec.why_recommended.tiebreaker_chain}")
    print(f"      immediate_actions: {len(rec.immediate_actions)}개")
    print(f"      monitoring_kpis: {len(rec.monitoring_kpis)}개")

    # 4) 9블록 빌드 + 저장
    print("[4/4] 9블록 빌드...")
    meta = _build_meta(BASE_CI)
    current_state = _build_current_state_block(BASE_CI)
    cause = _build_cause_block(BASE_CI)
    cascade = _build_cascade_block(BASE_CI)
    action_options = [_build_current_state_option(BASE_CI, current_state["kpi"])]
    for c in candidates:
        action_options.append(_build_action_option(c, current_state["kpi"], decision_info))
    recommendation = _build_recommendation_block(rec, decision_info, top_candidate)
    decision_meta = _build_decision_meta(decision_info)

    approval_info = {
        "status": "승인",
        "approved_by": "AUTO",
        "approved_role": "SYSTEM",
        "approved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "comment": "자동 승인 (AUTO_APPROVE 모드)",
        "rejection_reason": None,
    }

    result_v2 = {
        "meta": meta,
        "current_state": current_state,
        "cause": cause,
        "cascade": cascade,
        "action_options": action_options,
        "recommendation": recommendation,
        "decision_meta": decision_meta,
        "data_quality": data_quality,
        "approval_info": approval_info,
    }

    out_dir = ROOT / "compare_agent_out"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compare_v2_DefMEt_FE_118_{scenario}_{ts}.json"
    out_path.write_text(json.dumps(result_v2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      저장: {out_path}")

    # 어설션 검증
    print("[검증]")
    _assert_schema(result_v2, scenario)
    print(f"\nHITL 프롬프트 미리보기:\n")
    print(_build_hitl_prompt({**result_v2, "data_quality": data_quality, "decision_meta": decision_meta}))

    return out_path


def _assert_schema(result: dict, scenario: str) -> None:
    """plan.md의 검증 항목 모두 통과 확인."""
    # 9개 최상위 블록
    expected_keys = {
        "meta", "current_state", "cause", "cascade", "action_options",
        "recommendation", "decision_meta", "data_quality", "approval_info",
    }
    missing = expected_keys - set(result.keys())
    assert not missing, f"missing top-level keys: {missing}"
    print(f"  ✓ 9개 최상위 블록 모두 존재")

    # action_options[0] = 현재상태
    opts = result["action_options"]
    assert opts[0]["label"] == "현재상태", f"[0].label != '현재상태': {opts[0]['label']}"
    assert opts[0]["is_baseline"] is True
    print(f"  ✓ action_options[0].label == '현재상태'")

    # action_options[1] = A
    a_opt = opts[1]
    assert a_opt["label"] == "A", f"[1].label != 'A': {a_opt['label']}"
    print(f"  ✓ action_options[1].label == 'A' (배지 미포함)")

    # target_toolgroups는 list
    assert isinstance(a_opt["target_toolgroups"], list)
    assert len(a_opt["target_toolgroups"]) == 7
    print(f"  ✓ A.target_toolgroups가 list (길이 {len(a_opt['target_toolgroups'])})")

    # params.release_interval_minutes
    assert a_opt["params"]["release_interval_minutes"] == 62.2
    print(f"  ✓ A.params.release_interval_minutes == 62.2")

    # kpi_impact.q_time_min.now == 145.3
    assert a_opt["kpi_impact"]["q_time_min"]["now"] == 145.3
    print(f"  ✓ A.kpi_impact.q_time_min.now == 145.3 (현재상태 기준점 적용)")

    # 시나리오별 검증
    dq = result["data_quality"]
    rec = result["recommendation"]
    if scenario == "no_effect":
        assert a_opt["badge"] == "tentative"
        assert a_opt["is_recommended"] is True
        print(f"  ✓ A.badge == 'tentative' (no_effect 케이스)")
        assert dq["status"] == "warning"
        print(f"  ✓ data_quality.status == 'warning'")
        assert rec["recommendation_status"] == "tentative_no_effect"
        chain = rec["why_recommended"]["tiebreaker_chain"]
        assert chain, "tiebreaker_chain이 비어있음"
        print(f"  ✓ recommendation.why_recommended.tiebreaker_chain: {chain}")
    elif scenario == "clear_winner":
        assert a_opt["badge"] == "ai_recommended"
        assert dq["status"] == "ok"
        assert rec["recommendation_status"] == "ai_recommended"
        print(f"  ✓ A.badge == 'ai_recommended', data_quality.status == 'ok'")
    elif scenario == "equivalent":
        assert a_opt["badge"] == "equivalent_tiebreak"
        assert rec["recommendation_status"] == "equivalent_tiebreak"
        print(f"  ✓ A.badge == 'equivalent_tiebreak'")

    # monitoring_kpis가 객체
    if rec.get("monitoring_kpis"):
        m = rec["monitoring_kpis"][0]
        assert isinstance(m, dict) and set(m.keys()) >= {"kpi", "target", "check_after_min"}
        print(f"  ✓ recommendation.monitoring_kpis[0] 객체 (keys: kpi/target/check_after_min)")

    # headline 길이
    h = rec["headline"]
    print(f"  ✓ recommendation.headline 길이 {len(h)}자")


def main() -> None:
    scenarios = sys.argv[1:] if len(sys.argv) > 1 else ["no_effect"]
    for s in scenarios:
        if s not in SCENARIOS:
            print(f"unknown scenario: {s}, available: {list(SCENARIOS)}")
            continue
        run_scenario(s)


if __name__ == "__main__":
    main()
