"""
solution_generator 수동 테스트 스크립트.

시나리오를 바꿔가며 실행하려면 SCENARIO 변수를 수정한다.
실행: .venv/bin/python3 scripts/run_solution_generator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.schemas.alert import BottleneckAlert, CascadeImpact, SeverityLevel
from agents.schemas.cause import (
    CauseJudgment,
    CauseReport,
    ConsensusResult,
    SHAPFeature,
    TrendInsight,
)
from agents.solution_generator.rule_engine import clip_interval_pct, generate_candidates
from agents.verification_agent.sim_executor import (
    _DUE_THRESHOLD_CRITICAL_MIN,
    _DUE_THRESHOLD_WARN_MIN,
    _make_lot_actions,
)

# ── mock lot 데이터 (DB 없이 lot 레벨 액션 시뮬용) ────────────────────────────
# t0 = 72000.0 기준 time_to_due 계산
_T0 = 72000.0
_MOCK_LOTS = [
    {"lot_id": "Lot_001", "due_date_sim": _T0 + 2000.0,  "priority": 10, "is_super_hot": False, "route_id": "Route_A"},  # 위험
    {"lot_id": "Lot_002", "due_date_sim": _T0 + 6000.0,  "priority": 10, "is_super_hot": False, "route_id": "Route_A"},  # 경고
    {"lot_id": "Lot_003", "due_date_sim": _T0 + 15000.0, "priority": 10, "is_super_hot": False, "route_id": "Route_A"},  # 안전
    {"lot_id": "Lot_004", "due_date_sim": _T0 + 3500.0,  "priority": 20, "is_super_hot": False, "route_id": "Route_B"},  # 위험 (HotLot)
    {"lot_id": "Lot_005", "due_date_sim": _T0 + 25000.0, "priority": 10, "is_super_hot": False, "route_id": "Route_A"},  # 안전
]

# ─────────────────────────────────────────────────────────────────────────────
# 시나리오 선택 (1~5 중 하나로 변경)
# ─────────────────────────────────────────────────────────────────────────────
SCENARIO = 5

SCENARIOS = {
    # 단일 원인: WIP 과부하
    1: dict(
        toolgroup="LithoTrack_FE_95",
        primary_category="WIP_누적",
        secondary_causes=[],
        confidence="HIGH",
        reasoning="최근 2시간 wip가 임계값 대비 38% 초과. wip_delta_120=+4.2로 증가 추세 확인.",
        cause_summary=(
            "LithoTrack_FE_95 TG에서 WIP 과부하가 주원인으로 판정됩니다. "
            "현재 wip=18.4(임계값 13.3 대비 +38%), 최근 2시간 증가율 wip_delta_120=+4.2로 "
            "지속 악화 추세입니다. 투입 간격 조정을 통한 WIP 억제가 필요합니다."
        ),
        shap_top=[
            ("wip",            +0.4821, 18.4),
            ("wip_delta_120",  +0.2103,  4.2),
            ("wait_ratio",     +0.0812,  0.31),
            ("utilization_avg",+0.0441,  0.82),
        ],
        at_risk_lots=0.0,
        probability=0.91,
    ),

    # 단일 원인: 대기 누적 (q_time 급증)
    2: dict(
        toolgroup="Diffusion_FE_120",
        primary_category="대기_누적",
        secondary_causes=[],
        confidence="HIGH",
        reasoning="q_time_min이 CQT 임계값의 83%에 도달. wait_ratio도 0.68로 급증.",
        cause_summary=(
            "Diffusion_FE_120에서 대기 누적이 주원인입니다. "
            "q_time_min=41.2분(임계값 50분의 83%), wait_ratio=0.68로 "
            "두 지표 모두 임계 수준에 근접했습니다. "
            "현재 대기 중인 lot의 우선 처리가 시급합니다."
        ),
        shap_top=[
            ("q_time_min",           +0.5130, 41.2),
            ("wait_ratio",           +0.3022,  0.68),
            ("q_time_min_delta_120", +0.1841,  8.5),
            ("wip",                  +0.0302, 11.1),
        ],
        at_risk_lots=3.0,
        probability=0.88,
    ),

    # 복합 원인 S1: WIP_누적 + 공급_부족
    3: dict(
        toolgroup="Implant_128",
        primary_category="WIP_누적",
        secondary_causes=["공급_부족"],
        confidence="HIGH",
        reasoning=(
            "wip 임계 초과(+52%)와 동시에 available_tool_ratio=0.27로 장비 3대 다운. "
            "WIP 과부하와 공급 부족이 동시 발생한 최악 조합."
        ),
        cause_summary=(
            "Implant_128에서 WIP 과부하(wip=20.3, +52% 초과)와 장비 공급 부족 "
            "(available_tool_ratio=0.27, 장비 3대 고장)이 동시 발생했습니다. "
            "두 원인이 상호 악화시키는 구조로 즉각적인 강력 조치가 필요합니다."
        ),
        shap_top=[
            ("wip",                  +0.5241, 20.3),
            ("available_tool_ratio", +0.3812,  0.27),
            ("wip_delta_120",        +0.1503,  5.1),
            ("utilization_avg",      +0.0821,  0.94),
        ],
        at_risk_lots=7.0,
        probability=0.96,
    ),

    # 복합 원인 S3: 설비_포화 + 대기_누적 → HITL
    4: dict(
        toolgroup="Litho_FE_92",
        primary_category="설비_포화",
        secondary_causes=["대기_누적"],
        confidence="MEDIUM",
        reasoning=(
            "utilization_avg=0.97로 가동률 한계. wait_ratio=0.71로 대기 급증. "
            "장비 증설 없이는 투입 조정만으로 근본 해결 불가."
        ),
        cause_summary=(
            "Litho_FE_92의 가동률이 0.97로 사실상 포화 상태이며, "
            "대기 비율(wait_ratio=0.71)도 동시에 급증하고 있습니다. "
            "투입 간격 조정으로 단기 완화는 가능하나, "
            "장비 증설 또는 PM 일정 재조정 없이는 구조적 해결이 어렵습니다."
        ),
        shap_top=[
            ("utilization_avg",           +0.4931,  0.97),
            ("max_util",                  +0.3102,  0.99),
            ("wait_ratio",                +0.2241,  0.71),
            ("utilization_avg_delta_120", +0.0803,  0.05),
        ],
        at_risk_lots=2.0,
        probability=0.83,
    ),

    # 복합 원인 S4: 3개 이상 동시 악화 → HITL
    5: dict(
        toolgroup="DE_FE_71",
        primary_category="WIP_누적",
        secondary_causes=["대기_누적", "공급_부족"],
        confidence="HIGH",
        reasoning=(
            "wip +61% 초과, q_time_min=48분, available_tool_ratio=0.31. "
            "3개 카테고리 동시 악화. FAB 전체에 미치는 파급 위험 매우 높음."
        ),
        cause_summary=(
            "DE_FE_71에서 WIP 과부하(+61%), 대기 급증(q_time_min=48분), "
            "장비 공급 부족(available_tool_ratio=0.31)이 동시에 발생했습니다. "
            "세 카테고리의 동시 악화로 파급 위험이 매우 높으며 "
            "운영자 즉각 개입이 강력히 권고됩니다."
        ),
        shap_top=[
            ("wip",                  +0.4812, 21.5),
            ("q_time_min",           +0.3021, 48.0),
            ("available_tool_ratio", +0.2731,  0.31),
            ("wait_ratio",           +0.1502,  0.73),
        ],
        at_risk_lots=12.0,
        probability=0.98,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(s: dict) -> tuple[BottleneckAlert, CauseReport]:
    alert = BottleneckAlert(
        toolgroup=s["toolgroup"],
        severity=SeverityLevel.CRITICAL,
        composite_score=0.92,
        probability=s["probability"],
        snapshot_time=72000.0,
        impact=CascadeImpact(
            capacity_stress_score=0.85,
            ct_increase_min=45.0,
            at_risk_lots=s["at_risk_lots"],
            affected_tgs=[],
            impact_score=0.88,
        ),
    )
    cause = CauseReport(
        toolgroup=s["toolgroup"],
        snapshot_time=72000.0,
        shap_top=[
            SHAPFeature(feature=f, shap_value=sv, kpi_value=kv)
            for f, sv, kv in s["shap_top"]
        ],
        trend_top=[],
        upstream_suspects=[],
        sim_forecast=None,
        consensus=ConsensusResult(
            agreed_features=[],
            conflicted_features=[],
            upstream_aligns=False,
            sim_aligns=False,
        ),
        judgment=CauseJudgment(
            primary_category=s["primary_category"],
            primary_cause=s["shap_top"][0][0],
            primary_confidence=s["confidence"],   # type: ignore[arg-type]
            primary_reasoning=s["reasoning"],
            secondary_causes=s["secondary_causes"],
            cause_summary=s["cause_summary"],
        ),
        cause_summary=s["cause_summary"],
    )
    return alert, cause


def print_separator(char="─", width=60):
    print(char * width)


def main():
    s = SCENARIOS[SCENARIO]
    print_separator("═")
    print(f"  시나리오 {SCENARIO}: {s['toolgroup']} / {s['primary_category']}", end="")
    if s["secondary_causes"]:
        print(f" + {', '.join(s['secondary_causes'])}", end="")
    print()
    print_separator("═")

    alert, cause = build_inputs(s)

    # ── 1. 규칙 엔진 ────────────────────────────────────────────────────────
    print("\n[1단계] 규칙 엔진 실행 중...")
    result = generate_candidates(alert, cause)

    if result is None:
        print("  ⚠ judgment=None → 대응안 생성 불가")
        return

    print(f"  시나리오 감지: {_infer_scenario(s)}")

    print()
    for lv, label in [("conservative","보수안"), ("standard","표준안"), ("aggressive","강화안")]:
        p = result[lv]
        pct = p["release_interval_delta_pct"]
        pri = p["priority_direction"] or "변경없음"
        sh  = "ON" if p["superhotlot_enable"] else "OFF"
        print(f"  {label}: interval +{pct:.0f}%  |  priority {pri}  |  superhotlot {sh}")

    # ── 2. 대응안별 파라미터 + lot 레벨 적용 결과 ──────────────────────────────
    print()
    print_separator()
    LEVELS = [("conservative","보수안"), ("standard","표준안"), ("aggressive","강화안")]
    for lv, label in LEVELS:
        p = result[lv]
        clipped = clip_interval_pct(p["release_interval_delta_pct"])

        print(f"\n  ▶ {label} (rank {LEVELS.index((lv,label))+1})")
        print(f"    release_interval : +{clipped:.0f}%")
        print(f"    priority_dir     : {p['priority_direction'] or '변경없음'}")
        print(f"    superhotlot      : {'ON' if p['superhotlot_enable'] else 'OFF'}")
        print(f"    target_kpi       : {p['target_kpi']}")

        # lot 레벨 액션 시뮬 (mock lots 기준)
        lot_actions = _make_lot_actions(
            lots=_MOCK_LOTS,
            t0=_T0,
            priority_direction=p["priority_direction"],
            superhotlot_enable=p["superhotlot_enable"],
            tg=alert.toolgroup,
        )

        if lot_actions:
            print(f"    --- lot 레벨 액션 (mock, t0={_T0:.0f}) ---")
            print(f"    {'lot_id':<20} {'action':<16} {'payload'}")
            for row in lot_actions:
                payload = json.loads(row["payload_json"])
                ttd = next(
                    (l["due_date_sim"] - _T0 for l in _MOCK_LOTS if l["lot_id"] == row["lot_id"]), 0
                )
                zone = (
                    "위험" if ttd <= _DUE_THRESHOLD_CRITICAL_MIN
                    else "경고" if ttd <= _DUE_THRESHOLD_WARN_MIN
                    else "안전"
                )
                print(f"    {row['lot_id']:<20} {row['action_kind']:<16} {payload}  [{zone} ttd={ttd:.0f}분]")
        else:
            print(f"    --- lot 레벨 액션 없음 (모든 lot 안전 구간 또는 조건 미충족) ---")

    print()
    print_separator("═")


def _infer_scenario(s: dict) -> str:
    if not s["secondary_causes"]:
        return "없음 (단일 원인)"
    all_cats = {s["primary_category"]} | set(s["secondary_causes"])
    if len(all_cats) >= 3:
        return "S4 (3개 이상 동시 악화)"
    if {"WIP_누적","공급_부족"}.issubset(all_cats):
        return "S1 (WIP_누적 + 공급_부족)"
    if {"대기_누적","WIP_누적"}.issubset(all_cats):
        return "S2 (대기_누적 + WIP_누적)"
    if {"설비_포화","공급_부족"}.issubset(all_cats):
        return "S5 (설비_포화 + 공급_부족)"
    if "설비_포화" in all_cats:
        return "S3 (설비_포화 포함 복합)"
    return "미분류"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SCENARIO = int(sys.argv[1])
    main()
