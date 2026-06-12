"""AG-004 solution_generator 단위 테스트 (T-01 ~ T-10).

LLM 호출 없이 동작하도록 작성한다 (OPENAI_USAGE_RULES.md §4).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.schemas.alert import BottleneckAlert, CascadeImpact, SeverityLevel
from agents.schemas.cause import CauseReport, ConsensusResult, SHAPFeature
from agents.schemas.solution import SimParamDelta, SolutionCandidate
from agents.solution_generator.llm_generator import generate_texts
from agents.solution_generator.rule_engine import (
    _SEVERITY_PCT,
    clip_interval_pct,
    generate_candidates,
)


# ── 픽스처 헬퍼 ───────────────────────────────────────────────────────────────

def _alert(
    severity: SeverityLevel = SeverityLevel.CRITICAL,
    toolgroup: str = "TG_TEST",
    at_risk_lots: float = 0.0,
) -> BottleneckAlert:
    return BottleneckAlert(
        toolgroup=toolgroup,
        severity=severity,
        composite_score=0.90,
        probability=0.85,
        snapshot_time=3000.0,
        impact=CascadeImpact(
            capacity_stress_score=0.8,
            ct_increase_min=30.0,
            at_risk_lots=at_risk_lots,
            affected_tgs=[],
            impact_score=0.7,
        ),
    )


def _cause(
    features: list[tuple[str, float, float]],
    toolgroup: str = "TG_TEST",
) -> CauseReport:
    """features: [(feature_name, shap_value, kpi_value), ...]"""
    return CauseReport(
        toolgroup=toolgroup,
        snapshot_time=3000.0,
        shap_top=[
            SHAPFeature(feature=f, shap_value=sv, kpi_value=kv)
            for f, sv, kv in features
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
        cause_summary="테스트용 원인 요약",
    )


# ── T-01: 기본 생성 ───────────────────────────────────────────────────────────

def test_t01_basic_generation():
    """CRITICAL wip 병목 → 대응안 3개 (보수/표준/강화) 생성."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=1.0)
    cause = _cause([("wip", 0.45, 5.0), ("wait_ratio", 0.20, 0.8)])

    result = generate_candidates(alert, cause)

    assert set(result.keys()) >= {"conservative", "standard", "aggressive"}
    for lv in ("conservative", "standard", "aggressive"):
        assert "release_interval_delta_pct" in result[lv]
        assert "priority_direction" in result[lv]
        assert "superhotlot_enable" in result[lv]
        assert "target_kpi" in result[lv]

    # CRITICAL severity: 보수=15%, 표준=22%, 강화=28%
    assert result["conservative"]["release_interval_delta_pct"] == 15.0
    assert result["standard"]["release_interval_delta_pct"] == 22.0
    assert result["aggressive"]["release_interval_delta_pct"] == 28.0


# ── T-02: 피처 파라미터 매핑 ──────────────────────────────────────────────────

@pytest.mark.parametrize("feature,expected_conservative_pct,expected_priority", [
    ("wip",                 8.0,  None),   # pct_level=same, LOW conservative=5%… HIGH conservative=12%
    ("wait_ratio",          8.0,  None),
    ("q_time_min",          0.0,  "UP"),   # pct_level=down → conservative=0%
    ("max_util",            8.0,  None),
    ("utilization_avg",     8.0,  None),
    ("available_tool_ratio", 12.0, None),  # pct_level=up → conservative=S_pct=12%
])
def test_t02_feature_parameter_mapping(feature, expected_conservative_pct, expected_priority):
    """각 피처별 규칙 테이블이 올바른 conservative 파라미터를 반환하는지 확인."""
    # MEDIUM severity: C=8, S=12, A=18
    alert = _alert(SeverityLevel.MEDIUM, at_risk_lots=0.0)
    cause = _cause([(feature, 0.50, 5.0)])

    result = generate_candidates(alert, cause)
    cons = result["conservative"]

    assert cons["release_interval_delta_pct"] == pytest.approx(expected_conservative_pct)
    assert cons["priority_direction"] == expected_priority


# ── T-03: 피처별 내구성 ───────────────────────────────────────────────────────

@pytest.mark.parametrize("feature", [
    "wip", "wait_ratio", "q_time_min", "max_util", "utilization_avg", "available_tool_ratio",
])
def test_t03_feature_robustness(feature):
    """6개 피처 각각에 대해 generate_candidates가 예외 없이 동작하는지 확인."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=0.0)
    cause = _cause([(feature, 0.30, 2.0)])

    result = generate_candidates(alert, cause)

    for lv in ("conservative", "standard", "aggressive"):
        assert isinstance(result[lv]["superhotlot_enable"], bool)


# ── T-04: 출력 shape ─────────────────────────────────────────────────────────

def test_t04_output_shape():
    """SolutionCandidate 스키마 통과 + model_dump()가 compare_agent 연동 가능한 dict 반환."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=2.0)
    cause = _cause([("wip", 0.40, 4.5)])
    result = generate_candidates(alert, cause)

    for rank, lv in enumerate(("conservative", "standard", "aggressive"), start=1):
        p = result[lv]
        candidate = SolutionCandidate(
            rank=rank,
            name=f"{lv} 조정안",
            target_kpi=p["target_kpi"],
            params=SimParamDelta(
                release_interval_delta_pct=p["release_interval_delta_pct"],
                priority_direction=p["priority_direction"],
                superhotlot_enable=p["superhotlot_enable"],
            ),
            expected_effect="기대 효과",
            rationale="선택 근거",
        )
        dumped = candidate.model_dump()
        assert dumped["rank"] == rank
        assert "params" in dumped
        assert "expected_effect" in dumped
        assert "rationale" in dumped


# ── T-05: Pydantic 검증 ───────────────────────────────────────────────────────

def test_t05_pydantic_valid():
    """SimParamDelta 유효한 값은 ValidationError 없이 통과."""
    p = SimParamDelta(
        release_interval_delta_pct=15.0,
        priority_direction="UP",
        superhotlot_enable=True,
    )
    assert p.priority_direction == "UP"
    assert p.superhotlot_enable is True


def test_t05_pydantic_invalid_priority():
    """SimParamDelta priority_direction에 잘못된 값 → ValidationError."""
    with pytest.raises(ValidationError):
        SimParamDelta(priority_direction="INVALID")


def test_t05_solution_candidate_defaults():
    """SolutionCandidate 기본값이 올바른지 확인."""
    c = SolutionCandidate()
    assert c.rank == 1
    assert c.rationale == ""
    assert c.params.superhotlot_enable is False


# ── T-06: SUPERHOTLOT 조건 ────────────────────────────────────────────────────

def test_t06_superhotlot_eligible_false():
    """eligible=False 피처(wip conservative)는 superhotlot=False."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=10.0)
    cause = _cause([("wip", 0.50, 8.0)])
    result = generate_candidates(alert, cause)
    # wip conservative eligible=False
    assert result["conservative"]["superhotlot_enable"] is False


def test_t06_superhotlot_cond4_aggressive():
    """at_risk_lots > 0 → aggressive wip CRITICAL/HIGH에서 superhotlot=True (cond4)."""
    for severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
        alert = _alert(severity, at_risk_lots=1.0)
        cause = _cause([("wip", 0.50, 8.0)])
        result = generate_candidates(alert, cause)
        # wip aggressive eligible=True, severity not LOW
        assert result["aggressive"]["superhotlot_enable"] is True, f"failed for {severity}"


def test_t06_superhotlot_severity_gate_standard():
    """eligible=True이어도 severity gate(MEDIUM)가 standard를 차단."""
    # available_tool_ratio standard eligible=True
    alert = _alert(SeverityLevel.MEDIUM, at_risk_lots=5.0)
    cause = _cause([("available_tool_ratio", 0.50, 0.30)])
    result = generate_candidates(alert, cause)
    # MEDIUM은 standard severity gate에 걸림 ({HIGH, CRITICAL}에 없음)
    assert result["standard"]["superhotlot_enable"] is False


def test_t06_superhotlot_cond2_aggressive():
    """avail < 0.40 → aggressive available_tool_ratio MEDIUM에서 superhotlot=True (cond2)."""
    # available_tool_ratio aggressive eligible=True, MEDIUM not LOW → gate pass
    # cond2: avail < 0.40 → kpi_value=0.35
    alert = _alert(SeverityLevel.MEDIUM, at_risk_lots=0.0)
    cause = _cause([("available_tool_ratio", 0.45, 0.35)])
    result = generate_candidates(alert, cause)
    assert result["aggressive"]["superhotlot_enable"] is True


# ── T-08: post-processing clip ────────────────────────────────────────────────

def test_t08_clip_above_severity_max():
    """제안값 35%가 severity 상한(CRITICAL=28%)으로 clip."""
    assert clip_interval_pct(35.0, SeverityLevel.CRITICAL) == pytest.approx(28.0)


def test_t08_clip_absolute_max():
    """제안값 35%가 LOW severity 상한(12%)으로 clip (절대 상한 30%보다 작음)."""
    assert clip_interval_pct(35.0, SeverityLevel.LOW) == pytest.approx(12.0)


def test_t08_clip_no_clip_needed():
    """제안값이 상한 이내이면 그대로 통과."""
    assert clip_interval_pct(20.0, SeverityLevel.HIGH) == pytest.approx(20.0)


def test_t08_absolute_max_cap():
    """제안값 35%가 HIGH severity 상한(23%)보다 크더라도 절대 상한 30%가 아닌 severity 상한 적용."""
    assert clip_interval_pct(35.0, SeverityLevel.HIGH) == pytest.approx(23.0)


# ── T-09: LLM 없이 동작 ──────────────────────────────────────────────────────

def test_t09_llm_fallback_no_api_key():
    """OPENAI_API_KEY 없을 때 _safe_fallback 텍스트가 반환되고 파이프라인 계속."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=0.0)
    cause = _cause([("wip", 0.40, 5.0)])
    candidates = generate_candidates(alert, cause)

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        texts = generate_texts(alert, cause, candidates)

    assert set(texts.keys()) == {"conservative", "standard", "aggressive"}
    for lv in ("conservative", "standard", "aggressive"):
        assert "expected_effect" in texts[lv]
        assert "rationale" in texts[lv]
        assert isinstance(texts[lv]["expected_effect"], str)
        assert isinstance(texts[lv]["rationale"], str)


def test_t09_llm_fallback_placeholder_key():
    """'your_'로 시작하는 플레이스홀더 API 키도 fallback 처리."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=3.0)
    cause = _cause([("q_time_min", 0.60, 45.0)])
    candidates = generate_candidates(alert, cause)

    with patch.dict(os.environ, {"OPENAI_API_KEY": "your_key_here"}):
        texts = generate_texts(alert, cause, candidates)

    for lv in ("conservative", "standard", "aggressive"):
        assert texts[lv]["rationale"] != ""


# ── T-10: 복수 피처 시나리오 ──────────────────────────────────────────────────

def test_t10_scenario_s4_triple_degradation():
    """S4: wip+wait_ratio+q_time_min 동시 악화 → HITL=True, 전 레벨 aggressive pct."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause([
        ("wip",        0.40, 6.0),
        ("wait_ratio", 0.35, 0.9),
        ("q_time_min", 0.30, 50.0),
    ])

    result = generate_candidates(alert, cause)

    assert result["hitl_escalation_recommended"] is True
    agg_pct = _SEVERITY_PCT[SeverityLevel.CRITICAL]["aggressive"]
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["release_interval_delta_pct"] == pytest.approx(agg_pct)
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True


def test_t10_scenario_s2_wait_qtime():
    """S2: wait_ratio+q_time_min 악화(wip 없음) → HITL=False, priority=UP, superhotlot=True."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=0.0)
    cause = _cause([
        ("wait_ratio", 0.45, 0.85),
        ("q_time_min", 0.35, 40.0),
    ])

    result = generate_candidates(alert, cause)

    assert result["hitl_escalation_recommended"] is False
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True


def test_t10_scenario_s3_util_saturation():
    """S3: max_util+utilization_avg 동시 포화 → HITL=True."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=0.0)
    cause = _cause([
        ("max_util",         0.50, 0.95),
        ("utilization_avg",  0.30, 0.88),
    ])

    result = generate_candidates(alert, cause)

    assert result["hitl_escalation_recommended"] is True
    # aggressive만 superhotlot=True; 나머지는 max_util 규칙 따름 (eligible=False)
    assert result["aggressive"]["superhotlot_enable"] is True
