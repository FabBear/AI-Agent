"""AG-004 solution_generator 단위 테스트."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.schemas.alert import BottleneckAlert, CascadeImpact, SeverityLevel
from agents.schemas.cause import (
    CauseJudgment,
    CauseReport,
    ConsensusResult,
    SHAPFeature,
)
from agents.schemas.solution import SimParamDelta, SolutionCandidate
from agents.solution_generator.llm_generator import generate_texts
from agents.solution_generator.rule_engine import (
    _RELEASE_PCT,
    _SEVERITY_PCT,
    clip_interval_pct,
    generate_candidates,
)


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


def _judgment(
    primary_category: str = "WIP_누적",
    secondary_causes: list[str] | None = None,
    confidence: str = "HIGH",
) -> CauseJudgment:
    return CauseJudgment(
        primary_category=primary_category,
        primary_cause="wip",
        primary_confidence=confidence,  # type: ignore[arg-type]
        primary_reasoning=f"{primary_category} 악화 감지 (테스트)",
        secondary_causes=secondary_causes or [],
        cause_summary=f"{primary_category} 기반 판정 (테스트)",
    )


def _cause(
    primary_category: str = "WIP_누적",
    secondary_causes: list[str] | None = None,
    toolgroup: str = "TG_TEST",
    shap_features: list[tuple[str, float, float]] | None = None,
) -> CauseReport:
    shap = [
        SHAPFeature(feature=f, shap_value=sv, kpi_value=kv)
        for f, sv, kv in (shap_features or [("wip", 0.45, 5.0)])
    ]
    return CauseReport(
        toolgroup=toolgroup,
        snapshot_time=3000.0,
        shap_top=shap,
        trend_top=[],
        upstream_suspects=[],
        sim_forecast=None,
        consensus=ConsensusResult(
            agreed_features=[],
            conflicted_features=[],
            upstream_aligns=False,
            sim_aligns=False,
        ),
        judgment=_judgment(primary_category, secondary_causes),
        cause_summary=f"{primary_category} 악화 — 테스트용 원인 요약",
    )


def _cause_no_judgment(toolgroup: str = "TG_TEST") -> CauseReport:
    return CauseReport(
        toolgroup=toolgroup,
        snapshot_time=3000.0,
        shap_top=[SHAPFeature(feature="wip", shap_value=0.45, kpi_value=5.0)],
        trend_top=[],
        upstream_suspects=[],
        sim_forecast=None,
        consensus=ConsensusResult(
            agreed_features=[],
            conflicted_features=[],
            upstream_aligns=False,
            sim_aligns=False,
        ),
        judgment=None,
        cause_summary="판정 없음",
    )


def test_t01_basic_generation():
    """CRITICAL WIP_누적 → 대응안 3개 생성."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("WIP_누적")

    result = generate_candidates(alert, cause)

    assert result is not None
    assert set(result.keys()) >= {"conservative", "standard", "aggressive"}
    assert result["conservative"]["release_interval_delta_pct"] == pytest.approx(15.0)
    assert result["standard"]["release_interval_delta_pct"] == pytest.approx(22.0)
    assert result["aggressive"]["release_interval_delta_pct"] == pytest.approx(28.0)


@pytest.mark.parametrize("category,expected_cons_pct,expected_cons_priority", [
    ("WIP_누적", 15.0, None),
    ("대기_누적", 0.0, "UP"),
    ("설비_포화", 15.0, None),
    ("공급_부족", 22.0, None),
])
def test_t02_category_parameter_mapping(category, expected_cons_pct, expected_cons_priority):
    """카테고리 규칙 테이블이 CRITICAL conservative 파라미터를 반환한다."""
    result = generate_candidates(_alert(SeverityLevel.CRITICAL), _cause(category))

    assert result is not None
    cons = result["conservative"]
    assert cons["release_interval_delta_pct"] == pytest.approx(expected_cons_pct)
    assert cons["priority_direction"] == expected_cons_priority


def test_t02_severity_parameter_mapping():
    """동일 카테고리도 severity에 따라 조정폭이 달라진다."""
    result = generate_candidates(_alert(SeverityLevel.MEDIUM), _cause("WIP_누적"))

    assert result is not None
    assert result["conservative"]["release_interval_delta_pct"] == pytest.approx(8.0)
    assert result["standard"]["release_interval_delta_pct"] == pytest.approx(12.0)
    assert result["aggressive"]["release_interval_delta_pct"] == pytest.approx(18.0)


@pytest.mark.parametrize("category", ["WIP_누적", "대기_누적", "설비_포화", "공급_부족"])
def test_t03_category_robustness(category):
    """4개 카테고리 각각에 대해 generate_candidates가 예외 없이 동작한다."""
    result = generate_candidates(_alert(SeverityLevel.HIGH), _cause(category))

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert isinstance(result[lv]["superhotlot_enable"], bool)
        assert result[lv]["target_kpi"] != ""


def test_t04_output_shape():
    """SolutionCandidate 스키마가 expected_effect/rationale까지 직렬화한다."""
    result = generate_candidates(_alert(SeverityLevel.HIGH, at_risk_lots=2.0), _cause("WIP_누적"))
    assert result is not None

    p = result["standard"]
    candidate = SolutionCandidate(
        rank=2,
        name="표준 조정안",
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
    assert dumped["rank"] == 2
    assert "params" in dumped
    assert dumped["expected_effect"] == "기대 효과"
    assert dumped["rationale"] == "선택 근거"


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
    """SolutionCandidate 기본값이 올바르다."""
    c = SolutionCandidate()
    assert c.rank == 1
    assert c.rationale == ""
    assert c.params.superhotlot_enable is False


def test_t06_superhotlot_ineligible_category():
    """설비_포화는 eligible=False라 at_risk_lots>0이어도 False."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=10.0),
        _cause("설비_포화"),
    )

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["superhotlot_enable"] is False


def test_t06_superhotlot_zero_at_risk():
    """at_risk_lots=0이면 eligible=True 카테고리라도 False."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0),
        _cause("공급_부족"),
    )

    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is False


def test_t06_superhotlot_eligible_triggered():
    """공급_부족 + at_risk_lots>0 → standard/aggressive superhotlot=True."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=3.0),
        _cause("공급_부족"),
    )

    assert result is not None
    assert result["conservative"]["superhotlot_enable"] is False
    assert result["standard"]["superhotlot_enable"] is True
    assert result["aggressive"]["superhotlot_enable"] is True


def test_t06_superhotlot_severity_gate_standard():
    """MEDIUM에서는 standard superhotlot severity gate가 차단한다."""
    result = generate_candidates(
        _alert(SeverityLevel.MEDIUM, at_risk_lots=5.0),
        _cause("공급_부족"),
    )

    assert result is not None
    assert result["standard"]["superhotlot_enable"] is False
    assert result["aggressive"]["superhotlot_enable"] is True


def test_t07_judgment_none_returns_none():
    """judgment=None이면 generate_candidates()가 None을 반환한다."""
    result = generate_candidates(_alert(SeverityLevel.CRITICAL), _cause_no_judgment())
    assert result is None


def test_t08_clip_above_severity_max():
    """제안값 35%가 severity 상한으로 clip된다."""
    assert clip_interval_pct(35.0, SeverityLevel.CRITICAL) == pytest.approx(28.0)
    assert clip_interval_pct(35.0, SeverityLevel.LOW) == pytest.approx(12.0)
    assert clip_interval_pct(35.0, SeverityLevel.HIGH) == pytest.approx(23.0)


def test_t08_clip_no_clip_needed():
    """제안값이 상한 이내이면 그대로 통과한다."""
    assert clip_interval_pct(20.0, SeverityLevel.HIGH) == pytest.approx(20.0)


def test_t08_clip_default_is_critical():
    """기존 호출 호환을 위해 severity 생략 시 CRITICAL 상한을 사용한다."""
    assert clip_interval_pct(35.0) == pytest.approx(28.0)


def test_t09_llm_fallback_no_api_key():
    """OPENAI_API_KEY 없을 때 fallback 텍스트가 반환된다."""
    alert = _alert(SeverityLevel.HIGH, at_risk_lots=0.0)
    cause = _cause("WIP_누적")
    candidates = generate_candidates(alert, cause)
    assert candidates is not None

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        texts = generate_texts(alert, cause, candidates)

    assert set(texts.keys()) == {"conservative", "standard", "aggressive"}
    for lv in ("conservative", "standard", "aggressive"):
        assert isinstance(texts[lv]["expected_effect"], str)
        assert isinstance(texts[lv]["rationale"], str)


def test_t09_llm_fallback_placeholder_key():
    """'your_'로 시작하는 플레이스홀더 API 키도 fallback 처리한다."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=3.0)
    cause = _cause("대기_누적", shap_features=[("q_time_min", 0.60, 45.0)])
    candidates = generate_candidates(alert, cause)
    assert candidates is not None

    with patch.dict(os.environ, {"OPENAI_API_KEY": "your_key_here"}):
        texts = generate_texts(alert, cause, candidates)

    for lv in ("conservative", "standard", "aggressive"):
        assert texts[lv]["rationale"] != ""


def test_t10_s1_wip_supply():
    """S1: WIP_누적 + 공급_부족 → 전 레벨 강화안 수치 + SUPERHOTLOT=True."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=5.0),
        _cause("WIP_누적", secondary_causes=["공급_부족"]),
    )

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["release_interval_delta_pct"] == pytest.approx(_RELEASE_PCT["aggressive"])
        assert result[lv]["superhotlot_enable"] is True


def test_t10_s2_wait_wip():
    """S2: 대기_누적 + WIP_누적 → 전 레벨 PRIORITY=UP + SUPERHOTLOT=True."""
    result = generate_candidates(
        _alert(SeverityLevel.HIGH, at_risk_lots=0.0),
        _cause("대기_누적", secondary_causes=["WIP_누적"]),
    )

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True


def test_t10_s3_util_saturation_hitl():
    """S3: 설비_포화 복합 원인은 aggressive SUPERHOTLOT와 HITL 권고를 만든다."""
    result = generate_candidates(
        _alert(SeverityLevel.HIGH, at_risk_lots=0.0),
        _cause("설비_포화", secondary_causes=["대기_누적"]),
    )

    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is True
    assert result["hitl_escalation_recommended"] is True


def test_t10_s4_triple():
    """S4: 3개 이상 카테고리 동시 악화 → 전 레벨 강화안 수치 + PRIORITY=UP."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0),
        _cause("WIP_누적", secondary_causes=["대기_누적", "공급_부족"]),
    )

    assert result is not None
    agg_pct = _SEVERITY_PCT[SeverityLevel.CRITICAL]["aggressive"]
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["release_interval_delta_pct"] == pytest.approx(agg_pct)
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True
    assert result["hitl_escalation_recommended"] is True


def test_t10_s5_util_supply():
    """S5: 설비_포화 + 공급_부족 → 전 레벨 SUPERHOTLOT=True."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=1.0),
        _cause("설비_포화", secondary_causes=["공급_부족"]),
    )

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["superhotlot_enable"] is True


def test_t10_feature_name_normalization():
    """secondary_causes에 피처명이 와도 카테고리명과 동일하게 처리한다."""
    result = generate_candidates(
        _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0),
        _cause("WIP_누적", secondary_causes=["max_util_delta_120"]),
    )

    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is True
