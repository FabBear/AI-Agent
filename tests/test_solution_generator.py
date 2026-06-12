"""AG-004 solution_generator 단위 테스트 (T-01 ~ T-10).

LLM 호출 없이 동작하도록 작성한다 (OPENAI_USAGE_RULES.md §4).
"""

from __future__ import annotations

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
from agents.solution_generator.rule_engine import (
    _RELEASE_PCT,
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


def _judgment(
    primary_category: str = "WIP_누적",
    secondary_causes: list[str] | None = None,
    confidence: str = "HIGH",
) -> CauseJudgment:
    return CauseJudgment(
        primary_category=primary_category,
        primary_cause="",  # 대표 피처명 — rule_engine은 미사용
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
    """primary_category와 optional secondary_causes로 CauseReport 생성."""
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


# ── T-01: 기본 생성 ───────────────────────────────────────────────────────────

def test_t01_basic_generation():
    """CRITICAL WIP_누적 → 대응안 3개 (보수/표준/강화) 생성."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("WIP_누적")

    result = generate_candidates(alert, cause)

    assert result is not None
    assert set(result.keys()) >= {"conservative", "standard", "aggressive"}
    for lv in ("conservative", "standard", "aggressive"):
        assert "release_interval_delta_pct" in result[lv]
        assert "priority_direction" in result[lv]
        assert "superhotlot_enable" in result[lv]
        assert "target_kpi" in result[lv]

    # WIP_누적 pct_level=same, CRITICAL: C=15, S=22, A=28
    assert result["conservative"]["release_interval_delta_pct"] == pytest.approx(15.0)
    assert result["standard"]["release_interval_delta_pct"] == pytest.approx(22.0)
    assert result["aggressive"]["release_interval_delta_pct"] == pytest.approx(28.0)


# ── T-02: 카테고리 파라미터 매핑 ──────────────────────────────────────────────

@pytest.mark.parametrize("category,expected_cons_pct,expected_cons_priority", [
    # WIP_누적: pct_level=same, priority conservative=None
    ("WIP_누적",  15.0, None),
    # 대기_누적: pct_level=down → conservative=0%, priority conservative=UP
    ("대기_누적",  0.0, "UP"),
    # 설비_포화: pct_level=same, priority conservative=None
    ("설비_포화", 15.0, None),
    # 공급_부족: pct_level=up → conservative=S_pct=22%, priority conservative=None
    ("공급_부족", 22.0, None),
])
def test_t02_category_parameter_mapping(category, expected_cons_pct, expected_cons_priority):
    """카테고리별 규칙 테이블이 올바른 CRITICAL conservative 파라미터를 반환하는지 확인."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause(category)

    result = generate_candidates(alert, cause)

    assert result is not None
    cons = result["conservative"]
    assert cons["release_interval_delta_pct"] == pytest.approx(expected_cons_pct)
    assert cons["priority_direction"] == expected_cons_priority


# ── T-03: 카테고리 내구성 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("category", ["WIP_누적", "대기_누적", "설비_포화", "공급_부족"])
def test_t03_category_robustness(category):
    """4개 카테고리 각각에 대해 generate_candidates가 예외 없이 동작하는지 확인."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause(category)

    result = generate_candidates(alert, cause)

    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert isinstance(result[lv]["superhotlot_enable"], bool)
        assert result[lv]["target_kpi"] != ""


# ── T-04: 출력 shape ─────────────────────────────────────────────────────────

def test_t04_output_shape():
    """SolutionCandidate 스키마 통과 + model_dump()가 per-TG 포맷으로 반환."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=2.0)
    cause = _cause("WIP_누적")
    result = generate_candidates(alert, cause)
    assert result is not None

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
        )
        dumped = candidate.model_dump()
        assert dumped["rank"] == rank
        assert "params" in dumped
        assert "target_kpi" in dumped


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
    assert c.params.superhotlot_enable is False


# ── T-06: SUPERHOTLOT 조건 ────────────────────────────────────────────────────

def test_t06_superhotlot_ineligible_category():
    """설비_포화는 모든 레벨 eligible=False → at_risk_lots>0이어도 False."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=10.0)
    cause = _cause("설비_포화")
    result = generate_candidates(alert, cause)
    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["superhotlot_enable"] is False


def test_t06_superhotlot_zero_at_risk():
    """at_risk_lots=0이면 eligible=True 카테고리라도 False."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("공급_부족")  # aggressive eligible=True
    result = generate_candidates(alert, cause)
    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is False


def test_t06_superhotlot_eligible_triggered():
    """공급_부족 + at_risk_lots>0 → aggressive superhotlot=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=3.0)
    cause = _cause("공급_부족")
    result = generate_candidates(alert, cause)
    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is True
    assert result["standard"]["superhotlot_enable"] is True   # standard also eligible
    assert result["conservative"]["superhotlot_enable"] is False  # conservative not eligible


def test_t06_superhotlot_wip_aggressive_only():
    """WIP_누적 + at_risk_lots>0 → aggressive만 True, conservative/standard는 False."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=5.0)
    cause = _cause("WIP_누적")
    result = generate_candidates(alert, cause)
    assert result is not None
    assert result["conservative"]["superhotlot_enable"] is False
    assert result["standard"]["superhotlot_enable"] is False
    assert result["aggressive"]["superhotlot_enable"] is True


# ── T-07: judgment=None 처리 ─────────────────────────────────────────────────

def test_t07_judgment_none_returns_none():
    """judgment=None이면 generate_candidates()가 None을 반환한다."""
    alert = _alert(SeverityLevel.CRITICAL)
    cause = _cause_no_judgment()
    result = generate_candidates(alert, cause)
    assert result is None


# ── T-08: post-processing clip ────────────────────────────────────────────────

def test_t08_clip_above_max():
    """제안값 35%가 상한(28%)으로 clip."""
    assert clip_interval_pct(35.0) == pytest.approx(28.0)


def test_t08_clip_no_clip_needed():
    """제안값이 상한 이내이면 그대로 통과."""
    assert clip_interval_pct(20.0) == pytest.approx(20.0)


def test_t08_absolute_max_cap():
    """절대 상한(30%)을 초과하는 값은 28%(_MAX_PCT)로 clip."""
    assert clip_interval_pct(31.0) == pytest.approx(28.0)



# ── T-10: 복합 원인 시나리오 ──────────────────────────────────────────────────

def test_t10_s1_wip_supply():
    """S1: WIP_누적 + 공급_부족 → 전 레벨 강화안 수치 + SUPERHOTLOT=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=5.0)
    cause = _cause("WIP_누적", secondary_causes=["공급_부족"])

    result = generate_candidates(alert, cause)
    assert result is not None

    agg_pct = _RELEASE_PCT["aggressive"]
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["release_interval_delta_pct"] == pytest.approx(agg_pct)
        assert result[lv]["superhotlot_enable"] is True


def test_t10_s2_wait_wip():
    """S2: 대기_누적 + WIP_누적 → 전 레벨 PRIORITY=UP + SUPERHOTLOT=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("대기_누적", secondary_causes=["WIP_누적"])

    result = generate_candidates(alert, cause)
    assert result is not None

    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True


def test_t10_s2_conservative_interval_max():
    """S2: conservative interval은 기본값보다 작아지지 않는다 (max() 보정)."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("대기_누적", secondary_causes=["WIP_누적"])

    result = generate_candidates(alert, cause)
    assert result is not None
    cons_pct = _RELEASE_PCT["conservative"]
    assert result["conservative"]["release_interval_delta_pct"] >= cons_pct


def test_t10_s3_util_saturation():
    """S3: 설비_포화 + 대기_누적 → aggressive만 SUPERHOTLOT=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("설비_포화", secondary_causes=["대기_누적"])

    result = generate_candidates(alert, cause)
    assert result is not None
    assert result["aggressive"]["superhotlot_enable"] is True
    assert result["conservative"]["superhotlot_enable"] is False
    assert result["standard"]["superhotlot_enable"] is False


def test_t10_s4_triple():
    """S4: 3개 이상 카테고리 동시 악화 → 전 레벨 강화안 수치 + PRIORITY=UP + SUPERHOTLOT=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("WIP_누적", secondary_causes=["대기_누적", "공급_부족"])

    result = generate_candidates(alert, cause)
    assert result is not None

    agg_pct = _RELEASE_PCT["aggressive"]
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["release_interval_delta_pct"] == pytest.approx(agg_pct)
        assert result[lv]["priority_direction"] == "UP"
        assert result[lv]["superhotlot_enable"] is True


def test_t10_s5_util_supply():
    """S5: 설비_포화 + 공급_부족 → 전 레벨 SUPERHOTLOT=True."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=1.0)
    cause = _cause("설비_포화", secondary_causes=["공급_부족"])

    result = generate_candidates(alert, cause)
    assert result is not None
    for lv in ("conservative", "standard", "aggressive"):
        assert result[lv]["superhotlot_enable"] is True


def test_t10_single_cause_no_scenario():
    """단일 원인(secondary_causes 없음) → 시나리오 감지 안 됨."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("WIP_누적")

    result = generate_candidates(alert, cause)
    assert result is not None


def test_t10_feature_name_normalization():
    """secondary_causes에 피처명(max_util_delta_120)이 와도 카테고리명과 동일하게 처리."""
    # max_util_delta_120 → 설비_포화, 즉 primary=WIP_누적 + 설비_포화 → S3
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=0.0)
    cause = _cause("WIP_누적", secondary_causes=["max_util_delta_120"])

    result = generate_candidates(alert, cause)
    assert result is not None
    # WIP_누적 + 설비_포화 → S3: aggressive만 superhotlot
    assert result["aggressive"]["superhotlot_enable"] is True
    assert result["conservative"]["superhotlot_enable"] is False


def test_t10_delta_feature_same_as_base():
    """wip_delta_120은 wip와 동일 카테고리(WIP_누적)로 취급한다."""
    alert = _alert(SeverityLevel.CRITICAL, at_risk_lots=5.0)
    # primary=WIP_누적, secondary=wip_delta_120(→WIP_누적) → 중복 → 단일 카테고리 → 시나리오 없음
    cause = _cause("WIP_누적", secondary_causes=["wip_delta_120"])

    result = generate_candidates(alert, cause)
    assert result is not None
