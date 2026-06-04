"""composite_score 기반 심각도 분류 테스트."""

from agents.cascade_analyzer.scorer import assign_severity
from agents.schemas.alert import SeverityLevel


def test_critical_threshold():
    assert assign_severity(0.80) == SeverityLevel.CRITICAL
    assert assign_severity(0.75) == SeverityLevel.CRITICAL
    assert assign_severity(1.00) == SeverityLevel.CRITICAL


def test_high_threshold():
    assert assign_severity(0.74) == SeverityLevel.HIGH
    assert assign_severity(0.55) == SeverityLevel.HIGH


def test_medium_threshold():
    assert assign_severity(0.54) == SeverityLevel.MEDIUM
    assert assign_severity(0.35) == SeverityLevel.MEDIUM


def test_low_threshold():
    assert assign_severity(0.34) == SeverityLevel.LOW
    assert assign_severity(0.00) == SeverityLevel.LOW


def test_boundary_values():
    # 경계값: CRITICAL_SCORE=0.75, HIGH_SCORE=0.55, MEDIUM_SCORE=0.35
    assert assign_severity(0.749) == SeverityLevel.HIGH
    assert assign_severity(0.549) == SeverityLevel.MEDIUM
    assert assign_severity(0.349) == SeverityLevel.LOW
