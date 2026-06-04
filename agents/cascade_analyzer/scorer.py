"""prob + impact_score → composite_score → SeverityLevel."""

from agents import config
from agents.schemas.alert import BottleneckAlert, CascadeImpact, SeverityLevel
from agents.schemas.alert import PotentialBottleneck


def assign_severity(composite_score: float) -> SeverityLevel:
    if composite_score >= config.CRITICAL_SCORE:
        return SeverityLevel.CRITICAL
    if composite_score >= config.HIGH_SCORE:
        return SeverityLevel.HIGH
    if composite_score >= config.MEDIUM_SCORE:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def build_alert(pb: PotentialBottleneck, impact: CascadeImpact) -> BottleneckAlert:
    composite = round(
        config.PROB_WEIGHT * pb.probability + config.IMPACT_WEIGHT * impact.impact_score,
        4,
    )
    return BottleneckAlert(
        toolgroup=pb.toolgroup,
        severity=assign_severity(composite),
        composite_score=composite,
        probability=pb.probability,
        impact=impact,
        snapshot_time=pb.snapshot_time,
    )
