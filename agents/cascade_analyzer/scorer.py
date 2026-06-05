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
    # downstream이 없는 TG(마지막 공정 등)는 prob를 그대로 composite_score로 사용
    if not impact.affected_tgs:
        composite = round(pb.probability, 4)
    else:
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
