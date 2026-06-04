"""세 지표(capacity stress / CT 증가 / lot risk)를 계산하고 가중합 impact_score를 반환한다."""

from agents import config
from agents.schemas.alert import CascadeImpact
from agents.schemas.kpi import ToolGroupKPI


def compute_impact(
    bottleneck_tg: ToolGroupKPI,
    downstream: list[tuple[str, int]],  # (toolgroup, hop)
    kpi_map: dict[str, ToolGroupKPI],
) -> CascadeImpact:
    """
    bottleneck_tg    : 병목으로 판단된 TG의 KPI
    downstream       : dag_builder.get_downstream_tgs() 결과
    kpi_map          : {toolgroup: ToolGroupKPI} 현재 스냅샷 전체
    """
    affected_tgs = [tg for tg, _ in downstream]
    downstream_kpis = [kpi_map[tg] for tg, _ in downstream if tg in kpi_map]

    capacity_stress = _capacity_stress(downstream_kpis)
    ct_increase_min = _ct_increase(bottleneck_tg, downstream)
    at_risk_lots = _lot_risk(bottleneck_tg, downstream_kpis)

    # 정규화 [0, 1]
    cap_norm = min(capacity_stress, 1.0)
    ct_norm = min(ct_increase_min / config.CT_BASELINE_MIN, 1.0)
    lot_norm = min(at_risk_lots / config.LOT_BASELINE, 1.0)

    impact_score = config.W_CAPACITY * cap_norm + config.W_CT * ct_norm + config.W_LOT * lot_norm

    return CascadeImpact(
        capacity_stress_score=round(cap_norm, 4),
        ct_increase_min=round(ct_increase_min, 2),
        at_risk_lots=round(at_risk_lots, 1),
        affected_tgs=affected_tgs,
        impact_score=round(impact_score, 4),
    )


def _capacity_stress(downstream_kpis: list[ToolGroupKPI]) -> float:
    """
    후속 공정들의 평균 용량 포화도.
    utilization_avg / (available_tool_ratio + ε) — 1.0 이상이면 포화.
    [0, 1]로 클리핑.
    """
    if not downstream_kpis:
        return 0.0
    scores = []
    for kpi in downstream_kpis:
        stress = kpi.utilization_avg / max(kpi.available_tool_ratio, 0.01)
        scores.append(min(stress, 2.0) / 2.0)  # 2.0 이상은 완전 포화
    return sum(scores) / len(scores)


def _ct_increase(bottleneck_tg: ToolGroupKPI, downstream: list[tuple[str, int]]) -> float:
    """
    병목 TG의 q_time_min을 출발점으로, 홉마다 CT_DECAY 비율로 감쇠하며 전파.
    전체 후속 공정의 CT 증가 합산(분).
    """
    if not downstream:
        return 0.0
    source_ct = bottleneck_tg.q_time_min
    total = sum(source_ct * (config.CT_DECAY**hop) for _, hop in downstream)
    return total


def _lot_risk(bottleneck_tg: ToolGroupKPI, downstream_kpis: list[ToolGroupKPI]) -> float:
    """
    병목 TG + 직후 후속 공정들의 WIP 합계.
    """
    return bottleneck_tg.wip + sum(k.wip for k in downstream_kpis)
