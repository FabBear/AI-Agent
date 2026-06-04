"""DAG를 역방향으로 탐색해 WIP을 과공급 중인 업스트림 TG를 찾는다."""

import networkx as nx

from agents import config
from agents.schemas.kpi import ToolGroupKPI

_MAX_UPSTREAM_HOPS = 3


def find_upstream_suspects(
    G: nx.DiGraph,
    toolgroup: str,
    kpi_map: dict[str, ToolGroupKPI],
) -> list[str]:
    """
    업스트림 TG 중 utilization이 높고 WIP이 많아 병목 TG로 물량을 밀어넣고 있을
    가능성이 높은 TG 목록을 반환한다.
    """
    if toolgroup not in G:
        return []

    G_rev = G.reverse(copy=False)
    queue = [(toolgroup, 0)]
    seen = {toolgroup}
    suspects: list[tuple[str, float]] = []  # (tg, stress_score)

    while queue:
        node, hop = queue.pop(0)
        if hop >= _MAX_UPSTREAM_HOPS:
            continue
        for upstream in G_rev.successors(node):
            if upstream in seen:
                continue
            seen.add(upstream)
            queue.append((upstream, hop + 1))

            kpi = kpi_map.get(upstream)
            if kpi is None:
                continue
            # 높은 utilization + 낮은 tool 가용 = 생산은 하는데 빠져나갈 곳이 없어 쌓임
            stress = kpi.utilization_avg * (1.0 - kpi.available_tool_ratio + 1e-3)
            if kpi.utilization_avg >= config.U_HI or kpi.wip >= config.WIP_THR * 3:
                suspects.append((upstream, stress))

    suspects.sort(key=lambda x: x[1], reverse=True)
    return [tg for tg, _ in suspects[:5]]
