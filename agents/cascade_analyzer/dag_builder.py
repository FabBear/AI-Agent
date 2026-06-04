"""lot_events.csv의 실제 공정 라우팅에서 TG 순서 DAG를 구성한다."""

from pathlib import Path

import networkx as nx
import pandas as pd

_dag_cache: nx.DiGraph | None = None


def build_dag(csv_dir: str | Path) -> nx.DiGraph:
    """
    lot_events.csv의 step_seq 순서로 DirectedGraph를 구성한다.
    노드 = toolgroup, 엣지 = 공정 흐름 방향 (A → B: lot이 A 다음 B로 이동)
    """
    global _dag_cache
    if _dag_cache is not None:
        return _dag_cache

    csv_dir = Path(csv_dir)
    df = pd.read_csv(
        csv_dir / "lot_events.csv",
        usecols=["route_id", "step_seq", "tool_group"],
    )
    df = df[df["step_seq"].notna() & df["tool_group"].notna()].copy()
    df["step_seq"] = df["step_seq"].astype(float)

    G = nx.DiGraph()

    for _, group in df.groupby("route_id"):
        seq = group.drop_duplicates("step_seq").sort_values("step_seq")["tool_group"].tolist()
        for i in range(len(seq) - 1):
            src, dst = seq[i], seq[i + 1]
            if src != dst:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += 1
                else:
                    G.add_edge(src, dst, weight=1)

    _dag_cache = G
    return G


def get_downstream_tgs(G: nx.DiGraph, source: str, max_hops: int) -> list[tuple[str, int]]:
    """
    source에서 BFS로 max_hops 이내의 후속 TG 목록을 반환한다.
    Returns: [(toolgroup, hop_distance), ...]
    """
    if source not in G:
        return []

    visited: list[tuple[str, int]] = []
    queue = [(source, 0)]
    seen = {source}

    while queue:
        node, hop = queue.pop(0)
        if hop > 0:
            visited.append((node, hop))
        if hop < max_hops:
            for neighbor in G.successors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, hop + 1))

    return visited
