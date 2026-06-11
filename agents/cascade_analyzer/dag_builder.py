"""공정 흐름 DAG 구성 — Backend DB td_route_step 조회.

캐시: 프로세스 당 1회만 빌드 (_dag_cache).
"""

from __future__ import annotations

import os
from itertools import groupby
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from agents.logger import get_logger

_log = get_logger(__name__)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_BACKEND_DB_URL = os.getenv(
    "BACKEND_DATABASE_URL",
    "postgresql+psycopg://fabbear_user@localhost:5432/fabbear",
)

_dag_cache: nx.DiGraph | None = None
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_BACKEND_DB_URL, pool_pre_ping=True)
    return _engine


_DAG_SQL = text("""
SELECT
    pr.product_type AS route_id,
    rs.step_seq,
    tg.tg_code      AS tool_group
FROM td_route_step rs
JOIN tm_tool_group    tg ON tg.tg_id    = rs.tg_id
JOIN tm_product_route pr ON pr.route_id = rs.route_id
WHERE pr.is_active = TRUE
ORDER BY pr.product_type, rs.step_seq
""")


def _build_graph(triples: list[tuple[str, float, str]]) -> nx.DiGraph:
    """(route_id, step_seq, tool_group) 리스트에서 DAG를 구성한다."""
    G = nx.DiGraph()
    triples.sort(key=lambda x: (x[0], x[1]))
    for _route, group in groupby(triples, key=lambda x: x[0]):
        seen_seq: dict[float, str] = {}
        for _, seq, tg in group:
            seen_seq.setdefault(seq, tg)
        seq_list = [seen_seq[k] for k in sorted(seen_seq)]
        for i in range(len(seq_list) - 1):
            src, dst = seq_list[i], seq_list[i + 1]
            if src != dst:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += 1
                else:
                    G.add_edge(src, dst, weight=1)
    return G


def build_dag(csv_dir=None) -> nx.DiGraph:
    """Backend DB td_route_step에서 공정 흐름 DAG를 로드한다."""
    global _dag_cache
    if _dag_cache is not None:
        return _dag_cache

    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_DAG_SQL).fetchall()

    if not rows:
        raise RuntimeError("[DAG] Backend DB에 route_step 데이터가 없습니다.")

    triples = [(r.route_id, float(r.step_seq), r.tool_group) for r in rows]
    G = _build_graph(triples)
    _log.info(f"[DAG] 로드 완료 — {G.number_of_nodes()}개 TG, {G.number_of_edges()}개 엣지")
    _dag_cache = G
    return G


def get_downstream_tgs(G: nx.DiGraph, source: str, max_hops: int) -> list[tuple[str, int]]:
    """source에서 BFS로 max_hops 이내의 후속 TG 목록을 반환한다."""
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
