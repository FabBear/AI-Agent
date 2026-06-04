from __future__ import annotations


def rank_score(state: dict) -> dict:
    """
    q_time 개선 기반 순위 결정 — LLM 불필요

    1순위: avg_queue_time_min 개선량 (클수록 유리)
    2순위 (동점): throughput_delta (클수록 유리)
    3순위 (동점): wip_count 개선량 (클수록 유리)
    """
    candidates = state["action_candidates"]

    def rank_key(c):
        kd = c["kpi_delta"]
        qtime = abs(kd.get("avg_queue_time_min", 0))
        throughput = kd.get("throughput_delta", 0)
        wip = abs(kd.get("wip_count", 0))
        return (qtime, throughput, wip)

    sorted_candidates = sorted(candidates, key=rank_key, reverse=True)
    rank_map = {c["label"]: i + 1 for i, c in enumerate(sorted_candidates)}

    scored = []
    for c in candidates:
        rank = rank_map[c["label"]]
        scored.append({
            "label": c["label"],
            "rank": rank,
            "badge": "✅ AI추천" if rank == 1 else "⚪ 선택가능",
        })

    return {"scored_actions": scored}
