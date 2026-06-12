"""대응안 비교 점수 모듈 (1단계: 다기준 composite score).

verification_agent가 만든 paired t-test 통계(`kpi_stats`)를 그대로 받아
5개 KPI를 모두 사용한 가중 점수로 후보 순위를 매긴다.

수식 개요
---------
  composite = Σ_kpi ( w_kpi · normalized_improvement(kpi) · confidence(kpi) )
            − λ · uncertainty_penalty
  uncertainty_penalty = Σ_kpi ( w_kpi · normalized_ci_width(kpi) )

  - normalized_improvement : 후보들 간 min-max 정규화 (방향 보정 포함)
  - confidence(kpi)        : max(0, 1 − paired_t_p),  p값 없으면 0.5
  - normalized_ci_width    : 후보들 간 min-max 정규화 (전부 0이면 0)
"""

from __future__ import annotations

from agents import config

# 5개 KPI 방향 (config.KPI_DIRECTIONS_COMPARE 참조).
# lower_better → mean_delta<0 이 개선, higher_better → mean_delta>0 이 개선.
KPI_DIRECTIONS: dict[str, str] = config.KPI_DIRECTIONS_COMPARE

# KPI별 composite score 가중치 (config 개별 상수에서 매핑).
KPI_WEIGHTS: dict[str, float] = {
    "q_time_min": config.W_KPI_Q_TIME,
    "wip": config.W_KPI_WIP,
    "wait_ratio": config.W_KPI_WAIT_RATIO,
    "utilization_avg": config.W_KPI_UTIL,
    "available_tool_ratio": config.W_KPI_AVAIL,
}

# 불확실성(CI 폭) 페널티 가중치.
UNCERTAINTY_PENALTY_WEIGHT: float = config.W_KPI_UNCERTAINTY

# 의사결정 임계값.
EQUIVALENCE_SCORE_EPS: float = config.COMPARE_EQUIVALENCE_EPS   # 통계적 동등 임계값
NO_EFFECT_SCORE_MAX: float = config.COMPARE_NO_EFFECT_MAX       # 효과 미검증 임계값

# KPI별 "최소 의미있는 변화량". min-max 정규화 시 후보들 간 spread가 이 값 이하면
# 정규화 분모를 이 값으로 끌어올려, 작은 noise가 0↔1로 증폭되는 걸 막는다.
MIN_KPI_SPREAD: dict[str, float] = {
    "q_time_min": config.MIN_DELTA_Q_TIME,
    "wip": config.MIN_DELTA_WIP,
    "wait_ratio": config.MIN_DELTA_WAIT_RATIO,
    "utilization_avg": config.MIN_DELTA_UTIL,
    "available_tool_ratio": config.MIN_DELTA_AVAIL,
}

# action_kind 운영 메타데이터 (config.ACTION_KIND_METADATA 참조).
ACTION_KIND_METADATA: dict[str, dict] = config.ACTION_KIND_METADATA

# 기존 호출 호환용 — effort만 추출한 view
ACTION_KIND_EFFORT: dict[str, int] = {k: v["effort"] for k, v in ACTION_KIND_METADATA.items()}


def _improvement(kpi: str, mean_delta: float) -> float:
    """양수일수록 '개선'으로 통일."""
    if KPI_DIRECTIONS.get(kpi, "lower_better") == "lower_better":
        return -mean_delta
    return mean_delta


def get_action_metadata(action_kind: str) -> dict:
    """action_kind → 메타데이터 (effort/scope/reversibility/description_ko)."""
    return dict(ACTION_KIND_METADATA.get(action_kind, ACTION_KIND_METADATA["UNKNOWN"]))


def extract_tradeoffs(candidate: dict) -> list[dict]:
    """후보의 score_breakdown에서 verdict='worsened'인 KPI 자동 추출.

    severity 판정:
      - significant : 통계적 유의 (confidence ≥ 0.8 = p ≤ 0.2)
      - large       : |mean_delta| ≥ KPI별 MIN_KPI_SPREAD
      - both        → "major", 아니면 → "minor"

    Returns:
        [{kpi, mean_delta, severity, confidence, verdict}, ...]
        악화 KPI가 없으면 빈 리스트.
    """
    breakdown = candidate.get("score_breakdown", {}).get("kpi_contributions", {})
    tradeoffs: list[dict] = []
    for kpi, kc in breakdown.items():
        if kc.get("verdict") != "worsened":
            continue
        mean_d = kc.get("mean_delta", 0.0) or 0.0
        conf = kc.get("confidence", 0.0) or 0.0
        threshold = MIN_KPI_SPREAD.get(kpi, 0.0)
        is_significant = conf >= 0.8
        is_large = abs(mean_d) >= threshold
        severity = "major" if (is_significant and is_large) else "minor"
        tradeoffs.append({
            "kpi": kpi,
            "mean_delta": round(float(mean_d), 4),
            "severity": severity,
            "confidence": round(float(conf), 4),
            "verdict": "worsened",
        })
    # major 먼저, 그 다음 confidence 높은 순
    tradeoffs.sort(key=lambda t: (0 if t["severity"] == "major" else 1, -t["confidence"]))
    return tradeoffs


def _ci_width(stats: dict) -> float:
    lo = stats.get("ci_lo")
    hi = stats.get("ci_hi")
    if lo is None or hi is None:
        return 0.0
    try:
        return max(0.0, float(hi) - float(lo))
    except (TypeError, ValueError):
        return 0.0


def _normalize_minmax(values: list[float], min_spread: float = 0.0) -> list[float]:
    """min-max → [0, 1]. 분모는 max(hi-lo, min_spread).

    min_spread > 0이면 후보들 간 차이가 그 값 이하일 때 정규화가 1.0까지 안 늘어남.
    예: spread=2, min_spread=5 → 결과 범위 [0, 0.4]. 작은 noise가 큰 점수 차이로
    증폭되는 걸 막아 등가 케이스가 자연스럽게 잡힘.

    값이 모두 같으면: 0이면 0, 그 외엔 0.5.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    spread = hi - lo
    if spread < 1e-9:
        if abs(hi) < 1e-9:
            return [0.0] * len(values)
        return [0.5] * len(values)
    denom = max(spread, min_spread)
    return [(v - lo) / denom for v in values]


def _confidence(stats: dict) -> float:
    import math
    p = stats.get("paired_t_p")
    if p is None:
        return 0.5
    try:
        fp = float(p)
        if math.isnan(fp):
            return 0.5
        return max(0.0, min(1.0, 1.0 - fp))
    except (TypeError, ValueError):
        return 0.5


def _action_effort(candidate: dict) -> int:
    return ACTION_KIND_EFFORT.get(candidate.get("action_kind", "UNKNOWN"), 99)


def _tiebreaker_release_interval(candidate: dict) -> float:
    """Release Interval 변동폭 절대값 — 작을수록 운영 충격 적음.

    plan_meta에 release_interval_delta_min(글로벌 플랜)이 있으면 절대값 사용.
    release_interval_delta_pct(per-TG)는 절대값 사용. 둘 다 없으면 999.
    """
    pm = candidate.get("plan_meta") or {}
    delta_min = pm.get("release_interval_delta_min")
    if delta_min is not None:
        try:
            return abs(float(delta_min))
        except (TypeError, ValueError):
            pass
    delta_pct = pm.get("release_interval_delta_pct")
    if delta_pct is not None:
        try:
            return abs(float(delta_pct))
        except (TypeError, ValueError):
            pass
    return 999.0


def _apply_tiebreaker_chain(group: list[dict]) -> tuple[dict, list[dict]]:
    """동률 후보 그룹에 tie-breaker 체인 적용.

    체인 순서:
      1. operational_effort (action_kind 기반 운영 부담) — 작은 값 우선
      2. release_interval_delta (release_interval 변동폭) — 작은 값 우선
      3. composite_score — 큰 값 우선 (최종 fallback)

    Returns:
        (winner, evaluated_steps) — evaluated_steps는 각 단계 결과 기록
    """
    evaluated: list[dict] = []
    pool = list(group)

    # Step 1: operational_effort
    effort_values = {c["label"]: _action_effort(c) for c in pool}
    min_effort = min(effort_values.values())
    effort_winners = [c for c in pool if effort_values[c["label"]] == min_effort]
    evaluated.append({
        "step": "operational_effort",
        "values": effort_values,
        "result": effort_winners[0]["label"] if len(effort_winners) == 1 else "tied",
    })
    if len(effort_winners) == 1:
        return effort_winners[0], evaluated
    pool = effort_winners

    # Step 2: release_interval_delta
    ri_values = {c["label"]: round(_tiebreaker_release_interval(c), 4) for c in pool}
    min_ri = min(ri_values.values())
    ri_winners = [c for c in pool if ri_values[c["label"]] == min_ri]
    evaluated.append({
        "step": "release_interval_delta",
        "values": ri_values,
        "result": ri_winners[0]["label"] if len(ri_winners) == 1 else "tied",
    })
    if len(ri_winners) == 1:
        return ri_winners[0], evaluated
    pool = ri_winners

    # Step 3: composite_score (최종 fallback, 큰 쪽)
    score_values = {c["label"]: round(c.get("composite_score", 0.0), 4) for c in pool}
    winner = max(pool, key=lambda c: c.get("composite_score", 0.0))
    evaluated.append({
        "step": "composite_score",
        "values": score_values,
        "result": winner["label"],
    })
    return winner, evaluated


def _detect_decision_status(enriched: list[dict]) -> dict:
    """등가 클래스 감지 → 최종 추천 후보 + 결정 캐비엇.

    decision_status:
      - "no_meaningful_effect"  : 최고 점수가 임계값 이하 → 효과 미검증
      - "equivalent_candidates" : 1위와 점수 차이가 ε 이하인 후보가 2개 이상 → 통계적 동등
      - "clear_winner"          : 1위가 명확히 우위

    Returns:
        {
          decision_status, top_label, equivalent_set,
          tiebreaker_used, tiebreaker_chain_evaluated, decision_caveat
        }
    """
    if not enriched:
        return {
            "decision_status": "no_meaningful_effect",
            "top_label": None,
            "equivalent_set": [],
            "tiebreaker_used": None,
            "tiebreaker_chain_evaluated": [],
            "decision_caveat": "후보가 없습니다.",
        }

    sorted_by_score = sorted(enriched, key=lambda c: -c.get("composite_score", 0.0))
    top_score = sorted_by_score[0].get("composite_score", 0.0)

    # Case 1: 효과 미검증
    if top_score <= NO_EFFECT_SCORE_MAX:
        all_unchanged = all(
            all(
                kc.get("verdict") == "unchanged"
                for kc in c.get("score_breakdown", {}).get("kpi_contributions", {}).values()
            )
            for c in enriched
        )
        labels = [c["label"] for c in enriched]
        winner, evaluated = _apply_tiebreaker_chain(enriched)
        # 최종적으로 사용된 단계명
        tiebreaker_used = next(
            (s["step"] for s in evaluated if s["result"] not in ("tied",)),
            "operational_effort",
        )
        caveat = (
            "모든 후보에서 시뮬레이션상 통계적으로 유의미한 KPI 개선이 관측되지 않았습니다 "
            "(전 KPI verdict=unchanged). "
            f"tie-breaker({tiebreaker_used})로 {winner['label']}을(를) 잠정 추천하나 추가 검증이 필요합니다."
            if all_unchanged
            else f"최고 점수가 {top_score:.3f}로 매우 낮습니다. 의사결정 시 주의가 필요합니다."
        )
        return {
            "decision_status": "no_meaningful_effect",
            "top_label": winner["label"],
            "equivalent_set": labels,
            "tiebreaker_used": tiebreaker_used,
            "tiebreaker_chain_evaluated": evaluated,
            "decision_caveat": caveat,
        }

    # Case 2: 통계적 동등 후보들
    equivalent = [
        c for c in sorted_by_score
        if top_score - c.get("composite_score", 0.0) <= EQUIVALENCE_SCORE_EPS
    ]
    if len(equivalent) >= 2:
        labels = [c["label"] for c in equivalent]
        winner, evaluated = _apply_tiebreaker_chain(equivalent)
        tiebreaker_used = next(
            (s["step"] for s in evaluated if s["result"] not in ("tied",)),
            "operational_effort",
        )
        return {
            "decision_status": "equivalent_candidates",
            "top_label": winner["label"],
            "equivalent_set": labels,
            "tiebreaker_used": tiebreaker_used,
            "tiebreaker_chain_evaluated": evaluated,
            "decision_caveat": (
                f"후보 {', '.join(labels)}는 composite_score 차이가 {EQUIVALENCE_SCORE_EPS} 이내로 "
                f"통계적 동등 수준입니다. tie-breaker({tiebreaker_used})로 {winner['label']}을(를) 추천합니다."
            ),
        }

    # Case 3: 명확한 1위
    winner = sorted_by_score[0]
    return {
        "decision_status": "clear_winner",
        "top_label": winner["label"],
        "equivalent_set": [winner["label"]],
        "tiebreaker_used": None,
        "tiebreaker_chain_evaluated": [],
        "decision_caveat": "",
    }


# ── 추천 상태 derive (badge 표시 분리) ───────────────────────────────────────

def derive_recommendation_status(candidate: dict, decision_info: dict) -> dict:
    """후보 1개의 추천 메타데이터 — UI 표시 분리.

    Returns:
        {
          "is_recommended": bool,
          "recommendation_status": "ai_recommended" | "tentative_no_effect" | "equivalent_tiebreak" | None,
          "badge": "ai_recommended" | "tentative" | "equivalent_tiebreak" | None,
        }
    """
    if candidate.get("label") != decision_info.get("top_label"):
        return {"is_recommended": False, "recommendation_status": None, "badge": None}

    ds = decision_info.get("decision_status", "clear_winner")
    if ds == "clear_winner":
        return {
            "is_recommended": True,
            "recommendation_status": "ai_recommended",
            "badge": "ai_recommended",
        }
    if ds == "equivalent_candidates":
        return {
            "is_recommended": True,
            "recommendation_status": "equivalent_tiebreak",
            "badge": "equivalent_tiebreak",
        }
    if ds == "no_meaningful_effect":
        return {
            "is_recommended": True,
            "recommendation_status": "tentative_no_effect",
            "badge": "tentative",
        }
    return {"is_recommended": False, "recommendation_status": None, "badge": None}


def compute_composite_scores(candidates: list[dict]) -> tuple[list[dict], dict]:
    """5개 KPI 기반 composite score 계산 + 등가 클래스 감지.

    Args:
        candidates: `_build_action_candidates` 출력. 각 항목은 `kpi_stats` 보유.

    Returns:
        (enriched, decision_info)
          enriched: 입력과 같은 길이. 각 항목에 다음 추가:
            composite_score, score_breakdown, rank, badge,
            is_top, is_equivalent_to_top
          decision_info: _detect_decision_status() 결과
    """
    if not candidates:
        return [], {
            "decision_status": "no_meaningful_effect",
            "top_label": None,
            "equivalent_set": [],
            "tiebreaker_used": None,
            "decision_caveat": "후보가 없습니다.",
        }

    # 1) KPI별 수치 수집
    per_kpi_improvements: dict[str, list[float]] = {}
    per_kpi_confidences: dict[str, list[float]] = {}
    per_kpi_ci_widths: dict[str, list[float]] = {}
    per_kpi_mean_deltas: dict[str, list[float]] = {}
    per_kpi_verdicts: dict[str, list[str]] = {}

    for kpi in KPI_WEIGHTS:
        means, imps, confs, widths, verdicts = [], [], [], [], []
        for c in candidates:
            stats = c.get("kpi_stats", {}).get(kpi, {}) or {}
            try:
                mean_d = float(stats.get("mean_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                mean_d = 0.0
            means.append(mean_d)
            imps.append(_improvement(kpi, mean_d))
            confs.append(_confidence(stats))
            widths.append(_ci_width(stats))
            verdicts.append(stats.get("verdict", "unknown"))
        per_kpi_mean_deltas[kpi] = means
        per_kpi_improvements[kpi] = imps
        per_kpi_confidences[kpi] = confs
        per_kpi_ci_widths[kpi] = widths
        per_kpi_verdicts[kpi] = verdicts

    # 2) min-max 정규화 (KPI별, min_spread 적용으로 작은 차이 증폭 방지)
    per_kpi_norm = {
        k: _normalize_minmax(v, min_spread=MIN_KPI_SPREAD.get(k, 0.0))
        for k, v in per_kpi_improvements.items()
    }
    per_kpi_ci_norm = {k: _normalize_minmax(v) for k, v in per_kpi_ci_widths.items()}

    # 3) 후보별 composite
    enriched: list[dict] = []
    for i, c in enumerate(candidates):
        kpi_contribs: dict[str, dict] = {}
        raw = 0.0
        ci_penalty_sum = 0.0
        for kpi, w in KPI_WEIGHTS.items():
            imp_norm = per_kpi_norm[kpi][i]
            conf = per_kpi_confidences[kpi][i]
            contrib = w * imp_norm * conf
            raw += contrib
            ci_penalty_sum += w * per_kpi_ci_norm[kpi][i]
            kpi_contribs[kpi] = {
                "weight": w,
                "mean_delta": round(per_kpi_mean_deltas[kpi][i], 4),
                "improvement_norm": round(imp_norm, 4),
                "confidence": round(conf, 4),
                "ci_width": round(per_kpi_ci_widths[kpi][i], 4),
                "verdict": per_kpi_verdicts[kpi][i],
                "contribution": round(contrib, 4),
            }
        penalty = UNCERTAINTY_PENALTY_WEIGHT * ci_penalty_sum
        final = max(0.0, raw - penalty)
        enriched.append({
            **c,
            "composite_score": round(final, 4),
            "score_breakdown": {
                "kpi_contributions": kpi_contribs,
                "uncertainty_penalty": round(penalty, 4),
                "raw_score": round(raw, 4),
                "final_score": round(final, 4),
                "weights": dict(KPI_WEIGHTS),
                "penalty_weight": UNCERTAINTY_PENALTY_WEIGHT,
            },
        })

    # 4) rank (score 순서 — 통계적 우수성)
    sorted_idx = sorted(range(len(enriched)), key=lambda i: -enriched[i]["composite_score"])
    rank_map = {idx: r + 1 for r, idx in enumerate(sorted_idx)}
    for i, c in enumerate(enriched):
        c["rank"] = rank_map[i]

    # 5) decision_info — 등가 클래스 감지 + 운영 부담 tie-break
    decision_info = _detect_decision_status(enriched)
    top_label = decision_info.get("top_label")
    equivalent_set = set(decision_info.get("equivalent_set", []))

    # 6) is_top + action_metadata + tradeoffs 부여 (badge는 derive_recommendation_status에서 분리 산출)
    for c in enriched:
        is_top = (c["label"] == top_label)
        c["is_top"] = is_top
        c["is_equivalent_to_top"] = c["label"] in equivalent_set
        c["action_metadata"] = get_action_metadata(c.get("action_kind", "UNKNOWN"))
        c["tradeoffs"] = extract_tradeoffs(c)

    return enriched, decision_info
