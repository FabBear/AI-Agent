#!/usr/bin/env python3
"""비교분석 Agent (Agent 05) — PipelineState 통합 노드 (compare/2.0)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents import config

if TYPE_CHECKING:
    from agents.state import PipelineState

_ROOT = Path(__file__).parent.parent.parent
COMPARE_OUT_DIR = _ROOT / "compare_agent_out"

load_dotenv(_ROOT / ".env")

_llm: Optional[ChatOpenAI] = None

SCHEMA_VERSION = "compare/2.0"

# decision_status 코드 → 한국어 라벨 (HITL 콘솔용)
_DECISION_STATUS_LABEL = {
    "clear_winner": "명확한 1위",
    "equivalent_candidates": "통계적 동등 (tie-break 적용)",
    "no_meaningful_effect": "효과 미검증 (잠정 추천)",
}

# badge 코드 → 한국어 라벨 (HITL 콘솔용)
_BADGE_LABEL = {
    "ai_recommended": "AI 추천",
    "tentative": "잠정 추천 (효과 미검증)",
    "equivalent_tiebreak": "잠정 추천 (통계적 동등)",
}


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "  프로젝트 루트의 .env 파일에 OPENAI_API_KEY를 추가하세요."
            )
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_completion_tokens=1500,
        )
    return _llm


# ── 입력 변환 헬퍼 ────────────────────────────────────────────────────────────

def _build_action_candidates(verified_candidates: list[dict]) -> list[dict]:
    """verified_candidates → compare action_candidates 변환 (plan_meta 포함)."""
    candidates = []
    for vc in verified_candidates:
        action_rows = vc.get("action_rows", [])
        action_kind = action_rows[0]["action_kind"] if action_rows else "UNKNOWN"
        kpi_stats = vc.get("kpi_stats", {})
        p_val = vc.get("paired_t_p")
        confidence = (
            round(max(0.0, min(1.0, 1.0 - float(p_val))), 4)
            if p_val is not None else 0.5
        )
        candidates.append({
            "label": vc["label"],
            "action_kind": action_kind,
            "description": vc.get("name", ""),
            "simulation_confidence": confidence,
            "kpi_stats": kpi_stats,
            "verdict": vc.get("verdict", "unknown"),
            "paired_n": vc.get("paired_n", 0),
            "paired_t_p": p_val,
            "plan_meta": vc.get("plan_meta") or {},
            "per_tg_forecasts": vc.get("per_tg_forecasts") or {},
            "aggregation_rule": vc.get("aggregation_rule") or "",
        })
    return candidates


def _build_bottleneck_info(toolgroup: str, alert, kpi) -> dict:
    info: dict = {
        "tool_group": toolgroup,
        "risk_score": round(float(alert.composite_score), 4),
    }
    if kpi:
        info.update({
            "wip_count": int(kpi.wip),
            "avg_queue_time_min": round(float(kpi.q_time_min), 2),
            "wait_ratio": round(float(kpi.wait_ratio), 4),
            "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
            "utilization_avg": round(float(kpi.utilization_avg), 4),
        })
    return info


def _rank_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """다기준 composite score + tie-breaker chain.

    Returns:
        (enriched, scored, decision_info) — scored는 dashboard용 간이 요약
    """
    from agents.compare_agent.scorer import compute_composite_scores

    enriched, decision_info = compute_composite_scores(candidates)
    scored = [
        {
            "label": c["label"],
            "rank": c["rank"],
            "composite_score": c["composite_score"],
            "is_top": c.get("is_top", False),
        }
        for c in enriched
    ]
    return enriched, scored, decision_info


# ── 원인분석 컨텍스트 추출 ────────────────────────────────────────────────────

def _build_cause_context(toolgroup: str, cause_reports: list) -> dict:
    """CauseReport 리스트에서 해당 TG의 핵심 원인분석 정보 추출."""
    for report in cause_reports:
        tg = (
            getattr(report, "toolgroup", None)
            if not isinstance(report, dict)
            else report.get("toolgroup")
        )
        if tg != toolgroup:
            continue

        def _attr(obj, key, default=None):
            return getattr(obj, key, default) if not isinstance(obj, dict) else obj.get(key, default)

        shap_top = []
        for s in (_attr(report, "shap_top", []) or [])[:3]:
            if isinstance(s, dict):
                shap_top.append(s)
            else:
                shap_top.append({
                    "feature": s.feature,
                    "shap_value": round(float(s.shap_value), 4),
                    "kpi_value": round(float(s.kpi_value), 4),
                })

        trend_top = []
        for t in (_attr(report, "trend_top", []) or [])[:2]:
            if isinstance(t, dict):
                trend_top.append({"feature": t.get("feature", ""), "slope_per_hour": t.get("slope_per_hour", 0.0)})
            else:
                trend_top.append({"feature": t.feature, "slope_per_hour": round(float(t.slope_per_hour), 4)})

        consensus_raw = _attr(report, "consensus", {}) or {}
        def _c(key, default=None):
            return consensus_raw.get(key, default) if isinstance(consensus_raw, dict) else getattr(consensus_raw, key, default)

        g_star_sig_kpis = []
        for k in (_c("g_star_sig_kpis") or []):
            if isinstance(k, dict):
                g_star_sig_kpis.append(k)
            else:
                g_star_sig_kpis.append({
                    "kpi": k.kpi,
                    "delta_mean": round(float(k.delta_mean), 4),
                    "t_p_adj": round(float(k.t_p_adj), 4),
                    "significant": k.significant,
                })

        judgment_raw = _attr(report, "judgment", None)
        judgment = None
        if judgment_raw is not None:
            if isinstance(judgment_raw, dict):
                judgment = judgment_raw
            else:
                judgment = {
                    "primary_category": getattr(judgment_raw, "primary_category", ""),
                    "primary_cause": getattr(judgment_raw, "primary_cause", ""),
                    "primary_confidence": getattr(judgment_raw, "primary_confidence", "LOW"),
                    "primary_reasoning": getattr(judgment_raw, "primary_reasoning", ""),
                    "secondary_causes": list(getattr(judgment_raw, "secondary_causes", None) or []),
                    "dismissed": list(getattr(judgment_raw, "dismissed", None) or []),
                    "dismissed_reason": getattr(judgment_raw, "dismissed_reason", ""),
                    "cause_summary": getattr(judgment_raw, "cause_summary", ""),
                }

        categories = []
        for cat in (_attr(report, "cause_categories", []) or []):
            if isinstance(cat, dict):
                categories.append(cat)
            else:
                categories.append({
                    "name": cat.name,
                    "features": cat.features,
                    "shap_share_pct": round(float(cat.shap_share_pct), 1),
                    "n_trend_significant": cat.n_trend_significant,
                    "g_star_confirmed": cat.g_star_confirmed,
                    "total_score": round(float(cat.total_score), 3),
                    "confidence": cat.confidence,
                })

        sf_raw = _attr(report, "sim_forecast", None)
        sim_forecast = (
            sf_raw.model_dump() if sf_raw is not None and hasattr(sf_raw, "model_dump") else sf_raw
        )
        judgment_raw = _attr(report, "judgment", None)
        primary_cause_category = ""
        if judgment_raw is not None:
            primary_cause_category = (
                judgment_raw.get("primary_category", "")
                if isinstance(judgment_raw, dict)
                else getattr(judgment_raw, "primary_category", "")
            ) or ""

        return {
            "cause_summary": _attr(report, "cause_summary", "") or "",
            "primary_cause_category": primary_cause_category,
            "shap_top": shap_top,
            "trend_top": trend_top,
            "upstream_suspects": list(_attr(report, "upstream_suspects", []) or [])[:3],
            "consensus_summary": _c("summary", ""),
            "consensus_confidence": _c("confidence_level", "LOW"),
            "g_star_confirmed": _c("g_star_confirmed", False),
            "g_star_proba": _c("g_star_proba", 0.0),
            "g_star_sig_kpis": g_star_sig_kpis,
            "judgment": judgment,
            "cause_categories": categories,
            "sim_forecast": sim_forecast,
        }
    return {}


# ── 9블록 빌더 (compare/2.0) ─────────────────────────────────────────────────

_KPI_UNITS = {
    "wip": "lots",
    "q_time_min": "min",
    "wait_ratio": "ratio",
    "utilization_avg": "ratio",
    "available_tool_ratio": "ratio",
}

# bottleneck_info 필드명 ↔ canonical KPI 이름
_BN_KPI_MAP = {
    "wip": "wip_count",
    "q_time_min": "avg_queue_time_min",
    "wait_ratio": "wait_ratio",
    "utilization_avg": "utilization_avg",
    "available_tool_ratio": "available_tool_ratio",
}


def _build_meta(ci: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_type": ci.get("scenario_type", "global_plan"),
        "scenario_name": ci.get("scenario_name", ci.get("process_name", "")),
        "anchor_toolgroup": ci.get("anchor_toolgroup", ci.get("toolgroup", "")),
        "target_toolgroups": list(ci.get("target_toolgroups") or [ci.get("toolgroup", "")]),
        "severity": ci.get("severity", ""),
        "snapshot_time": ci.get("snapshot_time", 0.0),
        "t0": ci.get("t0", 0.0),
        "horizon_min": ci.get("horizon_min", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _build_current_state_block(ci: dict) -> dict:
    """현재 상태 카드 — KPI 절대값 + 2h 자연 진행 예측."""
    bn = ci.get("bottleneck_info", {})

    kpi: dict = {}
    for canonical, bn_key in _BN_KPI_MAP.items():
        value = bn.get(bn_key)
        if value is not None:
            kpi[canonical] = {"value": value, "unit": _KPI_UNITS.get(canonical, "")}
    kpi["risk_score"] = {"value": bn.get("risk_score", 0.0), "unit": "score"}

    # 2h 자연 진행 예측 — cause_context.sim_forecast에서 가져옴
    cc = ci.get("cause_context") or {}
    sf = cc.get("sim_forecast")
    natural_forecast = None
    if sf:
        gets_worse = sf.get("gets_worse", False)
        kpi_changes: dict = {}
        for k, v in (sf.get("kpi_delta") or {}).items():
            if isinstance(v, dict):
                kpi_changes[k] = {
                    "now": v.get("now"),
                    "after": v.get("future"),
                    "delta": v.get("delta"),
                    "pct_change": v.get("pct_change"),
                    "reliability": v.get("reliability", "MED"),
                }
        natural_forecast = {
            "gets_worse": gets_worse,
            "label": "악화 예상" if gets_worse else "안정 또는 완화 예상",
            "kpi": kpi_changes,
        }

    return {"kpi": kpi, "natural_forecast_2h": natural_forecast}


def _build_cause_block(ci: dict) -> dict:
    """원인 분석 패널 — SHAP, trend, G*, LLM 판정, 카테고리 수렴, 시뮬 예측 전체 포함."""
    cc = ci.get("cause_context") or {}
    if not cc:
        return {
            "summary": "",
            "shap_top": [],
            "trend_top": [],
            "upstream_suspects": [],
            "consensus": {"confidence": "LOW", "summary": ""},
            "g_star": {"confirmed": False, "proba": 0.0, "sig_kpis": []},
            "judgment": None,
            "cause_categories": [],
            "sim_forecast": None,
        }
    return {
        "summary": cc.get("cause_summary", ""),
        "shap_top": cc.get("shap_top", []),
        "trend_top": cc.get("trend_top", []),
        "upstream_suspects": cc.get("upstream_suspects", []),
        "consensus": {
            "confidence": cc.get("consensus_confidence", "LOW"),
            "summary": cc.get("consensus_summary", ""),
        },
        "g_star": {
            "confirmed": cc.get("g_star_confirmed", False),
            "proba": cc.get("g_star_proba", 0.0),
            "sig_kpis": cc.get("g_star_sig_kpis", []),
        },
        "judgment": cc.get("judgment"),
        "cause_categories": cc.get("cause_categories", []),
        "sim_forecast": cc.get("sim_forecast"),
    }


def _build_cascade_block(ci: dict) -> dict:
    """연쇄 영향 패널 — affected_toolgroups, CT 증가, 위험 Lot."""
    cd = ci.get("cascade_impact") or {}
    if not cd:
        return {
            "affected_toolgroups": [],
            "ct_increase_min": 0.0,
            "at_risk_lots": 0,
            "capacity_stress_score": 0.0,
            "impact_score": 0.0,
        }
    return {
        "affected_toolgroups": list(cd.get("affected_tgs", []) or []),
        "ct_increase_min": cd.get("ct_increase_min", 0.0),
        "at_risk_lots": cd.get("at_risk_lots", 0),
        "capacity_stress_score": cd.get("capacity_stress_score", 0.0),
        "impact_score": cd.get("impact_score", 0.0),
    }


def _summarize_plan_meta(pm: dict) -> str:
    """plan_meta → 한 줄 description (40자 이내 목표)."""
    if not pm:
        return ""
    parts: list[str] = []
    if pm.get("release_interval_minutes") is not None:
        cur = pm.get("current_interval_minutes", "-")
        tgt = pm["release_interval_minutes"]
        parts.append(f"Release Interval {cur}→{tgt}분")
    elif pm.get("release_interval_delta_pct") is not None:
        parts.append(f"Release Interval Δ{pm['release_interval_delta_pct']}%")
    if pm.get("superhotlot_enable"):
        parts.append("SUPERHOTLOT")
    return " · ".join(parts) if parts else ""


def _option_params(pm: dict) -> dict:
    """plan_meta → action_options[].params (프론트엔드용)."""
    if not pm:
        return {}
    out: dict = {}
    for k in (
        "release_interval_minutes", "current_interval_minutes", "release_interval_delta_min",
        "release_interval_delta_pct", "lot_priority_rule", "superhotlot_enable",
    ):
        if pm.get(k) is not None:
            out[k] = pm[k]
    return out


def _build_kpi_impact(current_state_kpi: dict, kpi_contribs: dict) -> dict:
    """KPI별 {now, after, delta, pct_change, verdict, confidence} 구조 생성."""
    impact: dict = {}
    for kpi_name, kc in kpi_contribs.items():
        now_val = current_state_kpi.get(kpi_name, {}).get("value")
        delta = float(kc.get("mean_delta", 0.0) or 0.0)
        after = (now_val + delta) if isinstance(now_val, (int, float)) else None
        if isinstance(now_val, (int, float)) and now_val != 0:
            pct = round((delta / now_val) * 100, 2)
        else:
            pct = 0.0
        impact[kpi_name] = {
            "now": round(now_val, 4) if isinstance(now_val, float) else now_val,
            "after": round(after, 4) if isinstance(after, float) else after,
            "delta": round(delta, 4),
            "pct_change": pct,
            "verdict": kc.get("verdict", "unknown"),
            "confidence": round(float(kc.get("confidence", 0.0)), 4),
            "ci_width": round(float(kc.get("ci_width", 0.0)), 4),
        }
    return impact


def _build_anchor_tg_kpi_impact(
    per_tg_forecasts: dict,
    anchor_toolgroup: str,
    kpi_contribs: dict,
) -> dict:
    """복합 TG 표시용 KPI: 평균 없이 anchor TG의 현재 → 대응안 변화만 사용."""
    forecast = per_tg_forecasts.get(anchor_toolgroup) or next(
        iter(per_tg_forecasts.values()),
        {},
    )
    current = forecast.get("current") or {}
    action = forecast.get("action") or {}
    impact: dict = {}
    for kpi_name in _KPI_UNITS:
        now_val = current.get(kpi_name)
        after = action.get(kpi_name)
        if not isinstance(now_val, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = float(after) - float(now_val)
        pct_change = (delta / float(now_val) * 100) if now_val else 0.0
        contribution = kpi_contribs.get(kpi_name) or {}
        impact[kpi_name] = {
            "now": round(float(now_val), 4),
            "after": round(float(after), 4),
            "delta": round(delta, 4),
            "pct_change": round(pct_change, 2),
            "verdict": contribution.get("verdict", "unknown"),
            "confidence": round(float(contribution.get("confidence", 0.0)), 4),
            "ci_width": round(float(contribution.get("ci_width", 0.0)), 4),
        }
    return impact


def _build_current_state_option(ci: dict, current_state_kpi: dict) -> dict:
    """action_options[0] — 현재 상태 유지 (baseline 옵션)."""
    severity = ci.get("severity", "")
    severity_msg = {
        "Critical": "심각한 병목 상황이 지속·악화될 위험이 높습니다",
        "High": "병목 상황이 지속될 가능성이 높습니다",
        "Medium": "병목이 유지되거나 점진적으로 심화될 수 있습니다",
    }.get(severity, "병목 상황이 지속될 수 있습니다")

    # baseline kpi_impact: now=after, delta=0
    kpi_impact: dict = {}
    for k, info in current_state_kpi.items():
        if k == "risk_score":
            continue
        val = info["value"]
        kpi_impact[k] = {
            "now": val, "after": val, "delta": 0.0, "pct_change": 0.0,
            "verdict": "baseline", "confidence": 1.0, "ci_width": 0.0,
        }

    return {
        "label": "현재상태",
        "kind": "NO_ACTION",
        "description": "조치 없음 — 병목 유지",
        "target_toolgroups": [],
        "params": {},
        "kpi_impact": kpi_impact,
        "operational": {"effort": 0, "scope": "none", "reversibility": "high"},
        "composite_score": 0.0,
        "simulation": {"paired_n": 0, "verdict": "baseline", "paired_t_p": None},
        "is_recommended": False,
        "recommendation_status": None,
        "badge": None,
        "is_baseline": True,
        "tradeoffs": [],
        "outcome_if_kept": f"추가 조치 없을 경우 {severity_msg}.",
    }


def _build_action_option(
    c: dict,
    current_state_kpi: dict,
    decision_info: dict,
    anchor_toolgroup: str = "",
) -> dict:
    """action_options[i] — 후보 1개 (A, B, ...)."""
    from agents.compare_agent.scorer import derive_recommendation_status

    md = c.get("action_metadata") or {}
    sb = c.get("score_breakdown") or {}
    pm = c.get("plan_meta") or {}
    rec_meta = derive_recommendation_status(c, decision_info)
    per_tg_forecasts = c.get("per_tg_forecasts") or {}
    kpi_contribs = sb.get("kpi_contributions", {})
    kpi_impact = (
        _build_anchor_tg_kpi_impact(
            per_tg_forecasts,
            anchor_toolgroup,
            kpi_contribs,
        )
        if per_tg_forecasts
        else _build_kpi_impact(current_state_kpi, kpi_contribs)
    )

    return {
        "label": c["label"],
        "kind": c.get("action_kind", "UNKNOWN"),
        "description": _summarize_plan_meta(pm) or c.get("description", "")[:60],
        "target_toolgroups": list(pm.get("target_toolgroups", []) or []),
        "params": _option_params(pm),
        "kpi_impact": kpi_impact,
        "operational": {
            "effort": md.get("effort"),
            "scope": md.get("scope"),
            "reversibility": md.get("reversibility"),
        },
        "composite_score": c.get("composite_score", 0.0),
        "simulation": {
            "paired_n": c.get("paired_n", 0),
            "verdict": c.get("verdict", "unknown"),
            "paired_t_p": c.get("paired_t_p"),
            "simulation_confidence": c.get("simulation_confidence"),
        },
        "is_recommended": rec_meta["is_recommended"],
        "recommendation_status": rec_meta["recommendation_status"],
        "badge": rec_meta["badge"],
        "is_baseline": False,
        "tradeoffs": c.get("tradeoffs", []),
        "per_tg_forecasts": per_tg_forecasts,
        "aggregation_rule": c.get("aggregation_rule") or "",
        "comparison_basis": (
            f"{anchor_toolgroup or '대표 TG'} 현재 상태 대비 대응안 2시간 후"
            if per_tg_forecasts
            else "현재 상태 대비"
        ),
    }


def _build_recommendation_block(
    rec_obj, decision_info: dict, top_candidate: dict,
) -> dict:
    """recommendation 블록 — LLM 출력 + recommended_label/status 보강."""
    rec_dict = rec_obj.model_dump()
    rec_dict["recommended_label"] = decision_info.get("top_label") or top_candidate.get("label", "")
    rec_dict["recommendation_status"] = {
        "clear_winner": "ai_recommended",
        "equivalent_candidates": "equivalent_tiebreak",
        "no_meaningful_effect": "tentative_no_effect",
    }.get(decision_info.get("decision_status", ""), "ai_recommended")
    return rec_dict


def _build_decision_meta(decision_info: dict) -> dict:
    """decision_meta 블록 — 감사·디버깅용."""
    return {
        "decision_status": decision_info.get("decision_status", ""),
        "top_label": decision_info.get("top_label"),
        "equivalent_set": decision_info.get("equivalent_set", []),
        "tiebreaker_used": decision_info.get("tiebreaker_used"),
        "tiebreaker_chain_evaluated": decision_info.get("tiebreaker_chain_evaluated", []),
        "decision_caveat": decision_info.get("decision_caveat", ""),
    }


def _build_data_quality(candidates: list[dict]) -> dict:
    """data_quality 블록 — 모든 mean_delta=0 + paired_n>0 감지."""
    if not candidates:
        return {"status": "ok", "warnings": [], "raw_diagnostics": {}}

    all_zero = True
    for c in candidates:
        contribs = (c.get("score_breakdown") or {}).get("kpi_contributions", {})
        if not contribs:
            all_zero = False
            break
        for kc in contribs.values():
            if abs(float(kc.get("mean_delta", 0.0) or 0.0)) >= 1e-9:
                all_zero = False
                break
        if not all_zero:
            break

    paired_ns = [c.get("paired_n", 0) for c in candidates]
    min_n = min(paired_ns) if paired_ns else 0
    labels = [c["label"] for c in candidates]

    if all_zero and min_n > 0:
        return {
            "status": "warning",
            "warnings": [{
                "code": "SIM_KPI_IDENTICAL",
                "severity": "high",
                "message": (
                    f"baseline ↔ whatif 모든 KPI mean_delta=0 (paired_n={min_n}). "
                    "시뮬 엔진이 whatif 액션을 무시하는 것으로 의심됩니다."
                ),
                "suspect_component": (
                    "agents/verification_agent/sim_executor.py · scripts/run_sim_forward_once.py"
                ),
            }],
            "raw_diagnostics": {
                "all_kpi_deltas_zero": True,
                "min_paired_n": min_n,
                "candidates_checked": labels,
            },
        }

    return {
        "status": "ok",
        "warnings": [],
        "raw_diagnostics": {
            "all_kpi_deltas_zero": False,
            "min_paired_n": min_n,
            "candidates_checked": labels,
        },
    }


# ── HITL 콘솔 출력 ────────────────────────────────────────────────────────────

def _build_rag_block(rag_evidence: dict | None) -> list[str]:
    """MVP HITL용 RAG 요약: 공통 사례, 후보별 판단, 인사이트."""
    if not rag_evidence:
        return []

    candidates = rag_evidence.get("candidates") or []
    common_hits = rag_evidence.get("common_hits") or []
    reference_hits: list[dict] = []
    seen_case_ids: set[str] = set()
    for candidate in candidates:
        for hit in (candidate.get("candidate_hits") or candidate.get("hits") or [])[:3]:
            case_id = str(hit.get("case_id") or "")
            if not case_id or case_id in seen_case_ids:
                continue
            reference_hits.append(hit)
            seen_case_ids.add(case_id)
            if len(reference_hits) >= 6:
                break
        if len(reference_hits) >= 6:
            break
    if not reference_hits:
        reference_hits = list(common_hits[:6])
    if not candidates or (not reference_hits and not common_hits):
        return ["", "[ RAG 유사 사례 참고 ]", "  비교 가능한 유사 사례가 없습니다."]

    is_demo = any(
        "DEMO 합성 데이터" in (hit.get("text") or "")
        for hit in (reference_hits or common_hits)
    )
    heading = "[ RAG 유사 사례 참고 ]"
    if is_demo:
        heading = "[ RAG 유사 사례 참고 - DEMO 합성 데이터 ]"
    lines = ["", heading]

    summary_by_id: dict[str, str] = {}
    for candidate in candidates:
        evidence = candidate.get("evidence") or {}
        for case in evidence.get("case_summaries", []):
            case_id = str(case.get("case_id", ""))
            if case_id and case_id not in summary_by_id:
                summary_by_id[case_id] = case.get("summary", "")

    _label_ko = {"conservative": "보수안", "standard": "표준안", "aggressive": "강화안"}

    def _extract_label(title: str) -> str:
        m = re.search(r'\b(conservative|standard|aggressive)\b', title, re.IGNORECASE)
        return m.group(1).lower() if m else ""

    def _outcome_sentence(summary: str) -> str:
        parts = [s.strip().rstrip(".") for s in summary.split(". ") if len(s.strip()) > 10]
        if len(parts) >= 2:
            return f"{parts[-2]} → {parts[-1]}"
        return parts[-1] if parts else summary[:150]

    def _format_candidate_refs(candidate: dict) -> str:
        refs: list[str] = []
        for hit in (candidate.get("candidate_hits") or candidate.get("hits") or [])[:2]:
            case_id = str(hit.get("case_id", "")).strip()
            if case_id and case_id not in refs:
                refs.append(case_id)
        if refs:
            return "검색 사례: " + ", ".join(refs)

        evidence = candidate.get("evidence") or {}
        claim_ids: list[str] = []
        for claim in evidence.get("claims", [])[:2]:
            for case_id in claim.get("case_ids", [])[:2]:
                case_id = str(case_id).strip()
                if case_id and case_id not in claim_ids:
                    claim_ids.append(case_id)
        if claim_ids:
            return "근거 사례: " + ", ".join(claim_ids)
        return "근거 사례: 없음"

    top_hits = (reference_hits or common_hits)[:6]
    tg_codes = [h.get("tg_code", "") for h in top_hits]
    all_same_tg = len(set(tg_codes)) == 1 and bool(tg_codes[0])

    if all_same_tg:
        tg = tg_codes[0]
        cause = (top_hits[0].get("bottleneck_cause_type") or "").strip()
        group_header = f"  ▸ {tg} {cause} 복합 대응 참조 사례" if cause else f"  ▸ {tg} 복합 대응 참조 사례"
        lines.append(group_header)
        for hit in top_hits:
            case_id = str(hit.get("case_id", "")).strip()
            outcome = _outcome_sentence(hit.get("cause_summary") or "")
            source = hit.get("report_url") or hit.get("source_path") or ""
            source_tag = f" [보기: {source}]" if source else ""
            lines.append(f"    {outcome}{source_tag}")
            if case_id:
                lines.append(f"      근거 사례: {case_id}")
    else:
        for hit in top_hits:
            case_id = str(hit.get("case_id", ""))
            summary = hit.get("cause_summary") or summary_by_id.get(case_id, "")
            title = hit.get("report_title") or case_id
            source = hit.get("report_url") or hit.get("source_path")
            lines.append(f"  - {summary}")
            lines.append(
                f"    근거: {title}" + (f" [보기: {source}]" if source else "")
            )
            if case_id:
                lines.append(f"    근거 사례: {case_id}")

    effect_labels = {
        "high": "높음",
        "medium": "보통",
        "low": "낮음",
        "unknown": "판단 불가",
    }
    risk_labels = {
        "high": "높음",
        "medium": "보통",
        "low": "낮음",
        "unknown": "판단 불가",
    }
    lines += ["", "  ▸ 대응안별 사례 비교"]
    for candidate in candidates:
        label = candidate.get("label", "?")
        evidence = candidate.get("evidence")
        if not evidence:
            lines.append(f"    {label}: 비교 가능한 사례 부족")
            continue
        lines.append(
            f"    {label}: "
            f"효과 {effect_labels.get(evidence.get('effect_outlook'), '판단 불가')} · "
            f"리스크 {risk_labels.get(evidence.get('risk_level'), '판단 불가')}"
        )
        summary = evidence.get("candidate_summary")
        if summary:
            lines.append(f"      {summary}")
        lines.append(f"      {_format_candidate_refs(candidate)}")

    comparison = rag_evidence.get("comparison") or {}
    insight = comparison.get("rag_summary") or comparison.get("overall_comment")
    if insight:
        lines += ["", "[ RAG 인사이트 ]", f"  {insight}"]
    return lines


def _build_rag_statistical_summary(ci: dict) -> str:
    """RAG 인사이트에 전달할 120분 통계 요약."""
    decision = ci.get("decision_info") or {}
    status = decision.get("decision_status", "")
    status_text = {
        "clear_winner": "후보 간 점수 우위가 확인됨",
        "equivalent_candidates": "후보 간 우위가 뚜렷하지 않음",
        "no_meaningful_effect": "의미 있는 개선 효과를 확인하지 못함",
    }.get(status, "판정이 명확하지 않음")
    lines = [f"120분 paired 검증: {status_text}."]
    for candidate in ci.get("action_candidates", []):
        contributions = (
            (candidate.get("score_breakdown") or {}).get("kpi_contributions")
            or {}
        )
        kpi_text = ", ".join(
            f"{kpi} Δ{float((contributions.get(kpi) or {}).get('mean_delta', 0.0)):+.4f}"
            f"({(contributions.get(kpi) or {}).get('verdict', 'unknown')})"
            for kpi in (
                "q_time_min",
                "wip",
                "wait_ratio",
                "utilization_avg",
                "available_tool_ratio",
            )
        )
        lines.append(
            f"{candidate.get('label', '?')}: "
            f"score {float(candidate.get('composite_score', 0.0)):.3f}, "
            f"{kpi_text}"
        )
    return "\n".join(lines)


def _sanitize_candidate_evidence(
    evidence: dict | None,
    hits: list[dict],
) -> dict | None:
    if not evidence:
        return None
    allowed_ids = {
        str(hit.get("case_id"))
        for hit in hits
        if hit.get("case_id")
    }
    sanitized = dict(evidence)
    sanitized["case_summaries"] = [
        case
        for case in evidence.get("case_summaries", [])
        if str(case.get("case_id")) in allowed_ids
    ][:6]
    sanitized["claims"] = [
        {
            **claim,
            "case_ids": [
                str(case_id)
                for case_id in claim.get("case_ids", [])
                if str(case_id) in allowed_ids
            ],
        }
        for claim in evidence.get("claims", [])[:3]
        if any(
            str(case_id) in allowed_ids
            for case_id in claim.get("case_ids", [])
        )
    ]
    return sanitized


def _sanitize_rag_comparison(
    comparison: dict | None,
    candidates: list[dict],
) -> dict | None:
    if not comparison:
        return None
    allowed_labels = {
        str(candidate.get("label"))
        for candidate in candidates
    }
    allowed_case_ids = {
        str(hit.get("case_id"))
        for candidate in candidates
        for hit in candidate.get("hits", [])
        if hit.get("case_id")
    }
    ranking = []
    for item in comparison.get("ranking", []):
        if str(item.get("label")) not in allowed_labels:
            continue
        ranking.append({
            **item,
            "case_ids": [
                str(case_id)
                for case_id in item.get("case_ids", [])
                if str(case_id) in allowed_case_ids
            ],
        })
    return {**comparison, "ranking": ranking}


def _fallback_rag_comparison(candidates: list[dict]) -> dict | None:
    summaries = [
        f"{candidate.get('label')}: "
        f"{(candidate.get('evidence') or {}).get('candidate_summary', '')}"
        for candidate in candidates
        if candidate.get("evidence")
    ]
    if not summaries:
        return None
    return {
        "ranking_status": "insufficient",
        "ranking": [],
        "rag_summary": " ".join(summaries),
        "overall_comment": "RAG는 과거 사례 참고 정보이며 통계 점수와 합산하지 않습니다.",
    }


def _build_hitl_prompt(
    result: dict,
    rag_evidence: dict | None = None,
) -> str:
    """compare/2.0 result → HITL 콘솔 텍스트."""
    meta = result["meta"]
    cs = result["current_state"]
    rec = result["recommendation"]
    options = result["action_options"]
    dq = result["data_quality"]
    dm = result["decision_meta"]

    lines: list[str] = []
    lines += ["", "=" * 58]
    target_tgs = meta.get("target_toolgroups", [])
    anchor = meta.get("anchor_toolgroup", "")
    tgs_summary = (
        f"anchor: {anchor} (대상 TG {len(target_tgs)}개)" if len(target_tgs) > 1 else anchor
    )
    lines.append(f"[비교분석 완료] {meta.get('scenario_name', '')} — {tgs_summary} 관리자 승인 필요")
    lines.append("=" * 58)
    lines.append(
        f"[ 의사결정 상태 ] "
        f"{_DECISION_STATUS_LABEL.get(dm['decision_status'], dm['decision_status'])}"
    )
    if dm.get("decision_caveat"):
        lines.append(f"  {dm['decision_caveat']}")

    lines += ["", "[ 현재 상태 (기준점) ]"]
    for k, info in cs.get("kpi", {}).items():
        lines.append(f"  {k:24s} {info['value']} {info.get('unit', '')}")
    nf = cs.get("natural_forecast_2h")
    if nf:
        lines.append(f"  2h 자연 진행: {nf['label']}")

    # 데이터 품질 경고
    if dq.get("status") == "warning":
        lines += ["", "[ ⚠️ 데이터 품질 경고 ]"]
        for w in dq.get("warnings", []):
            lines.append(f"  · [{w['code']}] {w['message']}")

    # AI 추천
    lines += ["", f"[ AI 추천 ]  (신뢰도 {rec.get('confidence_level', '-')})"]
    rec_label = rec.get("recommended_label", "")
    rec_status = rec.get("recommendation_status", "")
    badge_label = _BADGE_LABEL.get(rec_status, rec_status)
    lines.append(f"  추천 후보: {rec_label}  [{badge_label}]")
    lines.append(f"  {rec.get('headline', '')}")
    lines.append(f"  → {rec.get('primary_reason', '')}")

    # 추천 메커니즘
    wr = rec.get("why_recommended") or {}
    if wr:
        lines += ["", "[ 추천 메커니즘 ]"]
        lines.append(f"  selected_by: {wr.get('selected_by')}")
        chain = wr.get("tiebreaker_chain") or []
        if chain:
            lines.append(f"  tiebreaker_chain: {' → '.join(chain)}")
        if wr.get("explanation"):
            lines.append(f"  설명: {wr['explanation']}")

    # 즉시 실행 단계
    if rec.get("immediate_actions"):
        lines += ["", "[ 즉시 실행 단계 ]"]
        for idx, step in enumerate(rec["immediate_actions"], 1):
            lines.append(f"  {idx}. {step}")

    # 모니터링
    if rec.get("monitoring_kpis"):
        lines += ["", "[ 조치 후 모니터링 ]"]
        for m in rec["monitoring_kpis"]:
            if isinstance(m, dict):
                lines.append(
                    f"  · {m.get('kpi')}: {m.get('target')}  "
                    f"({m.get('check_after_min')}분 후 확인)"
                )

    # 롤백 조건
    if rec.get("rollback_condition"):
        lines += ["", "[ 롤백 조건 ]", f"  ⚠️ {rec['rollback_condition']}"]

    # 트레이드오프
    if rec.get("tradeoffs"):
        lines += ["", "[ 받아들이는 트레이드오프 ]"]
        for t in rec["tradeoffs"]:
            lines.append(f"  · {t}")

    # 다른 후보 선택 안 한 이유
    if rec.get("why_not_others"):
        lines += ["", "[ 다른 후보 선택 안 한 이유 ]"]
        for label, reason in rec["why_not_others"].items():
            lines.append(f"  · {label}: {reason}")

    # 주의사항
    if rec.get("caveats"):
        lines += ["", "[ 주의사항 ]"]
        for c_text in rec["caveats"]:
            lines.append(f"  · {c_text}")

    # 대응안 요약 테이블
    lines += ["", "[ 대응안 요약 ]"]
    for opt in options:
        badge_disp = ""
        if opt.get("badge"):
            badge_disp = f" [{_BADGE_LABEL.get(opt['badge'], opt['badge'])}]"
        score = opt.get("composite_score") or 0.0
        q_impact = opt.get("kpi_impact", {}).get("q_time_min", {})
        wip_impact = opt.get("kpi_impact", {}).get("wip", {})
        sim_conf = (opt.get("simulation") or {}).get("simulation_confidence")
        conf_disp = f"{int((sim_conf or 0) * 100)}%" if sim_conf is not None else "-"
        lines.append(
            f"  {opt['label']:8s}{badge_disp}  score {score:.3f}  "
            f"q_time Δ{q_impact.get('delta', 0):+.1f}분  "
            f"WIP Δ{wip_impact.get('delta', 0):+.1f}  "
            f"신뢰도 {conf_disp}"
        )
        for target_tg, forecast in (opt.get("per_tg_forecasts") or {}).items():
            current = forecast.get("current") or {}
            action = forecast.get("action") or current
            lines.append(f"    └ {target_tg}")
            lines.append(
                f"       q_time {current.get('q_time_min', '-')}→{action.get('q_time_min', '-')}분  "
                f"WIP {current.get('wip', '-')}→{action.get('wip', '-')}  "
                f"wait {current.get('wait_ratio', '-')}→{action.get('wait_ratio', '-')}"
            )
            lines.append(
                f"       util {current.get('utilization_avg', '-')}→{action.get('utilization_avg', '-')}  "
                f"avail {current.get('available_tool_ratio', '-')}→{action.get('available_tool_ratio', '-')}"
            )

    lines += _build_rag_block(rag_evidence)

    choices = " / ".join(
        [
            *(str(option.get("label")) for option in options if option.get("label")),
            "반려",
        ]
    )
    lines += ["", f"승인할 대응안을 입력하세요 ({choices}): ", ""]
    return "\n".join(lines)


# ── Pipeline 노드 ─────────────────────────────────────────────────────────────

def compare_rank(state: "PipelineState") -> dict:
    """Agent 5-1: 입력 변환 + 대응안 순위 결정 (LLM 불필요)."""
    from agents.logger import get_logger
    from agents.verification_agent.sim_executor import HORIZON_MIN
    _log = get_logger(__name__)

    verification_results = state.get("verification_results", [])
    if not verification_results:
        _log.info("[Compare] verification_results 없음 — 스킵")
        return {"compare_inputs": []}

    alerts = state.get("alerts", [])
    kpi_snapshot = state.get("kpi_snapshot", [])
    kpi_map = {k.toolgroup: k for k in kpi_snapshot}
    t0 = kpi_snapshot[0].snapshot_time if kpi_snapshot else 0.0

    compare_inputs: list[dict] = []

    global_plan_groups = [g for g in verification_results if "plan_id" in g]

    if global_plan_groups:
        from agents.schemas.alert import SeverityLevel
        critical_alerts = [a for a in alerts if a.severity == SeverityLevel.CRITICAL]
        anchor_alert = (
            max(critical_alerts, key=lambda a: a.composite_score)
            if critical_alerts else None
        )
        if anchor_alert is None:
            _log.warning("[Compare] GlobalSolutionPlan: CRITICAL anchor alert 없음 — 스킵")
        else:
            anchor_tg = anchor_alert.toolgroup
            target_tgs = global_plan_groups[0].get("target_toolgroups", [])
            all_verified: list[dict] = []
            for g in global_plan_groups:
                all_verified.extend(g.get("verified_candidates", []))
            candidates = _build_action_candidates(all_verified)
            candidates, scored, decision_info = _rank_candidates(candidates)
            first_forecasts = (
                candidates[0].get("per_tg_forecasts") if candidates else {}
            ) or {}
            target_toolgroup_states = {
                target_tg: {
                    "current": forecast.get("current") or {},
                    "no_action": forecast.get("no_action") or {},
                }
                for target_tg, forecast in first_forecasts.items()
            }
            compare_inputs.append({
                "toolgroup": anchor_tg,
                "process_name": "글로벌 플랜 A/B",
                "scenario_type": "global_plan",
                "scenario_name": "글로벌 플랜 A/B",
                "anchor_toolgroup": anchor_tg,
                "target_toolgroups": list(target_tgs),
                "severity": anchor_alert.severity.value,
                "snapshot_time": float(global_plan_groups[0].get("snapshot_time", t0)),
                "t0": float(t0),
                "horizon_min": HORIZON_MIN,
                "bottleneck_info": _build_bottleneck_info(anchor_tg, anchor_alert, kpi_map.get(anchor_tg)),
                "cascade_impact": anchor_alert.impact.model_dump(),
                "cause_context": _build_cause_context(anchor_tg, state.get("cause_reports", [])),
                "action_candidates": candidates,
                "scored_actions": scored,
                "decision_info": decision_info,
                "target_toolgroup_states": target_toolgroup_states,
                "comparison_basis": (
                    "대응안 2시간 후와 무대응 2시간 후의 차이"
                    if target_toolgroup_states
                    else "현재 상태 대비 대응안 변화"
                ),
            })

    _log.info(f"[Compare] rank 완료 — {len(compare_inputs)}개 공정")
    return {"compare_inputs": compare_inputs}


def compare_llm(state: "PipelineState") -> dict:
    """Agent 5-2: LLM 추천 근거 생성 + 9블록 출력 포맷 구성 (compare/2.0)."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    compare_inputs = state.get("compare_inputs", [])
    if not compare_inputs:
        return {"compare_formatted": []}

    compare_formatted: list[dict] = []

    for ci in compare_inputs:
        tg = ci["toolgroup"]
        candidates = ci["action_candidates"]
        decision_info = ci.get("decision_info", {}) or {}
        decision_status = decision_info.get("decision_status", "clear_winner")

        # data_quality 사전 계산 → LLM 프롬프트에도 전달
        data_quality = _build_data_quality(candidates)
        ci_with_dq_hint = {**ci, "data_quality_hint": data_quality}

        # top_candidate 결정
        top_label = decision_info.get("top_label")
        top_candidate = next(
            (c for c in candidates if c["label"] == top_label),
            candidates[0] if candidates else {},
        )

        # LLM 호출 (recommendation 모듈 내부에서 모든 케이스 분기 처리)
        from agents.compare_agent.recommendation import generate_recommendation
        recommendation_obj = generate_recommendation(
            ci=ci_with_dq_hint,
            candidates=candidates,
            decision_info=decision_info,
            top_candidate=top_candidate,
            llm=_get_llm(),
        )
        _log.info(
            f"[Compare] {tg} 추천 생성 완료 — decision={decision_status} "
            f"confidence={recommendation_obj.confidence_level}"
        )

        # 9블록 빌드
        meta = _build_meta(ci)
        current_state = _build_current_state_block(ci)
        cause = _build_cause_block(ci)
        cascade = _build_cascade_block(ci)
        baseline_option = _build_current_state_option(ci, current_state["kpi"])
        target_states = ci.get("target_toolgroup_states") or {}
        if target_states:
            baseline_option["per_tg_forecasts"] = {
                target_tg: {
                    "current": values.get("current") or {},
                    "no_action": values.get("no_action") or {},
                    "action": values.get("no_action") or {},
                }
                for target_tg, values in target_states.items()
            }
        action_options = [baseline_option]
        scored_map = {
            s["label"]: s
            for s in ci.get("scored_actions", [])
            if isinstance(s, dict) and s.get("label")
        }
        for c in candidates:
            opt = _build_action_option(
                c,
                current_state["kpi"],
                decision_info,
                anchor_toolgroup=ci.get("anchor_toolgroup") or tg,
            )
            scored = scored_map.get(c["label"], {})
            if scored.get("badge") and not opt.get("badge"):
                opt["badge"] = scored["badge"]
            action_options.append(opt)
        recommendation = _build_recommendation_block(recommendation_obj, decision_info, top_candidate)
        decision_meta = _build_decision_meta(decision_info)

        rag_context = state.get("rag_context") or []
        rag_evidence = next(
            (
                item
                for item in rag_context
                if item.get("toolgroup") == tg
            ),
            None,
        )

        result_v2 = {
            "meta": meta,
            "current_state": current_state,
            "cause": cause,
            "cascade": cascade,
            "action_options": action_options,
            "recommendation": recommendation,
            "decision_meta": decision_meta,
            "data_quality": data_quality,
            "rag_evidence": rag_evidence,
            "target_toolgroup_states": ci.get("target_toolgroup_states") or {},
            "comparison_basis": ci.get("comparison_basis") or "",
        }
        COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        (COMPARE_OUT_DIR / f"compare_debug_{tg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps(result_v2, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        hitl_prompt = _build_hitl_prompt(
            result_v2,
            rag_evidence=rag_evidence,
        )

        compare_formatted.append({
            "toolgroup": tg,
            "result_v2": result_v2,
            "action_candidates": candidates,
            "decision_info": decision_info,
            "hitl_prompt": hitl_prompt,
        })

        _log.info(f"[Compare] {tg} 포맷 완료")

    return {"compare_formatted": compare_formatted}


def compare_rag(state: "PipelineState") -> dict:
    """TG와 원인으로 공통 사례를 찾고 후보별 효과·리스크를 비교한다."""
    from agents.compare_agent.rag_evaluator import (
        build_action_profile,
        build_plan_description,
        build_plan_query,
        compare_candidate_evidence,
        evaluate_candidate_evidence,
    )
    from agents.logger import get_logger
    from app.services.qdrant_case_service import QdrantCaseService

    log = get_logger(__name__)
    compare_inputs = state.get("compare_inputs", [])
    if not compare_inputs:
        return {"rag_context": []}

    try:
        qdrant = QdrantCaseService()
    except Exception:
        log.exception("[RAG] Qdrant 초기화 실패")
        return {"rag_context": []}

    def _candidate_family(candidate: dict) -> str:
        profile = build_action_profile(candidate)
        interval = float(profile.get("interval_pct") or 0.0)
        has_priority = bool(profile.get("has_priority"))
        has_superhotlot = bool(profile.get("has_superhotlot"))
        desc = f"{candidate.get('description', '')} {build_plan_description(candidate)}".lower()
        if has_superhotlot or "aggressive" in desc or interval >= 28:
            return "aggressive"
        if has_priority and interval >= 20:
            return "standard"
        if interval <= 16 and not has_priority and not has_superhotlot:
            return "conservative"
        return "unknown"

    def _score_hit(candidate: dict, hit: dict) -> tuple[float, float]:
        family = _candidate_family(candidate)
        text = " ".join([
            str(hit.get("report_title", "")),
            str(hit.get("source_path", "")),
            str(hit.get("cause_summary", "")),
            str(hit.get("text", "")),
        ]).lower()
        q_score = float(hit.get("score", 0.0) or 0.0)
        boost = 0.0

        def has(*terms: str) -> bool:
            return any(term.lower() in text for term in terms)

        if family == "aggressive":
            if has("aggressive"):
                boost += 100.0
            if has("+28%", "28%"):
                boost += 60.0
            if has("superhotlot", "shl"):
                boost += 50.0
            if has("priority 30"):
                boost += 20.0
        elif family == "standard":
            if has("standard"):
                boost += 100.0
            if has("+22%", "+23%", "22%", "23%"):
                boost += 60.0
            if has("priority 20"):
                boost += 30.0
            if has("superhotlot 없음", "superhotlot 없", "superhotlot 없음", "없음"):
                boost += 20.0
        elif family == "conservative":
            if has("conservative"):
                boost += 100.0
            if has("+16%", "16%"):
                boost += 60.0
            if has("우선순위 없음", "priority 없음", "priority 없음", "priority none"):
                boost += 30.0
            if has("superhotlot 없음", "superhotlot 없", "없음"):
                boost += 20.0

        # 같은 TG/원인 사례가 있으면 소폭 보너스
        if str(hit.get("tg_code", "")) == str(candidate.get("toolgroup", "")):
            boost += 10.0
        if str(hit.get("bottleneck_cause_type", "")) and str(hit.get("bottleneck_cause_type", "")) in text:
            boost += 5.0

        return (boost, q_score)

    def _rank_hits(candidate: dict, hits: list[dict]) -> list[dict]:
        ranked = sorted(
            hits,
            key=lambda hit: _score_hit(candidate, hit),
            reverse=True,
        )
        return ranked

    cause_reports = state.get("cause_reports", [])
    kpi_snapshot = state.get("kpi_snapshot", [])
    kpi_map = {k.toolgroup: k for k in kpi_snapshot}

    rag_context: list[dict] = []
    for ci in compare_inputs:
        tg = ci.get("toolgroup", "")
        cause = ci.get("cause_context") or {}
        bottleneck = ci.get("bottleneck_info") or {}
        target_tgs: list[str] = ci.get("target_toolgroups") or [tg]

        all_tg_contexts = {
            t: _build_cause_context(t, cause_reports)
            for t in target_tgs
        }

        if len(target_tgs) > 1:
            # 복합 TG: 모든 TG를 함께 묘사하는 단일 통합 쿼리
            tg_lines = []
            for t in target_tgs:
                ctx = all_tg_contexts.get(t, {})
                kpi = kpi_map.get(t)
                tg_lines.append(
                    f"툴그룹: {t}, 원인: {ctx.get('primary_cause_category', '')}, "
                    f"원인 요약: {ctx.get('cause_summary', '')}, "
                    f"WIP: {int(kpi.wip) if kpi else 0}, "
                    f"대기: {round(kpi.q_time_min, 1) if kpi else 0}분"
                )
            common_query = "복합 병목 동시 대응\n" + "\n".join(tg_lines)
            common_hits = qdrant.search_similar(
                common_query,
                top_k=10,
                min_score=0.55,
                cause_type_filter=None,
                tg_filter=None,
            )
            current_state = " | ".join(
                f"툴그룹 {t}(원인: {all_tg_contexts.get(t, {}).get('primary_cause_category', '')}), "
                f"WIP {int(kpi_map.get(t).wip) if kpi_map.get(t) else 0}, "
                f"대기 {round(kpi_map.get(t).q_time_min, 1) if kpi_map.get(t) else 0}분"
                for t in target_tgs
            )
        else:
            # 단일 TG: 기존 로직
            common_query = (
                f"툴그룹: {tg}\n"
                f"원인 유형: {cause.get('primary_cause_category', '')}\n"
                f"원인 요약: {cause.get('cause_summary', '')}\n"
                f"현재 상태: WIP {bottleneck.get('wip_count', 0)}, "
                f"대기 {bottleneck.get('avg_queue_time_min', 0)}분, "
                f"이용률 {bottleneck.get('utilization_avg', 0)}"
            )
            common_hits = qdrant.search_similar(
                common_query,
                top_k=10,
                min_score=0.55,
                cause_type_filter=(
                    cause.get("primary_cause_category") or None
                ),
                tg_filter=tg or None,
            )
            current_state = (
                f"툴그룹 {tg}, 원인 {cause.get('cause_summary', '')}, "
                f"WIP {bottleneck.get('wip_count', 0)}, "
                f"대기 {bottleneck.get('avg_queue_time_min', 0)}분"
            )

        shared_hits = common_hits
        candidates_rag: list[dict] = []
        for candidate in ci.get("action_candidates", []):
            profile = build_action_profile(candidate)
            plan_description = build_plan_description(candidate)
            candidate_query = build_plan_query(ci, candidate)
            cause_filter = cause.get("primary_cause_category") or None
            tg_filter = tg or None if len(target_tgs) == 1 else None
            candidate_hits = qdrant.search_similar(
                candidate_query,
                top_k=10,
                min_score=0.55,
                cause_type_filter=cause_filter if len(target_tgs) == 1 else None,
                tg_filter=tg_filter,
            )
            candidate_hits = _rank_hits(candidate, candidate_hits)
            hits_for_candidate = candidate_hits or common_hits
            evidence = None
            if hits_for_candidate:
                result = evaluate_candidate_evidence(
                    plan_description=plan_description,
                    current_state_summary=current_state,
                    hits=hits_for_candidate,
                    llm=_get_llm(),
                    action_profile=profile,
                )
                evidence = _sanitize_candidate_evidence(
                    result.model_dump() if result else None,
                    hits_for_candidate,
                )
            candidates_rag.append({
                "label": candidate.get("label", "?"),
                "profile": profile,
                "plan_description": plan_description,
                "hits": hits_for_candidate,
                "shared_hits": common_hits,
                "candidate_hits": candidate_hits,
                "family": _candidate_family(candidate),
                "evidence": evidence,
            })

        comparison_result = compare_candidate_evidence(
            candidate_evidence=candidates_rag,
            statistical_summary=_build_rag_statistical_summary(ci),
            llm=_get_llm(),
            current_context=current_state,
        )
        comparison = _sanitize_rag_comparison(
            comparison_result.model_dump() if comparison_result else None,
            candidates_rag,
        ) or _fallback_rag_comparison(candidates_rag)
        rag_context.append({
            "toolgroup": tg,
            "common_hits": common_hits,
            "candidates": candidates_rag,
            "comparison": comparison,
        })
    return {"rag_context": rag_context}


def compare_hitl(state: "PipelineState") -> dict:
    """Agent 5-3: 관리자 승인(HITL) — Webhook / Auto / Terminal 모드."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    compare_formatted = state.get("compare_formatted", [])
    if not compare_formatted:
        return {"compare_results": []}

    webhook_mode = os.environ.get("WEBHOOK_MODE", "").lower() in ("1", "true", "yes")
    auto_approve = os.environ.get("AUTO_APPROVE", "").lower() in ("1", "true", "yes")

    if webhook_mode:
        return _hitl_webhook(state, compare_formatted, _log)
    elif auto_approve:
        return _hitl_auto(compare_formatted, _log)
    else:
        return _hitl_terminal(compare_formatted, _log)


# ── HITL 모드별 구현 ──────────────────────────────────────────────────────────

def _hitl_webhook(state: "PipelineState", compare_formatted: list[dict], _log) -> dict:
    """Webhook 모드: 상태 직렬화 → hitl_pending/ 저장 → HITL 승인 대기."""
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    hitl_token = str(_uuid.uuid4())

    pending = {
        "hitl_token": hitl_token,
        "created_at": _dt.now(_tz.utc).isoformat(),
        "compare_formatted": compare_formatted,
        "alerts": [a.model_dump() for a in state.get("alerts", [])],
        "kpi_snapshot": [k.model_dump() for k in state.get("kpi_snapshot", [])],
        "prev_kpi_snapshot": [k.model_dump() for k in state.get("prev_kpi_snapshot", [])],
        "cause_reports": [r.model_dump() for r in state.get("cause_reports", [])],
        "cascade_report": state.get("cascade_report"),
        "solution_candidates": state.get("solution_candidates", []),
    }

    pending_dir = _ROOT / "hitl_pending"
    pending_dir.mkdir(exist_ok=True)
    pending_path = pending_dir / f"{hitl_token}.json"
    pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info(f"[HITL-Webhook] 상태 저장: {pending_path.name}")

    _notify_spring_boot(hitl_token, compare_formatted, _log)

    print(f"\n{'='*60}")
    print(f"[분석 완료 — HITL 대기 중]")
    print(f"  token : {hitl_token}")
    print(f"  파일  : {pending_path}")
    print(f"  보고서 생성: python run_report.py --token {hitl_token}")
    print(f"{'='*60}\n")

    return {"compare_results": [], "hitl_approved": None, "hitl_token": hitl_token}


def _notify_spring_boot(hitl_token: str, compare_formatted: list[dict], _log) -> None:
    """Spring Boot에 HITL 대기 요청 전송 (실패해도 분석 파이프라인 계속 진행).

    compare/2.0 스키마 그대로 송신. Spring Boot 매핑은 별도 작업.
    """
    import urllib.request as _req
    import urllib.error as _err
    from datetime import datetime as _dt, timezone as _tz

    spring_url = os.environ.get("SPRING_BOOT_URL", "").strip()
    if not spring_url:
        _log.debug("[HITL-Webhook] SPRING_BOOT_URL 미설정 — Spring Boot 알림 스킵")
        return

    internal_token = os.environ.get("INTERNAL_API_TOKEN", "")
    results = [cf["result_v2"] for cf in compare_formatted]
    payload = json.dumps({
        "hitlToken": hitl_token,
        "schemaVersion": SCHEMA_VERSION,
        "toolgroups": [cf["toolgroup"] for cf in compare_formatted],
        "severity": (
            results[0]["meta"]["severity"] if results else "CRITICAL"
        ),
        "compareResults": results,
    }, ensure_ascii=False).encode("utf-8")

    request = _req.Request(
        f"{spring_url}/api/internal/agent-hitl/pending",
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Internal-Token": internal_token,
            "X-Event-Timestamp": _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        method="POST",
    )
    try:
        with _req.urlopen(request, timeout=10) as resp:
            _log.info(f"[HITL-Webhook] Spring Boot 등록 완료: HTTP {resp.status}")
    except _err.HTTPError as e:
        _log.warning(f"[HITL-Webhook] Spring Boot 등록 실패: HTTP {e.code} — 분석 파이프라인은 계속 진행")
    except Exception as e:
        _log.warning(f"[HITL-Webhook] Spring Boot 연결 실패: {e} — 분석 파이프라인은 계속 진행")


def _hitl_auto(compare_formatted: list[dict], _log) -> dict:
    """AUTO_APPROVE 모드: AI 추천 대응안 자동 승인."""
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_results: list[dict] = []

    for cf in compare_formatted:
        tg = cf["toolgroup"]
        print(cf["hitl_prompt"])

        rec_label = cf["result_v2"]["recommendation"].get("recommended_label", "")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[AUTO-APPROVE] AI 추천 대응안 자동 승인: {rec_label}")
        approval_info = {
            "status": "승인",
            "approved_by": "AUTO",
            "approved_role": "SYSTEM",
            "approved_at": now,
            "comment": "자동 승인 (AUTO_APPROVE 모드)",
            "rejection_reason": None,
            "selected_label": rec_label,
        }
        compare_results.append(_build_compare_result(cf, approval_info))
        _log.info(f"[Compare] {tg} AUTO-APPROVE 완료")

    _log.info(f"[Compare] 완료 — {len(compare_results)}개 공정")
    return {"compare_results": compare_results}


def _hitl_terminal(compare_formatted: list[dict], _log) -> dict:
    """Terminal 모드: 터미널 입력으로 승인/반려."""
    COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare_results: list[dict] = []

    for cf in compare_formatted:
        tg = cf["toolgroup"]
        print(cf["hitl_prompt"])

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        option_labels = {opt["label"].upper() for opt in cf["result_v2"]["action_options"]}
        valid = option_labels | {"반려"}
        while True:
            raw = input(">>> ").strip().upper()
            if raw in valid or raw == "현재상태":
                break
            print(f"유효하지 않은 입력입니다. ({' / '.join(sorted(valid))}) 중 선택하세요.")

        if raw == "반려":
            approval_info = {
                "status": "반려",
                "approved_by": None,
                "approved_role": None,
                "approved_at": now,
                "comment": None,
                "rejection_reason": input("반려 사유를 입력하세요 >>> ").strip(),
            }
        else:
            approver = input("승인자 이름을 입력하세요 >>> ").strip()
            role = input("직책을 입력하세요 >>> ").strip()
            comment = input("의견 (없으면 Enter) >>> ").strip() or "즉시 적용 승인"
            approval_info = {
                "status": "승인",
                "approved_by": approver,
                "approved_role": role,
                "approved_at": now,
                "comment": comment,
                "rejection_reason": None,
                "selected_label": raw.lower(),
            }

        compare_results.append(_build_compare_result(cf, approval_info))
        _log.info(f"[Compare] {tg} HITL 완료")

    _log.info(f"[Compare] 완료 — {len(compare_results)}개 공정")
    return {"compare_results": compare_results}


def _build_compare_result(cf: dict, approval_info: dict) -> dict:
    """compare_formatted + approval_info → compare_result (파일 저장 포함)."""
    tg = cf["toolgroup"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_v2 = cf["result_v2"]
    # approval_info를 result_v2 최상위에 합쳐 저장
    final_output = {**result_v2, "approval_info": approval_info}

    json_path = COMPARE_OUT_DIR / f"compare_v2_{tg}_{ts}.json"
    json_path.write_text(
        json.dumps(final_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[저장 완료] {json_path}")

    return {
        "toolgroup": tg,
        "json_output_path": str(json_path),
        "result_v2": result_v2,
        "approval_info": approval_info,
    }
