#!/usr/bin/env python3
"""비교분석 Agent (Agent 05) — PipelineState 통합 노드 (compare/2.0)."""

from __future__ import annotations

import json
import os
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

        return {
            "cause_summary": _attr(report, "cause_summary", "") or "",
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


def _build_action_option(c: dict, current_state_kpi: dict, decision_info: dict) -> dict:
    """action_options[i] — 후보 1개 (A, B, ...)."""
    from agents.compare_agent.scorer import derive_recommendation_status

    md = c.get("action_metadata") or {}
    sb = c.get("score_breakdown") or {}
    pm = c.get("plan_meta") or {}
    rec_meta = derive_recommendation_status(c, decision_info)

    return {
        "label": c["label"],
        "kind": c.get("action_kind", "UNKNOWN"),
        "description": _summarize_plan_meta(pm) or c.get("description", "")[:60],
        "target_toolgroups": list(pm.get("target_toolgroups", []) or []),
        "params": _option_params(pm),
        "kpi_impact": _build_kpi_impact(current_state_kpi, sb.get("kpi_contributions", {})),
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

def _build_hitl_prompt(result: dict) -> str:
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

    # 현재 상태
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

    lines += ["", "승인할 대응안을 입력하세요 (현재상태 / A / B / C / 반려): ", ""]
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
        action_options = [_build_current_state_option(ci, current_state["kpi"])]
        scored_map = {
            s["label"]: s
            for s in ci.get("scored_actions", [])
            if isinstance(s, dict) and s.get("label")
        }
        for c in candidates:
            opt = _build_action_option(c, current_state["kpi"], decision_info)
            scored = scored_map.get(c["label"], {})
            if scored.get("badge") and not opt.get("badge"):
                opt["badge"] = scored["badge"]
            action_options.append(opt)
        recommendation = _build_recommendation_block(recommendation_obj, decision_info, top_candidate)
        decision_meta = _build_decision_meta(decision_info)

        result_v2 = {
            "meta": meta,
            "current_state": current_state,
            "cause": cause,
            "cascade": cascade,
            "action_options": action_options,
            "recommendation": recommendation,
            "decision_meta": decision_meta,
            "data_quality": data_quality,
        }

        hitl_prompt = _build_hitl_prompt(result_v2)

        compare_formatted.append({
            "toolgroup": tg,
            "result_v2": result_v2,
            "action_candidates": candidates,
            "decision_info": decision_info,
            "hitl_prompt": hitl_prompt,
        })

        _log.info(f"[Compare] {tg} 포맷 완료")

    return {"compare_formatted": compare_formatted}


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
    """Webhook 모드: 상태 직렬화 → hitl_pending/ 저장 → Phase 1 종료."""
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
    print(f"[Phase 1 완료] HITL 대기 중")
    print(f"  token : {hitl_token}")
    print(f"  파일  : {pending_path}")
    print(f"  Phase 2 실행: python run_phase2.py --token {hitl_token}")
    print(f"{'='*60}\n")

    return {"compare_results": [], "hitl_approved": None, "hitl_token": hitl_token}


def _notify_spring_boot(hitl_token: str, compare_formatted: list[dict], _log) -> None:
    """Spring Boot에 HITL 대기 요청 전송 (실패해도 Phase 1 계속 진행).

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
        _log.warning(f"[HITL-Webhook] Spring Boot 등록 실패: HTTP {e.code} — Phase 1은 계속 진행")
    except Exception as e:
        _log.warning(f"[HITL-Webhook] Spring Boot 연결 실패: {e} — Phase 1은 계속 진행")


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
