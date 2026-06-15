"""단계별 중간 JSON 저장 — B-mode 프론트엔드 push용.

Stage 1: cascade 완료 → stage1_alert_t{t}.json
Stage 2: cause 완료   → stage2_cause_t{t}.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agents.logger import get_logger
from agents.state import PipelineState

_log = get_logger(__name__)
_OUT = Path(__file__).parent.parent / "stage_out"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _snapshot_time(state: PipelineState) -> int:
    snap = state.get("kpi_snapshot") or []
    return int(snap[0].snapshot_time) if snap else 0


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info(f"[stage_writer] 저장: {path.name}")


# ── Stage 1: 병목감지 + 확산분석 ──────────────────────────────────────────────

def emit_stage1(state: PipelineState) -> PipelineState:
    t = _snapshot_time(state)
    kpi_map = {k.toolgroup: k for k in (state.get("kpi_snapshot") or [])}

    alerts_json = []
    for a in (state.get("alerts") or []):
        kpi = kpi_map.get(a.toolgroup)
        entry: dict = {
            "toolgroup": a.toolgroup,
            "severity": a.severity.value,
            "composite_score": round(float(a.composite_score), 4),
            "probability": round(float(a.probability), 4),
            "impact": {
                "at_risk_lots": int(a.impact.at_risk_lots),
                "ct_increase_min": round(float(a.impact.ct_increase_min), 1),
                "impact_score": round(float(a.impact.impact_score), 4),
                "affected_count": len(a.impact.affected_tgs),
                "affected_tgs": list(a.impact.affected_tgs),
            },
        }
        if kpi:
            entry["kpi"] = {
                "wip": int(kpi.wip),
                "utilization_avg": round(float(kpi.utilization_avg), 4),
                "wait_ratio": round(float(kpi.wait_ratio), 4),
                "q_time_min": round(float(kpi.q_time_min), 1),
                "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
            }
        alerts_json.append(entry)

    _save(
        _OUT / f"stage1_alert_t{t}.json",
        {
            "event": "bottleneck_detected",
            "snapshot_time": t,
            "generated_at": _now(),
            "alerts": alerts_json,
        },
    )
    return state


# ── Stage 2: 원인 분석 ────────────────────────────────────────────────────────

def emit_stage2(state: PipelineState) -> PipelineState:
    t = _snapshot_time(state)

    cause_json = []
    for r in (state.get("cause_reports") or []):
        total_shap = sum(abs(f.shap_value) for f in r.shap_top) or 1.0

        shap_top = [
            {
                "rank": i + 1,
                "feature": f.feature,
                "kpi_value": round(float(f.kpi_value), 4),
                "shap_value": round(float(f.shap_value), 4),
                "contribution_pct": round(abs(f.shap_value) / total_shap * 100, 1),
            }
            for i, f in enumerate(r.shap_top)
        ]

        trend_top = [
            {
                "feature": tr.feature,
                "slope_per_hour": round(float(tr.slope_per_hour), 4),
                "r2": round(float(tr.r2), 3),
                "significant": tr.significant,
            }
            for tr in r.trend_top
        ]

        categories = [
            {
                "name": c.name,
                "features": c.features,
                "shap_share_pct": c.shap_share_pct,
                "n_trend_significant": c.n_trend_significant,
                "upstream_match": c.upstream_match,
                "g_star_confirmed": c.g_star_confirmed,
                "total_score": c.total_score,
                "confidence": c.confidence,
            }
            for c in (r.cause_categories or [])
        ]

        judgment = None
        if r.judgment:
            j = r.judgment
            judgment = {
                "primary_category": j.primary_category,
                "primary_cause": j.primary_cause,
                "primary_confidence": j.primary_confidence,
                "primary_reasoning": j.primary_reasoning,
                "secondary_causes": list(j.secondary_causes or []),
                "dismissed": list(j.dismissed or []),
                "dismissed_reason": j.dismissed_reason,
            }

        # G* 통계 검정 결과
        g_star = None
        if r.consensus:
            con = r.consensus
            sig_kpis = [
                {
                    "kpi": k.kpi,
                    "delta_mean": round(float(k.delta_mean), 4),
                    "t_p_adj": round(float(k.t_p_adj), 4),
                    "significant": k.significant,
                }
                for k in (con.g_star_sig_kpis or [])
            ]
            g_star = {
                "confirmed": con.g_star_confirmed,
                "proba": round(float(con.g_star_proba or 0.0), 4),
                "sig_kpis": sig_kpis,
            }

        # 시뮬 예측 (1200min forward sim)
        sim_forecast = None
        if r.sim_forecast:
            sf = r.sim_forecast
            sim_forecast = {
                "t0": sf.t0,
                "t_future": sf.t_future,
                "gets_worse": sf.gets_worse,
                "kpi_delta": {
                    k: {
                        "now": round(float(v.now), 4),
                        "future": round(float(v.future), 4),
                        "delta": round(float(v.delta), 4),
                        "pct_change": round(float(v.pct_change), 2),
                    }
                    for k, v in sf.kpi_delta.items()
                },
            }

        cause_json.append({
            "toolgroup": r.toolgroup,
            "snapshot_time": r.snapshot_time,
            "judgment": judgment,
            "categories": categories,
            "shap_top": shap_top,
            "trend_top": trend_top,
            "upstream_suspects": list(r.upstream_suspects or []),
            "g_star": g_star,
            "sim_forecast": sim_forecast,
            "cause_summary": r.cause_summary or "",
        })

    _save(
        _OUT / f"stage2_cause_t{t}.json",
        {
            "event": "cause_analyzed",
            "snapshot_time": t,
            "generated_at": _now(),
            "cause_reports": cause_json,
        },
    )
    return state


# ── Stage 3: 대응안 비교/승인 ─────────────────────────────────────────────────

def emit_stage3(state: PipelineState) -> PipelineState:
    t = _snapshot_time(state)

    _save(
        _OUT / f"stage3_action_t{t}.json",
        {
            "event": "action_approved",
            "snapshot_time": t,
            "generated_at": _now(),
            "hitl_approved": state.get("hitl_approved"),
            "compare_results": state.get("compare_results") or [],
        },
    )
    return state
