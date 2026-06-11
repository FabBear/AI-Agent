#!/usr/bin/env python3
"""최종보고서 생성 Agent (Agent 06) — PipelineState 통합 노드."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.state import PipelineState

from agents.report_agent.writer import write_summary, write_diffusion, write_cause, write_actions

_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = _ROOT / "report_agent_out"


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _build_draft_item(compare_result: dict, alert, kpi, prev_kpi, cause_report, kpi_map: dict | None = None, detected_at: str = "") -> dict:
    """compare_result + PipelineState 데이터 → report_draft 초기 항목."""
    tg = compare_result["toolgroup"]
    if not detected_at:
        detected_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    bottleneck_info: dict = {
        "tool_group": tg,
        "risk_score": round(float(alert.composite_score) * 100, 1),
        "delayed_orders": int(alert.impact.at_risk_lots),
    }
    if kpi:
        bottleneck_info.update({
            "avg_queue_time_min": round(float(kpi.q_time_min), 1),
            "peak_q_time_min": round(float(kpi.q_time_min), 1),
            "utilization_pct": round(float(kpi.utilization_avg) * 100, 1),
            "load_ratio": round(float(kpi.wait_ratio), 4),
            "wip_count": int(kpi.wip),
            "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
        })

    fab_kpi: dict = {}
    if kpi:
        fab_kpi = {
            "wip_total": int(kpi.wip),
            "utilization_avg_pct": round(float(kpi.utilization_avg) * 100, 1),
            "q_time_min": round(float(kpi.q_time_min), 1),
            "wait_ratio": round(float(kpi.wait_ratio), 4),
        }

    bottleneck_trend = [
        {
            "snapshot_time_min": k.snapshot_time,
            "q_time_min": round(float(k.q_time_min), 1),
            "utilization": round(float(k.utilization_avg), 4),
            "wip": int(k.wip),
        }
        for k in filter(None, [prev_kpi, kpi])
    ]

    affected_tgs = (alert.impact.affected_tgs or [])[:20]

    affected_processes = []
    for t in affected_tgs:
        tg_kpi = (kpi_map or {}).get(t)
        proc: dict = {"process": t, "status": "영향"}
        if tg_kpi:
            proc["utilization_pct"] = round(float(tg_kpi.utilization_avg) * 100, 1)
            proc["wait_ratio"] = round(float(tg_kpi.wait_ratio), 2)
            proc["wip"] = int(tg_kpi.wip)
        affected_processes.append(proc)

    sim_forecast = cause_report.sim_forecast if cause_report else None
    if sim_forecast:
        kd = sim_forecast.kpi_delta
        horizon_min = int(sim_forecast.t_future - sim_forecast.t0)
        forward_sim: dict = {
            "horizon_min": horizon_min,
            "results": [
                {
                    "toolgroup": tg,
                    "q_time_future": round(float(kd["q_time_min"].future), 1) if "q_time_min" in kd else None,
                    "wait_ratio_future": round(float(kd["wait_ratio"].future), 3) if "wait_ratio" in kd else None,
                    "wip_future": int(kd["wip"].future) if "wip" in kd else None,
                    "y_bottleneck": 1 if sim_forecast.gets_worse else 0,
                }
            ],
        }
    else:
        forward_sim = {}

    diffusion_analysis = {
        "is_bottleneck": True,
        "bottleneck_location": tg,
        "diffusion_path": [tg] + affected_tgs[:3],
        "affected_processes": affected_processes,
        "forward_simulation": forward_sim,
        "line_stop_expected_min": int(alert.impact.ct_increase_min),
        "risk_level": alert.severity.value,
    }

    cause_analysis: dict = {}
    shap_analysis: dict = {}
    feature_trend: list[dict] = []

    if cause_report and cause_report.shap_top:
        total = sum(abs(f.shap_value) for f in cause_report.shap_top) or 1.0
        shap_top = [
            {
                "rank": i + 1,
                "feature": f.feature,
                "kpi_value": round(float(f.kpi_value), 4),
                "shap_value": round(float(f.shap_value), 4),
                "contribution_pct": round(abs(f.shap_value) / total * 100, 1),
            }
            for i, f in enumerate(cause_report.shap_top)
        ]

        consensus_dict: dict | None = None
        if hasattr(cause_report, "consensus") and cause_report.consensus:
            c = cause_report.consensus
            consensus_dict = {
                "confidence_level": c.confidence_level,
                "agreed_features": c.agreed_features,
                "conflicted_features": c.conflicted_features,
                "g_star_confirmed": c.g_star_confirmed,
                "g_star_proba": c.g_star_proba,
                "summary": c.summary,
            }

        categories_list = [
            {
                "name": cat.name,
                "features": cat.features,
                "shap_share_pct": cat.shap_share_pct,
                "n_trend_significant": cat.n_trend_significant,
                "upstream_match": cat.upstream_match,
                "g_star_confirmed": cat.g_star_confirmed,
                "total_score": cat.total_score,
                "confidence": cat.confidence,
            }
            for cat in (cause_report.cause_categories or [])
        ]

        judgment_dict: dict | None = None
        if cause_report.judgment:
            j = cause_report.judgment
            judgment_dict = {
                "primary_category": j.primary_category,
                "primary_cause": j.primary_cause,
                "primary_confidence": j.primary_confidence,
                "primary_reasoning": j.primary_reasoning,
                "secondary_causes": j.secondary_causes,
                "dismissed": j.dismissed,
            }

        cause_analysis = {
            "shap_top": shap_top,
            "categories": categories_list,
            "judgment": judgment_dict,
            "summary": cause_report.cause_summary or "",
            "consensus": consensus_dict,
        }

        shap_analysis = {
            "model": "XGBoost",
            "snapshot_time": cause_report.snapshot_time,
            "toolgroup": tg,
            "top_features": [
                {
                    "feature": f.feature,
                    "value": round(float(f.kpi_value), 4),
                    "shap": round(float(f.shap_value), 4),
                    "share_abs_pct": round(abs(f.shap_value) / total * 100, 1),
                    "direction": "병목 쪽으로 기여(+)" if f.shap_value > 0 else "병목 완화(-)",
                }
                for f in cause_report.shap_top
            ],
        }

    if cause_report and cause_report.trend_top:
        n = max((len(t.values) for t in cause_report.trend_top), default=0)
        feat_map = {t.feature: t.values for t in cause_report.trend_top}
        for i in range(n):
            row: dict = {"time_label": f"T-{(n - 1 - i) * 60}분"}
            for feat, vals in feat_map.items():
                row[feat] = round(float(vals[i]), 4) if i < len(vals) else None
            feature_trend.append(row)

    return {
        "toolgroup": tg,
        "process_name": tg,
        "severity": alert.severity.value,
        "detected_at": detected_at,
        "bottleneck_info": bottleneck_info,
        "fab_kpi": fab_kpi,
        "bottleneck_trend": bottleneck_trend,
        "tool_status": [],
        "affected_lots_detail": [],
        "diffusion_analysis": diffusion_analysis,
        "cause_analysis": cause_analysis,
        "shap_analysis": shap_analysis,
        "feature_trend": feature_trend,
        "action_effects": compare_result.get("action_effects", []),
        "recommendation": compare_result.get("recommendation", {}),
        "approval_info": compare_result.get("approval_info", {}),
        "section_header": "",
        "section_review": "",
        "section_summary": "",
        "section_diffusion": "",
        "section_cause": "",
        "section_actions": "",
    }


# ── Pipeline 노드 ─────────────────────────────────────────────────────────────

def report_prepare(state: "PipelineState") -> dict:
    """Agent 6-1: 보고서 초안 생성 + 헤더/승인 섹션 (LLM 불필요)."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    compare_results = state.get("compare_results", [])
    if not compare_results:
        _log.info("[Report] compare_results 없음 — 스킵")
        return {"report_draft": []}

    alerts = state.get("alerts", [])
    kpi_snapshot = state.get("kpi_snapshot", [])
    prev_kpi_snapshot = state.get("prev_kpi_snapshot", [])
    cause_reports = state.get("cause_reports", [])

    alert_map = {a.toolgroup: a for a in alerts}
    kpi_map = {k.toolgroup: k for k in kpi_snapshot}
    prev_kpi_map = {k.toolgroup: k for k in prev_kpi_snapshot}
    cause_map = {r.toolgroup: r for r in cause_reports}

    report_draft: list[dict] = []
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for cr in compare_results:
        tg = cr["toolgroup"]
        alert = alert_map.get(tg)
        if alert is None:
            _log.warning(f"[Report] {tg}: alert 없음 — 스킵")
            continue

        item = _build_draft_item(
            compare_result=cr,
            alert=alert,
            kpi=kpi_map.get(tg),
            prev_kpi=prev_kpi_map.get(tg),
            cause_report=cause_map.get(tg),
            kpi_map=kpi_map,
            detected_at=detected_at,
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        sev = item["severity"]
        badge = {"Critical": "🚨 CRITICAL", "High": "🔴 HIGH", "Medium": "🟡 MEDIUM", "Low": "🟢 LOW"}.get(sev, sev)
        item["section_header"] = (
            f"# FAB 병목 대응 보고서\n\n"
            f"| 항목 | 내용 |\n"
            f"|------|------|\n"
            f"| 공정명 | `{item['process_name']}` |\n"
            f"| 심각도 | **{badge}** |\n"
            f"| 탐지시각 | {item['detected_at']} |\n"
            f"| 보고서 생성일시 | {now} |\n\n"
            f"---"
        )

        ai = item.get("approval_info") or {}
        if ai.get("status") == "반려":
            item["section_review"] = (
                f"## 검토 결과\n\n"
                f"| 항목 | 내용 |\n|------|------|\n"
                f"| 상태 | **🔴 반려** |\n"
                f"| 검토자 | {ai.get('approved_by') or '-'} ({ai.get('approved_role') or '-'}) |\n"
                f"| 반려일시 | {ai.get('approved_at', '-')} |\n"
                f"| 반려 사유 | {ai.get('rejection_reason', '-')} |\n\n"
                f"---"
            )
        else:
            item["section_review"] = (
                f"## 검토 결과\n\n"
                f"| 항목 | 내용 |\n|------|------|\n"
                f"| 상태 | **🟢 승인** |\n"
                f"| 승인자 | {ai.get('approved_by') or '-'} ({ai.get('approved_role') or '-'}) |\n"
                f"| 승인일시 | {ai.get('approved_at', '-')} |\n"
                f"| 의견 | {ai.get('comment') or '-'} |\n\n"
                f"---"
            )

        report_draft.append(item)

    _log.info(f"[Report] prepare 완료 — {len(report_draft)}개 공정")
    return {"report_draft": report_draft}


def report_summary(state: "PipelineState") -> dict:
    """Agent 6-2: 요약 섹션 LLM 작성."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = [dict(item) for item in state.get("report_draft", [])]
    for item in draft:
        try:
            item["section_summary"] = write_summary(item)["section_summary"]
            _log.info(f"[Report] {item['toolgroup']} summary 완료")
        except Exception as e:
            _log.error(f"[Report] {item['toolgroup']} summary 실패: {e}")
    return {"report_draft": draft}


def report_diffusion(state: "PipelineState") -> dict:
    """Agent 6-3: 확산 영향 섹션 LLM 작성."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = [dict(item) for item in state.get("report_draft", [])]
    for item in draft:
        try:
            item["section_diffusion"] = write_diffusion(item)["section_diffusion"]
            _log.info(f"[Report] {item['toolgroup']} diffusion 완료")
        except Exception as e:
            _log.error(f"[Report] {item['toolgroup']} diffusion 실패: {e}")
    return {"report_draft": draft}


def report_cause(state: "PipelineState") -> dict:
    """Agent 6-4: 원인 분석 섹션 LLM 작성."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = [dict(item) for item in state.get("report_draft", [])]
    for item in draft:
        try:
            item["section_cause"] = write_cause(item)["section_cause"]
            _log.info(f"[Report] {item['toolgroup']} cause 완료")
        except Exception as e:
            _log.error(f"[Report] {item['toolgroup']} cause 실패: {e}")
    return {"report_draft": draft}


def report_actions(state: "PipelineState") -> dict:
    """Agent 6-5: 대응안 섹션 LLM 작성."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = [dict(item) for item in state.get("report_draft", [])]
    for item in draft:
        try:
            item["section_actions"] = write_actions(item)["section_actions"]
            _log.info(f"[Report] {item['toolgroup']} actions 완료")
        except Exception as e:
            _log.error(f"[Report] {item['toolgroup']} actions 실패: {e}")
    return {"report_draft": draft}


def report_save(state: "PipelineState") -> dict:
    """Agent 6-6: 보고서 조립 + 저장 → report_results."""
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = state.get("report_draft", [])
    if not draft:
        return {"report_results": []}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_results: list[dict] = []

    for item in draft:
        tg = item["toolgroup"]
        sections = [
            item.get("section_header", ""),
            item.get("section_review", ""),
            item.get("section_summary", ""),
            item.get("section_diffusion", ""),
            item.get("section_cause", ""),
            item.get("section_actions", ""),
        ]
        final_report = "\n\n".join(s for s in sections if s)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"report_{tg}_{ts}"
        md_path = REPORTS_DIR / f"{base}.md"
        md_path.write_text(final_report, encoding="utf-8")

        json_path = REPORTS_DIR / f"{base}.json"
        json_path.write_text(
            json.dumps({
                "meta": {
                    "process_name": item.get("process_name", "-"),
                    "severity": item.get("severity", "-"),
                    "detected_at": item.get("detected_at", "-"),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "bottleneck_info": item.get("bottleneck_info", {}),
                "fab_kpi": item.get("fab_kpi", {}),
                "diffusion_analysis": item.get("diffusion_analysis", {}),
                "cause_analysis": item.get("cause_analysis", {}),
                "action_effects": item.get("action_effects", []),
                "recommendation": item.get("recommendation", {}),
                "approval_info": item.get("approval_info", {}),
                "full_markdown": final_report,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"보고서 저장 완료: {md_path}")
        report_results.append({
            "toolgroup": tg,
            "output_path": str(md_path),
            "json_output_path": str(json_path),
        })
        _log.info(f"[Report] {tg} 저장 완료 → {md_path}")

    _log.info(f"[Report] 완료 — {len(report_results)}개 보고서")

    # 검증 시뮬 데이터 정리 (보고서 저장 후)
    verify_results = state["verification_results"]
    scenario_ids = [r.get("whatif_scenario_id") for r in verify_results if r.get("whatif_scenario_id")]
    if scenario_ids:
        try:
            from agents.verification_agent.sim_executor import cleanup_verify_scenarios
            cleanup_verify_scenarios(scenario_ids)
        except Exception as e:
            _log.warning(f"[Report] VERIFY 데이터 정리 실패 (무시): {e}")

    return {"report_results": report_results}
