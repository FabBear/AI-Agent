#!/usr/bin/env python3
"""최종보고서 생성 Agent (Agent 06) — PipelineState 통합 노드."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.state import PipelineState

from agents.report_agent.adapter import (
    has_actions,
    normalize_compare_result,
    sanitize_for_json,
)
from agents.report_agent.builder import build_report_v2
from agents.report_agent.writer import narrate_with_reflection, render_sections

_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = _ROOT / "report_agent_out"


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _build_draft_item(compare_result: dict, alert, kpi, prev_kpi, cause_report, kpi_map: dict | None = None, detected_at: str = "") -> dict:
    """compare_result + PipelineState 데이터 → report_draft 초기 항목.

    compare_result는 Phase 1/Phase 2 경로에 따라 shape이 다르다.
    adapter.normalize_compare_result로 한 번 정규화한 뒤 사용한다.
    """
    tg = compare_result["toolgroup"]
    normalized = normalize_compare_result(compare_result)
    action_options = normalized["action_options"]
    recommendation = normalized["recommendation"]
    decision_meta  = normalized["decision_meta"]
    approval_info  = normalized["approval_info"]
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
    trend_stats: list[dict] = []

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
                "upstream_aligns": c.upstream_aligns,
                "sim_aligns": c.sim_aligns,
                "g_star_confirmed": c.g_star_confirmed,
                "g_star_proba": c.g_star_proba,
                "g_star_upstream_confirmed": c.g_star_upstream_confirmed,
                "g_star_sig_kpis": [
                    {
                        "kpi": k.kpi,
                        "delta_mean": round(float(k.delta_mean), 4),
                        "t_p_adj": round(float(k.t_p_adj), 4),
                        "significant": k.significant,
                    }
                    for k in (c.g_star_sig_kpis or [])
                ],
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
                "dismissed_reason": j.dismissed_reason,
            }

        evidence_list: list[dict] = [
            {
                "feature": e.feature,
                "votes": e.votes,
                "confidence": e.confidence,
                "shap_value": e.shap_value,
                "trend_slope": e.trend_slope,
                "trend_r2": e.trend_r2,
                "trend_significant": e.trend_significant,
                "upstream_match": e.upstream_match,
                "g_star_p_value": e.g_star_p_value,
                "g_star_significant": e.g_star_significant,
            }
            for e in (cause_report.evidence_bundle or [])
        ]

        cause_analysis = {
            "shap_top": shap_top,
            "categories": categories_list,
            "summary": cause_report.cause_summary or "",
            "consensus": consensus_dict,
            "judgment": judgment_dict,
            "evidence_bundle": evidence_list,
            "upstream_suspects": cause_report.upstream_suspects or [],
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

    actions_available = has_actions(normalized)

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
        "action_effects": action_options,
        "recommendation": recommendation,
        "decision_info": decision_meta,
        "approval_info": approval_info,
        # 업스트림에 승인 가능한 후보가 없으면 writer가 '대응안' 섹션을
        # 빈 표/None 으로 만드는 대신 안내 문구로 처리하도록 신호.
        "data_sufficiency": {
            "actions_available": actions_available,
        },
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

        # JSON v2 객체 빌드 — 결정론적(코드만, LLM 없음). sections는
        # 이후 LLM 노드들이 채운 마크다운 문자열로 report_save에서 합쳐 넣는다.
        normalized = normalize_compare_result(cr)
        snapshot_time = kpi_map[tg].snapshot_time if tg in kpi_map else None
        report_v2 = build_report_v2(
            tg=tg,
            alert=alert,
            kpi=kpi_map.get(tg),
            prev_kpi=prev_kpi_map.get(tg),
            cause_report=cause_map.get(tg),
            kpi_map=kpi_map,
            normalized_compare=normalized,
            detected_at=detected_at,
            snapshot_time=snapshot_time,
        )
        # ReportV2 객체 자체를 부착 — 직렬화는 저장 시점에 한 번만.
        item["report_v2"] = report_v2

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
    """Agent 6-2: **단일 LLM 호출**로 4개 섹션 narrative 동시 생성 +
    결정론적 표와 합쳐 4개 섹션 모두 채운다.

    PR #3 변경: 이전에는 summary/diffusion/cause/actions가 각각 LLM을
    호출(총 4회)했다. 지금은 여기서 한 번만 호출하고 모든 섹션을 채우며,
    이후 report_diffusion/report_cause/report_actions 노드는 passthrough이다.
    LangGraph 노드 이름은 보존(pipeline.py 무변경)하기 위해 함수 자체는 유지.
    """
    from agents.logger import get_logger
    _log = get_logger(__name__)

    draft = [dict(item) for item in state.get("report_draft", [])]
    for item in draft:
        rv2 = item.get("report_v2")
        if rv2 is None:
            _log.warning(f"[Report] {item.get('toolgroup', '?')} report_v2 없음 — 스킵")
            continue
        try:
            historical = state.get("historical_context")
            narration = narrate_with_reflection(rv2, historical=historical)
            sections = render_sections(rv2, narration)
            item["section_summary"]   = sections["summary"]
            item["section_diffusion"] = sections["diffusion"]
            item["section_cause"]     = sections["cause"]
            item["section_actions"]   = sections["actions"]
            _log.info(f"[Report] {item['toolgroup']} narrate+render 완료 (단일 LLM 호출)")
        except Exception as e:
            # 결정론적 fallback: LLM이 죽어도 표는 코드가 만들었으므로
            # 빈 narration으로 렌더링하면 표만 있는 보고서가 나온다.
            _log.error(f"[Report] {item['toolgroup']} narrate 실패: {e}")
            try:
                from agents.report_agent.writer import _fallback_narration
                fb = _fallback_narration(rv2)
                sections = render_sections(rv2, fb)
                item["section_summary"]   = sections["summary"]
                item["section_diffusion"] = sections["diffusion"]
                item["section_cause"]     = sections["cause"]
                item["section_actions"]   = sections["actions"]
            except Exception as e2:
                _log.error(f"[Report] {item['toolgroup']} fallback 렌더도 실패: {e2}")
    return {"report_draft": draft}


def _report_passthrough(state: "PipelineState") -> dict:
    """report_summary가 이미 모든 섹션을 채웠으므로 그대로 통과시킨다.

    LangGraph 그래프의 4-step LLM 체인을 1-step LLM + 3-step no-op으로
    바꾸기 위한 어댑터. pipeline.py(외부 파일)를 안 건드리는 게 목적.
    """
    return {"report_draft": list(state.get("report_draft", []))}


# pipeline.py가 import하는 노드 이름을 보존한다. 실제 구현은 passthrough.
report_diffusion = _report_passthrough
report_cause     = _report_passthrough
report_actions   = _report_passthrough


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
        # ── JSON v2 (1등 시민) ───────────────────────────────────────────
        # report_prepare에서 부착한 ReportV2 객체에 마크다운 섹션을
        # 채워 넣고 dict로 직렬화한다.
        report_v2 = item.get("report_v2")
        if report_v2 is not None:
            # 마크다운 섹션 주입 (LLM 노드들이 채운 결과)
            report_v2.sections.header    = item.get("section_header", "")
            report_v2.sections.review    = item.get("section_review", "")
            report_v2.sections.summary   = item.get("section_summary", "")
            report_v2.sections.diffusion = item.get("section_diffusion", "")
            report_v2.sections.cause     = item.get("section_cause", "")
            report_v2.sections.actions   = item.get("section_actions", "")
            report_v2.rendered.markdown  = final_report

            v2_payload = report_v2.model_dump(mode="json")
        else:
            v2_payload = None

        # ── Legacy 키 (백엔드 호환 유지용) ───────────────────────────────
        # PR #2 동안은 함께 출력 — backend가 v2로 마이그레이션하면 제거 예정.
        # 업스트림(특히 compare_agent의 paired_t_p)이 NaN을 흘려보낸다.
        # 표준 JSON에는 NaN/Inf 토큰이 없어 Spring Jackson/JSON.parse가 깨지므로
        # 직렬화 직전에 None으로 sanitize하고 allow_nan=False로 재발을 막는다.
        payload = sanitize_for_json({
            "schema_version": "report/1.0",
            # v2 — 새 1등 시민
            **(v2_payload or {}),
            # legacy — backend 마이그레이션 끝나면 제거
            "legacy": {
                "bottleneck_info":   item.get("bottleneck_info", {}),
                "fab_kpi":           item.get("fab_kpi", {}),
                "diffusion_analysis": item.get("diffusion_analysis", {}),
                "cause_analysis":    item.get("cause_analysis", {}),
                "action_effects":    item.get("action_effects", []),
                "recommendation":    item.get("recommendation", {}),
                "decision_info":     item.get("decision_info", {}),
                "approval_info":     item.get("approval_info", {}),
                "data_sufficiency":  item.get("data_sufficiency", {}),
                "full_markdown":     final_report,
            },
        })
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
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
    verify_results = state.get("verification_results", [])
    scenario_ids = [r.get("whatif_scenario_id") for r in verify_results if r.get("whatif_scenario_id")]
    if scenario_ids:
        try:
            from agents.verification_agent.sim_executor import cleanup_verify_scenarios
            cleanup_verify_scenarios(scenario_ids)
        except Exception as e:
            _log.warning(f"[Report] VERIFY 데이터 정리 실패 (무시): {e}")

    return {"report_results": report_results}
