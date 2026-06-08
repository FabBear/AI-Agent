"""Agent 5(compare_results) + PipelineState → Agent 6(ReportState) 형식 변환."""

from __future__ import annotations

from datetime import datetime

from agents.schemas.alert import BottleneckAlert
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI


def _bottleneck_info(
    alert: BottleneckAlert,
    kpi: ToolGroupKPI | None,
) -> dict:
    info: dict = {
        "tool_group": alert.toolgroup,
        "risk_score": round(float(alert.composite_score) * 100, 1),
        "delayed_orders": int(alert.impact.at_risk_lots),
    }
    if kpi:
        info.update({
            "avg_queue_time_min": round(float(kpi.q_time_min), 1),
            "peak_q_time_min": round(float(kpi.max_avg_q_time), 1),
            "utilization_pct": round(float(kpi.utilization_avg) * 100, 1),
            "load_ratio": round(float(kpi.wait_ratio), 4),
            "wip_count": int(kpi.wip),
            "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
        })
    return info


def _fab_kpi(kpi: ToolGroupKPI | None) -> dict:
    if kpi is None:
        return {}
    return {
        "wip_total": int(kpi.wip),
        "utilization_avg_pct": round(float(kpi.utilization_avg) * 100, 1),
        "q_time_min": round(float(kpi.q_time_min), 1),
        "wait_ratio": round(float(kpi.wait_ratio), 4),
        "available_tool_ratio": round(float(kpi.available_tool_ratio), 4),
    }


def _bottleneck_trend(
    kpi: ToolGroupKPI | None,
    prev_kpi: ToolGroupKPI | None,
) -> list[dict]:
    rows = []
    for k in filter(None, [prev_kpi, kpi]):
        rows.append({
            "snapshot_time_min": k.snapshot_time,
            "q_time_min": round(float(k.q_time_min), 1),
            "utilization": round(float(k.utilization_avg), 4),
            "wip": int(k.wip),
            "wait_ratio": round(float(k.wait_ratio), 4),
        })
    return rows


def _diffusion_analysis(alert: BottleneckAlert) -> dict:
    affected_tgs = alert.impact.affected_tgs or []
    path = [f"{alert.toolgroup} → {tg}" for tg in affected_tgs[:3]]
    affected = [{"process": tg, "status": "영향"} for tg in affected_tgs]
    return {
        "is_bottleneck": True,
        "bottleneck_location": alert.toolgroup,
        "diffusion_path": path,
        "affected_processes": affected,
        "forward_simulation": {},
        "line_stop_expected_min": int(alert.impact.ct_increase_min),
        "risk_level": alert.severity.value,
        "capacity_stress_score": round(float(alert.impact.capacity_stress_score), 4),
        "impact_score": round(float(alert.impact.impact_score), 4),
    }


def _cause_analysis(cause_report: CauseReport | None) -> list[dict]:
    if not cause_report:
        return []
    total = sum(abs(f.shap_value) for f in cause_report.shap_top) or 1.0
    rows = []
    for i, f in enumerate(cause_report.shap_top):
        rows.append({
            "rank": i + 1,
            "cause": f.feature,
            "contribution_pct": round(abs(f.shap_value) / total * 100, 1),
            "recommended_action": "",
            "similar_case": "없음",
        })
    if cause_report.cause_summary:
        rows.append({"summary": cause_report.cause_summary})
    return rows


def _shap_analysis(cause_report: CauseReport | None) -> dict:
    if not cause_report or not cause_report.shap_top:
        return {}
    total = sum(abs(f.shap_value) for f in cause_report.shap_top) or 1.0
    features = []
    for f in cause_report.shap_top:
        features.append({
            "feature": f.feature,
            "value": round(float(f.kpi_value), 4),
            "shap": round(float(f.shap_value), 4),
            "share_abs_pct": round(abs(f.shap_value) / total * 100, 1),
            "direction": "병목 쪽으로 기여(+)" if f.shap_value > 0 else "병목 완화(-)",
        })
    return {
        "model": "XGBoost",
        "snapshot_time": cause_report.snapshot_time,
        "toolgroup": cause_report.toolgroup,
        "top_features": features,
    }


def _feature_trend(cause_report: CauseReport | None) -> list[dict]:
    if not cause_report or not cause_report.trend_top:
        return []
    n = max(len(t.values) for t in cause_report.trend_top) if cause_report.trend_top else 0
    if n == 0:
        return []
    feature_map: dict[str, list[float]] = {t.feature: t.values for t in cause_report.trend_top}
    rows = []
    for i in range(n):
        row: dict = {"time_label": f"T-{(n - 1 - i) * 60}분"}
        for feat, vals in feature_map.items():
            row[feat] = round(float(vals[i]), 4) if i < len(vals) else None
        rows.append(row)
    return rows


def build_report_input(
    compare_result: dict,
    alert: BottleneckAlert,
    kpi: ToolGroupKPI | None,
    prev_kpi: ToolGroupKPI | None,
    cause_report: CauseReport | None,
) -> dict:
    """
    compare_results 항목 + PipelineState 데이터 → ReportState 입력 dict 변환.

    Args:
        compare_result: compare_results 리스트의 툴그룹 단위 항목
        alert:          해당 툴그룹의 BottleneckAlert
        kpi:            현재 시각 ToolGroupKPI
        prev_kpi:       이전 시각 ToolGroupKPI (없으면 None)
        cause_report:   해당 툴그룹의 CauseReport (없으면 None)
    """
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "process_name": compare_result["toolgroup"],
        "severity": alert.severity.value,
        "detected_at": detected_at,
        "bottleneck_info": _bottleneck_info(alert, kpi),
        "fab_kpi": _fab_kpi(kpi),
        "bottleneck_trend": _bottleneck_trend(kpi, prev_kpi),
        "tool_status": [],
        "affected_lots_detail": [],
        "diffusion_analysis": _diffusion_analysis(alert),
        "cause_analysis": _cause_analysis(cause_report),
        "shap_analysis": _shap_analysis(cause_report),
        "feature_trend": _feature_trend(cause_report),
        "action_effects": compare_result.get("action_effects", []),
        "recommendation": compare_result.get("recommendation", {}),
        "approval_info": compare_result.get("approval_info", {}),
    }
