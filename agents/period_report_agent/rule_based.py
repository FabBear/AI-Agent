"""기간 종합 보고서(REPORT_PERIOD_SUMMARY)의 룰베이스 baseline.

리포트 화면에서 사용자가 월간/기간을 선택하면, 그 기간의 병목 대응 이력을 모아
임원 요약형 운영 브리핑으로 집계한다."""

import re
from typing import Any

from agents.agent_task.helpers import backend_context, to_int
from agents.agent_task.schemas import (
    AgentArtifact,
    AgentReferences,
    AgentTaskAgentRequest,
    AgentTaskResult,
    EvidenceItem,
    Propagation,
    ResponseDirection,
)


def _count_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _top_count_text(items: Any, empty: str = "없음") -> str:
    if not isinstance(items, list) or not items:
        return empty
    parts: list[str] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tgName") or "-")
        count = to_int(item.get("count"))
        parts.append(f"{name} {count}건")
    return ", ".join(parts) if parts else empty


def _item_name(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value)
    return None


def _signed_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "확인 필요"
    try:
        return f"{float(value):+.2f}{suffix}"
    except (TypeError, ValueError):
        return "확인 필요"


def _month_title(req: AgentTaskAgentRequest, date_range: dict[str, Any]) -> str:
    month = str(req.params.get("month") or "")
    match = re.match(r"^(\d{4})-(\d{2})$", month)
    if match:
        return f"{match.group(1)}년 {int(match.group(2))}월 월간 운영 브리핑"
    from_text = str(date_range.get("from") or "선택 기간")
    to_text = str(date_range.get("to") or "")
    return f"{from_text}~{to_text} 월간 운영 브리핑"


def _derive_top_counts(items: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for item in items:
        name = str(item.get(key) or "")
        if not name:
            continue
        current = counts.setdefault(name, {"name": name, "areaName": item.get("areaName"), "count": 0})
        current["count"] = to_int(current["count"]) + 1
    rows = list(counts.values())
    rows.sort(key=lambda row: (-to_int(row.get("count")), str(row.get("name") or "")))
    return rows[:limit]


def _build_monthly_report_summary(req: AgentTaskAgentRequest, date_range: dict[str, Any], items: list[dict[str, Any]]) -> AgentTaskResult:
    backend = backend_context(req.context)
    history_summary = _count_map(backend.get("historySummary"))
    risk_counts = _count_map(history_summary.get("riskCounts"))
    decision_counts = _count_map(history_summary.get("decisionCounts"))
    top_tgs = history_summary.get("topToolGroups")
    top_areas = history_summary.get("topAreas")
    if not isinstance(top_tgs, list):
        top_tgs = _derive_top_counts(items, "tgName", 8)
    if not isinstance(top_areas, list):
        top_areas = _derive_top_counts(items, "areaName", 5)

    total = to_int(history_summary.get("totalCases"), len(items))
    critical = to_int(risk_counts.get("CRITICAL"), len([item for item in items if item.get("riskGrade") == "CRITICAL"]))
    high = to_int(risk_counts.get("HIGH"), len([item for item in items if item.get("riskGrade") == "HIGH"]))
    approved = to_int(decision_counts.get("APPROVED"), len([item for item in items if item.get("decision") == "APPROVED"]))
    rejected = to_int(decision_counts.get("REJECTED"), len([item for item in items if item.get("decision") == "REJECTED"]))
    report_count = to_int(history_summary.get("reportCount"), len([item for item in items if item.get("hasReport")]))
    top_tg_text = _top_count_text(top_tgs)
    avg_wait = _signed_metric(history_summary.get("avgEstAvgWaitDelta"))
    avg_delivery = _signed_metric(history_summary.get("avgEstDeliveryComplianceDelta"), "%p")
    title = _month_title(req, date_range)

    if total == 0:
        summary = f"{title} 대상 기간에는 집계된 병목 대응 케이스가 없습니다. 월간 회의에서는 데이터 수집 상태와 필터 조건을 먼저 확인하는 것이 좋습니다."
    else:
        summary = (
            f"{title}: 병목 대응 케이스 {total}건 중 Critical {critical}건, High {high}건이 확인됐고 "
            f"HITL 승인 {approved}건/반려 {rejected}건입니다. 반복 노출 TG는 {top_tg_text}이며, "
            f"평균 기대효과는 대기시간 {avg_wait}, 납기준수율 {avg_delivery}입니다."
        )

    return AgentTaskResult(
        summary=summary,
        evidence=[
            EvidenceItem(label="월간 케이스", value=f"{total}건", description="선택 월 detectedAt 기준 전체 병목 대응 케이스 수입니다."),
            EvidenceItem(label="Critical / High", value=f"{critical} / {high}건", description="임원 보고 우선순위가 높은 위험 등급입니다.", severity="critical" if critical else "info"),
            EvidenceItem(label="승인 / 반려", value=f"{approved} / {rejected}건", description="HITL 의사결정 결과입니다."),
            EvidenceItem(label="리포트 보유", value=f"{report_count}건", description="td_response_report가 생성된 케이스 수입니다."),
            EvidenceItem(label="반복 TG", value=top_tg_text, description="월간 반복 병목 후보로 우선 회고할 TG입니다.", severity="warning" if total else "info"),
            EvidenceItem(label="평균 기대효과", value=f"대기 {avg_wait}, 납기 {avg_delivery}", description="승인 대응안의 평균 예상 KPI 변화입니다."),
        ],
        propagation=Propagation(
            summary="반복 등장하는 Area/TG는 다음 월 운영 회의에서 구조적 병목 후보로 분리해 봐야 합니다.",
            affectedProcesses=[
                name for item in top_areas[:5]
                if isinstance(item, dict) and (name := _item_name(item, "name", "areaName"))
            ],
            affectedToolGroups=[
                name for item in top_tgs[:8]
                if isinstance(item, dict) and (name := _item_name(item, "name", "tgName"))
            ],
            horizon=f"{date_range.get('from', '월초')}~{date_range.get('to', '월말')}",
        ),
        responseDirections=[
            ResponseDirection(title="반복 병목 우선순위 확정", description="상위 반복 TG를 다음 월 개선 과제로 분리하고 담당 부서를 지정합니다."),
            ResponseDirection(title="승인 대응안 효과 점검", description="승인된 대응안의 예상 개선 폭과 재발 여부를 월간 회의에서 함께 확인합니다."),
            ResponseDirection(title="운영 기준 보정 검토", description="Critical 반복 구간은 threshold, 알림, HITL 검토 기준을 재점검합니다."),
        ],
        references=AgentReferences(
            caseIds=[str(item.get("caseId")) for item in items[:10] if isinstance(item, dict) and item.get("caseId")]
        ),
        followUpPrompts=["임원 보고용 5줄 요약으로 바꿔줘.", "반복 TG 원인을 자세히 정리해줘.", "승인 대응안 효과를 표로 정리해줘."],
        artifacts=[AgentArtifact(type="MONTHLY_REPORT", title=title, description="달력 월 기준 임원 요약형 운영 브리핑")],
    )


def build_period_report_baseline(req: AgentTaskAgentRequest) -> AgentTaskResult:
    date_range = req.params.get("dateRange") or req.context.get("dateRange") or {}
    from_text = date_range.get("from", "선택 기간") if isinstance(date_range, dict) else "선택 기간"
    to_text = date_range.get("to", "") if isinstance(date_range, dict) else ""
    history = backend_context(req.context).get("history")
    items = history.get("items") if isinstance(history, dict) and isinstance(history.get("items"), list) else req.context.get("items")
    items = items if isinstance(items, list) else []
    if str(req.params.get("intent") or "").lower() == "monthly" or str(req.params.get("periodType") or "").upper() == "MONTHLY":
        return _build_monthly_report_summary(req, date_range if isinstance(date_range, dict) else {}, items)
    critical = [item for item in items if isinstance(item, dict) and item.get("riskGrade") == "CRITICAL"]
    approved = [item for item in items if isinstance(item, dict) and item.get("decision") == "APPROVED"]
    top_targets = [str(item.get("tgName")) for item in items if isinstance(item, dict) and item.get("tgName")]
    return AgentTaskResult(
        summary=f"{from_text}~{to_text} 기간 리포트 {len(items)}건을 기준으로 반복 병목과 대응 이력을 요약했습니다.",
        evidence=[
            EvidenceItem(label="리포트 수", value=f"{len(items)}건", description="선택 기간 대응 이력입니다."),
            EvidenceItem(label="Critical", value=f"{len(critical)}건", description="우선 보고 대상입니다.", severity="critical"),
            EvidenceItem(label="승인 조치", value=f"{len(approved)}건", description="HITL에서 선택된 대응안입니다."),
        ],
        propagation=Propagation(
            summary="반복 등장하는 TG/Area는 구조적 병목 후보로 별도 회고가 필요합니다.",
            affectedToolGroups=top_targets[:8],
            horizon="선택 기간",
        ),
        responseDirections=[
            ResponseDirection(title="반복 병목 분리", description="동일 TG 반복 여부를 정책 점검 대상으로 분리합니다."),
            ResponseDirection(title="조치 효과 회고", description="승인 대응안별 KPI 변화와 재발 여부를 확인합니다."),
        ],
        references=AgentReferences(
            caseIds=[str(item.get("caseId")) for item in items[:10] if isinstance(item, dict) and item.get("caseId")]
        ),
        followUpPrompts=["반복 병목 TG를 순위로 정리해줘.", "운영 회의용 5문장 요약으로 바꿔줘."],
        artifacts=[AgentArtifact(type="PERIOD_REPORT", title="기간 요약 보고서", description="기간별 병목 요약")],
    )
