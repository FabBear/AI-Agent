"""기간 이슈 보고서 에이전트의 도구 — LLM이 골라 호출.

리포트 화면이 넘긴 기간 이력(history)·집계(historySummary)를 슬라이스한다. 모두 실데이터만
반환 → 환각 가드레일. 케이스 단건 상세 설명은 챗봇(get_case_detail)이 담당."""

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

from agents.agent_task.context import TaskContext
from agents.agent_task.helpers import pct, to_int
from agents.period_report_agent.rule_based import (
    _count_map,
    _derive_top_counts,
    _signed_metric,
    _top_count_text,
)

TOOL_LABELS = {
    "get_period_summary": "기간 집계 조회",
    "get_period_report_evidence": "기간 케이스 리포트 근거 조회",
    "get_repeat_bottlenecks": "반복 병목 TG 조회",
    "get_action_effectiveness": "승인 대응안 효과 조회",
    "get_period_cases": "기간 케이스 목록 조회",
    "get_case_effectiveness": "케이스 조치 효과(실측) 조회",
    "get_case_resolution": "케이스 해결 타임라인 조회",
    "get_case_cause": "케이스 원인·모델 신뢰도 조회",
}
KST = ZoneInfo("Asia/Seoul")


def _delta(value, suffix: str = "") -> str:
    if value is None:
        return "확인필요"
    try:
        return f"{float(value):+.2f}{suffix}"
    except (TypeError, ValueError):
        return "확인필요"


def _dt(value) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, datetime):
            dt = value
        else:
            return "-"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def _case_label(row) -> str:
    return f"{row['tg_name'] or row['tg_code']}({row['area_name']})"


def _items(ctx: TaskContext) -> list[dict[str, Any]]:
    history = ctx.live.get("history")
    items = history.get("items") if isinstance(history, dict) and isinstance(history.get("items"), list) else ctx.context.get("items")
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def _period_reports(ctx: TaskContext) -> list[dict[str, Any]]:
    reports = ctx.live.get("periodReports")
    return [it for it in reports if isinstance(it, dict)] if isinstance(reports, list) else []


def get_period_summary(ctx: TaskContext) -> str:
    hs = _count_map(ctx.live.get("historySummary"))
    items = _items(ctx)
    risk = _count_map(hs.get("riskCounts"))
    decision = _count_map(hs.get("decisionCounts"))
    status = _count_map(hs.get("statusCounts"))
    total = to_int(hs.get("totalCases"), len(items))
    if total == 0:
        return "선택 기간에 집계된 병목 대응 케이스가 없습니다."
    critical = to_int(risk.get("CRITICAL"), len([i for i in items if i.get("riskGrade") == "CRITICAL"]))
    high = to_int(risk.get("HIGH"), len([i for i in items if i.get("riskGrade") == "HIGH"]))
    approved = to_int(decision.get("APPROVED"), len([i for i in items if i.get("decision") == "APPROVED"]))
    rejected = to_int(decision.get("REJECTED"), len([i for i in items if i.get("decision") == "REJECTED"]))
    awaiting = to_int(status.get("AWAITING_HITL"))
    report_count = to_int(hs.get("reportCount"), len([i for i in items if i.get("hasReport")]))
    return (
        f"기간 케이스 {total}건 — Critical {critical}/High {high}, "
        f"승인 {approved}/반려 {rejected}/승인대기 {awaiting}, 리포트 {report_count}건."
    )


def get_period_report_evidence(ctx: TaskContext) -> str:
    reports = _period_reports(ctx)
    if not reports:
        return "선택 기간에 연결된 케이스 리포트(td_response_report)가 없습니다. 케이스 이력과 집계 중심으로 작성해야 합니다."
    lines = []
    for report in reports[:8]:
        tg = report.get("tgName") or "TG 미상"
        risk = report.get("riskGrade") or "위험도 미상"
        summary = str(report.get("summary") or report.get("rootCauseText") or "요약 없음").strip()
        lines.append(f"{tg}({risk}) report={report.get('reportId')}: {summary[:180]}")
    return f"기간 케이스 리포트 {len(reports)}건: " + " / ".join(lines)


def get_repeat_bottlenecks(ctx: TaskContext) -> str:
    hs = _count_map(ctx.live.get("historySummary"))
    top_tgs = hs.get("topToolGroups")
    top_areas = hs.get("topAreas")
    if not isinstance(top_tgs, list):
        top_tgs = _derive_top_counts(_items(ctx), "tgName", 8)
    if not isinstance(top_areas, list):
        top_areas = _derive_top_counts(_items(ctx), "areaName", 5)
    tg_txt = _top_count_text(top_tgs)
    area_txt = _top_count_text(top_areas)
    if tg_txt == "없음" and area_txt == "없음":
        return "반복 병목으로 집계된 TG/Area가 없습니다."
    return f"반복 병목 TG: {tg_txt} / 반복 Area: {area_txt}"


def get_action_effectiveness(ctx: TaskContext) -> str:
    hs = _count_map(ctx.live.get("historySummary"))
    decision = _count_map(hs.get("decisionCounts"))
    items = _items(ctx)
    approved = to_int(decision.get("APPROVED"), len([i for i in items if i.get("decision") == "APPROVED"]))
    avg_wait = _signed_metric(hs.get("avgEstAvgWaitDelta"))
    avg_delivery = _signed_metric(hs.get("avgEstDeliveryComplianceDelta"), "%p")
    return f"승인 대응안 {approved}건 평균 기대효과 — 대기시간 {avg_wait}, 납기준수율 {avg_delivery}."


def get_period_cases(ctx: TaskContext) -> str:
    items = _items(ctx)
    if not items:
        return "기간 내 케이스 목록이 없습니다."
    sample = []
    for i in items[:8]:
        sample.append(f"{i.get('tgName', '?')}({i.get('riskGrade', '?')}/{i.get('decision', '미결')})")
    return f"기간 케이스 {len(items)}건: " + ", ".join(sample)


async def get_case_effectiveness(ctx: TaskContext, case_ref: str = "") -> str:
    """승인된 대응안의 예상(est) vs 실측(actual) KPI 변화 — '조치가 실제 효과 있었나'."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 조회할 수 없습니다."
    row = await repo.case_detail(ctx.fab_id, str(case_ref or "").strip())
    if not row:
        return f"'{case_ref or '최근'}' 케이스를 찾지 못했습니다."
    label = _case_label(row)
    plans = await repo.case_plans(row["case_id"])
    selected = next((p for p in plans if p["selected"]), None)
    if selected is None:
        return f"{label} 케이스: 승인(선택)된 대응안이 없어 효과 검증 불가."
    est = f"대기 {_delta(selected['est_avg_wait_delta'])}일/처리량 {_delta(selected['est_throughput_delta'])}"
    if selected["validated_at"] is None and selected["actual_avg_wait_delta"] is None and selected["actual_throughput_delta"] is None:
        return f"{label} '{selected['plan_title']}' 승인 — 예상 {est}, 실측은 아직 미검증."
    act = f"대기 {_delta(selected['actual_avg_wait_delta'])}일/처리량 {_delta(selected['actual_throughput_delta'])}"
    return f"{label} '{selected['plan_title']}' — 예상({est}) vs 실측({act}), 검증 {_dt(selected['validated_at'])}."


async def get_case_resolution(ctx: TaskContext, case_ref: str = "") -> str:
    """케이스 감지→HITL 승인→해결 타임라인과 지연(SLA)."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 조회할 수 없습니다."
    row = await repo.case_detail(ctx.fab_id, str(case_ref or "").strip())
    if not row:
        return f"'{case_ref or '최근'}' 케이스를 찾지 못했습니다."
    label = _case_label(row)
    hitl = await repo.case_hitl(row["case_id"])
    latest = hitl[0] if hitl else None
    parts = [f"{label} 상태 {row['status']}", f"감지 {_dt(row['detected_at'])}"]
    if latest:
        parts.append(f"HITL {latest['decision']} {_dt(latest['decided_at'])}")
    parts.append(f"해결 {_dt(row['resolved_at'])}" if row["resolved_at"] else "미해결")
    return " · ".join(parts)


async def get_case_cause(ctx: TaskContext, case_ref: str = "") -> str:
    """대표 케이스의 원인 유형·리포트 원인요약 + ML 모델 신뢰도(accuracy/F1)."""
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 조회할 수 없습니다."
    row = await repo.case_detail(ctx.fab_id, str(case_ref or "").strip())
    if not row:
        return f"'{case_ref or '최근'}' 케이스를 찾지 못했습니다."
    parts = [f"{_case_label(row)} 원인유형 {row['bottleneck_cause_type'] or '미상'}, 위험 {pct(row['bottleneck_prob'])}/{row['risk_grade']}"]
    if row["root_cause_text"]:
        parts.append(f"원인요약: {str(row['root_cause_text'])[:200]}")
    if row["model_accuracy"] is not None:
        parts.append(f"모델 신뢰도 acc {_delta(row['model_accuracy']).lstrip('+')} / F1 {_delta(row['model_f1']).lstrip('+')}")
    return " — ".join(parts)


def build_report_tools(ctx: TaskContext) -> tuple[list, dict, dict]:
    """ctx에 바인딩된 (StructuredTool 목록, name→callable, name→라벨)을 반환."""

    def period_summary() -> str:
        return get_period_summary(ctx)

    def period_report_evidence() -> str:
        return get_period_report_evidence(ctx)

    def repeat_bottlenecks() -> str:
        return get_repeat_bottlenecks(ctx)

    def action_effectiveness() -> str:
        return get_action_effectiveness(ctx)

    def period_cases() -> str:
        return get_period_cases(ctx)

    async def case_effectiveness(case_ref: str = "") -> str:
        return await get_case_effectiveness(ctx, case_ref)

    async def case_resolution(case_ref: str = "") -> str:
        return await get_case_resolution(ctx, case_ref)

    async def case_cause(case_ref: str = "") -> str:
        return await get_case_cause(ctx, case_ref)

    tools = [
        StructuredTool.from_function(func=period_summary, name="get_period_summary",
            description="선택 기간의 전체 집계(케이스 수·위험등급·승인/반려/승인대기·리포트 수). 보고서의 출발점."),
        StructuredTool.from_function(func=period_report_evidence, name="get_period_report_evidence",
            description="선택 기간에 생성된 케이스 리포트(td_response_report)의 요약·원인 근거. 기간 이슈 보고서의 근거 섹션에 사용."),
        StructuredTool.from_function(func=repeat_bottlenecks, name="get_repeat_bottlenecks",
            description="기간 내 반복 등장한 TG/Area 순위. 구조적 병목 후보 식별에 사용."),
        StructuredTool.from_function(func=action_effectiveness, name="get_action_effectiveness",
            description="기간 전체 승인 대응안의 평균 기대효과(대기시간·납기준수율 델타). 거시 회고에 사용. 특정 케이스의 실측 효과는 get_case_effectiveness."),
        StructuredTool.from_function(func=period_cases, name="get_period_cases",
            description="기간 내 개별 케이스 목록(TG·위험등급·결정). 사례를 들거나 드릴다운할 케이스(TG명)를 고를 때 사용."),
        StructuredTool.from_function(coroutine=case_effectiveness, name="get_case_effectiveness",
            description="특정 케이스 승인 대응안의 예상(est) vs 실측(actual) KPI 변화 — '그 조치가 실제 효과 있었나'. case_ref=TG명/구역/케이스ID 접두(미지정 시 최근). DB 실조회."),
        StructuredTool.from_function(coroutine=case_resolution, name="get_case_resolution",
            description="특정 케이스의 감지→HITL 승인→해결 타임라인과 지연(SLA). case_ref=TG명/구역/케이스ID 접두. DB 실조회."),
        StructuredTool.from_function(coroutine=case_cause, name="get_case_cause",
            description="대표 케이스의 원인 유형·리포트 원인요약 + ML 모델 신뢰도(accuracy/F1). 사례 인용·근거 보강에 사용. case_ref=TG명/구역/케이스ID 접두. DB 실조회."),
    ]
    tool_fns = {
        "get_period_summary": period_summary,
        "get_period_report_evidence": period_report_evidence,
        "get_repeat_bottlenecks": repeat_bottlenecks,
        "get_action_effectiveness": action_effectiveness,
        "get_period_cases": period_cases,
        "get_case_effectiveness": case_effectiveness,
        "get_case_resolution": case_resolution,
        "get_case_cause": case_cause,
    }
    return tools, tool_fns, TOOL_LABELS
