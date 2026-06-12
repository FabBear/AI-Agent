"""공장 현황 브리핑 에이전트의 도구 — LLM이 골라 호출.

대부분 라이브 MES 스냅샷(req.context)을 슬라이스하고, 추세 1종만 TimescaleDB를 실조회한다
(쿼리는 챗봇과 동일한 ChatQueryRepository 재활용). 도구는 실데이터만 반환 → 환각 가드레일."""

from langchain_core.tools import StructuredTool

from agents.agent_task.context import TaskContext
from agents.agent_task.helpers import backend_list, num, pct, to_int
from agents.fab_briefing_agent.rule_based import (
    _down_by_tg,
    _fab_summary,
    _flatten_toolgroups,
    _process_summaries,
)

TOOL_LABELS = {
    "get_fab_overview": "전체 현황 조회",
    "get_down_tools_by_tg": "비가동 설비 TG별 조회",
    "get_wip_distribution": "구역별 WIP 분포 조회",
    "get_flow_risk": "흐름 위험(Q-time·셋업) 점검",
    "get_recent_cases": "최근 병목 케이스 이력 조회",
    "get_tg_trend": "TG 추세 조회",
    "get_bottleneck_risk_ranking": "ML 위험 TG 순위 조회",
    "get_tg_equipment_health": "TG 설비 health 조회",
    "get_area_risk_rollup": "구역 위험 롤업 조회",
}


def get_fab_overview(ctx: TaskContext) -> str:
    fab = _fab_summary(ctx.context)
    tss = fab.get("toolStatusSummary") if isinstance(fab.get("toolStatusSummary"), dict) else {}
    run, idle, setup, down = to_int(tss.get("RUN")), to_int(tss.get("IDLE")), to_int(tss.get("SETUP")), to_int(tss.get("DOWN"))
    total = run + idle + setup + down
    if not fab and not _flatten_toolgroups(ctx.context):
        return "라이브 MES 스냅샷이 없어 전체 현황을 집계할 수 없습니다."
    return (
        f"{fab.get('fabName') or 'FAB'} 전체 가동률 {pct(fab.get('utilizationRate'))}, "
        f"진행 WIP {to_int(fab.get('wipCount'))} Lot, "
        f"설비 {total}대(RUN {run}/IDLE {idle}/SETUP {setup}/DOWN {down}), "
        f"가용률 {pct(fab.get('avgAvailableToolRatio'))}."
    )


def get_down_tools_by_tg(ctx: TaskContext) -> str:
    tgs = _flatten_toolgroups(ctx.context)
    down = _down_by_tg(ctx.context, tgs)
    if not down:
        return "현재 비가동(DOWN)으로 집계된 설비가 없습니다."
    parts = [f"{d['tgName']}({d.get('areaName') or '구역?'}) {d['count']}대" for d in down[:6]]
    return f"비가동 설비 TG별({sum(d['count'] for d in down)}대): " + ", ".join(parts)


def get_wip_distribution(ctx: TaskContext) -> str:
    procs = _process_summaries(ctx.context)
    active = sorted(
        [p for p in procs if to_int(p.get("wipCount")) > 0],
        key=lambda p: to_int(p.get("wipCount")),
        reverse=True,
    )
    if not active:
        return "구역별 WIP가 거의 없어 라인 전반이 한산합니다."
    parts = [f"{p.get('areaName') or p.get('areaCode')} {to_int(p.get('wipCount'))}Lot(가동 {pct(p.get('avgUtilizationRate'))})" for p in active[:6]]
    return "구역별 WIP 분포: " + ", ".join(parts)


def get_flow_risk(ctx: TaskContext) -> str:
    tgs = _flatten_toolgroups(ctx.context)
    qrisk = sorted([t for t in tgs if num(t.get("avgQtimeMin")) > 0], key=lambda t: num(t.get("avgQtimeMin")), reverse=True)[:3]
    setup = sorted([t for t in tgs if num(t.get("setupRatio")) >= 0.2], key=lambda t: num(t.get("setupRatio")), reverse=True)[:3]
    parts: list[str] = []
    if qrisk:
        parts.append("Q-time 위험: " + ", ".join(f"{t.get('tgName') or t.get('tgCode')} {to_int(t.get('avgQtimeMin'))}분" for t in qrisk))
    if setup:
        parts.append("셋업 과다: " + ", ".join(f"{t.get('tgName') or t.get('tgCode')} {pct(t.get('setupRatio'))}" for t in setup))
    return " / ".join(parts) if parts else "Q-time·셋업 기준 두드러진 흐름 위험은 없습니다."


def get_recent_cases(ctx: TaskContext) -> str:
    cases = ctx.live.get("recentBottleneckCases")
    if not isinstance(cases, list) or not cases:
        return "최근 감지된 병목 케이스 이력이 없습니다."
    parts: list[str] = []
    for c in cases[:5]:
        if not isinstance(c, dict):
            continue
        prob = c.get("prob")
        prob_txt = f", 확률 {float(prob):.0%}" if prob not in (None, "") else ""
        parts.append(f"{c.get('tgCode', '?')} {c.get('riskGrade', '?')}{prob_txt} ({c.get('status', '?')})")
    return "최근 병목 케이스: " + "; ".join(parts)


def get_bottleneck_risk_ranking(ctx: TaskContext, limit: int = 5) -> str:
    """ML이 예측한 병목 위험(bottleneckProb) 상위 TG. 3D 뷰가 색칠하는 바로 그 위험."""
    tgs = _flatten_toolgroups(ctx.context)
    ranked = sorted([t for t in tgs if num(t.get("bottleneckProb")) > 0], key=lambda t: num(t.get("bottleneckProb")), reverse=True)
    if not ranked:
        return "ML 병목 위험으로 집계된 TG가 없습니다(스냅샷에 위험 신호 없음)."
    limit = max(1, min(10, to_int(limit, 5)))
    parts = [
        f"{t.get('tgName') or t.get('tgCode')}({t.get('areaName') or '구역?'}) 위험 {pct(t.get('bottleneckProb'))}/{t.get('riskGrade') or '-'}, 가동률 {pct(t.get('utilizationRate'))}, WIP {to_int(t.get('wipCount'))}"
        for t in ranked[:limit]
    ]
    return "ML 병목위험 상위 TG: " + "; ".join(parts)


def get_tg_equipment_health(ctx: TaskContext, tg: str = "") -> str:
    """특정 TG(미지정 시 최고위험 TG) 안 개별 설비의 status·가동률·셋업·다운 + TG OEE."""
    tgs = _flatten_toolgroups(ctx.context)
    if not tgs:
        return "TG 데이터가 없어 설비 health를 조회할 수 없습니다."
    key = (tg or "").strip().lower()
    target = None
    if key:
        for t in tgs:
            if any(key in str(t.get(f) or "").lower() for f in ("tgName", "tgCode", "areaName")):
                target = t
                break
    if target is None:
        target = max(tgs, key=lambda t: num(t.get("bottleneckProb")), default=None)
    if target is None:
        return "대상 TG를 찾지 못했습니다."
    tgid = str(target.get("tgId"))
    tools = [t for t in backend_list(ctx.context, "mesCurrent", "tools") if str(t.get("tgId")) == tgid]
    head = f"{target.get('tgName') or target.get('tgCode')} 설비 health"
    if target.get("oeeEstimate") not in (None, ""):
        head += f" (TG OEE {pct(target.get('oeeEstimate'))})"
    if not tools:
        return head + ": 개별 설비 데이터가 없습니다."
    order = {"DOWN": 0, "SETUP": 1, "RUN": 2, "IDLE": 3}
    tools.sort(key=lambda t: (order.get(str(t.get("status")), 4), -num(t.get("downRatio"))))
    parts = [f"{t.get('toolCode')}[{t.get('status')}] 가동 {pct(t.get('utilizationRate'))}, 셋업 {pct(t.get('setupRatio'))}, 다운 {pct(t.get('downRatio'))}" for t in tools[:8]]
    return head + " — " + "; ".join(parts)


def get_area_risk_rollup(ctx: TaskContext) -> str:
    """구역(Area)별 위험 롤업 — riskGrade·Critical/High 수·병목 TG 수·평균 Q-time."""
    procs = _process_summaries(ctx.context)
    if not procs:
        return "구역별 위험 롤업 데이터가 없습니다."

    def rc(p: dict) -> dict:
        return p.get("riskCounts") if isinstance(p.get("riskCounts"), dict) else {}

    ranked = sorted(procs, key=lambda p: (to_int(rc(p).get("CRITICAL")), to_int(rc(p).get("HIGH")), to_int(p.get("bottleneckToolGroupCount"))), reverse=True)
    parts = []
    for p in ranked[:5]:
        crit, high, bn = to_int(rc(p).get("CRITICAL")), to_int(rc(p).get("HIGH")), to_int(p.get("bottleneckToolGroupCount"))
        if crit == 0 and high == 0 and bn == 0:
            continue
        parts.append(f"{p.get('areaName') or p.get('areaCode')}: {p.get('riskGrade') or '-'} (Critical {crit}/High {high}, 병목TG {bn}, 평균Q {to_int(p.get('avgQtimeMin'))}분)")
    return ("구역 위험 롤업: " + "; ".join(parts)) if parts else "현재 위험(Critical/High) 구역이 없습니다."


async def get_tg_trend(ctx: TaskContext, tg: str = "", hours: int = 6) -> str:
    repo = await ctx.repo()
    if repo is None:
        return "FAB 정보가 없어 추세를 조회할 수 없습니다."
    hours = max(1, min(72, to_int(hours, 6)))
    bucket = 5 if hours <= 2 else (15 if hours <= 8 else 60)
    rows = await repo.kpi_trend(ctx.fab_id, str(tg or ""), hours, bucket, "tg")
    if not rows:
        return f"'{tg or '전체'}' 추세 데이터가 없습니다(과거 메트릭 부재 가능)."
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    lines = [f"[최근 {hours}시간 추세]"]
    for label, series in list(by_label.items())[:4]:
        first, last = series[0], series[-1]
        d_wip = float(last["wip"] or 0) - float(first["wip"] or 0)
        lines.append(
            f"- {label}: 가동률 {pct(last['util'])}, WIP {float(last['wip'] or 0):.0f}({d_wip:+.0f}), Q-time {float(last['qtime'] or 0):.0f}분"
        )
    return "\n".join(lines)


def build_briefing_tools(ctx: TaskContext) -> tuple[list, dict, dict]:
    """ctx에 바인딩된 (StructuredTool 목록, name→callable, name→라벨)을 반환."""

    def fab_overview() -> str:
        return get_fab_overview(ctx)

    def down_tools_by_tg() -> str:
        return get_down_tools_by_tg(ctx)

    def wip_distribution() -> str:
        return get_wip_distribution(ctx)

    def flow_risk() -> str:
        return get_flow_risk(ctx)

    def recent_cases() -> str:
        return get_recent_cases(ctx)

    def bottleneck_risk_ranking(limit: int = 5) -> str:
        return get_bottleneck_risk_ranking(ctx, limit)

    def tg_equipment_health(tg: str = "") -> str:
        return get_tg_equipment_health(ctx, tg)

    def area_risk_rollup() -> str:
        return get_area_risk_rollup(ctx)

    async def tg_trend(tg: str = "", hours: int = 6) -> str:
        return await get_tg_trend(ctx, tg, hours)

    tools = [
        StructuredTool.from_function(func=fab_overview, name="get_fab_overview",
            description="공장 전체 현황(가동률·진행 WIP·설비 RUN/IDLE/SETUP/DOWN·가용률). 브리핑의 출발점으로 먼저 호출."),
        StructuredTool.from_function(func=down_tools_by_tg, name="get_down_tools_by_tg",
            description="비가동(DOWN) 설비를 소속 툴그룹(TG)별로 분해. 전체 현황에서 비가동이 보이면 어디 몰렸는지 확인."),
        StructuredTool.from_function(func=wip_distribution, name="get_wip_distribution",
            description="구역(Area)별 WIP 분포와 가동률(라인 밸런스). WIP 쏠림/한산 구역 파악에 사용."),
        StructuredTool.from_function(func=flow_risk, name="get_flow_risk",
            description="Q-time이 쌓이는 TG와 셋업 비중이 과한 TG(흐름 위험). 대기/셋업 thrash 점검에 사용."),
        StructuredTool.from_function(func=recent_cases, name="get_recent_cases",
            description="최근 ML이 감지한 병목 케이스 이력(위험등급·확률·상태). 브리핑에 사실로 1줄 인용."),
        StructuredTool.from_function(func=bottleneck_risk_ranking, name="get_bottleneck_risk_ranking",
            description="ML이 예측한 병목 위험(bottleneckProb) 상위 TG 순위 — 3D 뷰가 색칠하는 바로 그 위험. '지금 가장 위험한 TG'를 짚을 때 사용. limit=상위 N(기본 5)."),
        StructuredTool.from_function(func=tg_equipment_health, name="get_tg_equipment_health",
            description="특정 TG 안 개별 설비의 status·가동률·셋업·다운율 + TG OEE. 위험/비가동 TG를 설비 레벨로 드릴다운할 때 사용. tg=TG코드/명(미지정 시 최고위험 TG)."),
        StructuredTool.from_function(func=area_risk_rollup, name="get_area_risk_rollup",
            description="구역(Area)별 위험 롤업 — riskGrade·Critical/High 케이스 수·병목 TG 수·평균 Q-time. 어느 구역이 위험한지 한눈에 볼 때 사용."),
        StructuredTool.from_function(coroutine=tg_trend, name="get_tg_trend",
            description="특정 TG/구역의 가동률·WIP·Q-time 시계열 추세(TimescaleDB 실조회). 비가동·WIP 쏠림이 보인 TG가 악화 중인지 확인. tg=TG코드/구역명, hours=조회시간(기본 6)."),
    ]
    tool_fns = {
        "get_fab_overview": fab_overview,
        "get_down_tools_by_tg": down_tools_by_tg,
        "get_wip_distribution": wip_distribution,
        "get_flow_risk": flow_risk,
        "get_recent_cases": recent_cases,
        "get_bottleneck_risk_ranking": bottleneck_risk_ranking,
        "get_tg_equipment_health": tg_equipment_health,
        "get_area_risk_rollup": area_risk_rollup,
        "get_tg_trend": tg_trend,
    }
    return tools, tool_fns, TOOL_LABELS
