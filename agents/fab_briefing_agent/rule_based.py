"""공장 현황 브리핑(FAB_SNAPSHOT_BRIEFING)의 룰베이스 baseline.

3D Fab View와 동일한 실시간 MES 스냅샷(mesCurrent)만 근거로, 현재 라인 상태를
결정론적으로 집계한다. 병목 '예측'이 아니라 '현황 요약'이다."""

from typing import Any

from agents.agent_task.helpers import (
    backend_context,
    backend_list,
    name_of,
    num,
    pct,
    to_int,
)
from agents.agent_task.schemas import (
    AgentArtifact,
    AgentReferences,
    AgentTaskAgentRequest,
    AgentTaskResult,
    EvidenceItem,
    Propagation,
    ResponseDirection,
    WatchToolGroup,
)


def _flatten_toolgroups(context: dict[str, Any]) -> list[dict[str, Any]]:
    mes_tgs = backend_list(context, "mesCurrent", "toolGroups")
    if mes_tgs:
        return mes_tgs
    equipment_tgs = backend_list(context, "equipmentCurrent", "toolGroups")
    if equipment_tgs:
        return equipment_tgs
    if isinstance(context.get("toolGroups"), list):
        return [item for item in context["toolGroups"] if isinstance(item, dict)]
    result: list[dict[str, Any]] = []
    for area in context.get("areas", []):
        if isinstance(area, dict):
            result.extend(item for item in area.get("toolGroups", []) if isinstance(item, dict))
    return result


def _fab_summary(context: dict[str, Any]) -> dict[str, Any]:
    mes = backend_context(context).get("mesCurrent")
    fab = mes.get("fab") if isinstance(mes, dict) else None
    return fab if isinstance(fab, dict) else {}


def _process_summaries(context: dict[str, Any]) -> list[dict[str, Any]]:
    mes = backend_context(context).get("mesCurrent")
    if isinstance(mes, dict) and isinstance(mes.get("processSummaries"), list):
        return [p for p in mes["processSummaries"] if isinstance(p, dict)]
    return []


def _mes_tools(context: dict[str, Any]) -> list[dict[str, Any]]:
    return backend_list(context, "mesCurrent", "tools")


def _tg_area(tg: dict[str, Any]) -> str | None:
    return str(tg.get("areaName") or tg.get("areaCode") or "") or None


def _trend(context: dict[str, Any], key: str) -> str | None:
    """trends[key] 12pt 시계열 → '증가/감소/보합'. 데이터 없으면 None."""
    mes = backend_context(context).get("mesCurrent")
    series = mes.get("trends", {}).get(key) if isinstance(mes, dict) and isinstance(mes.get("trends"), dict) else None
    if not isinstance(series, list):
        return None
    vals = [num(p.get("value")) for p in series if isinstance(p, dict)]
    if len(vals) < 3:
        return None
    last, prev_avg = vals[-1], sum(vals[:-1]) / len(vals[:-1])
    scale = max(abs(prev_avg), abs(last), 1e-9)
    ratio = (last - prev_avg) / scale
    if ratio > 0.15:
        return "증가"
    if ratio < -0.15:
        return "감소"
    return "보합"


def _down_by_tg(context: dict[str, Any], tgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DOWN tool을 소속 TG별로 그룹핑. mesCurrent.tools(status/downRatio) + toolGroups(tgId→이름) 근거."""
    name_by_id = {str(t.get("tgId")): name_of(t) for t in tgs}
    area_by_id = {str(t.get("tgId")): _tg_area(t) for t in tgs}
    counts: dict[str, int] = {}
    for tool in _mes_tools(context):
        if str(tool.get("status")) == "DOWN" or num(tool.get("downRatio")) >= 1.0:
            tgid = str(tool.get("tgId"))
            counts[tgid] = counts.get(tgid, 0) + 1
    out = [
        {"tgName": name_by_id.get(tgid, tgid), "areaName": area_by_id.get(tgid), "count": c}
        for tgid, c in counts.items()
    ]
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _watch_candidates(
    tgs: list[dict[str, Any]], down_by_tg: list[dict[str, Any]]
) -> list[WatchToolGroup]:
    """균형 기준(다운 + WIP 집중 + 셋업 과다 + Q-time 위험)으로 '살펴볼 TG' 후보 + 근거."""
    acc: dict[str, dict[str, Any]] = {}

    def add(tg_name: str, area: str | None, reason: str, severity: str) -> None:
        cur = acc.get(tg_name)
        if cur is None:
            acc[tg_name] = {"area": area, "reasons": [reason], "severity": severity}
        else:
            cur["reasons"].append(reason)
            if _SEV_ORDER.get(severity, 3) < _SEV_ORDER.get(cur["severity"], 3):
                cur["severity"] = severity

    for d in down_by_tg[:4]:
        add(d["tgName"], d.get("areaName"), f"비가동 설비 {d['count']}대", "critical" if d["count"] >= 3 else "warning")
    for t in sorted([t for t in tgs if to_int(t.get("wipCount")) > 0], key=lambda t: to_int(t.get("wipCount")), reverse=True)[:3]:
        add(name_of(t), _tg_area(t), f"WIP {to_int(t.get('wipCount'))} Lot 집중", "warning")
    for t in sorted([t for t in tgs if num(t.get("setupRatio")) >= 0.2], key=lambda t: num(t.get("setupRatio")), reverse=True)[:2]:
        add(name_of(t), _tg_area(t), f"셋업 비중 {pct(t.get('setupRatio'))}", "warning")
    for t in sorted([t for t in tgs if num(t.get("avgQtimeMin")) > 0], key=lambda t: num(t.get("avgQtimeMin")), reverse=True)[:2]:
        add(name_of(t), _tg_area(t), f"평균 대기 {to_int(t.get('avgQtimeMin'))}분", "warning")

    items = [
        WatchToolGroup(tgName=name, areaName=v["area"], reason=" · ".join(v["reasons"]), severity=v["severity"])
        for name, v in acc.items()
    ]
    items.sort(key=lambda w: _SEV_ORDER.get(w.severity, 3))
    return items[:6]


def _briefing_directions(down: int, top_area_name: str | None) -> list[ResponseDirection]:
    # '대응 방향'이 아니라 현장 작업자가 바로 확인하면 좋은 포인트(운영 사실 기반).
    directions: list[ResponseDirection] = []
    if down > 0:
        directions.append(ResponseDirection(
            title="비가동 설비 확인",
            description=f"현재 비가동(DOWN) 설비 {down}대가 있습니다. PM/고장 여부와 복구 예정을 점검하세요.",
        ))
    if top_area_name:
        directions.append(ResponseDirection(
            title="일감 집중 구역 점검",
            description=f"{top_area_name} 구역에 WIP가 가장 많이 몰려 있습니다. 진행 상태와 대기 Lot을 확인하세요.",
        ))
    directions.append(ResponseDirection(
        title="구역별 가동 현황 훑기",
        description="구역별 가동률과 가용 설비를 보고 바쁜 구역과 한가한 구역을 파악하세요.",
    ))
    return directions


def build_fab_briefing_baseline(req: AgentTaskAgentRequest) -> AgentTaskResult:
    # 병목 예측이 아니라 '현재 공장 전체 현황'을 현장 작업자가 한눈에 보도록,
    # 3D 뷰와 동일한 실시간 MES 스냅샷(mesCurrent)만 근거로 집계한다.
    ctx = req.context
    fab = _fab_summary(ctx)
    procs = _process_summaries(ctx)
    tgs = _flatten_toolgroups(ctx)
    tss = fab.get("toolStatusSummary") if isinstance(fab.get("toolStatusSummary"), dict) else {}
    run, idle, setup, down = to_int(tss.get("RUN")), to_int(tss.get("IDLE")), to_int(tss.get("SETUP")), to_int(tss.get("DOWN"))
    total_tools = run + idle + setup + down
    total_wip = to_int(fab.get("wipCount"))
    util = num(fab.get("utilizationRate"))
    avail = num(fab.get("avgAvailableToolRatio"))
    fab_name = str(fab.get("fabName") or "FAB")
    wip_trend = _trend(ctx, "wip")

    active_areas = sorted(
        [p for p in procs if to_int(p.get("wipCount")) > 0 or num(p.get("avgUtilizationRate")) > 0],
        key=lambda p: (to_int(p.get("wipCount")), num(p.get("avgUtilizationRate"))),
        reverse=True,
    )
    top_area = active_areas[0] if active_areas else None
    top_area_name = str(top_area.get("areaName") or top_area.get("areaCode")) if top_area else None
    busy_area_names = [str(p.get("areaName") or p.get("areaCode")) for p in active_areas[:5]]
    top_tg = max(
        tgs,
        key=lambda t: (to_int(t.get("wipCount")), num(t.get("bottleneckProb")), num(t.get("utilizationRate"))),
        default=None,
    )
    top_tg_name = name_of(top_tg, "") if top_tg else ""
    if not top_area_name and top_tg:
        top_area_name = _tg_area(top_tg)

    # mesCurrent.fab가 없으면 화면 toolGroups로 최소 집계(degraded).
    if total_tools == 0 and not fab:
        total_wip = sum(to_int(t.get("wipCount")) for t in tgs)
        util = (sum(num(t.get("utilizationRate")) for t in tgs) / len(tgs)) if tgs else 0.0

    down_by_tg = _down_by_tg(ctx, tgs)
    watch = _watch_candidates(tgs, down_by_tg)
    watch_names = {w.tg_name for w in watch}
    ref_tg_ids = [
        str(t.get("tgId")) for t in tgs
        if t.get("tgId") and (name_of(t) in watch_names or len(tgs) == 1)
    ]

    status_line = (
        f"설비 {total_tools}대 중 가동 {run} · 대기 {idle} · 셋업 {setup} · 비가동 {down}대"
        if total_tools else "설비 상태 데이터 확인 필요"
    )
    wip_clause = f"진행 중 WIP {total_wip} Lot" + (f"(최근 {wip_trend} 추세)" if wip_trend else "")
    summary = f"{fab_name} 현재 전체 가동률 {pct(util)}, {wip_clause}입니다. {status_line}."
    if top_area_name:
        summary += f" 일감은 주로 {top_area_name} 구역에 몰려 있습니다."
    if top_tg_name:
        summary += f" 우선 살펴볼 TG는 {top_tg_name}입니다."
    elif total_wip == 0:
        summary += " 현재 대부분 구역이 대기 상태로, 진행 중인 작업이 적습니다."

    wip_value = f"{total_wip} Lot" + (f" ({wip_trend})" if wip_trend else "")
    evidence = [
        EvidenceItem(label="전체 가동률", value=pct(util), description="현재 FAB 평균 설비 가동률입니다."),
        EvidenceItem(label="진행 중 WIP", value=wip_value, description="공정에 투입된 Lot 수와 최근 12시간 추세입니다."),
        EvidenceItem(label="가동/대기/비가동", value=f"{run} / {idle} / {down}대", description="RUN / IDLE / DOWN 설비 수입니다."),
        EvidenceItem(
            label="비가동 설비", value=f"{down}대",
            description="PM/고장 등으로 멈춰 있는 설비입니다. (사유 PM/BM·지속시간은 설비 상세에서 확인)",
            severity="warning" if down else "info",
        ),
        EvidenceItem(label="설비 가용률", value=pct(avail), description="정비/고장을 제외한 사용 가능 설비 비율입니다."),
    ]
    proc_summary = (
        "일감·가동이 있는 구역(라인 밸런스): " + ", ".join(busy_area_names) + " 순으로 활동이 많습니다."
        if busy_area_names else "현재 대부분 구역이 저부하/대기 상태로, 라인 전반이 한산합니다."
    )

    return AgentTaskResult(
        summary=summary,
        evidence=evidence,
        propagation=Propagation(
            summary=proc_summary,
            affectedProcesses=busy_area_names,
            affectedToolGroups=[],
            horizon=str(fab.get("measuredAt") or req.params.get("horizon") or "현재 스냅샷"),
        ),
        responseDirections=_briefing_directions(down, top_area_name),
        references=AgentReferences(tgIds=list(dict.fromkeys(ref_tg_ids))),
        followUpPrompts=[
            "비가동(DOWN) 설비를 TG별로 정리해줘.",
            "WIP가 가장 많은 구역과 추세를 알려줘.",
            "셋업 비중이 높은 설비그룹을 보여줘.",
        ],
        artifacts=[
            AgentArtifact(type="BRIEF", title="FAB 현황 브리핑", description="현재 스냅샷 기반 전체 현황 요약"),
        ],
        watchToolGroups=watch,
    )
