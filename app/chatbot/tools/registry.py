"""LLM에 노출할 도구 레지스트리 — ChatContext를 바인딩한 얇은 클로저로 StructuredTool을 만든다.

도구 본문은 도메인 모듈(tools/*.py)에 있고, 여기서는 ctx 주입 + 스키마/설명만 담당한다.
챗봇 도구는 전부 조회 전용(get_*/search_*) — 실행·변경성 동사는 등록하지 않는다."""

from langchain_core.tools import StructuredTool

from app.chatbot.context import ChatContext
from app.chatbot.tools import (
    bottleneck_cases,
    fab_status,
    knowledge,
    kpi_trend,
    lot_status,
    tool_activity,
)

TOOL_LABELS = {
    "search_knowledge": "사내 지식 검색 중",
    "get_fab_status": "실시간 현황 조회 중",
    "get_kpi_trend": "KPI 추세 조회 중",
    "get_top_toolgroups": "툴그룹 순위 조회 중",
    "get_tool_status": "설비 현황 조회 중",
    "get_lot_status": "WIP·대기 현황 조회 중",
    "search_bottleneck_cases": "병목 케이스 조회 중",
    "get_case_detail": "케이스 상세 조회 중",
    "get_tool_activity": "설비 활동 추이 조회 중",
    "compare_periods": "기간 비교 중",
}


def build_tools(ctx: ChatContext) -> tuple[list, dict]:
    """ctx에 바인딩된 (StructuredTool 목록, name→callable 맵)을 반환.
    StructuredTool은 깔끔한 시그니처(ctx 없는)에서 스키마를 추론하므로 얇은 클로저로 감싼다."""

    def search_knowledge(query: str) -> str:
        return knowledge.search_knowledge(ctx, query)

    def get_fab_status(area: str = "") -> str:
        return fab_status.get_fab_status(ctx, area)

    async def get_kpi_trend(area: str = "", hours: int = 6, group_by: str = "area") -> str:
        return await kpi_trend.get_kpi_trend(ctx, area, hours, group_by)

    async def compare_periods(area: str = "", hours: int = 4) -> str:
        return await kpi_trend.compare_periods(ctx, area, hours)

    async def get_lot_status(area: str = "") -> str:
        return await lot_status.get_lot_status(ctx, area)

    async def get_top_toolgroups(area: str = "", metric: str = "util", order: str = "desc", limit: int = 5) -> str:
        return await lot_status.get_top_toolgroups(ctx, area, metric, order, limit)

    async def get_tool_status(tg: str = "", metric: str = "util") -> str:
        return await lot_status.get_tool_status(ctx, tg, metric)

    async def search_bottleneck_cases(area: str = "", status: str = "") -> str:
        return await bottleneck_cases.search_bottleneck_cases(ctx, area, status)

    async def get_case_detail(case_ref: str = "") -> str:
        return await bottleneck_cases.get_case_detail(ctx, case_ref)

    async def get_tool_activity(tool: str = "", hours: int = 6) -> str:
        return await tool_activity.get_tool_activity(ctx, tool, hours)

    tools = [
        StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description="반도체 FAB 도메인 지식(개념/정의/원리/SOP/용어/설비·공정 설명)을 사내 지식베이스에서 검색. 정의·원리·방법·차이 질문에 사용.",
        ),
        StructuredTool.from_function(
            func=get_fab_status,
            name="get_fab_status",
            description="지금 공장의 실시간 현황(전체/구역별 WIP·가동률·가용률·설비상태). 현재 상태 질문에 사용. 특정 구역은 area에 구역명(예: Diffusion).",
        ),
        StructuredTool.from_function(
            coroutine=get_kpi_trend,
            name="get_kpi_trend",
            description="구역/TG의 KPI 시계열 추세(WIP·가동률·Q-time). '추세·변화·지난 N시간' 질문에 사용. area=구역명 또는 TG코드, hours=조회시간(기본 6), group_by='tg'면 툴그룹 단위 추세.",
        ),
        StructuredTool.from_function(
            coroutine=get_top_toolgroups,
            name="get_top_toolgroups",
            description="툴그룹(TG) 단위 현황·순위. '가동률이 가장 높은/낮은 툴그룹', 'WIP 많은 TG', 'Q-time 긴 TG' 등 TG 순위·비교·현황 질문에 사용. metric: util/wip/qtime/wait/bottleneck, order: desc/asc, area=구역명(생략 시 전체).",
        ),
        StructuredTool.from_function(
            coroutine=get_tool_status,
            name="get_tool_status",
            description="특정 툴그룹 안의 개별 툴(설비/기기)들의 현황(가동률·OEE·대기 Lot·셋업/다운). '그 TG 안의 툴들/설비별 상태/어떤 장비가 문제야' 질문에 사용. tg=TG코드나 TG명(구역명도 가능), metric=util/queue/qtime/down 중 정렬 기준.",
        ),
        StructuredTool.from_function(
            coroutine=get_lot_status,
            name="get_lot_status",
            description="구역/TG별 WIP·대기·Q-time 현황. '어디가 막혀있나/적체/밀림/쌓임/대기 Lot 많은 곳' 등 지금 물량이 몰린 위치 질문에 사용. area=구역명(생략 시 전체).",
        ),
        StructuredTool.from_function(
            coroutine=search_bottleneck_cases,
            name="search_bottleneck_cases",
            description="ML이 과거에 감지한 병목 '이벤트 이력'(케이스 기록: 위험등급·확률·상태) 조회 전용. '병목 케이스 있었어/이력/감지 기록' 질문에만 사용. 케이스 1건의 원인·대응안 상세는 get_case_detail, 지금 어디가 막혔는지는 get_lot_status를 써라.",
        ),
        StructuredTool.from_function(
            coroutine=get_case_detail,
            name="get_case_detail",
            description="특정 병목 케이스 1건의 상세 — 원인 분석(SHAP 주요 피처)·대응안과 예상/실측 효과·HITL 승인 결정·리포트 요약. '왜 병목이었어/원인/대응안/승인됐어/리포트 요약' 질문에 사용. case_ref=케이스ID 앞자리/TG코드/구역명, 생략 시 최신 케이스. 케이스 '목록·이력'은 search_bottleneck_cases.",
        ),
        StructuredTool.from_function(
            coroutine=get_tool_activity,
            name="get_tool_activity",
            description="특정 툴(설비) 1대의 최근 활동 추이(가동률·대기 Lot·Q-time·다운율 시계열). '이 설비 최근 어땠어/추이/괜찮았나' 질문에 사용. tool=툴 코드나 이름, hours=조회시간(기본 6). TG 안 여러 설비의 현재 상태 비교는 get_tool_status.",
        ),
        StructuredTool.from_function(
            coroutine=compare_periods,
            name="compare_periods",
            description="최근 N시간 vs 직전 N시간 비교(WIP·가동률·Q-time 델타). '아까/전 교대/오전 대비 어때, 나아졌어/나빠졌어' 질문에 사용. area=구역명, hours=비교 단위(기본 4).",
        ),
    ]

    tool_fns = {
        "search_knowledge": search_knowledge,
        "get_fab_status": get_fab_status,
        "get_kpi_trend": get_kpi_trend,
        "get_top_toolgroups": get_top_toolgroups,
        "get_tool_status": get_tool_status,
        "get_lot_status": get_lot_status,
        "search_bottleneck_cases": search_bottleneck_cases,
        "get_case_detail": get_case_detail,
        "get_tool_activity": get_tool_activity,
        "compare_periods": compare_periods,
    }
    return tools, tool_fns
