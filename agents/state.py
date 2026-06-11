from typing import TypedDict

from agents.schemas.alert import BottleneckAlert, PotentialBottleneck
from agents.schemas.cause import CauseReport
from agents.schemas.kpi import ToolGroupKPI


class PipelineState(TypedDict):
    case_id: str | None                 # 병목 케이스 ID (Spring Boot → HITL webhook 연결)
    kpi_snapshot: list[ToolGroupKPI]
    prev_kpi_snapshot: list[ToolGroupKPI]  # t-120분 스냅샷 (delta 피처용)
    potential_bottlenecks: list[PotentialBottleneck]
    alerts: list[BottleneckAlert]
    cause_reports: list[CauseReport]
    cascade_report: str | None
    current_release_interval: float | None  # 현재 Lot Release Interval (분), 없으면 None
    solution_candidates: list[dict]
    hitl_approved: bool | None
    hitl_token: str | None            # Webhook 모드 HITL 토큰 (Phase 1 → Phase 2 연결)
    verification_results: list[dict]  # 대응안 효과 검증 결과 (Agent 4 출력)
    compare_inputs: list[dict]        # Agent 5 중간: 툴그룹별 입력 + 순위
    compare_formatted: list[dict]     # Agent 5 중간: LLM 추천 + HITL 프롬프트
    compare_results: list[dict]       # 대응안 비교분석 결과 (Agent 5 출력)
    report_draft: list[dict]          # Agent 6 중간: 툴그룹별 섹션 누적
    report_results: list[dict]        # 최종보고서 생성 결과 (Agent 6 출력)
