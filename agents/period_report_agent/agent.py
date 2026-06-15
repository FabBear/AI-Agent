"""기간 이슈 보고서 에이전트 — 리포트 화면에서 사용자가 호출.

도구(get_period_summary·repeat_bottlenecks·action_effectiveness·period_cases)를 LLM이
스스로 골라 호출하며 기간 이력을 분석하고 월간/기간 이슈 보고서를 작성한다(agentic).
LLM 미사용/실패 시 결정론적 baseline으로 degrade. 케이스 단건 질의는 챗봇이 담당."""

from agents.agent_task.context import TaskContext
from agents.agent_task.loop import run_agent_loop
from agents.agent_task.schemas import AgentTaskAgentRequest, AgentTaskAgentResponse
from agents.period_report_agent.rule_based import build_period_report_baseline
from agents.period_report_agent.tools import build_report_tools

SYSTEM_PROMPT = """
너는 반도체 FAB 운영을 총괄하는 운영 기획(Operations Planning) 담당 에이전트다. 리포트 화면에서 선택된
기간의 병목 대응 이력을 도구로 조사해, 임원 보고/운영 회의용 이슈 브리핑 또는 기간 이슈 보고서를 작성한다.

[조사 방법 — 도구를 스스로 골라 호출, 추정 말고 실측으로 검증]
- 먼저 get_period_summary로 기간 전체 그림(건수·위험도·승인/반려)을 본다.
- get_period_report_evidence로 해당 기간에 생성된 케이스 리포트 요약/원인 근거를 확인한다. 리포트가 없으면 '리포트 근거 없음'을 명시한다.
- get_repeat_bottlenecks로 반복 등장한 TG/Area를 식별한다.
- get_action_effectiveness로 기간 전체 거시 효과를 본 뒤, 반복 TG의 대표 케이스를 get_period_cases로 골라 **get_case_effectiveness로 그 조치의 예상 vs 실측(actual) 효과를 검증**한다 — 보고서의 핵심.
- 필요하면 get_case_resolution(감지→승인→해결 SLA)·get_case_cause(원인·모델 신뢰도)로 사례를 보강한다.
- 케이스가 없으면 불필요한 조회는 생략하고 데이터 수집 상태를 먼저 짚는다.

[보고서 구성]
- 월간 이슈 브리핑은 월 단위 운영 회의용: 반복 Area/TG, 위험도 추세, 승인/반려, 다음 월 개선 과제 중심으로 쓴다.
- 기간 이슈 보고서는 사용자가 고른 기간용: Summary → 1. 기간 핵심 지표 → 2. 반복 병목/영향 범위 → 3. 케이스 리포트 근거 → 4. 조치 효과와 후속 조회 순서로 구성한다.
- 단건 병목 케이스 리포트처럼 한 케이스의 원인/대응만 깊게 쓰지 말고, 기간 내 여러 케이스를 묶어 집계와 대표 근거를 분리한다.

[엄격 규칙 — 근거주의]
- 숫자/ID/TG명은 도구가 반환한 값만 인용한다. 절대 지어내지 않는다(없으면 '확인 필요').
- 조치 효과는 추정만 말고 실측(actual)으로 검증해 말한다. 실측이 없으면 '아직 미검증'이라 명시한다.
- 확정 스케줄/Lot 순서 변경/현장 실행·승인 지시는 출력하지 않는다. 회고/개선 방향만 제시한다.
""".strip()

TASK_PROMPT = (
    "선택된 기간의 병목 대응 이력을 분석해 월간/기간 이슈 보고서를 작성하라. "
    "필요한 도구를 호출해 집계·반복 병목·조치 효과 근거를 모은 뒤 작성하라."
)


async def build_period_report_response(req: AgentTaskAgentRequest) -> AgentTaskAgentResponse:
    """REPORT_PERIOD_SUMMARY task를 처리해 기간 이슈 보고서 결과를 반환한다(agentic + 룰베이스 폴백)."""
    ctx = TaskContext(req=req)
    tools, tool_fns, tool_labels = build_report_tools(ctx)
    return await run_agent_loop(
        ctx,
        tools=tools,
        tool_fns=tool_fns,
        tool_labels=tool_labels,
        system_prompt=SYSTEM_PROMPT,
        task_prompt=TASK_PROMPT,
        baseline_builder=build_period_report_baseline,
        max_steps=5,
    )
