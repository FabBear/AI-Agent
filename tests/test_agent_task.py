"""사용자 호출형 Agent Task 회귀 — fab_briefing_agent / period_report_agent.

(챗봇 라우팅·후처리·STT 회귀는 test_chatbot_*.py / test_stt_correction.py 참조.)"""

from uuid import uuid4

import pytest

from agents.agent_task.llm import should_use_llm
from agents.agent_task.schemas import AgentTaskAgentRequest
from agents.fab_briefing_agent import build_fab_briefing_baseline, build_fab_briefing_response
from agents.period_report_agent import build_period_report_baseline


def test_fab_briefing_prefers_backend_context_tool_groups():
    target_tg_id = str(uuid4())
    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="FAB_SNAPSHOT_BRIEFING",
        sourcePage="FAB3D",
        context={
            "toolGroups": [
                {
                    "tgId": str(uuid4()),
                    "tgName": "FRONTEND_STALE_TG",
                    "utilizationRate": 0.1,
                    "wipCount": 1,
                    "bottleneckProb": 0.1,
                }
            ],
            "backendContext": {
                "mesCurrent": {
                    "toolGroups": [
                        {
                            "tgId": target_tg_id,
                            "tgName": "BACKEND_AUTHORITATIVE_TG",
                            "areaName": "Defect Metrology",
                            "utilizationRate": 0.94,
                            "wipCount": 51,
                            "bottleneckProb": 0.91,
                        }
                    ]
                }
            },
        },
        params={},
    )

    result = build_fab_briefing_baseline(request)

    assert result.watch_tool_groups[0].tg_name == "BACKEND_AUTHORITATIVE_TG"
    assert all(watch.tg_name != "FRONTEND_STALE_TG" for watch in result.watch_tool_groups)
    assert result.references.tg_ids == [target_tg_id]
    assert "BACKEND_AUTHORITATIVE_TG" in result.summary


@pytest.mark.asyncio
async def test_fab_briefing_falls_back_to_rule_based_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_TASK_USE_LLM", raising=False)
    monkeypatch.delenv("USER_INVOKED_AGENT_USE_LLM", raising=False)
    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="FAB_SNAPSHOT_BRIEFING",
        sourcePage="FAB3D",
        context={},
        params={},
    )

    response = await build_fab_briefing_response(request)

    assert response.status == "SUCCEEDED"
    assert response.result is not None
    # 키 없음 → agentic 루프 대신 결정론적 룰베이스로 degrade.
    step_names = [step.step_name for step in response.progress]
    assert "RULE_BASED" in step_names
    assert "TOOL_CALL" not in step_names


def test_period_report_does_not_emit_detailed_scheduling_commands():
    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="REPORT_PERIOD_SUMMARY",
        sourcePage="REPORT_ARCHIVE",
        context={"items": [{"caseId": str(uuid4()), "tgName": "DEF_MET_FE_118", "riskGrade": "CRITICAL"}]},
        params={},
    )

    result_text = build_period_report_baseline(request).model_dump_json()

    assert "Lot 순서 재배열" not in result_text
    assert "상세 스케줄 확정" not in result_text


def test_period_report_monthly_uses_history_summary_context():
    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="REPORT_PERIOD_SUMMARY",
        sourcePage="REPORT_ARCHIVE",
        context={
            "dateRange": {"from": "2026-06-01", "to": "2026-06-30"},
            "backendContext": {
                "history": {
                    "items": [
                        {"caseId": str(uuid4()), "tgName": "DefMet_FE_43", "riskGrade": "CRITICAL", "decision": "APPROVED"}
                    ]
                },
                "historySummary": {
                    "totalCases": 120,
                    "riskCounts": {"CRITICAL": 8, "HIGH": 37},
                    "statusCounts": {"RESOLVED": 108, "AWAITING_HITL": 5},
                    "decisionCounts": {"APPROVED": 74, "REJECTED": 12},
                    "reportCount": 91,
                    "topToolGroups": [{"name": "DefMet_FE_43", "areaName": "Defect Metrology", "count": 9}],
                    "topAreas": [{"name": "Defect Metrology", "count": 14}],
                    "avgEstAvgWaitDelta": -0.14,
                    "avgEstDeliveryComplianceDelta": 1.25,
                },
            },
        },
        params={
            "intent": "monthly",
            "periodType": "MONTHLY",
            "month": "2026-06",
            "reportTone": "EXECUTIVE",
        },
    )

    result = build_period_report_baseline(request)

    assert result.artifacts[0].type == "MONTHLY_REPORT"
    assert result.artifacts[0].title == "2026년 6월 월간 운영 브리핑"
    assert "120건" in result.summary
    assert "DefMet_FE_43 9건" in result.summary
    assert result.evidence[0].value == "120건"


def _briefing_ctx(toolgroups=None, procsummaries=None, tools=None):
    from agents.agent_task.context import TaskContext

    mes = {}
    if toolgroups is not None:
        mes["toolGroups"] = toolgroups
    if procsummaries is not None:
        mes["processSummaries"] = procsummaries
    if tools is not None:
        mes["tools"] = tools
    req = AgentTaskAgentRequest(
        taskId=uuid4(), fabId=uuid4(), userId=uuid4(),
        taskType="FAB_SNAPSHOT_BRIEFING", sourcePage="FAB3D",
        context={"backendContext": {"mesCurrent": mes}}, params={},
    )
    return TaskContext(req=req)


def test_briefing_risk_ranking_sorts_by_ml_probability():
    from agents.fab_briefing_agent.tools import get_bottleneck_risk_ranking

    ctx = _briefing_ctx(toolgroups=[
        {"tgId": "1", "tgName": "Litho", "areaName": "Litho", "bottleneckProb": 0.30, "riskGrade": "LOW", "utilizationRate": 0.5, "wipCount": 10},
        {"tgId": "2", "tgName": "DefMet", "areaName": "Defect", "bottleneckProb": 0.91, "riskGrade": "CRITICAL", "utilizationRate": 0.94, "wipCount": 51},
    ])
    out = get_bottleneck_risk_ranking(ctx)
    # 위험 높은 DefMet이 Litho보다 앞에 온다(ML 확률 정렬).
    assert "DefMet" in out and out.index("DefMet") < out.index("Litho")
    assert "91.0%" in out


def test_briefing_area_risk_rollup_summarizes_process_summaries():
    from agents.fab_briefing_agent.tools import get_area_risk_rollup

    ctx = _briefing_ctx(procsummaries=[
        {"areaName": "Litho", "riskGrade": "LOW", "riskCounts": {}, "bottleneckToolGroupCount": 0, "avgQtimeMin": 5},
        {"areaName": "Defect", "riskGrade": "CRITICAL", "riskCounts": {"CRITICAL": 2, "HIGH": 1}, "bottleneckToolGroupCount": 3, "avgQtimeMin": 42},
    ])
    out = get_area_risk_rollup(ctx)
    assert "Defect" in out and "Critical 2/High 1" in out
    assert "Litho" not in out  # 위험 0인 구역은 제외


class _FakeCaseRepo:
    def __init__(self, detail, plans=None, hitl=None):
        self._detail, self._plans, self._hitl = detail, plans or [], hitl or []

    async def case_detail(self, fab_id, ref):
        return self._detail

    async def case_plans(self, case_id):
        return self._plans

    async def case_hitl(self, case_id):
        return self._hitl


def _report_ctx_with_repo(repo):
    from agents.agent_task.context import TaskContext

    req = AgentTaskAgentRequest(
        taskId=uuid4(), fabId=uuid4(), userId=uuid4(),
        taskType="REPORT_PERIOD_SUMMARY", sourcePage="REPORT_ARCHIVE", context={}, params={},
    )
    ctx = TaskContext(req=req)
    ctx._repo = repo
    return ctx


@pytest.mark.asyncio
async def test_case_effectiveness_flags_unvalidated_then_reports_actual():
    from datetime import datetime

    from agents.period_report_agent.tools import get_case_effectiveness

    detail = {"case_id": "c1", "tg_name": "DefMet", "tg_code": "DM", "area_name": "Defect"}
    unvalidated = [{"selected": True, "plan_title": "투입조절", "est_avg_wait_delta": -0.5, "est_throughput_delta": 1.2,
                    "actual_avg_wait_delta": None, "actual_throughput_delta": None, "validated_at": None}]
    out1 = await get_case_effectiveness(_report_ctx_with_repo(_FakeCaseRepo(detail, unvalidated)))
    assert "투입조절" in out1 and "미검증" in out1

    validated = [{"selected": True, "plan_title": "투입조절", "est_avg_wait_delta": -0.5, "est_throughput_delta": 1.2,
                  "actual_avg_wait_delta": -0.4, "actual_throughput_delta": 1.0, "validated_at": datetime(2026, 6, 1)}]
    out2 = await get_case_effectiveness(_report_ctx_with_repo(_FakeCaseRepo(detail, validated)))
    assert "실측" in out2 and "-0.40" in out2


class _FakeAI:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class _FakeBound:
    """bind_tools 결과 — script 순서대로 ainvoke를 응답."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    async def ainvoke(self, messages):
        ai = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return ai


class _FakeStructured:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, messages):
        return self._result


class _FakeLLM:
    def __init__(self, script, result):
        self._script = script
        self._result = result

    def bind_tools(self, tools):
        return _FakeBound(self._script)

    def with_structured_output(self, schema):
        return _FakeStructured(self._result)


@pytest.mark.asyncio
async def test_fab_briefing_runs_agentic_tool_loop(monkeypatch):
    """LLM이 도구를 골라 호출 → progress에 결정 트레이스가 남고 결과가 작성된다."""
    from agents.agent_task import loop as loop_mod
    from agents.agent_task.schemas import AgentTaskResult, Propagation

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_TASK_USE_LLM", raising=False)

    final = AgentTaskResult(summary="현황 브리핑 완료", propagation=Propagation(summary="ok"))
    script = [
        _FakeAI(tool_calls=[{"name": "get_fab_overview", "args": {}, "id": "c1"}]),
        # overview 결과 보고 ML 위험 순위로 드릴다운(신규 도구) — 다단계 자율 결정.
        _FakeAI(tool_calls=[{"name": "get_bottleneck_risk_ranking", "args": {"limit": 5}, "id": "c2"}]),
        _FakeAI(tool_calls=[], content="충분"),
    ]
    monkeypatch.setattr(loop_mod, "make_chat_llm", lambda req: _FakeLLM(script, final))

    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="FAB_SNAPSHOT_BRIEFING",
        sourcePage="FAB3D",
        context={"backendContext": {"mesCurrent": {"fab": {"utilizationRate": 0.3, "wipCount": 12}}}},
        params={},
    )

    response = await build_fab_briefing_response(request)

    assert response.status == "SUCCEEDED"
    assert response.result.summary == "현황 브리핑 완료"
    step_names = [s.step_name for s in response.progress]
    assert "PLAN" in step_names
    assert step_names.count("TOOL_CALL") == 2  # LLM이 두 도구를 스스로 호출
    assert "COMPOSE" in step_names
    # 도구 호출 메시지에 사람이 읽을 라벨이 들어간다(화면 트레이스).
    tool_msgs = [s.message for s in response.progress if s.step_name == "TOOL_CALL"]
    assert any("전체 현황 조회" in m for m in tool_msgs)
    assert any("ML 위험 TG 순위 조회" in m for m in tool_msgs)


def test_llm_gate_defaults_on_with_key_and_is_overridable(monkeypatch):
    request = AgentTaskAgentRequest(
        taskId=uuid4(),
        fabId=uuid4(),
        userId=uuid4(),
        taskType="FAB_SNAPSHOT_BRIEFING",
        sourcePage="FAB3D",
        context={},
        params={},
    )

    # 키가 있으면 기본 ON(에이전트가 기본).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_TASK_USE_LLM", raising=False)
    monkeypatch.delenv("USER_INVOKED_AGENT_USE_LLM", raising=False)
    assert should_use_llm(request) is True

    # env로 끌 수 있다.
    monkeypatch.setenv("AGENT_TASK_USE_LLM", "false")
    assert should_use_llm(request) is False

    # params.useLlm=False가 우선한다.
    monkeypatch.delenv("AGENT_TASK_USE_LLM", raising=False)
    request.params["useLlm"] = False
    assert should_use_llm(request) is False

    # 키가 없으면 무조건 OFF.
    request.params.pop("useLlm")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert should_use_llm(request) is False
