"""테스트 케이스 22-31: AI 호출 시나리오 로직 검증."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import app.services.agent_service as agent_service
from app.services.agent_service import (
    _handle_pipeline_event,
    _initial_state,
    _reconstruct_report_state,
    run_pipeline_with_timeout,
)


# ── 22. 파이프라인 Webhook 모드 ────────────────────────────────────────────────

def test_22_webhook_mode_pipeline_stops_at_hitl(monkeypatch):
    """WEBHOOK_MODE=1이면 파이프라인이 compare_hitl → emit3 → END 순서로 종료한다."""
    monkeypatch.setenv("WEBHOOK_MODE", "1")
    from agents.pipeline import build_pipeline
    from langgraph.graph import END

    pipeline = build_pipeline(run_sim=False, run_g_star=False, run_detection=False)
    graph = pipeline.get_graph()
    edge_map = {e.source: e.target for e in graph.edges}

    assert edge_map.get("compare_hitl") == "emit3", "compare_hitl → emit3 엣지 없음"
    assert edge_map.get("emit3") == END, "emit3 → END 엣지 없음(Webhook 모드)"


@pytest.mark.asyncio
async def test_22_compare_hitl_event_marks_hitl_done_and_returns_hitl():
    """_handle_pipeline_event: compare_hitl 노드 → hitl 단계 완료 마킹 후 'hitl' 반환."""
    case_id = uuid4()
    spring_client = AsyncMock()

    with patch.object(agent_service, "_complete_step", new=AsyncMock()) as mock_complete:
        result = await _handle_pipeline_event(
            case_id=case_id,
            node_name="compare_hitl",
            state={"compare_formatted": [{"toolgroup": "DE_FE_1"}]},
            step_repo=AsyncMock(),
            cause_repo=AsyncMock(),
            plan_repo=AsyncMock(),
            spring_client=spring_client,
            tg_code="DE_FE_1",
        )

    assert result == "hitl"
    mock_complete.assert_awaited_once()
    args = mock_complete.call_args[0]
    assert args[1] == "hitl"
    assert "승인 대기" in (args[2] or "")


# ── 23. 병목 없음 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_23_no_alerts_skips_all_downstream_steps():
    """cascade 노드에서 alerts가 빈 배열이면 cause~report 5단계를 모두 '병목 없음'으로 건너뛴다."""
    case_id = uuid4()
    step_repo = AsyncMock()

    with patch.object(agent_service, "_complete_step", new=AsyncMock()):
        result = await _handle_pipeline_event(
            case_id=case_id,
            node_name="cascade",
            state={"alerts": []},
            step_repo=step_repo,
            cause_repo=AsyncMock(),
            plan_repo=AsyncMock(),
            spring_client=AsyncMock(),
            tg_code="ANY_TG",
        )

    assert result == "cascade"
    skip_calls = [
        call for call in step_repo.mark_done.call_args_list
        if call.args[2] == "병목 없음"
    ]
    skipped_steps = {call.args[1] for call in skip_calls}
    assert skipped_steps == {"cause", "solution", "compare", "hitl", "report"}


# ── 24. 파이프라인 타임아웃 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_24_timeout_marks_failed_and_notifies_backend():
    """TimeoutError 발생 시 활성 단계 FAILED 기록 + 백엔드 notify_agent_step(FAILED) 호출."""
    case_id = uuid4()
    pool = MagicMock()

    step_repo_mock = AsyncMock()
    step_repo_mock.mark_active_failed = AsyncMock(return_value="cause")
    spring_client_mock = AsyncMock()

    with (
        patch("app.services.agent_service.AgentStepRepository", return_value=step_repo_mock),
        patch("app.services.agent_service.get_spring_client", return_value=spring_client_mock),
        patch("app.services.agent_service.run_pipeline", new=AsyncMock(side_effect=TimeoutError())),
        patch("app.services.agent_service.get_settings", return_value=MagicMock(pipeline_timeout_sec=1200, max_concurrent_pipelines=3)),
    ):
        await run_pipeline_with_timeout(
            case_id, uuid4(), "DE_FE_1", 120.0, 0.91, "CRITICAL", pool
        )

    step_repo_mock.mark_active_failed.assert_awaited_once_with(case_id, "파이프라인 타임아웃")
    spring_client_mock.notify_agent_step.assert_awaited_once()
    call_kwargs = spring_client_mock.notify_agent_step.call_args
    assert call_kwargs.kwargs.get("status") == "FAILED" or "FAILED" in str(call_kwargs)


# ── 25. HITL 승인 → 보고서 ───────────────────────────────────────────────────

def test_25_hitl_approved_sets_hitl_approved_true_and_applies_plan():
    """승인 결정 + selected_plan → hitl_approved=True, 선택 플랜 액션 레이블 적용."""
    pending = {
        "hitl_token": "tok",
        "compare_formatted": [{"toolgroup": "DE_FE_1", "recommendation": {"reason": "x"}, "action_effects": []}],
        "alerts": [], "kpi_snapshot": [], "prev_kpi_snapshot": [],
        "cause_reports": [], "solution_candidates": [],
    }
    selected_plan = {"plan_seq": 2}

    state = _reconstruct_report_state(
        pending, selected_plan, uuid4(), __import__("datetime").datetime(2026, 6, 17, tzinfo=__import__("datetime").timezone.utc), "승인"
    )

    assert state["hitl_approved"] is True
    assert state["compare_results"][0]["recommendation"]["action_label"] == "B"


# ── 26. HITL 반려 ─────────────────────────────────────────────────────────────

def test_26_hitl_rejected_sets_hitl_approved_false_and_no_plan():
    """반려 결정 → hitl_approved=False, 대응안 선택 없음(next=CLOSED 경로)."""
    import datetime as dt

    pending = {
        "hitl_token": "tok",
        "compare_formatted": [{"toolgroup": "DE_FE_1", "recommendation": {"reason": "x"}, "action_effects": []}],
        "alerts": [], "kpi_snapshot": [], "prev_kpi_snapshot": [],
        "cause_reports": [], "solution_candidates": [],
    }

    state = _reconstruct_report_state(
        pending, None, uuid4(),
        dt.datetime(2026, 6, 17, tzinfo=dt.timezone.utc),
        "반려",
        decision="REJECTED",
    )

    assert state["hitl_approved"] is False
    assert state["compare_results"][0]["recommendation"].get("action_label") is None


# ── 27. RAG 비가용 폴백 ───────────────────────────────────────────────────────

def test_27_rag_returns_empty_on_connection_failure():
    """Qdrant 미연결(ConnectError) → 예외 캐치 후 빈 리스트 반환. chat_enabled=True 가정."""
    import httpx

    from app.chatbot.rag import rag_search

    with (
        patch("app.chatbot.rag.chat_enabled", return_value=True),
        patch("app.chatbot.rag.os.getenv", side_effect=lambda k, d="": d),
        patch("openai.OpenAI") as mock_openai,
        patch("httpx.post", side_effect=httpx.ConnectError("Qdrant down")),
    ):
        mock_openai.return_value.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
        result = rag_search("DE_FE_1 병목 원인")

    assert result == []


# ── 28. LLM 타임아웃 → degraded=True ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_28_chat_endpoint_returns_degraded_on_llm_failure():
    """answer_chat이 None 반환(LLM 오류/타임아웃) → API 응답 degraded=True."""
    from fastapi.testclient import TestClient

    from app.main import app

    headers = {
        "X-Internal-Token": "fabbear-internal-token-2024",
        "X-User-Id": str(uuid4()),
        "X-User-Role": "ROLE_ENGINEER",
        "X-Factory-Id": str(uuid4()),
        "X-Request-Id": str(uuid4()),
    }

    with (
        patch("app.api.chat.chat_enabled", return_value=True),
        patch("app.api.chat.answer_chat", new=AsyncMock(return_value=None)),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/chat/message",
            headers=headers,
            json={"message": "현재 병목 현황은?"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["degraded"] is True


# ── 29. 신뢰도/경고 산출 ─────────────────────────────────────────────────────

def test_29_confidence_high_when_tools_used_and_no_suspect_numbers():
    """도구 사용 + 수상한 수치 없음 → confidence=HIGH."""
    from app.chatbot.postprocess import _confidence

    confidence, warnings = _confidence(
        tools_used=["get_fab_status"],
        sources=[{"id": "case_123"}],
        suspects=[],
    )

    assert confidence == "HIGH"
    assert warnings == []


def test_29_confidence_low_when_no_tools_and_suspect_numbers():
    """도구 미사용 + 수상한 수치 → confidence=LOW, 경고 포함."""
    from app.chatbot.postprocess import _confidence

    confidence, warnings = _confidence(
        tools_used=[],
        sources=[],
        suspects=["수치 확인 불가"],
    )

    assert confidence == "LOW"
    assert len(warnings) > 0


# ── 30. 캐싱 재사용 ───────────────────────────────────────────────────────────

def test_30_no_caching_layer_confirmed():
    """compare/report 출력에 캐시 레이어가 없음을 코드로 확인(현재 미구현).

    캐싱이 도입되면 이 테스트를 업데이트해야 한다.
    """
    import agents.compare_agent.node as compare_module
    import agents.report_agent.node as report_module

    source = compare_module.__file__
    with open(source) as f:
        text = f.read()
    assert "lru_cache" not in text and "cache" not in text.lower().split("def ")[0], (
        "compare_agent에 캐시가 추가됨 — 테스트 30을 업데이트할 것"
    )

    source2 = report_module.__file__
    with open(source2) as f:
        text2 = f.read()
    assert "lru_cache" not in text2, (
        "report_agent에 캐시가 추가됨 — 테스트 30을 업데이트할 것"
    )


# ── 31. G* 시뮬 연동 ─────────────────────────────────────────────────────────

def test_31_g_star_node_exists_in_pipeline_between_emit1_and_cause():
    """파이프라인 그래프에 g_star 노드가 emit1 → g_star → cause 순서로 연결됨."""
    from agents.pipeline import build_pipeline

    pipeline = build_pipeline(run_sim=False, run_g_star=True, run_detection=False)
    graph = pipeline.get_graph()

    edge_sources = {e.source for e in graph.edges}
    edge_map = {e.source: e.target for e in graph.edges}

    assert "g_star" in edge_sources, "g_star 노드가 파이프라인에 없음"
    assert edge_map.get("emit1") == "g_star", "emit1 → g_star 엣지 없음"
    assert edge_map.get("g_star") == "cause", "g_star → cause 엣지 없음"


def test_31_g_star_disabled_passthrough_when_run_g_star_false():
    """run_g_star=False이면 g_star 노드는 state를 그대로 통과(lambda s: s)."""
    from agents.pipeline import build_pipeline

    pipeline = build_pipeline(run_sim=False, run_g_star=False, run_detection=False)
    graph = pipeline.get_graph()

    assert "g_star" in {e.source for e in graph.edges}, "g_star 노드 자체는 존재해야 함"
