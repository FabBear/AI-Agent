"""FastAPI endpoint contract tests."""

from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_db
from app.config import get_settings
from app.main import app

INTERNAL_HEADERS = {
    "X-User-Id": str(uuid4()),
    "X-User-Role": "ROLE_ADMIN",
    "X-Factory-Id": str(uuid4()),
    "X-Request-Id": str(uuid4()),
}


@pytest.fixture
async def client(monkeypatch):
    pool = AsyncMock()
    pool.fetch.return_value = []
    app.dependency_overrides[get_db] = lambda: pool
    monkeypatch.setattr("app.api.agent.run_pipeline_with_timeout", AsyncMock())
    monkeypatch.setattr("app.api.agent.run_post_hitl", AsyncMock())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def internal_token_headers() -> dict[str, str]:
    return {"X-Internal-Token": get_settings().internal_api_token}


@pytest.mark.asyncio
async def test_openapi_is_available(client: AsyncClient) -> None:
    response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_run_returns_accepted(client: AsyncClient) -> None:
    case_id = str(uuid4())
    response = await client.post(
        "/api/agent/run",
        headers=internal_token_headers(),
        json={
            "caseId": case_id,
            "tgId": str(uuid4()),
            "tgCode": "DIFFUSION_FE_127",
            "snapshotTime": 120.0,
            "bottleneckProb": 0.91,
            "riskGrade": "CRITICAL",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "success": True,
        "data": {"caseId": case_id, "status": "STARTED"},
    }


@pytest.mark.asyncio
async def test_user_invoked_agent_task_returns_structured_result(client: AsyncClient) -> None:
    task_id = str(uuid4())
    tg_id = str(uuid4())
    response = await client.post(
        "/api/agent/tasks",
        headers=internal_token_headers(),
        json={
            "taskId": task_id,
            "fabId": str(uuid4()),
            "userId": str(uuid4()),
            "taskType": "FAB_SNAPSHOT_BRIEFING",
            "sourcePage": "FAB3D",
            "context": {
                "backendContext": {
                    "mesCurrent": {
                        "toolGroups": [
                            {
                                "tgId": tg_id,
                                "tgName": "BACKEND_AUTHORITATIVE_TG",
                                "areaName": "Defect Metrology",
                                "utilizationRate": 0.94,
                                "wipCount": 51,
                                "bottleneckProb": 0.91,
                            }
                        ]
                    }
                }
            },
            # 결정론 검증을 위해 LLM(agentic) 경로를 끄고 룰베이스 결과를 단언한다.
            "params": {"useLlm": False},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["result"]["references"]["tgIds"] == [tg_id]
    assert "BACKEND_AUTHORITATIVE_TG" in body["result"]["summary"]

    follow_up = await client.get(f"/api/agent/tasks/{task_id}", headers=internal_token_headers())
    assert follow_up.status_code == 200
    assert follow_up.json()["result"]["summary"] == body["result"]["summary"]


@pytest.mark.asyncio
async def test_internal_token_is_required(client: AsyncClient) -> None:
    response = await client.post(
        "/api/ml/predict",
        json={"fabId": str(uuid4()), "snapshotTime": None},
    )

    assert response.status_code == 401
    assert response.json()["errorCode"] == "INVALID_INTERNAL_TOKEN"


@pytest.mark.asyncio
async def test_validation_error_uses_common_format(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent/run",
        headers=internal_token_headers(),
        json={},
    )

    assert response.status_code == 400
    assert response.json()["errorCode"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_hitl_result_uses_internal_headers(client: AsyncClient) -> None:
    case_id = str(uuid4())
    response = await client.post(
        "/api/agent/hitl-result",
        headers={**internal_token_headers(), **INTERNAL_HEADERS},
        json={
            "caseId": case_id,
            "decisionId": str(uuid4()),
            "decision": "APPROVED",
            "selectedPlanId": str(uuid4()),
            "comment": "승인",
            "decidedBy": INTERNAL_HEADERS["X-User-Id"],
            "decidedAt": "2026-06-10T03:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "caseId": case_id,
        "nextStep": "REPORT_GENERATION",
    }


@pytest.mark.asyncio
async def test_progress_returns_six_ddl_steps(client: AsyncClient) -> None:
    case_id = str(uuid4())
    pool = app.dependency_overrides[get_db]()
    pool.fetch.return_value = [
        {
            "step_order": index,
            "step_name": step_name,
            "status": "PENDING",
            "started_at": None,
            "completed_at": None,
            "output_summary": None,
            "attempt_no": 1,
        }
        for index, step_name in enumerate(
            [
                "DIFFUSION_ANALYSIS",
                "CAUSE_ANALYSIS",
                "ACTION_PLAN_GEN",
                "ACTION_PLAN_COMPARE",
                "HITL_WAITING",
                "REPORT_GEN",
            ],
            start=1,
        )
    ]
    response = await client.get(
        f"/api/agent/cases/{case_id}/progress",
        headers=INTERNAL_HEADERS,
    )

    assert response.status_code == 200
    steps = response.json()["data"]["steps"]
    assert [step["stepName"] for step in steps] == [
        "DIFFUSION_ANALYSIS",
        "CAUSE_ANALYSIS",
        "ACTION_PLAN_GEN",
        "ACTION_PLAN_COMPARE",
        "HITL_WAITING",
        "REPORT_GEN",
    ]
    assert all(step["status"] == "PENDING" for step in steps)


@pytest.mark.asyncio
async def test_predict_returns_empty_predictions_without_metrics(client: AsyncClient) -> None:
    response = await client.post(
        "/api/ml/predict",
        headers=internal_token_headers(),
        json={"fabId": str(uuid4()), "snapshotTime": None},
    )

    assert response.status_code == 200
    assert response.json()["data"]["predictions"] == []
    assert response.json()["data"]["highCriticalCount"] == 0


@pytest.mark.asyncio
async def test_chat_stub_uses_internal_user_headers(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat/message",
        headers=INTERNAL_HEADERS,
        json={
            "message": "현재 병목 공정은?",
            "sessionId": None,
            "fabId": INTERNAL_HEADERS["X-Factory-Id"],
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["references"] == []
