from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.spring_client import SpringClient


@pytest.mark.asyncio
async def test_spring_client_reuses_http_client() -> None:
    client = SpringClient()
    await client.close()
    http_client = AsyncMock()
    response = MagicMock()
    http_client.post.return_value = response
    client._client = http_client

    await client.notify_agent_step(uuid4(), "cause", "원인 분석 완료")
    await client.notify_agent_step(uuid4(), "solution", "대응안 생성 완료")

    assert http_client.post.await_count == 2
    assert response.raise_for_status.call_count == 2
    first_call = http_client.post.await_args_list[0]
    assert first_call.args[0] == "/api/internal/agent/step-done"
    assert first_call.kwargs["json"]["notificationType"] == "AGENT_STEP_UPDATED"


@pytest.mark.asyncio
async def test_spring_client_maps_terminal_notifications() -> None:
    client = SpringClient()
    await client.close()
    http_client = AsyncMock()
    response = MagicMock()
    http_client.post.return_value = response
    client._client = http_client

    await client.notify_agent_step(uuid4(), "hitl", "관리자 승인 대기")
    await client.notify_agent_step(uuid4(), "report", "보고서 생성 완료")

    hitl_payload = http_client.post.await_args_list[0].kwargs["json"]
    report_payload = http_client.post.await_args_list[1].kwargs["json"]
    assert hitl_payload["notificationType"] == "AGENT_HITL_WAITING"
    assert report_payload["notificationType"] == "AGENT_PIPELINE_DONE"
