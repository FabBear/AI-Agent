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
