"""챗봇 도구 인자/실행 안전성 회귀."""

from uuid import uuid4

import pytest

from app.chatbot.context import ChatContext
from app.chatbot.runner import run_tool_calls
from app.chatbot.tools.kpi_trend import get_kpi_trend
from app.chatbot.tools.lot_status import get_top_toolgroups
from app.chatbot.tools.params import safe_int


class FakeRepo:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def kpi_trend(self, fab_id, area, hours, bucket, group_by):
        self.calls.append(("kpi_trend", area, hours, bucket, group_by))
        return []

    async def top_toolgroups(self, fab_id, area, metric, order, limit):
        self.calls.append(("top_toolgroups", area, metric, order, limit))
        return []


def test_safe_int_accepts_human_strings_and_falls_back():
    assert safe_int("6시간", default=4, min_value=1, max_value=24) == 6
    assert safe_int("많이", default=5, min_value=1, max_value=15) == 5
    assert safe_int(999, default=5, min_value=1, max_value=15) == 15


@pytest.mark.asyncio
async def test_tools_coerce_string_hours_and_limit_without_errors():
    repo = FakeRepo()
    ctx = ChatContext(fab_id=uuid4())
    ctx._repo = repo

    assert "추세 데이터가 없습니다" in await get_kpi_trend(ctx, area="Litho", hours="6시간")
    assert "툴그룹 현황 데이터가 없습니다" in await get_top_toolgroups(ctx, limit="많이")

    assert repo.calls[0] == ("kpi_trend", "Litho", 6, 15, "area")
    assert repo.calls[1] == ("top_toolgroups", "", "util", "desc", 5)


@pytest.mark.asyncio
async def test_run_tool_calls_handles_unknown_and_failing_tools():
    async def broken_tool():
        raise RuntimeError("boom")

    agent = {"tool_fns": {"broken": broken_tool}, "tools_used": []}
    messages = []

    await run_tool_calls(agent, [{"name": "missing", "id": "1"}, {"name": "broken", "id": "2"}], messages)

    assert len(messages) == 2
    assert "알 수 없는 도구" in messages[0].content
    assert "오류가 발생했습니다" in messages[1].content
    assert agent["tools_used"] == ["broken"]
