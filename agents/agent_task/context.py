"""TaskContext — 한 번의 Agent Task 실행 동안 도구가 공유하는 상태 + 진행 트레이스.

에이전트 도구는 이 ctx를 받아 라이브 MES 컨텍스트(req.context)나 DB(repo)를 조회하고,
조회 한 건마다 progress step을 남긴다(=화면에 보이는 '에이전트의 결정 흔적')."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from agents.agent_task.schemas import AgentTaskAgentRequest, AgentTaskProgressStep

if TYPE_CHECKING:
    from app.repositories.chat_query_repository import ChatQueryRepository


@dataclass
class TaskContext:
    req: AgentTaskAgentRequest
    progress: list[AgentTaskProgressStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    _repo: object | None = field(default=None, repr=False)

    @property
    def context(self) -> dict[str, Any]:
        return self.req.context

    @property
    def live(self) -> dict[str, Any]:
        """Spring이 넣어준 authoritative 백엔드 컨텍스트(mesCurrent/history 등)."""
        backend = self.req.context.get("backendContext")
        return backend if isinstance(backend, dict) else {}

    @property
    def fab_id(self) -> UUID:
        return self.req.fab_id

    def trace(self, step_name: str, message: str) -> None:
        """진행 단계(에이전트의 관측/결정 한 줄)를 기록한다."""
        self.progress.append(
            AgentTaskProgressStep(
                stepName=step_name,
                status="SUCCEEDED",
                message=message,
                occurredAt=datetime.now(timezone.utc),
            )
        )

    async def repo(self) -> "ChatQueryRepository":
        """조회 전용 ChatQueryRepository(읽기 전용 풀). 요청 내 1회 생성 캐시."""
        if self._repo is None:
            from app.chatbot.db import get_chat_pool
            from app.repositories.chat_query_repository import ChatQueryRepository

            self._repo = ChatQueryRepository(await get_chat_pool())
        return cast("ChatQueryRepository", self._repo)
