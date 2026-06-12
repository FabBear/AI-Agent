"""ChatContext — 한 번의 챗 요청 동안 도구 함수가 공유하는 상태.

도구 모듈은 이 컨텍스트를 통해 live_status, fab_id, 카드, 출처, 호출 이력을 명시적으로 공유한다."""

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class ChatContext:
    live_status: str | None = None
    fab_id: UUID | None = None
    ui_cards: dict = field(default_factory=dict)
    used_hits: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    _repo: object | None = field(default=None, repr=False)

    def set_card(self, key: str, card: dict | None) -> None:
        """도구가 만든 카드를 sink에 등록(None이면 무시). 같은 key는 마지막 호출이 덮어씀."""
        if card:
            self.ui_cards[key] = card

    async def repo(self) -> "object | None":
        """챗봇 조회 전용 repository(없으면 None). read-only 풀 사용·요청 내 1회 생성 캐시."""
        if self.fab_id is None:
            return None
        if self._repo is None:
            from app.chatbot.db import get_chat_pool
            from app.repositories.chat_query_repository import ChatQueryRepository
            self._repo = ChatQueryRepository(await get_chat_pool())
        return self._repo
