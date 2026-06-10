"""Chat API stubs."""

from enum import Enum
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.deps import InternalUser, get_internal_user
from app.api.schemas import ApiModel
from app.common.responses import ApiResponse, success

router = APIRouter()


class ChatIntent(str, Enum):
    KPI_QUERY = "KPI_QUERY"
    CASE_SEARCH = "CASE_SEARCH"
    MANUAL_SEARCH = "MANUAL_SEARCH"


class ChatMessageRequest(ApiModel):
    message: str = Field(min_length=1)
    session_id: str | None = Field(default=None, alias="sessionId")
    fab_id: UUID = Field(alias="fabId")


class ChatReference(ApiModel):
    type: str
    id: str
    title: str
    excerpt: str


class ChatMessageResult(ApiModel):
    answer: str
    intent: ChatIntent
    references: list[ChatReference]


@router.post("/message", response_model=ApiResponse[ChatMessageResult])
async def send_message(
    _: ChatMessageRequest,
    __: Annotated[InternalUser, Depends(get_internal_user)],
) -> ApiResponse[ChatMessageResult]:
    return success(
        ChatMessageResult(
            answer="챗봇 서비스가 준비 중입니다.",
            intent=ChatIntent.MANUAL_SEARCH,
            references=[],
        )
    )
