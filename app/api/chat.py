"""Chat API — 도메인 LLM 어시스턴트."""

import json
import logging
from enum import Enum
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import Field

from app.api.deps import InternalUser, get_internal_user
from app.api.schemas import ApiModel
from app.chatbot.service import answer_chat, answer_chat_stream, chat_enabled, generate_title
from app.common.responses import ApiResponse, success

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatIntent(str, Enum):
    KPI_QUERY = "KPI_QUERY"
    CASE_SEARCH = "CASE_SEARCH"
    MANUAL_SEARCH = "MANUAL_SEARCH"


class ChatTurn(ApiModel):
    role: str
    content: str


class ChatMessageRequest(ApiModel):
    message: str = Field(min_length=1)
    session_id: str | None = Field(default=None, alias="sessionId")
    fab_id: UUID | None = Field(default=None, alias="fabId")
    history: list[ChatTurn] = Field(default_factory=list)
    context: str | None = None
    live_status: str | None = Field(default=None, alias="liveStatus")
    generate_title: bool = Field(default=False, alias="generateTitle")


class ChatReference(ApiModel):
    type: str
    id: str
    title: str
    excerpt: str


class ChatSource(ApiModel):
    title: str
    source_path: str | None = Field(default=None, alias="sourcePath")
    category: str | None = None


class ChatMessageResult(ApiModel):
    answer: str
    intent: ChatIntent
    references: list[ChatReference]
    degraded: bool = False
    title: str | None = None
    sources: list[ChatSource] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list, alias="followUps")
    spoken_summary: str | None = Field(default=None, alias="spokenSummary")
    ui: dict | None = None
    confidence: str | None = None
    warnings: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list, alias="toolsUsed")


@router.post("/message", response_model=ApiResponse[ChatMessageResult])
async def send_message(
    request: ChatMessageRequest,
    _: Annotated[InternalUser, Depends(get_internal_user)],
) -> ApiResponse[ChatMessageResult]:
    answer: str | None = None
    title: str | None = None
    sources: list[ChatSource] = []
    follow_ups: list[str] = []
    spoken_summary: str | None = None
    ui: dict | None = None
    confidence: str | None = None
    warnings: list[str] = []
    tools_used: list[str] = []
    if chat_enabled():
        try:
            history = [turn.model_dump() for turn in request.history]
            result = await answer_chat(
                request.message, history=history, context=request.context,
                live_status=request.live_status, fab_id=request.fab_id,
            )
            if result:
                answer = result["answer"]
                sources = [ChatSource.model_validate(s) for s in result["sources"]]
                follow_ups = list(result["followUps"])
                spoken_summary = result["spokenSummary"]
                ui = result["ui"]
                confidence = result["confidence"]
                warnings = list(result["warnings"])
                tools_used = list(result["toolsUsed"])
        except Exception:  # noqa: BLE001
            logger.exception("chat answer_chat 실패")
            answer = None
        if request.generate_title:
            title = generate_title(request.message)

    if answer:
        return success(ChatMessageResult(
            answer=answer, intent=ChatIntent.MANUAL_SEARCH, references=[], title=title,
            sources=sources, follow_ups=follow_ups, spoken_summary=spoken_summary, ui=ui,
            confidence=confidence, warnings=warnings, tools_used=tools_used,
        ))

    return success(ChatMessageResult(answer="", intent=ChatIntent.MANUAL_SEARCH, references=[], degraded=True, title=title))


@router.post("/stream")
async def stream_message(
    request: ChatMessageRequest,
    _: Annotated[InternalUser, Depends(get_internal_user)],
) -> StreamingResponse:
    """SSE 스트리밍 챗. 이벤트: stage / token / meta / error."""

    async def gen():
        try:
            history = [turn.model_dump() for turn in request.history]
            async for ev in answer_chat_stream(
                request.message, history=history, context=request.context,
                live_status=request.live_status, fab_id=request.fab_id,
            ):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if request.generate_title:
                title = generate_title(request.message)
                if title:
                    yield f"event: title\ndata: {json.dumps({'title': title}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception:  # noqa: BLE001
            logger.exception("chat stream 실패")
            yield "event: error\ndata: {\"message\": \"stream failed\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
