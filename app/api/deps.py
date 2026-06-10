"""FastAPI dependencies for authentication and database access."""

import secrets
from functools import lru_cache
from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import Depends, Header
from pydantic import BaseModel

from app.common.errors import AppError
from app.config import Settings, get_settings
from app.db.pool import get_pool
from app.services.predict_service import PredictService


class InternalUser(BaseModel):
    user_id: UUID
    role: str
    factory_id: UUID
    request_id: UUID


async def get_internal_user(
    user_id: Annotated[UUID, Header(alias="X-User-Id")],
    role: Annotated[str, Header(alias="X-User-Role")],
    factory_id: Annotated[UUID, Header(alias="X-Factory-Id")],
    request_id: Annotated[UUID, Header(alias="X-Request-Id")],
) -> InternalUser:
    return InternalUser(
        user_id=user_id,
        role=role,
        factory_id=factory_id,
        request_id=request_id,
    )


async def require_admin(
    user: Annotated[InternalUser, Depends(get_internal_user)],
) -> InternalUser:
    if user.role != "ROLE_ADMIN":
        raise AppError(403, "ADMIN_REQUIRED", "관리자 권한이 필요합니다.")
    return user


async def verify_internal_token(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> None:
    if token is None or not secrets.compare_digest(token, settings.internal_api_token):
        raise AppError(401, "INVALID_INTERNAL_TOKEN", "유효하지 않은 내부 토큰입니다.")


async def get_db() -> asyncpg.Pool:
    return await get_pool()


def get_predict_service() -> PredictService:
    return _get_predict_service()


@lru_cache
def _get_predict_service() -> PredictService:
    return PredictService(get_settings())
