"""FabBear FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.ml import router as ml_router
from app.common.errors import (
    AppError,
    app_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.db.pool import close_pool, get_pool


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    title="FabBear FastAPI",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(agent_router, prefix="/api/agent", tags=["agent"])
app.include_router(ml_router, prefix="/api/ml", tags=["ml"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
