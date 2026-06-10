"""Application exceptions and error response helpers."""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def error_body(error_code: str, message: str) -> dict[str, Any]:
    return {"success": False, "errorCode": error_code, "message": message}


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.error_code, exc.message),
    )


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first_error.get("loc", ()) if part != "body")
    message = first_error.get("msg", "요청 값이 올바르지 않습니다.")
    if location:
        message = f"{location}: {message}"
    return JSONResponse(
        status_code=400,
        content=error_body("VALIDATION_ERROR", message),
    )


async def unhandled_error_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_body("INTERNAL_SERVER_ERROR", "서버 내부 오류가 발생했습니다."),
    )
