"""Stable API error codes and sanitized error payloads."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        job_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.job_id = job_id
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.job_id:
            body["error"]["job_id"] = self.job_id
        if self.extra:
            body["error"].update(self.extra)
        return body


def raise_api(
    code: str,
    message: str,
    *,
    status_code: int = 400,
    job_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    raise ApiError(code, message, status_code=status_code, job_id=job_id, extra=extra)


async def api_error_handler(_request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = {"error": detail}
    else:
        body = {
            "error": {
                "code": "HTTP_ERROR",
                "message": str(detail),
            }
        }
    return JSONResponse(status_code=exc.status_code, content=body)
