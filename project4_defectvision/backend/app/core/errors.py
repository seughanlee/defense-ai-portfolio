"""Uniform error body: {"error": {"code", "message", "request_id", "details"?}}."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.logging import request_id_var

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message, self.details = status_code, code, message, details


def _body(code: str, message: str, details: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        err["details"] = details
    return {"error": err}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(_body("VALIDATION_ERROR", "Request validation failed", details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")  # details go to the log, not to the client
        return JSONResponse(_body("INTERNAL_ERROR", "Internal server error"), status_code=500)
