"""FastAPI application factory.

    uvicorn backend.app.main:app            # from project4_defectvision/, uses DV_* environment variables
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import sessionmaker

from backend.app.api.routes import public, router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import install_error_handlers
from backend.app.core.logging import configure_logging, log_fields, request_id_var
from backend.app.db.session import init_db, make_engine
from backend.app.services.model_service import ModelService
from backend.app.services.storage import ResultStorage

logger = logging.getLogger("defectvision.api")


def create_app(settings: Settings | None = None, *, load_model: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        if load_model:
            app.state.model_service.start_loading(background=True)
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="DefectVision AI Inspection API",
        version=settings.app_version,
        description="PatchCore-based visual inspection. Heatmaps are model estimates, not confirmed defects.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    init_db(app.state.engine)
    app.state.session_factory = sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.storage = ResultStorage(settings.storage_dir)
    app.state.model_service = ModelService(settings.model_artifact_dir, settings.model_device)

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"], expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        rid = request.headers.get("X-Request-ID", "")
        rid = rid if rid.isalnum() and len(rid) <= 64 else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            # Reject oversized uploads before the body is parsed. Chunked bodies are still capped on read.
            length = request.headers.get("content-length")
            limit = settings.max_upload_bytes + 64 * 1024  # multipart framing overhead
            if length and length.isdigit() and int(length) > limit:
                response = JSONResponse(
                    {"error": {"code": "FILE_TOO_LARGE", "message": f"Request body exceeds {limit} bytes",
                               "request_id": rid}}, status_code=413)
            else:
                try:
                    response = await call_next(request)
                except Exception:  # log with this request's id; never leak internals to the client
                    logger.exception("Unhandled error")
                    response = JSONResponse(
                        {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error",
                                   "request_id": rid}}, status_code=500)
            response.headers["X-Request-ID"] = rid
            log_fields(logger, logging.INFO, "request", method=request.method, path=request.url.path,
                       status=response.status_code, duration_ms=round((time.perf_counter() - t0) * 1000, 1))
            return response
        finally:
            request_id_var.reset(token)

    install_error_handlers(app)
    app.include_router(public)
    app.include_router(router)
    return app


def __getattr__(name: str):  # noqa: ANN202 - lazy module attribute so importing this file has no side effects
    if name == "app":
        return create_app()
    raise AttributeError(name)
