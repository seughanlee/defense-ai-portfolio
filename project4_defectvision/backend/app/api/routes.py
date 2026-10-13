from __future__ import annotations

import logging
import secrets
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.core.errors import APIError
from backend.app.core.logging import log_fields
from backend.app.db.models import Inspection
from backend.app.db.session import session_scope
from backend.app.schemas.inspection import HourBucket, InspectionPage, InspectionResult, MetricsSummary
from backend.app.services.image_validation import sanitize_filename, validate_image
from backend.app.services.model_service import ModelService
from backend.app.services.storage import ARTIFACT_NAMES, ResultStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- dependencies
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model(request: Request) -> ModelService:
    return request.app.state.model_service


def get_storage(request: Request) -> ResultStorage:
    return request.app.state.storage


def get_db(request: Request):  # noqa: ANN201 - generator dependency
    yield from session_scope(request.app.state.session_factory)


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise APIError(401, "UNAUTHORIZED", "Missing or invalid X-API-Key header")


DB = Annotated[Session, Depends(get_db)]
Model = Annotated[ModelService, Depends(get_model)]

public = APIRouter(prefix="/api/v1", tags=["health"])
router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


# ---------------------------------------------------------------- health
@public.get("/health")
def health(request: Request, model: Model) -> dict:
    """Liveness: the process is up. Says nothing about whether inspections can be served."""
    return {"status": "ok", "version": request.app.state.settings.app_version, "model": model.status()}


@public.get("/health/ready")
def ready(model: Model) -> JSONResponse:
    """Readiness: 200 only when a verified model is loaded; 503 otherwise (including demo mode)."""
    status = model.status()
    return JSONResponse({"ready": status["ready"], "model": status}, status_code=200 if status["ready"] else 503)


# ---------------------------------------------------------------- models
@router.get("/models", tags=["models"])
def models(model: Model) -> dict:
    status = model.status()
    items = []
    if model.predictor is not None:
        m = model.predictor.metadata
        cfg = m.get("config", {})
        items.append({
            "model_name": m.get("model_name"),
            "model_version": m.get("model_version"),
            "created_at": m.get("created_at"),
            "active": True,
            "backbone": cfg.get("model", {}).get("backbone"),
            "layers": cfg.get("model", {}).get("layers"),
            "coreset_ratio": cfg.get("model", {}).get("coreset_ratio"),
            "image_size": cfg.get("preprocessing", {}).get("image_size"),
            "dataset": {k: m.get("dataset", {}).get(k) for k in ("name", "category", "fingerprint_sha256", "counts")},
            "thresholds": m.get("thresholds"),
            "calibration": m.get("calibration"),
            "evaluation": m.get("evaluation", {"status": "NOT_RUN"}),
            "checksums": m.get("checksums"),
        })
    return {"state": status, "items": items}


# ---------------------------------------------------------------- inspections
@router.post("/inspections", status_code=201, response_model=InspectionResult, tags=["inspections"])
def create_inspection(
    db: DB,
    model: Model,
    settings: Annotated[Settings, Depends(get_settings)],
    storage: Annotated[ResultStorage, Depends(get_storage)],
    file: Annotated[UploadFile, File(description="PNG, JPEG or BMP image")],
) -> InspectionResult:
    # Fail fast before reading the body when no model can serve the request.
    model.require_ready()
    data = file.file.read(settings.max_upload_bytes + 1)
    img = validate_image(data, max_bytes=settings.max_upload_bytes, max_pixels=settings.max_image_pixels,
                         allowed_formats=settings.allowed_formats)
    out = model.inspect(img.rgb)
    meta = model.predictor.metadata

    row = Inspection(
        id=uuid.uuid4(), filename=sanitize_filename(file.filename), content_sha256=img.sha256,
        image_format=img.format, image_width=img.width, image_height=img.height,
        model_name=meta.get("model_name", "unknown"), model_version=meta.get("model_version", "unknown"),
        category=meta.get("dataset", {}).get("category", "unknown"),
        anomaly_score=out.score, image_threshold=out.threshold,
        prediction="abnormal" if out.is_anomalous else "normal",
        defect_area_ratio=out.defect_area_ratio, inference_time_ms=out.inference_time_ms,
    )
    images = {"original": img.rgb, "heatmap": out.heatmap, "overlay": out.overlay}
    if out.mask is not None:
        images["mask"] = out.mask
    storage.save(row.id, images)
    try:
        db.add(row)
        db.flush()  # surface DB errors here so stored files can be removed (commit happens in get_db)
    except Exception:
        storage.delete(row.id)
        raise
    log_fields(logger, logging.INFO, "Inspection stored", inspection_id=str(row.id), sha256=img.sha256[:12],
               size_bytes=img.size_bytes, prediction=row.prediction, score=round(out.score, 5),
               inference_ms=round(out.inference_time_ms, 1))
    return InspectionResult.model_validate(row).with_urls(out.mask is not None)


@router.get("/inspections", response_model=InspectionPage, tags=["inspections"])
def list_inspections(
    db: DB,
    storage: Annotated[ResultStorage, Depends(get_storage)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    prediction: Literal["normal", "abnormal"] | None = None,
    model_version: Annotated[str | None, Query(max_length=64)] = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> InspectionPage:
    conds = []
    if prediction:
        conds.append(Inspection.prediction == prediction)
    if model_version:
        conds.append(Inspection.model_version == model_version)
    if since:
        conds.append(Inspection.created_at >= since)
    if until:
        conds.append(Inspection.created_at < until)
    total = db.scalar(select(func.count()).select_from(Inspection).where(*conds)) or 0
    rows = db.scalars(
        select(Inspection).where(*conds).order_by(Inspection.created_at.desc())
        .offset((page - 1) * page_size).limit(page_size)
    ).all()
    items = [InspectionResult.model_validate(r).with_urls(storage.path(r.id, "mask").is_file()) for r in rows]
    return InspectionPage(items=items, total=total, page=page, page_size=page_size)


def _get_row(db: Session, inspection_id: uuid.UUID) -> Inspection:
    row = db.get(Inspection, inspection_id)
    if row is None:
        raise APIError(404, "INSPECTION_NOT_FOUND", f"No inspection with id {inspection_id}")
    return row


@router.get("/inspections/{inspection_id}", response_model=InspectionResult, tags=["inspections"])
def get_inspection(
    inspection_id: uuid.UUID, db: DB, storage: Annotated[ResultStorage, Depends(get_storage)]
) -> InspectionResult:
    row = _get_row(db, inspection_id)
    return InspectionResult.model_validate(row).with_urls(storage.path(row.id, "mask").is_file())


@router.get("/inspections/{inspection_id}/{artifact}", tags=["inspections"],
            responses={200: {"content": {"image/png": {}}}})
def get_inspection_image(
    inspection_id: uuid.UUID,
    artifact: Literal["original", "heatmap", "overlay", "mask"],
    db: DB,
    storage: Annotated[ResultStorage, Depends(get_storage)],
) -> FileResponse:
    _get_row(db, inspection_id)
    assert artifact in ARTIFACT_NAMES
    path = storage.path(inspection_id, artifact)
    if not path.is_file():
        raise APIError(404, "IMAGE_NOT_FOUND", f"No {artifact} image stored for this inspection")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# ---------------------------------------------------------------- metrics
@router.get("/metrics/summary", response_model=MetricsSummary, tags=["metrics"])
def metrics_summary(db: DB, model: Model) -> MetricsSummary:
    counts = dict(db.execute(select(Inspection.prediction, func.count()).group_by(Inspection.prediction)).all())
    normal, abnormal = counts.get("normal", 0), counts.get("abnormal", 0)
    total = normal + abnormal

    recent = db.scalars(select(Inspection.inference_time_ms).order_by(Inspection.created_at.desc()).limit(1000)).all()
    lat = np.asarray(recent, dtype=np.float64)
    latency = {
        "n": float(lat.size),
        "mean": float(lat.mean()) if lat.size else None,
        "p50": float(np.percentile(lat, 50)) if lat.size else None,
        "p95": float(np.percentile(lat, 95)) if lat.size else None,
    }

    now = datetime.now(UTC)
    start = (now - timedelta(hours=23)).replace(minute=0, second=0, microsecond=0)
    rows = db.execute(select(Inspection.created_at, Inspection.prediction).where(Inspection.created_at >= start)).all()
    tot, ab = Counter(), Counter()
    for created, pred in rows:
        created = created if created.tzinfo else created.replace(tzinfo=UTC)
        key = created.replace(minute=0, second=0, microsecond=0)
        tot[key] += 1
        ab[key] += pred == "abnormal"
    hourly = []
    for i in range(24):
        h = start + timedelta(hours=i)
        hourly.append(HourBucket(hour=h, total=tot[h], abnormal=ab[h],
                                 abnormal_rate=ab[h] / tot[h] if tot[h] else 0.0))

    by_version = dict(
        db.execute(select(Inspection.model_version, func.count()).group_by(Inspection.model_version)).all()
    )
    return MetricsSummary(total=total, normal=normal, abnormal=abnormal,
                          abnormal_rate=abnormal / total if total else None, latency_ms=latency,
                          hourly_last_24h=hourly, by_model_version=by_version, model=model.status())
