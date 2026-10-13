from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Inspection(Base):
    """One inspected image. Image files live in storage under ``<id>/``; only metadata is stored here."""

    __tablename__ = "inspections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    filename: Mapped[str] = mapped_column(String(255))  # sanitised, display only — never used as a path
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    image_format: Mapped[str] = mapped_column(String(16))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)

    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(64))

    anomaly_score: Mapped[float] = mapped_column(Float)
    image_threshold: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String(16), index=True)  # "normal" | "abnormal"
    defect_area_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float] = mapped_column(Float)
