from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InspectionResult(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    inspection_id: uuid.UUID = Field(validation_alias="id")
    created_at: datetime
    filename: str
    image_width: int
    image_height: int
    model_name: str
    model_version: str
    category: str
    anomaly_score: float = Field(description="Raw PatchCore distance. Not a probability; compare with threshold.")
    threshold: float = Field(validation_alias="image_threshold")
    prediction: Literal["normal", "abnormal"]
    defect_area_ratio: float | None = Field(
        None, description="Share of pixels above the pixel threshold. A model estimate, not a measured defect."
    )
    inference_time_ms: float
    original_url: str = ""
    heatmap_url: str = ""
    overlay_url: str = ""
    mask_url: str | None = None

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=UTC)  # SQLite returns naive datetimes

    def with_urls(self, has_mask: bool) -> InspectionResult:
        base = f"/api/v1/inspections/{self.inspection_id}"
        return self.model_copy(update={
            "original_url": f"{base}/original", "heatmap_url": f"{base}/heatmap",
            "overlay_url": f"{base}/overlay", "mask_url": f"{base}/mask" if has_mask else None,
        })


class InspectionPage(BaseModel):
    items: list[InspectionResult]
    total: int
    page: int
    page_size: int


class HourBucket(BaseModel):
    hour: datetime
    total: int
    abnormal: int
    abnormal_rate: float


class MetricsSummary(BaseModel):
    total: int
    normal: int
    abnormal: int
    abnormal_rate: float | None
    latency_ms: dict[str, float | None]
    hourly_last_24h: list[HourBucket]
    by_model_version: dict[str, int]
    model: dict[str, Any]
