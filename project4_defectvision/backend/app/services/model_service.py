"""Model lifecycle for the API process.

States: not_configured (no artifact set -> dashboard shows Demo Mode), loading, ready, failed.
Only ``ready`` serves predictions. The API never produces a score without a loaded, verified artifact.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from backend.app.core.errors import APIError
from backend.app.core.logging import log_fields

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class InspectionOutput:
    score: float
    threshold: float
    is_anomalous: bool
    defect_area_ratio: float | None
    inference_time_ms: float
    heatmap: np.ndarray  # RGB uint8, at the original image size
    overlay: np.ndarray
    mask: np.ndarray | None  # uint8 0/255, original size


class Predictor(Protocol):
    """What the API needs from a model; lets tests swap in a fake without torch."""

    metadata: dict[str, Any]

    def inspect(self, rgb: np.ndarray) -> InspectionOutput: ...


class PatchCorePredictor:
    def __init__(self, artifact_dir: Path, device: str | None) -> None:
        from ml.inference.patchcore import PatchCoreDetector

        self.detector, self.metadata = PatchCoreDetector.load(artifact_dir, device=device)
        if self.detector.image_threshold is None:
            raise ValueError("Artifact has no calibrated image threshold")

    def inspect(self, rgb: np.ndarray) -> InspectionOutput:
        from ml.visualization.overlay import colorize, model_map_range, normalize_map, overlay

        det = self.detector
        pred = det.predict(rgb)
        h, w = rgb.shape[:2]
        vmin, vmax = model_map_range(self.metadata.get("calibration", {}), det.pixel_threshold,
                                     float(pred.anomaly_map.max()) or 1.0)
        norm = cv2.resize(normalize_map(pred.anomaly_map, vmin, vmax), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = None
        if pred.defect_mask is not None:
            mask = cv2.resize(pred.defect_mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        return InspectionOutput(
            score=pred.score,
            threshold=float(det.image_threshold),
            is_anomalous=pred.is_anomalous,
            defect_area_ratio=None if pred.defect_mask is None else float(pred.defect_mask.mean()),
            inference_time_ms=pred.inference_time_ms,
            heatmap=colorize(norm),
            overlay=overlay(rgb, norm),
            mask=mask,
        )


@dataclass
class ModelService:
    artifact_dir: Path | None
    device: str | None = None
    state: ModelState = ModelState.NOT_CONFIGURED
    error: str | None = None
    load_seconds: float | None = None
    predictor: Predictor | None = None
    _infer_lock: threading.Lock = field(default_factory=threading.Lock)

    def start_loading(self, background: bool = True) -> None:
        if self.artifact_dir is None:
            self.state = ModelState.NOT_CONFIGURED
            log_fields(logger, logging.WARNING, "No model artifact configured: running in demo mode")
            return
        self.state = ModelState.LOADING
        if background:
            threading.Thread(target=self._load, name="model-loader", daemon=True).start()
        else:
            self._load()

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            predictor = PatchCorePredictor(self.artifact_dir, self.device)
        except Exception as exc:  # any failure leaves the API up but not ready
            self.state, self.error = ModelState.FAILED, f"{type(exc).__name__}: {exc}"
            logger.exception("Model loading failed")
            return
        self.predictor, self.load_seconds = predictor, round(time.perf_counter() - t0, 2)
        self.state = ModelState.READY
        meta = predictor.metadata
        log_fields(logger, logging.INFO, "Model ready", model_version=meta.get("model_version"),
                   load_seconds=self.load_seconds)

    def set_predictor(self, predictor: Predictor) -> None:
        """Install an already-built predictor (used by tests)."""
        self.predictor, self.state, self.error = predictor, ModelState.READY, None

    def status(self) -> dict[str, Any]:
        meta = self.predictor.metadata if self.predictor else {}
        return {
            "state": self.state.value,
            "ready": self.state is ModelState.READY,
            "demo_mode": self.state is not ModelState.READY,
            "error": self.error,
            "model_name": meta.get("model_name"),
            "model_version": meta.get("model_version"),
            "category": meta.get("dataset", {}).get("category"),
            "load_seconds": self.load_seconds,
        }

    def require_ready(self) -> None:
        if self.state is not ModelState.READY or self.predictor is None:
            raise APIError(503, "MODEL_NOT_READY", f"Model is not ready (state: {self.state.value})",
                           details={"state": self.state.value})

    def inspect(self, rgb: np.ndarray) -> InspectionOutput:
        self.require_ready()
        # torch inference is CPU-bound and the detector is not designed for concurrent calls: serialise.
        with self._infer_lock:
            return self.predictor.inspect(rgb)
