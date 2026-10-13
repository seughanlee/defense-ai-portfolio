from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.model_service import InspectionOutput


class FakePredictor:
    """Deterministic stand-in: score = mean brightness / 255, threshold 0.5. No torch involved."""

    metadata = {
        "model_name": "patchcore", "model_version": "fake-1", "created_at": "2026-10-13T00:00:00+00:00",
        "dataset": {"name": "synthetic", "category": "unit_test"},
        "config": {"model": {"backbone": "none", "layers": [], "coreset_ratio": 0.1},
                   "preprocessing": {"image_size": 32}},
        "thresholds": {"image": 0.5, "pixel": 0.5}, "calibration": {}, "evaluation": {"status": "NOT_RUN"},
        "checksums": {},
    }

    def inspect(self, rgb: np.ndarray) -> InspectionOutput:
        score = float(rgb.mean() / 255)
        h, w = rgb.shape[:2]
        return InspectionOutput(
            score=score, threshold=0.5, is_anomalous=score > 0.5, defect_area_ratio=0.0, inference_time_ms=1.5,
            heatmap=np.zeros((h, w, 3), np.uint8), overlay=rgb.copy(), mask=np.zeros((h, w), np.uint8),
        )


def png_bytes(value: int = 100, size: tuple[int, int] = (40, 30), fmt: str = "PNG", mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (value,) * len(mode) if mode != "L" else value).save(buf, format=fmt)
    return buf.getvalue()


def make_settings(tmp_path: Path, **kw) -> Settings:
    return Settings(env="test", database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                    storage_dir=tmp_path / "storage", log_level="WARNING", **kw)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def demo_client(settings: Settings):
    with TestClient(create_app(settings, load_model=False)) as c:
        yield c


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings, load_model=False)
    app.state.model_service.set_predictor(FakePredictor())
    with TestClient(app) as c:
        yield c
