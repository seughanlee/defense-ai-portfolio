"""API with a real PatchCore artifact (random backbone, synthetic data): wiring check, not a quality check."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.tests.conftest import make_settings
from ml.datasets.synthetic import SYNTHETIC_CATEGORY, make_synthetic_dataset
from ml.training.train_patchcore import train
from tests.conftest import DEFAULT_CONFIG, SMOKE_OVERRIDES

pytestmark = pytest.mark.slow


def test_real_artifact_served_end_to_end(tmp_path):
    root = tmp_path / "data"
    make_synthetic_dataset(root, n_train=10, n_test_good=2, n_test_defect=2, size=64, seed=0)
    artifact = train(root, SYNTHETIC_CATEGORY, DEFAULT_CONFIG, tmp_path / "models", "api-1", SMOKE_OVERRIDES)

    app = create_app(make_settings(tmp_path, model_artifact_dir=artifact), load_model=False)
    app.state.model_service.start_loading(background=False)
    with TestClient(app) as c:
        assert c.get("/api/v1/health/ready").status_code == 200
        img_path = next((root / SYNTHETIC_CATEGORY / "test" / "spot").glob("*.png"))
        r = c.post("/api/v1/inspections", files={"file": ("spot.png", img_path.read_bytes(), "image/png")})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["model_version"] == "api-1" and body["category"] == SYNTHETIC_CATEGORY
        assert body["prediction"] == ("abnormal" if body["anomaly_score"] > body["threshold"] else "normal")
        heat = Image.open(io.BytesIO(c.get(body["heatmap_url"]).content))
        assert heat.size == (64, 64)  # rendered back at the uploaded image's resolution
        models = c.get("/api/v1/models").json()["items"][0]
        assert models["checksums"]["memory_bank_sha256"] and models["dataset"]["category"] == SYNTHETIC_CATEGORY
