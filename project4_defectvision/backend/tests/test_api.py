import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.image_validation import sanitize_filename
from backend.app.services.model_service import ModelService, ModelState
from backend.tests.conftest import FakePredictor, make_settings, png_bytes


def upload(client, data: bytes, name: str = "part.png", ctype: str = "image/png"):
    return client.post("/api/v1/inspections", files={"file": (name, data, ctype)})


# ---------------------------------------------------------------- health / readiness / demo mode
def test_health_is_up_but_not_ready_without_model(demo_client):
    h = demo_client.get("/api/v1/health").json()
    assert h["status"] == "ok"
    assert h["model"]["state"] == "not_configured" and h["model"]["demo_mode"] is True
    r = demo_client.get("/api/v1/health/ready")
    assert r.status_code == 503 and r.json()["ready"] is False


def test_demo_mode_refuses_inspection_instead_of_inventing_scores(demo_client):
    r = upload(demo_client, png_bytes())
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_READY"
    assert demo_client.get("/api/v1/inspections").json()["total"] == 0
    assert demo_client.get("/api/v1/models").json()["items"] == []


def test_ready_when_model_loaded(client):
    assert client.get("/api/v1/health/ready").status_code == 200
    m = client.get("/api/v1/models").json()
    assert m["items"][0]["model_version"] == "fake-1" and m["state"]["ready"]


def test_model_loading_failure_is_reported(tmp_path):
    svc = ModelService(tmp_path / "missing_artifact")
    svc.start_loading(background=False)
    assert svc.state is ModelState.FAILED and "ArtifactError" in svc.error
    app = create_app(make_settings(tmp_path), load_model=False)
    app.state.model_service = svc
    with TestClient(app) as c:
        r = c.get("/api/v1/health/ready")
        assert r.status_code == 503 and r.json()["model"]["state"] == "failed"


# ---------------------------------------------------------------- inspection happy path
def test_inspection_roundtrip(client):
    r = upload(client, png_bytes(200, size=(40, 30)), name="nut_001.png")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["prediction"] == "abnormal" and body["anomaly_score"] == pytest.approx(200 / 255)
    assert body["threshold"] == 0.5 and body["model_version"] == "fake-1"
    assert body["image_width"] == 40 and body["image_height"] == 30 and body["filename"] == "nut_001.png"
    iid = body["inspection_id"]

    assert client.get(f"/api/v1/inspections/{iid}").json()["anomaly_score"] == body["anomaly_score"]
    for art in ("original", "heatmap", "overlay", "mask"):
        img = client.get(body[f"{art}_url"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(img.content)).size == (40, 30)


def test_normal_prediction_below_threshold(client):
    assert upload(client, png_bytes(20)).json()["prediction"] == "normal"


@pytest.mark.parametrize("fmt,mode", [("JPEG", "RGB"), ("BMP", "RGB"), ("PNG", "L")])
def test_accepted_formats_including_grayscale(client, fmt, mode):
    r = upload(client, png_bytes(50, fmt=fmt, mode=mode), name=f"x.{fmt.lower()}")
    assert r.status_code == 201, r.text


def test_list_pagination_and_filters(client):
    for v in (10, 20, 200, 210, 220):
        assert upload(client, png_bytes(v)).status_code == 201
    page = client.get("/api/v1/inspections", params={"page_size": 2}).json()
    assert page["total"] == 5 and len(page["items"]) == 2 and page["page"] == 1
    last = client.get("/api/v1/inspections", params={"page_size": 2, "page": 3}).json()
    assert len(last["items"]) == 1
    ab = client.get("/api/v1/inspections", params={"prediction": "abnormal"}).json()
    assert ab["total"] == 3 and all(i["prediction"] == "abnormal" for i in ab["items"])
    assert client.get("/api/v1/inspections", params={"model_version": "other"}).json()["total"] == 0
    assert client.get("/api/v1/inspections", params={"page_size": 1000}).status_code == 422


def test_metrics_summary(client):
    for v in (10, 200, 220):
        upload(client, png_bytes(v))
    s = client.get("/api/v1/metrics/summary").json()
    assert (s["total"], s["normal"], s["abnormal"]) == (3, 1, 2)
    assert s["abnormal_rate"] == pytest.approx(2 / 3)
    assert s["latency_ms"]["p95"] == pytest.approx(1.5)
    assert len(s["hourly_last_24h"]) == 24 and sum(b["total"] for b in s["hourly_last_24h"]) == 3
    assert s["by_model_version"] == {"fake-1": 3}


def test_metrics_summary_empty(client):
    s = client.get("/api/v1/metrics/summary").json()
    assert s["total"] == 0 and s["abnormal_rate"] is None and s["latency_ms"]["p95"] is None


def test_results_persist_across_restart(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, load_model=False)
    app.state.model_service.set_predictor(FakePredictor())
    with TestClient(app) as c:
        iid = upload(c, png_bytes()).json()["inspection_id"]
    with TestClient(create_app(settings, load_model=False)) as c:  # new process-equivalent, no model
        assert c.get(f"/api/v1/inspections/{iid}").status_code == 200
        assert c.get(f"/api/v1/inspections/{iid}/heatmap").status_code == 200


# ---------------------------------------------------------------- invalid uploads
def test_empty_file(client):
    r = upload(client, b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "EMPTY_FILE"


def test_non_image_with_image_extension_and_mime(client):
    r = upload(client, b"MZ\x90\x00 definitely not an image", name="evil.png", ctype="image/png")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_IMAGE"


def test_truncated_png(client):
    r = upload(client, png_bytes(size=(64, 64))[:60])
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_IMAGE"


def test_disallowed_format(client):
    r = upload(client, png_bytes(fmt="GIF", mode="L"), name="a.gif", ctype="image/gif")
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_IMAGE_FORMAT"


def test_missing_file_field(client):
    r = client.post("/api/v1/inspections", data={"x": "1"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_upload_size_limit(tmp_path):
    settings = make_settings(tmp_path, max_upload_mb=0.01)  # ~10 KB
    app = create_app(settings, load_model=False)
    app.state.model_service.set_predictor(FakePredictor())
    big = Image.frombytes("RGB", (200, 200), bytes((i * 7) % 256 for i in range(200 * 200 * 3)))
    buf = io.BytesIO()
    big.save(buf, format="BMP")  # uncompressed, ~120 KB
    with TestClient(app) as c:
        r = upload(c, buf.getvalue(), name="big.bmp")
        assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_pixel_limit_guards_decompression_bombs(tmp_path):
    app = create_app(make_settings(tmp_path, max_image_pixels=1000), load_model=False)
    app.state.model_service.set_predictor(FakePredictor())
    with TestClient(app) as c:
        r = upload(c, png_bytes(size=(100, 100)))  # tiny file, 10 000 pixels
        assert r.status_code == 413 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


# ---------------------------------------------------------------- lookup errors and path safety
def test_unknown_and_malformed_ids(client):
    r = client.get(f"/api/v1/inspections/{uuid.uuid4()}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "INSPECTION_NOT_FOUND"
    assert client.get("/api/v1/inspections/not-a-uuid").status_code == 422


@pytest.mark.parametrize("path", [
    "/api/v1/inspections/..%2F..%2Fsecrets/heatmap",
    "/api/v1/inspections/{iid}/..%2F..%2Ftest.db",
    "/api/v1/inspections/{iid}/metadata",
])
def test_path_traversal_is_rejected(client, path):
    iid = upload(client, png_bytes()).json()["inspection_id"]
    r = client.get(path.format(iid=iid))
    assert r.status_code in (404, 422) and r.headers["content-type"].startswith("application/json")


def test_filename_is_sanitised_for_display():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\temp\\a\x00b.png") == "ab.png"
    assert sanitize_filename("") == "upload" and len(sanitize_filename("a" * 1000)) == 255


# ---------------------------------------------------------------- auth, transactions, settings, logging
def test_api_key_required_when_configured(tmp_path):
    app = create_app(make_settings(tmp_path, api_key="k" * 20), load_model=False)
    app.state.model_service.set_predictor(FakePredictor())
    with TestClient(app) as c:
        assert c.get("/api/v1/health").status_code == 200  # health stays public for orchestrators
        r = c.get("/api/v1/inspections")
        assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
        assert c.get("/api/v1/inspections", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/v1/inspections", headers={"X-API-Key": "k" * 20}).status_code == 200


def test_db_failure_rolls_back_and_removes_stored_images(settings, monkeypatch, caplog):
    app = create_app(settings, load_model=False)
    app.state.model_service.set_predictor(FakePredictor())

    def boom(self, *a, **k):
        raise RuntimeError("db down")

    with TestClient(app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(Session, "flush", boom)
        r = upload(c, png_bytes(), name="x.png")
        monkeypatch.undo()
        assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR"
        assert "db down" not in r.text  # internals are not leaked to the client
        rid = r.json()["error"]["request_id"]
        assert rid != "-" and r.headers["X-Request-ID"] == rid  # the 500 can be traced in the logs
        assert c.get("/api/v1/inspections").json()["total"] == 0
    assert not any(settings.storage_dir.iterdir())


def test_production_settings_are_guarded():
    with pytest.raises(ValueError, match="DV_API_KEY"):
        Settings(env="prod", database_url="postgresql+psycopg://u:p@db/x")
    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings(env="prod", api_key="x" * 20)
    ok = Settings(env="prod", api_key="x" * 20, database_url="postgresql+psycopg://u:p@db/x")
    assert ok.env == "prod"


def test_request_id_echoed_and_upload_details_not_logged(client, caplog):
    caplog.set_level("INFO")
    r = client.post("/api/v1/inspections", files={"file": ("secret-name.png", png_bytes(), "image/png")},
                    headers={"X-Request-ID": "abc123"})
    assert r.status_code == 201 and r.headers["X-Request-ID"] == "abc123"
    text = " ".join(f"{rec.getMessage()} {getattr(rec, 'fields', '')}" for rec in caplog.records)
    assert "secret-name" not in text
    assert any(getattr(rec, "fields", {}).get("sha256") for rec in caplog.records)


def test_openapi_schema_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/health", "/api/v1/models", "/api/v1/inspections", "/api/v1/inspections/{inspection_id}",
              "/api/v1/metrics/summary"):
        assert p in paths
