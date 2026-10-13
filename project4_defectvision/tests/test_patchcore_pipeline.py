"""PatchCore wrapper and end-to-end pipeline on synthetic data with a random (offline) backbone.

These tests check shapes, persistence and safety behaviour only. Scores from a random backbone on
synthetic images say nothing about detection quality.
"""

import json
import shutil

import numpy as np
import pytest

from ml.datasets.mvtec import official_test_split, scan_category, split_train_val
from ml.evaluation.evaluate import evaluate
from ml.inference.patchcore import (
    MEMORY_BANK_FILE,
    METADATA_FILE,
    ArtifactError,
    NotFittedError,
    PatchCoreDetector,
)
from ml.training.train_patchcore import train
from tests.conftest import DEFAULT_CONFIG, SMOKE_OVERRIDES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def fitted(smoke_config, synthetic_root, category):
    samples = scan_category(synthetic_root, category)
    fit, val = split_train_val(samples, smoke_config.data.val_fraction, smoke_config.data.seed)
    det = PatchCoreDetector(smoke_config, seed=0)
    hash_before = det.backbone_sha256()
    det.fit([s.image_path for s in fit])
    det.calibrate([s.image_path for s in val])
    return det, samples, hash_before


@pytest.fixture(scope="session")
def saved_artifact(fitted, tmp_path_factory):
    det, _, _ = fitted
    out = tmp_path_factory.mktemp("artifact") / "v-test"
    det.save(out, model_version="v-test", dataset={"name": "synthetic", "category": "synthetic_part"})
    return out


def test_unfitted_model_refuses_to_predict(smoke_config):
    det = PatchCoreDetector(smoke_config)
    with pytest.raises(NotFittedError):
        det.predict(np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError):
        det.fit([])


def test_fit_builds_memory_bank_and_leaves_backbone_untouched(fitted):
    det, _, hash_before = fitted
    assert det.is_fitted and det.model.memory_bank.ndim == 2
    assert det.backbone_sha256() == hash_before  # BatchNorm statistics did not drift during fit


def test_output_shapes(fitted):
    det, samples, _ = fitted
    tests = official_test_split(samples)
    scores, maps = det.score([s.image_path for s in tests])
    size = det.cfg.preprocessing.image_size
    assert scores.shape == (len(tests),) and np.isfinite(scores).all()
    assert maps.shape == (len(tests), size, size)  # heatmap matches the model input resolution


def test_prediction_respects_threshold(fitted):
    det, samples, _ = fitted
    p = det.predict(samples[-1].image_path)
    assert det.image_threshold is not None and det.pixel_threshold is not None
    assert p.is_anomalous == (p.score > det.image_threshold)
    assert p.defect_mask.dtype == bool and p.defect_mask.shape == p.anomaly_map.shape
    assert p.inference_time_ms > 0


def test_save_load_roundtrip_gives_identical_scores(fitted, saved_artifact):
    det, samples, _ = fitted
    loaded, meta = PatchCoreDetector.load(saved_artifact)
    paths = [s.image_path for s in official_test_split(samples)]
    np.testing.assert_allclose(loaded.score(paths)[0], det.score(paths)[0], rtol=1e-5)
    assert meta["model_version"] == "v-test" and meta["evaluation"]["status"] == "NOT_RUN"
    assert loaded.image_threshold == det.image_threshold


def test_save_refuses_overwrite(fitted, saved_artifact):
    det, _, _ = fitted
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        det.save(saved_artifact, model_version="v-test", dataset={})


def test_missing_checkpoint_file(saved_artifact, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(saved_artifact, broken)
    (broken / MEMORY_BANK_FILE).unlink()
    with pytest.raises(ArtifactError, match="Missing artifact file"):
        PatchCoreDetector.load(broken)
    with pytest.raises(ArtifactError):
        PatchCoreDetector.load(tmp_path / "does_not_exist")


def test_tampered_memory_bank_rejected(saved_artifact, tmp_path):
    bad = tmp_path / "tampered"
    shutil.copytree(saved_artifact, bad)
    with open(bad / MEMORY_BANK_FILE, "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\x00\x01\x02\x03")
    with pytest.raises(ArtifactError, match="checksum"):
        PatchCoreDetector.load(bad)


def test_backbone_mismatch_rejected(saved_artifact, tmp_path):
    bad = tmp_path / "other_backbone"
    shutil.copytree(saved_artifact, bad)
    meta = json.loads((bad / METADATA_FILE).read_text())
    meta["checksums"]["backbone_sha256"] = "0" * 64
    (bad / METADATA_FILE).write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="Backbone weights differ"):
        PatchCoreDetector.load(bad)


def test_invalid_metadata_rejected(saved_artifact, tmp_path):
    bad = tmp_path / "bad_json"
    shutil.copytree(saved_artifact, bad)
    (bad / METADATA_FILE).write_text("{not json")
    with pytest.raises(ArtifactError, match="Invalid metadata"):
        PatchCoreDetector.load(bad)


def test_train_and_evaluate_end_to_end(synthetic_root, category, tmp_path):
    out = train(synthetic_root, category, DEFAULT_CONFIG, tmp_path / "models", "smoke", SMOKE_OVERRIDES)
    meta = json.loads((out / METADATA_FILE).read_text())
    assert meta["dataset"]["name"] == "synthetic"
    assert len(meta["dataset"]["split"]["val_files"]) > 0
    # held-out validation images were not used for the memory bank
    assert not set(meta["dataset"]["split"]["fit_files"]) & set(meta["dataset"]["split"]["val_files"])

    result = evaluate(out, synthetic_root, n_latency=3, n_panels=2)
    assert result["status"] == "SYNTHETIC_SMOKE_TEST" and result["warning"]
    for name in ("evaluation.json", "per_image.csv", "report.md"):
        assert (out / "evaluation" / name).is_file()
    assert len(list((out / "evaluation" / "panels").glob("*.png"))) == 2
    assert result["operational"]["latency_batch1"]["n"] == 3
    assert json.loads((out / METADATA_FILE).read_text())["evaluation"]["status"] == "SYNTHETIC_SMOKE_TEST"
