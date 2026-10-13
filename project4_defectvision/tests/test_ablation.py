"""Ablation support: seeded memory-bank sampling, validation-proxy anomalies, resource probes, runner."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.ablation.run_ablation import planned_runs
from ml.ablation.run_one import load_plan, run_overrides, stage_eval, stage_fit
from ml.ablation.summarize import summarize
from ml.config import load_config
from ml.datasets.mvtec import scan_category, split_train_val
from ml.datasets.synthetic_anomalies import KINDS, foreground_mask, make_validation_anomalies
from ml.evaluation.resources import current_rss_mb, peak_rss_mb
from ml.inference.patchcore import PatchCoreDetector
from tests.conftest import DEFAULT_CONFIG, SMOKE_OVERRIDES


def _img(seed=0, size=96):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 30, np.uint8)
    img[20:76, 20:76] = rng.integers(120, 200, (56, 56, 3), dtype=np.uint8)  # an "object" on dark background
    return img


# ---------------------------------------------------------------- synthetic validation anomalies
def test_synthetic_anomalies_are_deterministic_and_cover_all_kinds():
    imgs = [_img(i) for i in range(6)]
    a, b = make_validation_anomalies(imgs, seed=3), make_validation_anomalies(imgs, seed=3)
    assert [x.kind for x in a] == list(KINDS) * 2
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.image, y.image)
        np.testing.assert_array_equal(x.mask, y.mask)
    c = make_validation_anomalies(imgs, seed=4)
    assert any(not np.array_equal(x.mask, y.mask) for x, y in zip(a, c, strict=True))


def test_synthetic_anomaly_changes_pixels_only_inside_mask():
    imgs = [_img(i) for i in range(3)]
    for orig, s in zip(imgs, make_validation_anomalies(imgs, seed=0), strict=True):
        assert s.image.shape == orig.shape and s.image.dtype == np.uint8
        assert s.mask.any() and s.mask.mean() < 0.2
        np.testing.assert_array_equal(s.image[~s.mask], orig[~s.mask])
        assert not np.array_equal(s.image, orig)


def test_input_images_are_not_modified():
    imgs = [_img(0)]
    before = imgs[0].copy()
    make_validation_anomalies(imgs, seed=0)
    np.testing.assert_array_equal(imgs[0], before)


def test_foreground_mask_finds_object_and_falls_back_on_texture():
    fg = foreground_mask(_img(0))
    assert fg[48, 48] and not fg[2, 2]
    texture = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert foreground_mask(texture).all()  # no single background colour -> whole image allowed


# ---------------------------------------------------------------- resources
def test_memory_probes():
    assert peak_rss_mb() >= current_rss_mb() > 0


# ---------------------------------------------------------------- config + seeded sampling
def test_sampling_method_validated():
    with pytest.raises(ValueError, match="sampling"):
        load_config(DEFAULT_CONFIG, {**SMOKE_OVERRIDES, "model": {"sampling": "farthest"}})


def _fit(synthetic_root, category, **model):
    cfg = load_config(DEFAULT_CONFIG, {**SMOKE_OVERRIDES, "model": {**SMOKE_OVERRIDES["model"], **model}})
    fit, _ = split_train_val(scan_category(synthetic_root, category), 0.2, 42)
    det = PatchCoreDetector(cfg, seed=0)
    stats = det.fit([s.image_path for s in fit])
    return det.model.memory_bank.clone(), stats


@pytest.mark.slow
@pytest.mark.parametrize("sampling", ["kcenter_greedy", "random"])
def test_sampling_is_reproducible_for_a_seed(synthetic_root, category, sampling):
    a, stats = _fit(synthetic_root, category, sampling=sampling, sampling_seed=5)
    b, _ = _fit(synthetic_root, category, sampling=sampling, sampling_seed=5)
    c, _ = _fit(synthetic_root, category, sampling=sampling, sampling_seed=6)
    assert torch_equal(a, b)
    assert not torch_equal(a, c)
    assert a.shape[0] == int(stats["n_patches"] * 0.1)


@pytest.mark.slow
def test_full_bank_keeps_every_patch(synthetic_root, category):
    bank, stats = _fit(synthetic_root, category, coreset_ratio=1.0)
    assert bank.shape[0] == stats["n_patches"]


def torch_equal(a, b) -> bool:
    return a.shape == b.shape and bool((a == b).all())


# ---------------------------------------------------------------- plan + runner
@pytest.fixture
def tiny_plan(tmp_path, category):
    plan = {
        "base_config": str(DEFAULT_CONFIG),
        "baseline": {"model": {"pre_trained": False, "layers": ["layer2", "layer3"]},
                     "preprocessing": {"image_size": 64}, "runtime": {"batch_size": 4, "num_threads": 2}},
        "categories": [category],
        "validation_proxy": {"anomaly_seed": 1, "aupro_fpr_limit": 0.3},
        "latency": {"n_images": 2},
        "selection": {"metric": "val_image_auroc", "tolerance": 0.01, "tie_break": "latency_p95_ms"},
        "experiments": [
            {"id": "B0", "factor": "baseline", "seeds": [0, 1], "overrides": {}},
            {"id": "L-3", "factor": "layers", "overrides": {"model": {"layers": ["layer3"]}}},
            {"id": "X", "factor": "layers", "overrides": {}, "not_run": "excluded on purpose"},
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    return load_plan(path)


def test_overrides_merge_baseline_experiment_and_seed(tiny_plan):
    o = run_overrides(tiny_plan, "L-3", 7)
    assert o["model"]["layers"] == ["layer3"] and o["model"]["pre_trained"] is False
    assert o["model"]["sampling_seed"] == 7 and o["preprocessing"]["image_size"] == 64


def test_planned_runs_seed0_first_and_skip_not_run(tiny_plan, category):
    runs = planned_runs(tiny_plan, [category], None)
    assert runs == [("B0", category, 0), ("L-3", category, 0), ("B0", category, 1)]
    assert planned_runs(tiny_plan, [category], ["L-3"]) == [("L-3", category, 0)]


@pytest.mark.slow
def test_ablation_end_to_end_on_synthetic(tiny_plan, synthetic_root, category, tmp_path):
    out, results = tmp_path / "out", tmp_path / "results"
    for exp, seed in (("B0", 0), ("B0", 1), ("L-3", 0)):
        fit = stage_fit(tiny_plan, exp, category, seed, synthetic_root, out)
        assert fit["peak_rss_fit_mb"] > 0 and fit["memory_bank_mb"] > 0
        ev = stage_eval(tiny_plan, exp, category, seed, synthetic_root, out, keep_bank=False)
        assert ev["status"] == "OK" and ev["n_val_normal"] == 2
        assert ev["latency_p95_ms"] > 0 and ev["peak_rss_serve_mb"] >= ev["rss_after_load_mb"] * 0.5
        for k in ("val_image_auroc", "test_image_auroc", "test_pixel_auroc", "test_aupro"):
            assert ev[k] is None or 0 <= ev[k] <= 1

    summarize(tiny_plan, out, results)
    runs = pd.read_csv(results / "runs.csv")
    assert set(runs.status) == {"OK", "NOT_RUN"}
    assert runs.loc[runs.exp_id == "X", "reason"].iloc[0] == "excluded on purpose"
    summary = pd.read_csv(results / "summary.csv")
    b0 = summary[(summary.exp_id == "B0") & (summary.category == category)].iloc[0]
    assert b0.n_seeds == 2
    sel = json.loads((results / "selection.json").read_text())
    assert sel["status"] == "OK" and sel["selected"] in {"B0", "L-3"}
    assert "NOT_RUN" in (results / "tables.md").read_text(encoding="utf-8")
