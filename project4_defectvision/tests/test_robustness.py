import numpy as np
import pytest

from ml.datasets.mvtec import official_test_split, scan_category, split_train_val
from ml.evaluation.robustness import LEVELS, ROTATION_EXCLUDED, evaluate_robustness, perturb
from ml.inference.patchcore import PatchCoreDetector


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(LEVELS))
def test_perturbations_keep_shape_dtype_and_change_pixels(name):
    img = _img()
    out, _ = perturb(name, LEVELS[name][-1], img, None, seed=1)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_identity_levels_leave_image_unchanged():
    img = _img()
    assert np.array_equal(perturb("brightness", 1.0, img, None, 0)[0], img)
    assert np.array_equal(perturb("rotation", 0.0, img, None, 0)[0], img)


def test_noise_is_deterministic_per_seed():
    img = _img()
    a, b, c = (perturb("gaussian_noise", 10.0, img, None, s)[0] for s in (5, 5, 6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_rotation_moves_mask_with_image():
    img = np.zeros((64, 64, 3), np.uint8)
    img[10:20, 40:50] = 255
    mask = img[..., 0] > 0
    out, out_mask = perturb("rotation", 90.0, img, mask, 0)
    assert out_mask.dtype == bool and out_mask.sum() == pytest.approx(mask.sum(), rel=0.1)
    np.testing.assert_array_equal(out[..., 0] > 127, out_mask)  # the bright square and its mask stay aligned


def test_photometric_perturbations_leave_mask_alone():
    mask = np.zeros((48, 64), bool)
    mask[5:9, 5:9] = True
    assert perturb("jpeg", 50, _img(), mask, 0)[1] is mask


def test_orientation_defects_are_excluded_from_rotation():
    assert "flip" in ROTATION_EXCLUDED["metal_nut"] and "misplaced" in ROTATION_EXCLUDED["transistor"]


@pytest.mark.slow
def test_robustness_end_to_end_on_synthetic(smoke_config, synthetic_root, category):
    samples = scan_category(synthetic_root, category)
    fit, val = split_train_val(samples, 0.2, 42)
    det = PatchCoreDetector(smoke_config)
    det.fit([s.image_path for s in fit])
    det.calibrate([s.image_path for s in val])
    tests = official_test_split(samples)
    res = evaluate_robustness(det, tests, category, {"jpeg": (50,), "rotation": (5.0,)})
    assert list(res.corruption) == ["clean", "jpeg", "rotation"]
    assert (res.n_images == len(tests)).all()
    assert res[["image_auroc", "pixel_auroc", "fpr_at_threshold"]].stack().between(0, 1).all()
    per_type = res.attrs["per_type"]
    assert set(per_type.defect_type) == {"good", "scratch", "spot"}
