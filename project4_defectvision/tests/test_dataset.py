import shutil

import numpy as np
import pytest
from PIL import Image

from ml.datasets.mvtec import (
    DatasetError,
    load_image,
    official_test_split,
    scan_category,
    split_train_val,
    validate_category,
)
from ml.datasets.synthetic import make_synthetic_dataset


@pytest.fixture
def fresh(tmp_path, category):
    """A private copy of the synthetic dataset that a test may damage."""
    make_synthetic_dataset(tmp_path, n_train=6, n_test_good=2, n_test_defect=2, size=32, seed=1)
    return tmp_path, tmp_path / category


def test_missing_root_raises(tmp_path):
    with pytest.raises(DatasetError, match="Dataset root not found"):
        scan_category(tmp_path / "absent", "metal_nut")


def test_missing_category_lists_available(fresh, category):
    root, _ = fresh
    with pytest.raises(DatasetError, match=category):
        scan_category(root, "metal_nut")


def test_empty_train_dir_raises(fresh, category):
    root, base = fresh
    for p in (base / "train" / "good").iterdir():
        p.unlink()
    with pytest.raises(DatasetError, match="No training images"):
        scan_category(root, category)


def test_synthetic_dataset_validates(synthetic_root, category):
    r = validate_category(synthetic_root, category)
    assert r.ok, r.errors
    assert r.counts["train"] == {"good": 12}
    assert r.counts["test"] == {"good": 4, "scratch": 2, "spot": 2}
    assert r.fingerprint and len(r.fingerprint) == 64


def test_every_defect_has_mask_and_good_has_none(synthetic_root, category):
    tests = official_test_split(scan_category(synthetic_root, category))
    for s in tests:
        assert (s.mask_path is None) == (s.defect_type == "good")
        assert s.label == (0 if s.defect_type == "good" else 1)


def test_missing_mask_is_error(fresh, category):
    root, base = fresh
    next((base / "ground_truth" / "scratch").iterdir()).unlink()
    r = validate_category(root, category)
    assert any("Missing mask" in e for e in r.errors)
    assert r.fingerprint is None


def test_mask_size_mismatch_is_error(fresh, category):
    root, base = fresh
    mask = next((base / "ground_truth" / "spot").iterdir())
    Image.new("L", (10, 10)).save(mask)
    assert any("Mask size" in e for e in validate_category(root, category).errors)


def test_corrupt_image_is_error(fresh, category):
    root, base = fresh
    (base / "test" / "good" / "000.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    assert any("Corrupt" in e for e in validate_category(root, category).errors)


def test_train_test_duplicate_is_leakage(fresh, category):
    root, base = fresh
    shutil.copy(base / "train" / "good" / "000.png", base / "test" / "good" / "999.png")
    assert any("leakage" in e for e in validate_category(root, category).errors)


def test_split_is_deterministic_disjoint_and_excludes_test(synthetic_root, category):
    samples = scan_category(synthetic_root, category)
    fit1, val1 = split_train_val(samples, 0.25, seed=7)
    fit2, val2 = split_train_val(samples, 0.25, seed=7)
    assert [s.image_path for s in val1] == [s.image_path for s in val2]
    assert not {s.image_path for s in fit1} & {s.image_path for s in val1}
    assert len(fit1) + len(val1) == 12 and len(val1) == 3
    assert all(s.split == "train" for s in fit1 + val1)


def test_split_rejects_too_few_images(fresh, category):
    root, base = fresh
    for p in sorted((base / "train" / "good").iterdir())[1:]:
        p.unlink()
    with pytest.raises(DatasetError):
        split_train_val(scan_category(root, category), 0.2, seed=0)


def test_grayscale_loaded_as_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.full((8, 8), 100, np.uint8), mode="L").save(p)
    arr = load_image(p)
    assert arr.shape == (8, 8, 3) and arr.dtype == np.uint8
