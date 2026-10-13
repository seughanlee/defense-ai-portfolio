import numpy as np
import pytest

from ml.config import PreprocessingConfig
from ml.datasets.transforms import preprocess, resize_mask
from ml.evaluation.metrics import aupro, image_metrics, latency_summary, mean_iou, pixel_auroc


def test_preprocess_shape_and_normalisation():
    x = preprocess(np.full((40, 50, 3), 255, np.uint8), PreprocessingConfig(image_size=32))
    assert tuple(x.shape) == (3, 32, 32)
    # white pixel -> (1 - mean) / std
    assert x[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, rel=1e-4)


@pytest.mark.parametrize("bad", [np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8), np.zeros((8, 8, 3))])
def test_preprocess_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        preprocess(bad, PreprocessingConfig(image_size=32))


def test_resize_mask_stays_binary():
    m = np.zeros((100, 100), bool)
    m[10:30, 10:30] = True
    out = resize_mask(m, 64)
    assert out.dtype == bool and out.shape == (64, 64) and out.any()


def test_image_metrics_perfect_and_threshold():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = image_metrics(labels, scores, threshold=0.25)
    assert m["auroc"] == 1.0 and m["average_precision"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 0, "tp": 2}
    assert m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3)
    assert m["threshold"] == 0.25


def test_image_metrics_single_class_gives_none():
    m = image_metrics(np.zeros(4), np.arange(4.0), threshold=None)
    assert m["auroc"] is None and m["threshold"] is None


def test_pixel_metrics_perfect_map():
    masks = np.zeros((3, 16, 16), bool)
    masks[1, 2:6, 2:6] = True
    masks[2, 8:12, 8:14] = True
    masks[2, 0:2, 0:2] = True  # second region in the same image
    maps = masks.astype(np.float32) + 0.01 * np.random.default_rng(0).random(masks.shape)
    assert pixel_auroc(masks, maps) == 1.0
    assert aupro(masks, maps) == pytest.approx(1.0, abs=1e-3)
    assert mean_iou(masks, maps, pixel_threshold=0.5) == 1.0


def test_aupro_random_map_is_low():
    rng = np.random.default_rng(0)
    masks = np.zeros((4, 32, 32), bool)
    masks[:, 10:20, 10:20] = True
    # Random scores: PRO ~ FPR, so normalised area up to 0.3 is ~0.15.
    assert aupro(masks, rng.random(masks.shape)) == pytest.approx(0.15, abs=0.03)


def test_aupro_uses_8_connected_regions():
    # Diagonal pair (one region under 8-connectivity) + a separate 2x2 block.
    masks = np.zeros((1, 10, 10), bool)
    masks[0, 1, 1] = masks[0, 2, 2] = True
    masks[0, 6:8, 6:8] = True
    maps = np.full(masks.shape, 0.5, np.float32)  # all normal pixels tie
    maps[0, 1, 1] = 1.0  # half of the diagonal region detected
    maps[0, 6:8, 6:8] = 1.0  # block fully detected
    maps[0, 2, 2] = 0.0  # other half only found after every normal pixel
    # PRO stays at mean(0.5, 1.0) = 0.75 over FPR [0, 0.3]. With 4-connectivity it would be 2/3.
    assert aupro(masks, maps) == pytest.approx(0.75)


def test_pixel_metrics_without_defects_are_none():
    masks = np.zeros((2, 8, 8), bool)
    maps = np.random.default_rng(0).random(masks.shape)
    assert pixel_auroc(masks, maps) is None
    assert aupro(masks, maps) is None
    assert mean_iou(masks, maps, 0.5) is None


def test_latency_summary():
    s = latency_summary([10.0, 20.0, 30.0])
    assert s["p50_ms"] == 20.0 and s["throughput_img_per_s"] == pytest.approx(50.0)
    assert latency_summary([]) == {"status": "NOT_RUN"}
