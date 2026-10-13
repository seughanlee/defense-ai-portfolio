import numpy as np
import pytest

from ml.evaluation.threshold_stability import bootstrap_thresholds, order_stat_exceedance


def test_order_stat_exceedance_n44():
    r = order_stat_exceedance(44, 0.99)
    assert r["upper_order_stat"]["k"] == 44 and r["lower_order_stat"]["k"] == 43
    assert r["upper_order_stat"]["mean"] == pytest.approx(1 / 45)
    assert r["lower_order_stat"]["mean"] == pytest.approx(2 / 45)
    assert r["nominal_fpr"] == pytest.approx(0.01)


def test_theory_matches_simulation():
    # Exchangeable normals: draw a validation set, set the threshold, measure exceedance on fresh normals.
    rng = np.random.default_rng(0)
    n, q = 44, 0.99
    thr = np.quantile(rng.random((20000, n)), q, axis=1)
    fpr = 1 - thr  # uniform scores: P(new > thr) = 1 - thr
    r = order_stat_exceedance(n, q)
    assert r["upper_order_stat"]["mean"] <= fpr.mean() <= r["lower_order_stat"]["mean"]


def test_more_validation_images_tighten_the_threshold():
    a, b = order_stat_exceedance(10, 0.99), order_stat_exceedance(400, 0.99)
    assert a["lower_order_stat"]["mean"] > b["lower_order_stat"]["mean"]
    assert b["lower_order_stat"]["mean"] < 0.02  # converges towards the nominal 1 %


def test_invalid_inputs():
    with pytest.raises(ValueError):
        order_stat_exceedance(0, 0.99)


def test_bootstrap_shapes_and_determinism():
    rng = np.random.default_rng(1)
    val = rng.normal(0, 1, 44)
    test_scores = np.concatenate([rng.normal(0, 1, 20), rng.normal(4, 1, 30)])
    labels = np.array([0] * 20 + [1] * 30)
    a = bootstrap_thresholds(val, test_scores, labels, 0.99, [10, 44], n_boot=300, seed=3)
    b = bootstrap_thresholds(val, test_scores, labels, 0.99, [10, 44], n_boot=300, seed=3)
    assert a.equals(b)
    assert set(a.resampling) == {"subsample", "bootstrap"} and len(a) == 6
    thr = a[a.quantity == "threshold"].set_index("val_size")
    # smaller validation sets give a wider threshold spread
    assert (thr.loc[10, "p95"] - thr.loc[10, "p05"]) > (thr.loc[44, "p95"] - thr.loc[44, "p05"])
    assert a[a.quantity != "threshold"][["p05", "p95"]].stack().between(0, 1).all()
