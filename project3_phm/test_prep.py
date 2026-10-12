"""Leakage, windowing and metric checks (EVALUATION.md sec. 9). Run: python test_prep.py"""
import numpy as np

from metrics import interval_metrics, nasa_score
from prep import WINDOW, official_test_points, prepare, training_windows, truncation_points

P1 = prepare("FD001")
P4 = prepare("FD004")


def test_splits_are_disjoint_and_complete():
    for p in (P1, P4):
        s = p.splits
        assert not set(s["train"]) & set(s["val"]) and not set(s["train"]) & set(s["calib"]) \
            and not set(s["val"]) & set(s["calib"])
        assert len(s["train"]) + len(s["val"]) + len(s["calib"]) == p.train_df["unit"].nunique()


def test_normalization_uses_train_engines_only():
    for p in (P1, P4):
        rows = p.train_df[p.train_df["unit"].isin(p.splits["train"])]
        for cond, g in rows.groupby("cond"):
            assert np.allclose(g[p.sensors].mean(), 0, atol=1e-6), (p.subset, cond)
        other = p.train_df[~p.train_df["unit"].isin(p.splits["train"])]
        assert not np.allclose(other[p.sensors].mean(), 0, atol=1e-3)   # held-out engines were not used


def test_window_ends_at_the_right_cycle():
    w = training_windows(P1, "train")
    raw = P1.train_df
    i = 777
    u, c = w["unit"][i], w["cycle"][i]
    row = raw[(raw["unit"] == u) & (raw["cycle"] == c)][P1.sensors].to_numpy()[0]
    assert np.allclose(w["X"][i, -1, :len(P1.sensors)], row)
    life = raw[raw["unit"] == u]["cycle"].max()
    assert w["rul"][i] == life - c


def test_truncation_and_test_points():
    t = truncation_points(P1, "val")
    assert t["X"].shape[1] == WINDOW and (t["rul"] >= 1).all() and (t["cycle"] >= 31).all()
    tp = official_test_points(P1)
    assert len(tp["rul"]) == 100 and tp["X"].shape == (100, WINDOW, P1.n_channels)
    tp4 = official_test_points(P4)
    assert (tp4["length"] < WINDOW).sum() == 11                       # padded engines (data report)


def test_nasa_score_hand_values():
    assert np.allclose(nasa_score([-13, 10, 0], [0, 0, 0]), [np.e - 1, np.e - 1, 0])


def test_interval_coverage_on_synthetic_data():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100_000)
    m = interval_metrics(np.full_like(y, -1.6449), np.full_like(y, 1.6449), y)
    assert abs(m["coverage"] - 0.90) < 0.005


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
