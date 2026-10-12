"""Step 2: baselines evaluated on validation engines (the official test set is not touched here).

Usage: python baselines.py      # writes results/baselines.md
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from metrics import engine_bootstrap, point_metrics, rmse
from prep import CAP, prepare, training_windows, truncation_points

RESULTS = Path(__file__).parent / "results"
TEST_LIKE_MAX = 150     # view restricted to true RUL <= 150 (see note in the report)


def window_features(X):
    """Per-channel last value, mean, std and least-squares slope over the window."""
    W = X.shape[1]
    t = np.arange(W) - (W - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([X[:, -1], X.mean(1), X.std(1), slope])


def fit_predict(name, tr, ev):
    if name == "Constant (mean RUL)":
        return np.full(len(ev["rul"]), tr["y"].mean())
    if name == "Linear (last cycle)":
        m = Ridge(alpha=1.0).fit(tr["X"][:, -1], tr["y"])
        return m.predict(ev["X"][:, -1])
    if name == "GBM (window stats)":
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
        return m.fit(window_features(tr["X"]), tr["y"]).predict(window_features(ev["X"]))
    raise ValueError(name)


def main():
    RESULTS.mkdir(exist_ok=True)
    models = ["Constant (mean RUL)", "Linear (last cycle)", "GBM (window stats)"]
    lines = ["# Step 2: baselines (validation engines, K=10 truncation points per engine)\n",
             "Predictions are clipped to [0, 125]. RMSE uses the true (uncapped) RUL unless marked capped. "
             "95 % CI: engine-level bootstrap.\n"]
    for subset in ["FD001", "FD004"]:
        p = prepare(subset)
        tr, va = training_windows(p, "train"), truncation_points(p, "val")
        near = va["rul"] <= TEST_LIKE_MAX
        lines += [f"\n## {subset}  (train {len(p.splits['train'])} engines / {len(tr['y']):,} windows, "
                  f"val {len(p.splits['val'])} engines / {len(va['rul'])} points, "
                  f"{(~near).sum()} of them with true RUL > {TEST_LIKE_MAX})\n",
                  f"| Model | RMSE (all) | RMSE (true RUL ≤ {TEST_LIKE_MAX}) | 95 % CI | RMSE capped | MAE | NASA mean |",
                  "|---|---|---|---|---|---|---|"]
        for name in models:
            pred = np.clip(fit_predict(name, tr, va), 0, CAP)
            m = point_metrics(pred, va["rul"])
            idx = np.flatnonzero(near)
            ci = engine_bootstrap(lambda i: rmse(pred[idx][i], va["rul"][idx][i]), va["unit"][idx])
            lines.append(f"| {name} | {m['rmse']:.1f} | {rmse(pred[near], va['rul'][near]):.1f} | "
                         f"{ci[0]:.1f}–{ci[1]:.1f} | {m['rmse_capped']:.1f} | {m['mae']:.1f} | {m['nasa_mean']:.2f} |")
            print(subset, name, f"{m['rmse']:.1f}", f"{rmse(pred[near], va['rul'][near]):.1f}", f"{m['rmse_capped']:.1f}")
    (RESULTS / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
