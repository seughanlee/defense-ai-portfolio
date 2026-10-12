"""Step 5: robustness to sensor faults (FD001 validation) and to an unseen fault mode (FD001 -> FD003).
Nothing is refit: models, normalization and conformal calibration all come from FD001.

Usage: python step5_shift.py     # writes results/step5_shift.md and results/step5_shift.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import fit_predict
from eval_step3 import load_runs
from metrics import interval_metrics, rmse
from prep import CAP, official_test_points, prepare, training_windows, truncation_points
from uq import build_intervals, deep_outputs, safety_metrics

RESULTS = Path(__file__).parent / "results"
METHODS = ["Conformal (fixed)", "Conformal (normalized)", "Conformal (Mondrian)"]


def corrupt(X, p, kind, rng):
    """Sensor faults applied to normalized windows (channels follow p.sensors)."""
    X = X.copy()
    s = {name: i for i, name in enumerate(p.sensors)}
    if kind == "stuck s11 (Ps30)":            # sensor freezes at its healthy (training-mean) value
        X[:, :, s["s11"]] = 0.0
    elif kind == "noise x2 (all sensors)":    # noise std of every sensor doubles: add N(0, sigma) with the same size
        X += rng.normal(0, 1, X.shape).astype(np.float32) * 0.5
    elif kind == "drift s4 (T50)":            # slow calibration drift over the window, up to +2 std
        X[:, :, s["s4"]] += np.linspace(0, 2, X.shape[1], dtype=np.float32)[None, :]
    return X


def summarize(tag, models, gbm_fit, cal_out, cal_gbm, y_cal, X, y, units):
    ev = deep_outputs(models, X)
    ivs = build_intervals(cal_out, ev, y_cal)
    gbm_mu = np.clip(gbm_fit(X), 0, CAP)
    iv_g = build_intervals({"mu": cal_gbm}, {"mu": gbm_mu}, y_cal)
    rows = [dict(tag=tag, model="LSTM", method=m, rmse=rmse(np.clip(ev["mu"], 0, CAP), y),
                 sd=float(np.median(ev["sd"])), **interval_metrics(*ivs[m], y), **safety_metrics(*ivs[m], y, units))
            for m in METHODS]
    rows.append(dict(tag=tag, model="GBM", method="Conformal (Mondrian)", rmse=rmse(gbm_mu, y), sd=float("nan"),
                     **interval_metrics(*iv_g["Conformal (Mondrian)"], y),
                     **safety_metrics(*iv_g["Conformal (Mondrian)"], y, units)))
    return rows, ev


def main():
    rng = np.random.default_rng(0)
    p1 = prepare("FD001")
    models = [m for m, _ in load_runs("FD001", "lstm", "condition", p1.n_channels)]
    tr = training_windows(p1, "train")
    cal = truncation_points(p1, "calib", K=10)
    y_cal = np.minimum(cal["rul"], CAP)
    cal_out = deep_outputs(models, cal["X"])

    from baselines import window_features
    from sklearn.ensemble import HistGradientBoostingRegressor
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=0).fit(window_features(tr["X"]), tr["y"])
    gbm_fit = lambda X: gbm.predict(window_features(X))
    cal_gbm = np.clip(gbm_fit(cal["X"]), 0, CAP)

    va = truncation_points(p1, "val")
    y_va = np.minimum(va["rul"], CAP)
    rows = []
    r, ev_clean = summarize("FD001 val (clean)", models, gbm_fit, cal_out, cal_gbm, y_cal, va["X"], y_va, va["unit"])
    rows += r
    for kind in ["stuck s11 (Ps30)", "noise x2 (all sensors)", "drift s4 (T50)"]:
        r, _ = summarize(f"FD001 val, {kind}", models, gbm_fit, cal_out, cal_gbm, y_cal,
                         corrupt(va["X"], p1, kind, rng), y_va, va["unit"])
        rows += r

    # FD001 -> FD003: same sensors, normalizer and calibration as FD001
    p3 = prepare("FD003", norm_source=p1)
    p3.splits["all"] = np.sort(p3.train_df["unit"].unique())
    t3 = truncation_points(p3, "all", K=10)
    y3 = np.minimum(t3["rul"], CAP)
    r, ev3 = summarize("FD003 run-to-failure engines (K=10)", models, gbm_fit, cal_out, cal_gbm, y_cal, t3["X"], y3, t3["unit"])
    rows += r
    te3 = official_test_points(p3)
    r, _ = summarize("FD003 official test", models, gbm_fit, cal_out, cal_gbm, y_cal, te3["X"],
                     np.minimum(te3["rul"], CAP), te3["unit"])
    rows += r

    # Fault-mode proxy on FD003. A first attempt with s13 failed (it rises in every FD003 engine). A scan of
    # all sensors showed s7/s12/s20/s21 fall and s15 rises in 100 % of FD001 engines, while 44 % of FD003
    # engines move the opposite way. We group by the direction of s12 (phi, fuel flow / Ps30).
    life_delta = lambda df, s: df.groupby("unit")[s].apply(lambda x: x.iloc[-30:].mean() - x.iloc[:30].mean())
    delta = life_delta(p3.train_df, "s12")
    agree_s7 = float(((delta > 0) == (life_delta(p3.train_df, "s7") > 0)).mean())
    group = {u: ("FD001-like (s12 falls)" if d < 0 else "unseen (s12 rises)") for u, d in delta.items()}
    d1 = life_delta(p1.train_df[p1.train_df["unit"].isin(p1.splits["train"])], "s12")
    iv3 = build_intervals(cal_out, ev3, y_cal)
    lo, hi = iv3["Conformal (Mondrian)"]
    grp_rows = []
    for name in ["FD001-like (s12 falls)", "unseen (s12 rises)"]:
        m = np.array([group[u] == name for u in t3["unit"]])
        n_eng = len({u for u, gname in group.items() if gname == name})
        grp_rows.append(f"| {name} | {n_eng} | {rmse(np.clip(ev3['mu'][m], 0, CAP), y3[m]):.1f} | "
                        f"{((y3[m] >= lo[m]) & (y3[m] <= hi[m])).mean():.3f} | {(y3[m] < lo[m]).mean():.3f} | "
                        f"{np.median(ev3['sd'][m]):.2f} |")

    # Input-range guard: flag a window when any sensor in its last 10 cycles is far outside the FD001 training range.
    # It needs no model and no labels; the thresholds are fixed here, not tuned on FD003.
    def guard_rate(X, z):
        return float((np.abs(X[:, -10:, :len(p1.sensors)]).max(axis=(1, 2)) > z).mean())

    unseen = np.array([group[u] == "unseen (s12 rises)" for u in t3["unit"]])
    near = t3["rul"] <= 50
    guard_rows = []
    for z in [4.0, 6.0, 8.0]:
        guard_rows.append(
            f"| {z:g} | {guard_rate(va['X'], z):.3f} | {guard_rate(t3['X'][~unseen], z):.3f} | "
            f"{guard_rate(t3['X'][unseen], z):.3f} | {guard_rate(t3['X'][unseen & near], z):.3f} | "
            f"{guard_rate(t3['X'][~unseen & near], z):.3f} |")

    head = ("| Condition | Model | Method | RMSE | Median ensemble std | Coverage | Below lower (dangerous) | Mean width | "
            "Coverage RUL<25 | Worst engine |")
    lines = ["# Step 5: sensor faults and an unseen fault mode\n",
             "Everything (models, normalization, conformal calibration) comes from FD001. Target: RUL capped at 125.\n",
             head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['model']} | {r['method']} | {r['rmse']:.1f} | {r['sd']:.2f} | {r['coverage']:.3f} | "
                     f"{r['below']:.3f} | {r['width']:.1f} | {r['conditional']['0-25'][0]:.2f} | {r['engine_min']:.2f} |")
    lines += ["\n## FD003 fault-mode proxy (LSTM + Conformal Mondrian, run-to-failure engines)\n",
              f"FD001 training engines: s12 falls in {(d1 < 0).sum()} of {len(d1)} engines (HPC degradation only), "
              f"so 's12 rises' marks behaviour FD001 never showed (presumably the fan-degradation mode; FD003 has no "
              f"per-engine fault labels). The s12 direction agrees with the s7 direction in {agree_s7:.0%} of FD003 engines.\n",
              "| Group | Engines | RMSE | Coverage | Below lower (dangerous) | Median ensemble std |",
              "|---|---|---|---|---|---|", *grp_rows,
              "\n## Input-range guard (no model): share of windows flagged\n",
              "A window is flagged when any sensor in its last 10 cycles exceeds |z| in FD001-normalized units.\n",
              "| Threshold \\|z\\| | FD001 val (false alarms) | FD003 FD001-like | FD003 unseen | FD003 unseen, RUL ≤ 50 | "
              "FD003 FD001-like, RUL ≤ 50 |", "|---|---|---|---|---|---|", *guard_rows]
    (RESULTS / "step5_shift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Figure: per-condition RMSE / coverage / ensemble std for LSTM + Mondrian
    sel = [r for r in rows if r["model"] == "LSTM" and r["method"] == "Conformal (Mondrian)"]
    labels = [r["tag"].replace("FD001 val, ", "").replace(" (K=10)", "").replace("run-to-failure engines", "RTF") for r in sel]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ["rmse", "coverage", "sd"], ["RMSE (capped)", "90 % interval coverage", "Median ensemble std"]):
        ax.bar(range(len(sel)), [r[key] for r in sel], color=["C0"] + ["C1"] * 3 + ["C3"] * 2)
        ax.set(xticks=range(len(sel)), title=title)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        if key == "coverage":
            ax.axhline(0.9, color="k", ls="--", lw=1)
            ax.set_ylim(0, 1)
    fig.suptitle("LSTM ensemble + Mondrian conformal: blue = clean, orange = sensor faults, red = unseen fault mode")
    fig.tight_layout()
    fig.savefig(RESULTS / "step5_shift.png", dpi=110)


if __name__ == "__main__":
    main()
