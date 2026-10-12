"""Diagnose hypothesis H1 (1D-CNN beats GBM): paired engine-level bootstrap of the RMSE difference,
learning curves, and error by true-RUL bin. Validation engines only.

Usage: python diagnose_step3.py     # writes results/step3_diagnosis.md and results/step3_diagnosis.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import fit_predict
from eval_step3 import load_runs
from metrics import RUL_BINS, rmse
from prep import CAP, prepare, training_windows, truncation_points
from train import CKPT, predict

RESULTS = Path(__file__).parent / "results"


def paired_diff_ci(err_a, err_b, units, n_boot=2000, seed=0):
    """95 % CI of RMSE(a) - RMSE(b) on the same points, resampling whole engines."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    groups = [np.flatnonzero(units == u) for u in uniq]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(np.sqrt(np.mean(err_a[idx] ** 2)) - np.sqrt(np.mean(err_b[idx] ** 2)))
    return np.percentile(diffs, [2.5, 97.5]), float(np.mean(np.array(diffs) > 0))


def main():
    lines = ["# Step 3 diagnosis: why the 1D-CNN does not beat GBM\n",
             "All numbers on validation engines, target = RUL capped at 125. Deep models = 5-seed ensembles.\n",
             "## 1. Paired comparison with GBM (positive difference = model is worse than GBM)\n",
             "| Subset | Model | RMSE model | RMSE GBM | Difference | 95 % CI (engine bootstrap) | P(model worse) |",
             "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bin_rows = []
    for col, subset in enumerate(["FD001", "FD004"]):
        p = prepare(subset)
        va = truncation_points(p, "val")
        target = np.minimum(va["rul"], CAP)
        gbm = np.clip(fit_predict("GBM (window stats)", training_windows(p, "train"), va), 0, CAP)
        preds = {"GBM": gbm}
        for model in ["cnn", "lstm"]:
            runs = load_runs(subset, model, "condition", p.n_channels)
            preds[model.upper()] = np.mean([np.clip(predict(m, va["X"]), 0, CAP) for m, _ in runs], 0)
            (lo, hi), p_worse = paired_diff_ci(preds[model.upper()] - target, gbm - target, va["unit"])
            d = rmse(preds[model.upper()], target) - rmse(gbm, target)
            lines.append(f"| {subset} | {model.upper()} | {rmse(preds[model.upper()], target):.2f} | {rmse(gbm, target):.2f} | "
                         f"{d:+.2f} | {lo:+.2f} – {hi:+.2f} | {p_worse:.2f} |")

            # learning curves: mean over seeds
            hist = [json.loads((CKPT / f"{subset}_{model}_condition_s{s}.json").read_text())["history"] for s in range(5)]
            tr = np.mean([[h["train_rmse"] for h in hs] for hs in hist], 0)
            vl = np.mean([[h["val_rmse_capped"] for h in hs] for hs in hist], 0)
            ep = np.arange(1, len(tr) + 1)
            c = "C0" if model == "cnn" else "C1"
            axes[0, col].plot(ep, tr, "--", color=c, label=f"{model.upper()} train")
            axes[0, col].plot(ep, vl, "-", color=c, label=f"{model.upper()} val")
            best_ep = int(np.argmin(vl)) + 1
            bin_rows.append((subset, model.upper(), best_ep, vl.min(), vl[-1], tr[best_ep - 1], tr[-1]))
        axes[0, col].axhline(rmse(gbm, target), color="k", ls=":", label="GBM val")
        axes[0, col].set(title=f"{subset}: learning curves (mean of 5 seeds)", xlabel="Epoch", ylabel="RMSE (capped)", ylim=(8, 30))
        axes[0, col].legend(fontsize=8)

        # error by true-RUL bin
        names = list(preds)
        width = 0.25
        for j, name in enumerate(names):
            vals = []
            for a, b in RUL_BINS:
                m = (va["rul"] >= a) & (va["rul"] < b)
                vals.append(rmse(preds[name][m], target[m]) if m.any() else np.nan)
            axes[1, col].bar(np.arange(len(RUL_BINS)) + (j - 1) * width, vals, width, label=name)
            if col == 0 and j == 0:
                counts = [int(((va["rul"] >= a) & (va["rul"] < b)).sum()) for a, b in RUL_BINS]
        axes[1, col].set(xticks=range(len(RUL_BINS)), title=f"{subset}: RMSE by true-RUL bin",
                         xticklabels=[f"{a}-{b if np.isfinite(b) else '∞'}" for a, b in RUL_BINS], ylabel="RMSE (capped)")
        axes[1, col].legend(fontsize=8)
        lines_bins = []
        for a, b in RUL_BINS:
            m = (va["rul"] >= a) & (va["rul"] < b)
            lines_bins.append(f"| {a}–{b if np.isfinite(b) else '∞'} | {int(m.sum())} | " +
                              " | ".join(f"{rmse(preds[n][m], target[m]):.1f}" for n in names) + " |")
        lines += [f"\n### {subset}: RMSE by true-RUL bin\n", "| True RUL | Points | " + " | ".join(names) + " |",
                  "|---|---|" + "---|" * len(names), *lines_bins, ""]

    lines += ["\n## 2. Learning curves (mean of 5 seeds)\n",
              "| Subset | Model | Best epoch | Val at best | Val at epoch 40 | Train at best | Train at epoch 40 |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {s} | {m} | {be} | {vb:.2f} | {vl:.2f} | {tb:.2f} | {tl:.2f} |" for s, m, be, vb, vl, tb, tl in bin_rows]
    fig.tight_layout()
    fig.savefig(RESULTS / "step3_diagnosis.png", dpi=110)
    (RESULTS / "step3_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
