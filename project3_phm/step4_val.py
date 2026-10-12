"""Step 4 on validation engines: compare interval methods, pick one with the pre-registered rule
(val coverage >= 88 %, then the narrowest mean width), and draw the figures.

Usage: python step4_val.py     # writes results/step4_val.md and results/step4_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import fit_predict
from eval_step3 import load_runs
from metrics import RUL_BINS, interval_metrics
from prep import CAP, prepare, training_windows, truncation_points
from uq import ALPHA, build_intervals, deep_outputs, safety_metrics

RESULTS = Path(__file__).parent / "results"
MIN_COVERAGE = 0.88
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def evaluate_all(subset):
    p = prepare(subset)
    tr = training_windows(p, "train")
    va = truncation_points(p, "val")
    cals = {"K=10": truncation_points(p, "calib", K=10), "K=1": truncation_points(p, "calib", K=1)}
    y_va = np.minimum(va["rul"], CAP)
    models = [m for m, _ in load_runs(subset, "lstm", "condition", p.n_channels)]
    ev = {"LSTM": deep_outputs(models, va["X"], with_mc=True), "GBM": {"mu": None}}
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401  (fit_predict refits per call)
    gbm_ev = np.clip(fit_predict("GBM (window stats)", tr, va), 0, CAP)
    ev["GBM"] = {"mu": gbm_ev}
    rows, store = [], {}
    for k, cal in cals.items():
        y_cal = np.minimum(cal["rul"], CAP)
        cal_out = {"LSTM": deep_outputs(models, cal["X"]),
                   "GBM": {"mu": np.clip(fit_predict("GBM (window stats)", tr, cal), 0, CAP)}}
        for model in ["LSTM", "GBM"]:
            for name, (lo, hi) in build_intervals(cal_out[model], ev[model], y_cal).items():
                m = interval_metrics(lo, hi, y_va)
                s = safety_metrics(lo, hi, y_va, va["unit"])
                rows.append(dict(subset=subset, model=model, calib=k, method=name, n_cal=len(y_cal), **m, **s))
                store[(model, k, name)] = (lo, hi)
            if k == "K=10" and model == "LSTM":
                curves = {}
                for lv in LEVELS:
                    for name, (lo, hi) in build_intervals(cal_out[model], ev[model], y_cal, alpha=1 - lv).items():
                        curves.setdefault(name, []).append(float(((y_va >= lo) & (y_va <= hi)).mean()))
    return p, va, y_va, rows, store, curves, models


def fmt_row(r):
    c = r["conditional"]
    return (f"| {r['model']} | {r['method']} | {r['calib']} ({r['n_cal']}) | {r['coverage']:.3f} | {r['below']:.3f} | "
            f"{r['above']:.3f} | {r['width']:.1f} | {r['winkler']:.1f} | {c['0-25'][0]:.2f} | "
            f"{r['engine_min']:.2f} | {r['engines_below_80']}/{r['n_engines']} |")


def trajectory_plot(p, models, method_choice, path):
    """Full-life predictions for three validation engines with the chosen interval and fixed conformal."""
    cal = truncation_points(p, "calib", K=10)
    y_cal = np.minimum(cal["rul"], CAP)
    cal_out = deep_outputs(models, cal["X"])
    units = p.splits["val"][:3]
    full = training_windows(p, "val")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, u in zip(axes, units):
        m = full["unit"] == u
        ev = deep_outputs(models, full["X"][m])
        iv = build_intervals(cal_out, ev, y_cal)
        cyc = full["cycle"][m]
        for name, color in [("Conformal (fixed)", "0.75"), (method_choice, "C0")]:
            lo, hi = iv[name]
            ax.fill_between(cyc, lo, hi, color=color, alpha=0.45 if color == "C0" else 0.35, label=f"90 % {name}")
        ax.plot(cyc, np.minimum(full["rul"][m], CAP), "k-", lw=1.5, label="True RUL (capped)")
        ax.plot(cyc, ev["mu"], color="C3", lw=1, label="LSTM ensemble")
        ax.set(title=f"Validation engine {u}", xlabel="Cycle")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RUL [cycles]")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{p.subset}: whole-life predictions with 90 % intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    head = ("| Model | Method | Calibration (points) | Coverage | Below lower (dangerous) | Above upper (early) | "
            "Mean width | Winkler | Coverage RUL<25 | Worst engine | Engines <80 % |")
    lines = ["# Step 4: 90 % intervals on validation engines\n",
             "Target: RUL capped at 125 (the upper end 125 reads as \"125 cycles or more\"). "
             "Coverage = share of points whose true RUL lies inside the interval.\n",
             "- **Below lower**: true RUL is shorter than the lower bound, so the engine would fail earlier than promised. "
             "This is the safety-relevant miss; with a symmetric 90 % interval it should be about 5 %.",
             "- **Above upper**: the engine lasts longer than the upper bound; maintenance would come early (cost, not risk).",
             f"- Selection rule (EVALUATION.md sec. 5): coverage ≥ {MIN_COVERAGE:.0%} on validation, then the narrowest mean width.\n"]
    fig_c, axes_c = plt.subplots(1, 2, figsize=(12, 4.5))
    fig_b, axes_b = plt.subplots(1, 2, figsize=(13, 4.5))
    chosen, all_rows = {}, []
    for col, subset in enumerate(["FD001", "FD004"]):
        p, va, y_va, rows, store, curves, models = evaluate_all(subset)
        all_rows += rows
        lines += [f"\n## {subset}  (validation: {len(p.splits['val'])} engines, {len(y_va)} points)\n", head,
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [fmt_row(r) for r in rows]
        ok = [r for r in rows if r["coverage"] >= MIN_COVERAGE and r["calib"] == "K=10"]
        best = min(ok, key=lambda r: r["width"])
        chosen[subset] = best
        lines.append(f"\n**Selected for {subset}:** {best['model']} + {best['method']} (K=10): coverage {best['coverage']:.3f}, "
                     f"width {best['width']:.1f} cycles.\n")

        ax = axes_c[col]
        ax.plot([0.45, 1], [0.45, 1], ":", color="0.5")
        for name, cov in curves.items():
            ax.plot(LEVELS, cov, "o-", ms=4, label=name)
        ax.set(title=f"{subset}: nominal vs empirical coverage (LSTM, K=10)", xlabel="Nominal level",
               ylabel="Empirical coverage", xlim=(0.45, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes_b[col]
        sel = [r for r in rows if r["calib"] == "K=10" and r["model"] == "LSTM"]
        width = 0.8 / len(sel)
        for j, r in enumerate(sel):
            vals = [r["conditional"][f"{a}-{b}"][0] for a, b in RUL_BINS]
            ax.bar(np.arange(len(RUL_BINS)) + (j - len(sel) / 2 + 0.5) * width, vals, width, label=r["method"])
        ax.axhline(1 - ALPHA, color="k", ls="--", lw=1)
        ax.set(title=f"{subset}: coverage by true-RUL bin (LSTM, K=10)", ylim=(0, 1.05), xticks=range(len(RUL_BINS)),
               xticklabels=[f"{a}-{b if np.isfinite(b) else '∞'}" for a, b in RUL_BINS])
        ax.legend(fontsize=6, ncol=2)
        if subset == "FD001":
            trajectory_plot(p, models, best["method"] if best["model"] == "LSTM" else "Conformal (normalized)",
                            RESULTS / "step4_trajectories.png")
    fig_c.tight_layout()
    fig_c.savefig(RESULTS / "step4_coverage_curve.png", dpi=110)
    fig_b.tight_layout()
    fig_b.savefig(RESULTS / "step4_conditional.png", dpi=110)
    (RESULTS / "step4_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "step4_selected.json").write_text(json.dumps(
        {s: {k: r[k] for k in ["model", "method", "calib", "coverage", "width"]} for s, r in chosen.items()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
