"""Final Monte Carlo evaluation of all tracking filters on the evaluation seed.

Usage: python run_experiment.py [--runs 100] [--seed 0]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import chi2_interval, filter_specs, run_mc, summarize

OUT = Path(__file__).parent / "results"


def metrics_table(res, rows):
    man = res["man"]
    meas = res["meas_se"]
    lines = ["| Filter | Pos RMSE [m] | Straight | Turn | Turn onset (5 s) | Peak RMSE | Vel RMSE [m/s] | Avg NEES (ideal 4) | Peak NEES |",
             "|---|---|---|---|---|---|---|---|---|",
             f"| Raw measurement | {np.sqrt(meas.mean()):.1f} | {np.sqrt(meas[:, ~man].mean()):.1f} | "
             f"{np.sqrt(meas[:, man].mean()):.1f} | - | - | - | - | - |"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['pos']:.1f} | {r['straight']:.1f} | {r['turn']:.1f} | {r['onset']:.1f} | "
                     f"{r['peak']:.1f} | {r['vel']:.1f} | {r['nees']:.1f} | {r['nees_peak']:.1f} |")
    return "\n".join(lines)


def plot(res):
    t, man, names = res["t"], res["man"], res["names"]
    pos_rmse_t = np.sqrt(res["pos_se"].mean(0))
    avg_nees_t = res["nees"].mean(0)
    lo, hi = (b / res["runs"] for b in chi2_interval(4 * res["runs"]))

    def shade(ax):
        edges = np.flatnonzero(np.diff(man.astype(int)))
        for s, e in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[s + 1], t[e], color="0.85", zorder=0)

    truth, meas_pos, est = res["sample"]
    best = names.index("IMM CV+CT-EKF")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(meas_pos[:, 0] / 1e3, meas_pos[:, 1] / 1e3, ".", ms=3, color="0.6", label="Measurements")
    ax.plot(truth[:, 0] / 1e3, truth[:, 2] / 1e3, "k-", lw=2, label="Truth")
    ax.plot(est[best, :, 0] / 1e3, est[best, :, 2] / 1e3, "-", color="tab:red", label=f"{names[best]} estimate")
    ax.set(xlabel="x [km]", ylabel="y [km]", title="Sample trajectory (run 0)", aspect="equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.png", dpi=120)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    for i, name in enumerate(names):
        color = f"C{i}"   # same color per filter across panels
        axes[0].plot(t, pos_rmse_t[i], color=color, label=name)
        if not np.all(np.isnan(res["mu"][:, i])):
            axes[1].plot(t, np.nanmean(res["mu"][:, i], 0), color=color, label=name)
        axes[2].plot(t, avg_nees_t[i], color=color, label=name)
    axes[0].plot(t, np.sqrt(res["meas_se"].mean(0)), ":", color="0.5", label="Raw measurement")
    axes[0].set(ylabel="Position RMSE [m]", title=f"Monte Carlo ({res['runs']} runs, seed {res['seed']}), grey = maneuver")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="IMM P(maneuver models)", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    axes[2].axhline(lo, ls="--", color="k", lw=0.8)
    axes[2].axhline(hi, ls="--", color="k", lw=0.8)
    axes[2].set(ylabel="Average NEES", xlabel="Time [s]", yscale="log")
    for ax in axes:
        shade(ax)
    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    res = run_mc(filter_specs(), args.runs, args.seed)
    table = metrics_table(res, summarize(res))
    lo, hi = (b / args.runs for b in chi2_interval(4 * args.runs))
    (OUT / "metrics.md").write_text(
        f"Monte Carlo runs: {args.runs}, seed: {args.seed}\n\n"
        f"NEES 95% band (per step, averaged over runs): [{lo:.2f}, {hi:.2f}]\n\n{table}\n",
        encoding="utf-8")
    print(table)
    plot(res)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
