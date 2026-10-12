"""Off-design robustness: vary turn load factor while filter parameters stay fixed.

Usage: python robustness.py [--runs 30] [--seed 0]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import filter_specs, run_mc, summarize
from simulate import turn_scenario

OUT = Path(__file__).parent / "results"
G_LEVELS = [1, 2, 3, 4, 5, 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    specs = filter_specs()
    names = [n for n, _ in specs]
    by_g = {}
    for g in G_LEVELS:
        by_g[g] = summarize(run_mc(specs, args.runs, args.seed, turn_scenario(g, g)))
        print(f"{g}g: " + ", ".join(f"{r['name']} turn {r['turn']:.0f} / NEES peak {r['nees_peak']:.1f}"
                                     for r in by_g[g]))

    lines = [f"# Off-design robustness ({args.runs} runs, seed {args.seed})\n",
             "Both turns use the same load factor. Filter parameters are unchanged (tuned around 2-3g).\n",
             "Cells: turn-segment position RMSE [m] / peak NEES\n",
             "| Filter | " + " | ".join(f"{g}g" for g in G_LEVELS) + " |",
             "|---|" + "---|" * len(G_LEVELS)]
    for i, name in enumerate(names):
        cells = [f"{by_g[g][i]['turn']:.0f} / {by_g[g][i]['nees_peak']:.1f}" for g in G_LEVELS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(names):
        axes[0].plot(G_LEVELS, [by_g[g][i]["turn"] for g in G_LEVELS], "o-", label=name)
        axes[1].plot(G_LEVELS, [by_g[g][i]["nees_peak"] for g in G_LEVELS], "o-", label=name)
    axes[0].set(xlabel="Turn load factor [g]", ylabel="Turn-segment position RMSE [m]")
    axes[1].set(xlabel="Turn load factor [g]", ylabel="Peak average NEES", yscale="log")
    axes[1].axhline(4, ls="--", color="k", lw=0.8)
    for ax in axes:
        ax.axvspan(2, 3, color="0.9", zorder=0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Robustness to unmodeled turn rates (grey = tuning range)")
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=120)
    print(f"\nSaved {OUT / 'robustness.md'}")


if __name__ == "__main__":
    main()
