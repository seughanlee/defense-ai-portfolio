"""Parameter sweeps on a separate tuning seed (evaluation uses a different seed).

Usage: python sweep.py [--runs 50] [--seed 100]
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import filter_specs, run_mc, summarize

OUT = Path(__file__).parent / "results"
COLS = ["pos", "straight", "turn", "onset", "nees_peak"]
NEES_PEAK_MAX = 20.0   # consistency constraint: reject settings that are badly overconfident anywhere


def select(rows):
    """Lowest overall position RMSE among settings satisfying the NEES constraint."""
    ok = [r for r in rows if r["nees_peak"] <= NEES_PEAK_MAX]
    return min(ok or rows, key=lambda r: r["pos"])


def sweep(filter_name, grid, runs, seed, fixed=None):
    keys = list(grid)
    rows = []
    for values in itertools.product(*grid.values()):
        params = {**(fixed or {}), **dict(zip(keys, values))}
        specs = [s for s in filter_specs(**params) if s[0] == filter_name]
        m = summarize(run_mc(specs, runs, seed))[0]
        rows.append({**params, **{c: m[c] for c in COLS}})
        print(f"  {params} -> pos {m['pos']:.1f}, onset {m['onset']:.1f}, NEES peak {m['nees_peak']:.1f}")
    return keys, rows


def to_markdown(title, keys, rows):
    best = select(rows)
    head = "| " + " | ".join(keys + COLS) + " |"
    sep = "|" + "---|" * (len(keys) + len(COLS))
    body = []
    for r in rows:
        cells = [f"{r[k]:g}" for k in keys] + [f"{r[c]:.1f}" for c in COLS]
        mark = " **(selected)**" if r is best else ""
        body.append("| " + " | ".join(cells) + f" |{mark}")
    return f"## {title}\n\n" + "\n".join([head, sep, *body]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sections = [f"# Parameter sweeps (tuning seed {args.seed}, {args.runs} runs)\n",
                "Columns: position RMSE [m] overall / straight / turn / first 5 s of each turn, "
                "peak of run-averaged NEES.\n",
                f"Selection rule: lowest overall RMSE among settings with NEES peak <= {NEES_PEAK_MAX:g}.\n"]

    print("[1] CT-EKF process noise")
    keys, rows = sweep("IMM CV+CT-EKF",
                       {"sigma_w": [0.01, 0.02, 0.03, 0.05, 0.08], "sigma_a_ct_ekf": [1.0, 2.0, 5.0]},
                       args.runs, args.seed)
    sections.append(to_markdown("CT-EKF process noise", keys, rows))
    ekf_params = {k: select(rows)[k] for k in keys}

    print("[2] Fixed-CT turn rate / process noise")
    keys, rows = sweep("IMM CV+CT fixed",
                       {"omega_fixed": [0.08, 0.10, 0.12], "sigma_a_ct_fixed": [2.0, 5.0, 10.0]},
                       args.runs, args.seed)
    sections.append(to_markdown("Fixed CT model", keys, rows))
    fixed_params = {k: select(rows)[k] for k in keys}

    print(f"[3] Transition probabilities (CT-EKF IMM, {ekf_params})")
    keys, rows = sweep("IMM CV+CT-EKF",
                       {"p_cv": [0.80, 0.90, 0.95, 0.98], "p_man": [0.80, 0.90, 0.95]},
                       args.runs, args.seed, fixed=ekf_params)
    sections.append(to_markdown(f"Transition probabilities (with {ekf_params})", keys, rows))
    pi_params = {k: select(rows)[k] for k in keys}
    chosen = {**ekf_params, **fixed_params, **pi_params}
    sections.append(f"## Chosen parameters\n\n`{chosen}`\n")
    print(f"\nChosen: {chosen}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p_man, marker in zip([0.80, 0.90, 0.95], "osD"):
        pts = [r for r in rows if r["p_man"] == p_man]
        ax.plot([r["straight"] for r in pts], [r["nees_peak"] for r in pts], marker + "-", label=f"p_man={p_man}")
        for r in pts:
            ax.annotate(f"{r['p_cv']:g}", (r["straight"], r["nees_peak"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(NEES_PEAK_MAX, ls="--", color="k", lw=0.8, label="NEES constraint")
    ax.set(xlabel="Straight-segment RMSE [m]", ylabel="Peak average NEES",
           title="Precision vs. overconfidence (labels: p_cv)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sweep_transition.png", dpi=120)

    (OUT / "sweep.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"\nSaved {OUT / 'sweep.md'}")


if __name__ == "__main__":
    main()
