"""Live monitor for step-3 training: re-reads checkpoints/*.log every few seconds and redraws.

Usage: python monitor.py [--interval 5]
"""
import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from run_step3 import CONFIGS, SEEDS
from train import CKPT, run_name

RESULTS = Path(__file__).parent / "results"
LINE = re.compile(r"epoch\s+(\d+)\s+train\s+([\d.]+)\s+val\(capped\)\s+([\d.]+)\s+\(([\d.]+) s\)")
EPOCHS = 40


def read_log(name):
    path = CKPT / f"{name}.log"
    if not path.exists():
        return []
    return [tuple(float(v) for v in m.groups()) for m in LINE.finditer(path.read_text(encoding="utf-8", errors="ignore"))]


def gbm_baselines():
    """GBM val RMSE (capped) per subset from results/baselines.md."""
    out, subset = {}, None
    path = RESULTS / "baselines.md"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## FD"):
            subset = line.split()[1]
        elif line.startswith("| GBM") and subset:
            out[subset] = float(line.split("|")[5])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    gbm = gbm_baselines()

    plt.ion()
    fig = plt.figure(figsize=(14, 8.5))
    fig.canvas.manager.set_window_title("Project 3 - live training monitor")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 2, 2])
    ax_bar = fig.add_subplot(gs[0, :])
    ax_cfg = [fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(5)]
    ax_sum = fig.add_subplot(gs[2, 2])

    while plt.fignum_exists(fig.number):
        logs = {cfg: [read_log(run_name(*cfg, s)) for s in SEEDS] for cfg in CONFIGS}
        total_jobs = len(CONFIGS) * len(SEEDS)
        epochs_done = sum(len(r) for runs in logs.values() for r in runs)
        jobs_done = sum(len(r) >= EPOCHS for runs in logs.values() for r in runs)
        times = {cfg: [e[3] for r in runs for e in r] for cfg, runs in logs.items()}
        remaining = 0.0
        for cfg, runs in logs.items():
            per_epoch = (sum(times[cfg]) / len(times[cfg])) if times[cfg] else (
                sum(sum(t) for t in times.values()) / max(1, sum(len(t) for t in times.values())))
            remaining += sum(EPOCHS - len(r) for r in runs) * per_epoch
        progress = (CKPT / "step3_progress.txt")
        status = progress.read_text(encoding="utf-8").strip() if progress.exists() else ""
        finished = "DONE" in status

        ax_bar.clear()
        ax_bar.barh(0, total_jobs * EPOCHS, color="0.9")
        ax_bar.barh(0, epochs_done, color="C0")
        ax_bar.set(xlim=(0, total_jobs * EPOCHS), yticks=[], xticks=[])
        for s in ax_bar.spines.values():
            s.set_visible(False)
        head = "FINISHED" if finished else f"ETA {remaining / 60:.0f} min  |  {status.split(' (')[0]}"
        ax_bar.set_title(f"Runs {jobs_done}/{total_jobs}  ·  epochs {epochs_done}/{total_jobs * EPOCHS}  |  {head}  |  "
                         f"updated {time.strftime('%H:%M:%S')}", fontsize=10)

        best = {}
        for ax, cfg in zip(ax_cfg, CONFIGS):
            ax.clear()
            subset, model, norm = cfg
            best[cfg] = []
            for s, r in zip(SEEDS, logs[cfg]):
                if not r:
                    continue
                ax.plot([e[0] for e in r], [e[2] for e in r], color=f"C{s}", lw=1.2, label=f"seed {s}")
                b = min(e[2] for e in r)
                best[cfg].append(b)
                if len(r) < EPOCHS:
                    ax.plot(r[-1][0], r[-1][2], "o", color=f"C{s}", ms=5)       # run in progress
            if subset in gbm:
                ax.axhline(gbm[subset], ls="--", color="k", lw=1, label=f"GBM {gbm[subset]:.2f}")
            mean_txt = f"  mean best {sum(best[cfg]) / len(best[cfg]):.2f}" if best[cfg] else "  waiting"
            ax.set(title=f"{subset} {model.upper()} ({norm}){mean_txt}", xlim=(0.5, EPOCHS + 0.5), ylim=(10, 32))
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.3)
            if ax.lines:
                ax.legend(fontsize=7, loc="upper right")
        for ax in ax_cfg[3:]:
            ax.set_xlabel("Epoch")
        for ax in (ax_cfg[0], ax_cfg[3]):
            ax.set_ylabel("Val RMSE (capped)")

        ax_sum.clear()
        labels = [f"{c[0]}\n{c[1]}-{c[2][:4]}" for c in CONFIGS]
        for i, cfg in enumerate(CONFIGS):
            ax_sum.scatter([i] * len(best[cfg]), best[cfg], color="C0", s=18, zorder=3)
            if best[cfg]:
                ax_sum.hlines(sum(best[cfg]) / len(best[cfg]), i - 0.3, i + 0.3, color="C0", lw=2)
            if cfg[0] in gbm:
                ax_sum.hlines(gbm[cfg[0]], i - 0.4, i + 0.4, color="k", ls="--", lw=1)
        ax_sum.set(xticks=range(len(CONFIGS)), title="Best val RMSE per seed (dash = GBM)", ylim=(10, 32))
        ax_sum.set_xticklabels(labels, fontsize=7)
        ax_sum.title.set_fontsize(9)
        ax_sum.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
