"""Live training monitor: re-reads checkpoints/train_seed*.log every few seconds and redraws.

Usage: python monitor.py [--seeds 0 1 2] [--epochs 20] [--interval 5]
"""
import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

CKPT = Path(__file__).parent / "checkpoints"
LINE = re.compile(r"epoch\s+(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)\s+acc\s+([\d.]+)\s+\((\d+) s\)")


def read_log(seed):
    path = CKPT / f"train_seed{seed}.log"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    return [tuple(float(v) for v in m.groups()) for m in LINE.finditer(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    plt.ion()
    fig = plt.figure(figsize=(11, 7))
    fig.canvas.manager.set_window_title("Micro-Doppler CNN - live training monitor")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 3, 3])
    ax_bar = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1:, 0])
    ax_acc = fig.add_subplot(gs[1:, 1])

    while plt.fignum_exists(fig.number):
        logs = {s: read_log(s) for s in args.seeds}
        done_epochs = sum(len(v) for v in logs.values())
        total_epochs = args.epochs * len(args.seeds)
        times = [r[4] for v in logs.values() for r in v]
        eta_min = (total_epochs - done_epochs) * (sum(times) / len(times)) / 60 if times else float("nan")
        finished = (CKPT / "ALL_DONE.txt").exists()

        ax_bar.clear()
        for i, s in enumerate(args.seeds):
            n = len(logs[s])
            ax_bar.barh(i, args.epochs, color="0.9")
            ax_bar.barh(i, n, color=f"C{i}")
            best = max((r[3] for r in logs[s]), default=None)
            label = f"{n}/{args.epochs}" + (f"  best val acc {best:.3f}" if best is not None else "  waiting")
            ax_bar.text(args.epochs + 0.3, i, label, va="center", fontsize=9)
        ax_bar.set(yticks=range(len(args.seeds)), yticklabels=[f"seed {s}" for s in args.seeds],
                   xlim=(0, args.epochs * 1.45), xticks=[])
        status = "FINISHED" if finished else f"running - ETA {eta_min:.0f} min"
        ax_bar.set_title(f"Progress {done_epochs}/{total_epochs} epochs  |  {status}  |  "
                         f"updated {time.strftime('%H:%M:%S')}", fontsize=10)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        ax_loss.clear()
        ax_acc.clear()
        for i, s in enumerate(args.seeds):
            if not logs[s]:
                continue
            ep = [r[0] for r in logs[s]]
            ax_loss.plot(ep, [r[1] for r in logs[s]], "-", color=f"C{i}", label=f"seed {s} train")
            ax_loss.plot(ep, [r[2] for r in logs[s]], "--", color=f"C{i}", label=f"seed {s} val")
            ax_acc.plot(ep, [r[3] for r in logs[s]], "o-", color=f"C{i}", ms=3, label=f"seed {s}")
        ax_loss.set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss (solid = train, dashed = val)",
                    xlim=(0.5, args.epochs + 0.5))
        ax_acc.set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy",
                   xlim=(0.5, args.epochs + 0.5), ylim=(0, 1))
        ax_acc.axhline(0.2, ls=":", color="0.5", lw=0.8)
        for ax in (ax_loss, ax_acc):
            ax.grid(alpha=0.3)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.tight_layout()
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
