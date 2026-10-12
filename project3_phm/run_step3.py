"""Run every step-3 configuration sequentially; one log per run, progress in checkpoints/step3_progress.txt.

Usage (detached): Start-Process python -ArgumentList "-u","run_step3.py"
"""
import subprocess
import sys
import time
from pathlib import Path

from train import CKPT, run_name

CONFIGS = [("FD001", "cnn", "condition"), ("FD001", "lstm", "condition"),
           ("FD004", "cnn", "condition"), ("FD004", "cnn", "global"), ("FD004", "lstm", "condition")]
SEEDS = range(5)


def main():
    CKPT.mkdir(exist_ok=True)
    jobs = [(c, s) for c in CONFIGS for s in SEEDS]
    progress = CKPT / "step3_progress.txt"
    t_start = time.time()
    for i, ((subset, model, norm), seed) in enumerate(jobs):
        name = run_name(subset, model, norm, seed)
        progress.write_text(f"{i}/{len(jobs)} running {name} (elapsed {time.time() - t_start:.0f} s)\n")
        with open(CKPT / f"{name}.log", "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", "-W", "ignore", "train.py", "--subset", subset, "--model", model,
                            "--norm", norm, "--seed", str(seed)], stdout=log, stderr=subprocess.STDOUT,
                           cwd=Path(__file__).parent, check=False)
    progress.write_text(f"{len(jobs)}/{len(jobs)} DONE (elapsed {time.time() - t_start:.0f} s)\n")


if __name__ == "__main__":
    main()
