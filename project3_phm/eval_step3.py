"""Step 3 summary on validation engines: deep models (mean ± std over seeds, ensemble) vs the GBM baseline.

Usage: python eval_step3.py     # writes results/step3_val.md
"""
import json
from pathlib import Path

import numpy as np
import torch

from baselines import fit_predict
from metrics import engine_bootstrap, point_metrics, rmse
from models import MODELS
from prep import CAP, prepare, training_windows, truncation_points
from train import CKPT, predict

RESULTS = Path(__file__).parent / "results"
CONFIGS = [("FD001", "cnn", "condition"), ("FD001", "lstm", "condition"),
           ("FD004", "cnn", "condition"), ("FD004", "cnn", "global"), ("FD004", "lstm", "condition")]


def load_runs(subset, model, norm, n_channels):
    runs = []
    for meta_path in sorted(CKPT.glob(f"{subset}_{model}_{norm}_s*.json")):
        m = MODELS[model](n_channels)
        m.load_state_dict(torch.load(meta_path.with_suffix(".pt")))
        runs.append((m, json.loads(meta_path.read_text())))
    return runs


def main():
    lines = ["# Step 3: deep models on validation engines\n",
             "Selection metric (EVALUATION.md sec. 5): RMSE against the capped true RUL. Predictions clipped to [0, 125].\n",
             "| Subset | Model | Normalization | Seeds | RMSE capped (mean ± std) | Ensemble RMSE capped | 95 % CI (ensemble) | "
             "RMSE true RUL ≤ 150 (ensemble) | Best epoch (mean) |",
             "|---|---|---|---|---|---|---|---|---|"]
    prepared, baselines = {}, {}
    for subset, model, norm in CONFIGS:
        key = (subset, norm)
        if key not in prepared:
            p = prepare(subset, per_condition=norm == "condition")
            prepared[key] = (p, truncation_points(p, "val"))
        p, va = prepared[key]
        if subset not in baselines and norm == "condition":
            pred = np.clip(fit_predict("GBM (window stats)", training_windows(p, "train"), va), 0, CAP)
            baselines[subset] = rmse(pred, np.minimum(va["rul"], CAP))
        runs = load_runs(subset, model, norm, p.n_channels)
        if not runs:
            continue
        capped = np.minimum(va["rul"], CAP)
        preds = np.stack([np.clip(predict(m, va["X"]), 0, CAP) for m, _ in runs])
        per_seed = np.array([rmse(pr, capped) for pr in preds])
        ens = preds.mean(0)
        ci = engine_bootstrap(lambda i: rmse(ens[i], capped[i]), va["unit"])
        near = va["rul"] <= 150
        best_ep = np.mean([int(np.argmin([h["val_rmse_capped"] for h in meta["history"]])) + 1 for _, meta in runs])
        lines.append(f"| {subset} | {model} | {norm} | {len(runs)} | {per_seed.mean():.2f} ± {per_seed.std():.2f} | "
                     f"{rmse(ens, capped):.2f} | {ci[0]:.1f}–{ci[1]:.1f} | {rmse(ens[near], va['rul'][near]):.2f} | {best_ep:.0f} |")
    lines += ["\n## GBM baseline (same validation points)\n", "| Subset | RMSE capped |", "|---|---|"]
    lines += [f"| {s} | {v:.2f} |" for s, v in baselines.items()]
    (RESULTS / "step3_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
