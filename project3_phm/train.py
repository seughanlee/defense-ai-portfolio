"""Train one window regressor; keep the epoch with the lowest validation RMSE on the capped target.

Usage: python train.py --subset FD001 --model cnn --seed 0 [--norm condition|global] [--epochs 40]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metrics import rmse
from models import MODELS, n_params
from prep import CAP, prepare, training_windows, truncation_points

CKPT = Path(__file__).parent / "checkpoints"


def run_name(subset, model, norm, seed):
    return f"{subset}_{model}_{norm}_s{seed}"


@torch.no_grad()
def predict(model, X, batch=1024):
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--model", default="cnn", choices=list(MODELS))
    ap.add_argument("--norm", default="condition", choices=["condition", "global"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    CKPT.mkdir(exist_ok=True)
    name = run_name(args.subset, args.model, args.norm, args.seed)

    p = prepare(args.subset, per_condition=args.norm == "condition")
    tr, va = training_windows(p, "train"), truncation_points(p, "val")
    Xtr, ytr = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])
    va_capped = np.minimum(va["rul"], CAP)
    model = MODELS[args.model](p.n_channels)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"{name}: {n_params(model):,} params, {len(ytr):,} windows, {p.n_channels} channels", flush=True)

    best, history = np.inf, []
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(ytr), generator=gen)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = torch.mean(((model(Xtr[idx]) - ytr[idx]) / CAP) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val = rmse(np.clip(predict(model, va["X"]), 0, CAP), va_capped)
        history.append(dict(epoch=epoch + 1, train_rmse=float(np.sqrt(total / len(ytr)) * CAP), val_rmse_capped=val))
        print(f"epoch {epoch + 1:2d}  train {history[-1]['train_rmse']:.2f}  val(capped) {val:.2f}  "
              f"({time.time() - t0:.1f} s)", flush=True)
        if val < best:
            best = val
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
    (CKPT / f"{name}.json").write_text(json.dumps(dict(args=vars(args), best_val_rmse_capped=best,
                                                       n_params=n_params(model), history=history), indent=1))
    print(f"best val RMSE (capped) {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
