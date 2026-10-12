"""Train the CNN on the train split, keep the checkpoint with the best validation accuracy.

Usage: python train.py [--seed 0] [--epochs 20]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import CLASSES
from dataset import load
from models import MicroDopplerCNN, n_params

CKPT = Path(__file__).parent / "checkpoints"
MAX_FREQ_SHIFT = 4      # bins (~1.25 kHz, ~19 m/s of bulk velocity)


def augment(x, gen):
    """Circular time shift (stationary micro-motion) + small Doppler shift (bulk velocity)."""
    t = int(torch.randint(0, x.shape[-1], (1,), generator=gen))
    f = int(torch.randint(-MAX_FREQ_SHIFT, MAX_FREQ_SHIFT + 1, (1,), generator=gen))
    return torch.roll(x, shifts=(f, t), dims=(-2, -1))


@torch.no_grad()
def predict_logits(model, X, batch=256):
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + batch])[:, None]) for i in range(0, len(X), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    CKPT.mkdir(exist_ok=True)

    tr, va = load("train"), load("val")
    Xtr, ytr = torch.from_numpy(tr["X"])[:, None], torch.from_numpy(tr["y"]).long()
    model = MicroDopplerCNN(len(CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(Xtr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    print(f"params: {n_params(model):,}")

    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.stack([augment(x, gen) for x in Xtr[idx]])
            loss = loss_fn(model(xb), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        val_logits = predict_logits(model, va["X"])
        val_loss = loss_fn(val_logits, torch.from_numpy(va["y"]).long()).item()
        val_acc = (val_logits.argmax(1).numpy() == va["y"]).mean()
        history.append(dict(epoch=epoch + 1, train_loss=total / len(Xtr), val_loss=val_loss, val_acc=float(val_acc)))
        print(f"epoch {epoch + 1:2d}  train {total / len(Xtr):.3f}  val {val_loss:.3f}  acc {val_acc:.3f}  "
              f"({time.time() - t0:.0f} s)")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), CKPT / f"cnn_seed{args.seed}.pt")

    (CKPT / f"history_seed{args.seed}.json").write_text(json.dumps(history, indent=1))
    print(f"best val acc {best:.3f}")


if __name__ == "__main__":
    main()
