"""Unknown-target detection: output-space scores (MSP, energy) vs feature-space scores
(Mahalanobis, kNN) on the penultimate CNN features.

Usage: python ood.py      # needs checkpoints/cnn_seed*.pt; writes results/ood.md and results/ood_scores.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from config import CLASSES, OOD_CLASS
from dataset import load
from evaluate import logsumexp, ood_metrics, softmax
from models import MicroDopplerCNN
from train import CKPT

RESULTS = Path(__file__).parent / "results"
KNN_K = 50


@torch.no_grad()
def features_and_logits(model, X, batch=256):
    """Penultimate features (after global average pooling) and logits."""
    model.eval()
    feats, logits = [], []
    for i in range(0, len(X), batch):
        h = model.head[:2](model.features(torch.from_numpy(X[i:i + batch])[:, None]))
        feats.append(h)
        logits.append(model.head[2:](h))
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def mahalanobis_score(f_train, y_train, f):
    """-min_c (f - mu_c)^T Sigma^-1 (f - mu_c) with a shared, shrinkage-estimated covariance."""
    mus = np.stack([f_train[y_train == c].mean(0) for c in range(len(CLASSES))])
    centered = f_train - mus[y_train]
    prec = LedoitWolf().fit(centered).precision_
    d = np.stack([np.einsum("ij,jk,ik->i", f - mu, prec, f - mu) for mu in mus], 1)
    return -d.min(1), np.linalg.cond(np.cov(centered.T))


def knn_score(f_train, f, k=KNN_K):
    """-(distance to the k-th nearest training feature) on L2-normalized features."""
    norm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    tr, q = norm(f_train), norm(f)
    sims = q @ tr.T                                    # cosine similarity; distance = sqrt(2 - 2 sim)
    kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
    return -np.sqrt(np.clip(2 - 2 * kth, 0, None))


def main():
    tr, te, ood = load("train"), load("test"), load("ood")
    temps = [s["temperature"] for s in json.loads((RESULTS / "metrics.json").read_text())["seeds"]]
    seeds = sorted(int(p.stem.split("seed")[1]) for p in CKPT.glob("cnn_seed*.pt"))
    methods = ["MSP", "Energy", "Mahalanobis", "kNN"]
    per_seed = {m: [] for m in methods}
    conds = []
    for seed, T in zip(seeds, temps):
        model = MicroDopplerCNN(len(CLASSES))
        model.load_state_dict(torch.load(CKPT / f"cnn_seed{seed}.pt"))
        f_tr, _ = features_and_logits(model, tr["X"])
        f_te, z_te = features_and_logits(model, te["X"])
        f_ood, z_ood = features_and_logits(model, ood["X"])
        maha_te, cond = mahalanobis_score(f_tr, tr["y"], f_te)
        maha_ood, _ = mahalanobis_score(f_tr, tr["y"], f_ood)
        conds.append(cond)
        scores = {
            "MSP": (softmax(z_te / T).max(1), softmax(z_ood / T).max(1)),
            "Energy": (logsumexp(z_te), logsumexp(z_ood)),
            "Mahalanobis": (maha_te, maha_ood),
            "kNN": (knn_score(f_tr, f_te), knn_score(f_tr, f_ood)),
        }
        for m in methods:
            per_seed[m].append(ood_metrics(*scores[m]))
        if seed == seeds[0]:
            sample = scores
        print(f"seed {seed}: " + ", ".join(f"{m} {per_seed[m][-1][0]:.3f}" for m in methods))

    lines = [f"# Unknown-target detection ({OOD_CLASS}, n={len(ood['y'])}) vs in-distribution test (n={len(te['y'])})\n",
             f"Seeds {seeds}. Features: 96-d penultimate layer. kNN k={KNN_K}. "
             f"Covariance condition number before shrinkage: {np.mean(conds):.0f} (mean over seeds).\n",
             "| Score | Space | AUROC (mean ± std) | FPR@95TPR (mean ± std) |", "|---|---|---|---|"]
    space = {"MSP": "output", "Energy": "output", "Mahalanobis": "feature", "kNN": "feature"}
    for m in methods:
        a = np.array(per_seed[m])
        lines.append(f"| {m} | {space[m]} | {a[:, 0].mean():.3f} ± {a[:, 0].std():.3f} | "
                     f"{a[:, 1].mean():.3f} ± {a[:, 1].std():.3f} |")
    (RESULTS / "ood.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(methods), figsize=(13, 3.2))
    for ax, m in zip(axes, methods):
        s_in, s_out = sample[m]
        bins = np.linspace(min(s_in.min(), s_out.min()), max(s_in.max(), s_out.max()), 40)
        ax.hist(s_in, bins, alpha=0.6, density=True, label="known (test)")
        ax.hist(s_out, bins, alpha=0.6, density=True, label=f"unknown ({OOD_CLASS})")
        ax.set(title=f"{m}  AUROC {per_seed[m][0][0]:.2f}", yticks=[])
    axes[0].legend(fontsize=8)
    fig.suptitle("Score distributions, seed 0 (higher = more like training data)")
    fig.tight_layout()
    fig.savefig(RESULTS / "ood_scores.png", dpi=110)


if __name__ == "__main__":
    main()
