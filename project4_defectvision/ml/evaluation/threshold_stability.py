"""How much can a threshold calibrated on a few normal images be trusted?

The image threshold is a quantile of validation-normal scores. Two analyses, neither of which changes the
threshold (the test split is only used to *report* consequences):

1. Distribution-free expectation. If validation and future normal images are exchangeable, the chance that a
   new normal image exceeds the k-th smallest of n validation scores follows Beta(n + 1 - k, k). The 0.99
   quantile with linear interpolation lies between two order statistics, so the expected false-alarm rate is
   bracketed by the two Beta means. With n = 44 this is 2.2 % - 4.4 %, not the nominal 1 %.
2. Bootstrap. Resample validation scores (with replacement, and sub-sample smaller validation sets without
   replacement) to see how far the threshold moves and what that does to test false-alarm rate and recall.

    python -m ml.evaluation.threshold_stability --artifact artifacts/ablation/models/metal_nut/patchcore/B0-s0 \
        --data-root data/mvtec_ad
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

PROJECT_DIR = Path(__file__).resolve().parents[2]


def order_stat_exceedance(n: int, quantile: float, coverage: float = 0.9) -> dict:
    """Expected false-alarm rate of ``np.quantile(val, quantile)`` used as a threshold on new normals.

    np.quantile (linear) interpolates between the order statistics k_lo = floor(h) + 1 and k_hi = k_lo + 1
    (1-based), h = (n - 1) * quantile. The exceedance probability of the k-th order statistic of n exchangeable
    samples is Beta(n + 1 - k, k); the threshold lies between the two, so its false-alarm rate does too.
    """
    if n < 1 or not 0 < quantile <= 1:
        raise ValueError("need n >= 1 and 0 < quantile <= 1")
    h = (n - 1) * quantile
    k_lo = int(math.floor(h)) + 1
    k_hi = min(k_lo + 1, n)
    lo_tail = (1 - coverage) / 2

    def stats(k: int) -> dict:
        dist = beta(n + 1 - k, k)
        return {"k": k, "mean": float(dist.mean()), "low": float(dist.ppf(lo_tail)),
                "high": float(dist.ppf(1 - lo_tail))}

    return {"n": n, "quantile": quantile, "nominal_fpr": 1 - quantile, "coverage": coverage,
            "upper_order_stat": stats(k_hi), "lower_order_stat": stats(k_lo)}


def _rates(threshold: np.ndarray, test_scores: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal, defect = test_scores[test_labels == 0], test_scores[test_labels == 1]
    fpr = (normal[None, :] > threshold[:, None]).mean(axis=1)
    recall = (defect[None, :] > threshold[:, None]).mean(axis=1)
    return fpr, recall


def bootstrap_thresholds(val_scores: np.ndarray, test_scores: np.ndarray, test_labels: np.ndarray, quantile: float,
                         sizes: list[int], n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Threshold spread for the full validation set (bootstrap) and for smaller sets (sub-sampling)."""
    rng = np.random.default_rng(seed)
    n = len(val_scores)
    rows = []
    for m in sizes:
        if m > n:
            continue
        replace = m == n  # full size: bootstrap; smaller: draw without replacement from the real set
        idx = rng.choice(n, size=(n_boot, m), replace=True) if replace else np.argsort(rng.random((n_boot, n)))[:, :m]
        thr = np.quantile(val_scores[idx], quantile, axis=1)
        fpr, recall = _rates(thr, test_scores, test_labels)
        for name, v in (("threshold", thr), ("test_fpr", fpr), ("test_recall", recall)):
            rows.append({"val_size": m, "resampling": "bootstrap" if replace else "subsample", "quantity": name,
                         "p05": float(np.quantile(v, 0.05)), "median": float(np.median(v)),
                         "p95": float(np.quantile(v, 0.95)), "mean": float(v.mean())})
    return pd.DataFrame(rows)


def run(artifact: Path, data_root: Path, out_dir: Path, sizes: list[int], n_boot: int) -> dict:
    from ml.datasets.mvtec import category_dir, official_test_split, scan_category
    from ml.inference.patchcore import PatchCoreDetector

    det, meta = PatchCoreDetector.load(artifact)
    cat = meta["dataset"]["category"]
    q = meta["config"]["threshold"]["image_quantile"]
    train_good = category_dir(data_root, cat) / "train" / "good"
    val_paths = [train_good / f for f in meta["dataset"]["split"]["val_files"]]
    tests = official_test_split(scan_category(data_root, cat))
    val_scores, _ = det.score(val_paths)
    test_scores, _ = det.score([s.image_path for s in tests])
    labels = np.array([s.label for s in tests])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cat}_{meta['model_version']}"
    pd.DataFrame({"split": ["val"] * len(val_paths) + ["test"] * len(tests),
                  "file": [p.name for p in val_paths] + [f"{s.defect_type}/{s.image_path.name}" for s in tests],
                  "label": [0] * len(val_paths) + labels.tolist(),
                  "score": np.concatenate([val_scores, test_scores])}).to_csv(out_dir / f"{stem}_scores.csv",
                                                                              index=False)

    # The stored threshold must be reproduced exactly from the re-computed validation scores.
    reproduced = float(np.quantile(val_scores, q))
    boot = bootstrap_thresholds(val_scores, test_scores, labels, q, sizes, n_boot)
    boot.to_csv(out_dir / f"{stem}_bootstrap.csv", index=False)
    fpr_at, rec_at = _rates(np.array([det.image_threshold]), test_scores, labels)
    result = {
        "category": cat, "model_version": meta["model_version"], "quantile": q,
        "n_val_normal": len(val_paths), "n_test_normal": int((labels == 0).sum()), "n_test_defect": int(labels.sum()),
        "stored_threshold": det.image_threshold, "reproduced_threshold": reproduced,
        "threshold_reproduced": bool(np.isclose(reproduced, det.image_threshold, rtol=1e-5)),
        "test_fpr_at_stored": float(fpr_at[0]), "test_recall_at_stored": float(rec_at[0]),
        "expected_fpr_theory": order_stat_exceedance(len(val_paths), q),
        "theory_by_val_size": {m: order_stat_exceedance(m, q) for m in sizes if m <= len(val_paths)},
        "n_boot": n_boot,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data/mvtec_ad")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "results/threshold_stability")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30, 44])
    p.add_argument("--n-boot", type=int, default=5000)
    a = p.parse_args()
    r = run(a.artifact, a.data_root, a.out, a.sizes, a.n_boot)
    t = r["expected_fpr_theory"]
    print(json.dumps({k: r[k] for k in ("stored_threshold", "threshold_reproduced", "test_fpr_at_stored",
                                         "test_recall_at_stored")}, indent=2))
    print("expected FPR (theory): between {:.3f} and {:.3f}".format(t["upper_order_stat"]["mean"],
                                                                     t["lower_order_stat"]["mean"]))


if __name__ == "__main__":
    main()
