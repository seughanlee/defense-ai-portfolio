"""Classical baseline: hand-crafted micro-Doppler features + random forest.

Usage: python baseline.py      # saves class probabilities to results/baseline_probs.npz
"""
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataset import load
from features import handcrafted

RESULTS = Path(__file__).parent / "results"


def main():
    t0 = time.time()
    feats = {split: (handcrafted(d["X"]), d["y"]) for split, d in
             ((s, load(s)) for s in ["train", "val", "test", "snr_sweep", "ood"])}
    print(f"features: {feats['train'][0].shape[1]} dims ({time.time() - t0:.0f} s)")

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    rf.fit(*feats["train"])
    probs = {s: rf.predict_proba(feats[s][0]) for s in ["val", "test", "snr_sweep", "ood"]}
    for s in ["val", "test"]:
        print(f"{s} acc {(probs[s].argmax(1) == feats[s][1]).mean():.3f}")

    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "baseline_probs.npz", **probs)
    top = np.argsort(rf.feature_importances_)[::-1][:10]
    print("top feature indices:", top.tolist())


if __name__ == "__main__":
    main()
