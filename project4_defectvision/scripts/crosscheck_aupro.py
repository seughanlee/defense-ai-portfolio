"""Compare ml.evaluation.metrics.aupro with the official MVTec AD evaluation code.

The official code (mvtec_ad_evaluation, from MVTec) is NOT part of this repository. Download it from MVTec,
extract it anywhere, and pass that folder:

    python -I scripts/crosscheck_aupro.py --official <dir containing pro_curve_util.py>

Writes results/aupro_crosscheck.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ml.evaluation.metrics import aupro  # noqa: E402


def _import_official(folder: Path):
    # The official file imports scipy.ndimage.measurements, removed in recent SciPy; provide the same function.
    sys.modules.setdefault("scipy.ndimage.measurements", types.SimpleNamespace(label=ndimage.label))
    sys.path.insert(0, str(folder))
    import generic_util
    import pro_curve_util

    return pro_curve_util.compute_pro, generic_util.trapezoid


def _cases(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n, h, w = 30, 96, 96
    masks = np.zeros((n, h, w), bool)
    for i in range(n // 2, n):  # half the images defect-free, half with 1-2 regions
        for _ in range(rng.integers(1, 3)):
            y, x = rng.integers(5, h - 25, 2)
            masks[i, y : y + rng.integers(4, 20), x : x + rng.integers(4, 20)] = True
    masks[-1, 50:60, 50:60] = False
    masks[-1, 50, 50] = masks[-1, 51, 51] = True  # diagonal-only neighbours: one region under 8-connectivity
    blur = np.stack([ndimage.gaussian_filter(m.astype(np.float32), 3) for m in masks])
    return {
        "random": (masks, rng.random(masks.shape).astype(np.float32)),
        "blurred_gt+noise": (masks, blur + 0.15 * rng.random(masks.shape).astype(np.float32)),
        "near_perfect": (masks, masks + 0.01 * rng.random(masks.shape).astype(np.float32)),
        "shifted_blobs": (masks, np.roll(blur, 4, axis=2) + 0.05 * rng.random(masks.shape).astype(np.float32)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "results/aupro_crosscheck.csv")
    args = p.parse_args()
    compute_pro, trapezoid = _import_official(args.official)
    rows = []
    for name, (masks, maps) in _cases(np.random.default_rng(0)).items():
        with contextlib.redirect_stdout(io.StringIO()):  # official code prints progress
            fprs, pros = compute_pro(list(maps), list(masks.astype(np.uint8)))
        official = float(trapezoid(fprs, pros, x_max=0.3) / 0.3)
        ours = aupro(masks, maps, fpr_limit=0.3)
        rows.append({"case": name, "official": official, "ours": ours, "abs_diff": abs(ours - official)})
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
