"""Robustness: how do detection and false alarms change when test images are perturbed?

    python -m ml.evaluation.robustness --artifact <artifact dir> --data-root data/mvtec_ad

Rules:
- The threshold stored in the artifact is applied unchanged; nothing is re-calibrated on perturbed data.
- Geometric perturbations (rotation) are applied to the ground-truth mask with the same transform.
- Defect types whose *definition* is orientation or placement are excluded from rotation, because a rotated
  normal part can legitimately look like that defect (see ROTATION_EXCLUDED).
- Photometric perturbations can also change the meaning of colour/brightness defects; they are kept but
  reported per defect type so this can be seen.
- Noise is generated from a per-image seed, so every run perturbs every image identically.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ml.evaluation.metrics import image_metrics, pixel_auroc

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Defect types defined by orientation/placement: rotation would change what "normal" means for them.
ROTATION_EXCLUDED: dict[str, set[str]] = {"metal_nut": {"flip"}, "transistor": {"misplaced"}}

# Severity levels, chosen in advance (mild / moderate / strong). Values are relative to the image so that
# 700 px and 1024 px categories receive comparable perturbations.
LEVELS: dict[str, tuple[float, ...]] = {
    "brightness": (0.85, 0.7, 0.5),      # multiplicative gain (1.0 = unchanged); darker lighting
    "gaussian_noise": (5.0, 10.0, 20.0),  # std in grey levels (0-255)
    "gaussian_blur": (0.002, 0.004, 0.008),  # sigma as a fraction of the image side
    "jpeg": (75, 50, 25),                 # JPEG quality
    "rotation": (2.0, 5.0, 10.0),         # degrees, about the centre, reflect border
}


def _seed(name: str) -> int:
    return zlib.crc32(name.encode())


def brightness(img: np.ndarray, gain: float, **_) -> np.ndarray:
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, std: float, *, seed: int = 0, **_) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, rel_sigma: float, **_) -> np.ndarray:
    sigma = max(rel_sigma * max(img.shape[:2]), 0.1)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def jpeg(img: np.ndarray, quality: float, **_) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def rotate(img: np.ndarray, degrees: float, *, mask: bool = False, **_) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    if mask:
        out = cv2.warpAffine(img.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return out.astype(bool)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


CORRUPTIONS: dict[str, Callable[..., np.ndarray]] = {
    "brightness": brightness, "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur,
    "jpeg": jpeg, "rotation": rotate,
}
GEOMETRIC = {"rotation"}


def perturb(name: str, level: float, img: np.ndarray, mask: np.ndarray | None, seed: int):
    """Return (perturbed image, mask transformed consistently)."""
    out = CORRUPTIONS[name](img, level, seed=seed)
    if mask is not None and name in GEOMETRIC:
        mask = CORRUPTIONS[name](mask, level, mask=True)
    return out, mask


def evaluate_robustness(detector, samples, category: str, corruptions: dict[str, tuple[float, ...]]) -> pd.DataFrame:
    """Score clean and perturbed versions of ``samples`` (official test split) and summarise per setting."""
    from ml.datasets.mvtec import load_image, load_mask
    from ml.datasets.transforms import resize_mask

    size = detector.cfg.preprocessing.image_size
    thr = detector.image_threshold
    images = [load_image(s.image_path) for s in samples]
    masks = [load_mask(s.mask_path) if s.mask_path else None for s in samples]
    rows, per_type = [], []

    def summarise(name: str, level: float, keep: list[int], imgs: list[np.ndarray], msks: list) -> None:
        scores, maps = detector.score(imgs)
        labels = np.array([samples[i].label for i in keep])
        gt = np.stack([resize_mask(m, size) if m is not None else np.zeros((size, size), bool) for m in msks])
        im = image_metrics(labels, scores, thr)
        rows.append({"corruption": name, "level": level, "n_images": len(keep), "image_auroc": im["auroc"],
                     "fpr_at_threshold": im.get("false_positive_rate"), "recall_at_threshold": im.get("recall"),
                     "pixel_auroc": pixel_auroc(gt, maps), "mean_normal_score": float(scores[labels == 0].mean())})
        df = pd.DataFrame({"defect_type": [samples[i].defect_type for i in keep], "flagged": scores > thr})
        for t, g in df.groupby("defect_type"):
            per_type.append({"corruption": name, "level": level, "defect_type": t, "n": len(g),
                             "flagged_rate": float(g.flagged.mean())})

    all_idx = list(range(len(samples)))
    summarise("clean", 0.0, all_idx, images, masks)
    for name, levels in corruptions.items():
        excluded = ROTATION_EXCLUDED.get(category, set()) if name in GEOMETRIC else set()
        keep = [i for i in all_idx if samples[i].defect_type not in excluded]
        for level in levels:
            pairs = [perturb(name, level, images[i], masks[i], _seed(f"{samples[i].image_path.name}/{name}"))
                     for i in keep]
            summarise(name, level, keep, [p[0] for p in pairs], [p[1] for p in pairs])
    out = pd.DataFrame(rows)
    out.attrs["per_type"] = pd.DataFrame(per_type)
    return out


def main() -> None:
    from ml.datasets.mvtec import official_test_split, scan_category
    from ml.inference.patchcore import PatchCoreDetector

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data/mvtec_ad")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "results/robustness")
    p.add_argument("--corruptions", nargs="*", default=list(LEVELS))
    a = p.parse_args()
    det, meta = PatchCoreDetector.load(a.artifact)
    cat = meta["dataset"]["category"]
    tests = official_test_split(scan_category(a.data_root, cat))
    res = evaluate_robustness(det, tests, cat, {k: LEVELS[k] for k in a.corruptions})
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{cat}_{meta['model_version']}"
    res.to_csv(a.out / f"{stem}.csv", index=False)
    res.attrs["per_type"].to_csv(a.out / f"{stem}_per_type.csv", index=False)
    (a.out / f"{stem}.json").write_text(json.dumps({
        "category": cat, "model_version": meta["model_version"], "threshold": det.image_threshold,
        "levels": {k: LEVELS[k] for k in a.corruptions},
        "rotation_excluded_defect_types": sorted(ROTATION_EXCLUDED.get(cat, set())),
    }, indent=2), encoding="utf-8")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
