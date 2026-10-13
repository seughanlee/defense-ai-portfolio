"""Synthetic anomalies on held-out *validation* normal images, used as a model-selection proxy.

In the official MVTec AD split, defective images exist only in ``test``. Choosing hyperparameters with
test defects would make the test result optimistic, so ablation choices are made on validation normals
plus these generated defects instead. The memory bank never sees validation images, so a pasted patch is
genuinely "unseen" for the model.

The proxy is not the real defect distribution. How well its ranking agrees with the test ranking is
itself measured and reported (see docs/ABLATION_STUDY.md).

Kinds (cycled per image so every kind is represented):
- ``cutpaste``: a patch of the same image, flipped/rotated and brightness-shifted, pasted elsewhere
  (after Li et al., CutPaste, CVPR 2021)
- ``scar``: a thin rotated stroke with a random colour (scratch-like)
- ``noise_blob``: an irregular smooth blob filled with coloured noise (stain/contamination-like,
  after the Perlin-noise masks of DRAEM, ICCV 2021, simplified)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KINDS = ("cutpaste", "scar", "noise_blob")


@dataclass
class SyntheticAnomaly:
    image: np.ndarray  # RGB uint8 (H, W, 3)
    mask: np.ndarray  # bool (H, W)
    kind: str


def foreground_mask(image: np.ndarray, tol: float = 25.0) -> np.ndarray:
    """Crude object mask: pixels that differ from the median border colour.

    Keeps defects on the part (metal_nut, screw, ...) instead of the empty background. Falls back to the
    whole image when the result is implausible (texture-like categories have no single background).
    """
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]).astype(np.float32)
    bg = np.median(border, axis=0)
    diff = np.abs(image.astype(np.float32) - bg).max(axis=2)
    fg = cv2.morphologyEx((diff > tol).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    frac = fg.mean()
    return fg.astype(bool) if 0.05 <= frac <= 0.95 else np.ones(image.shape[:2], bool)


def _center(rng: np.random.Generator, fg: np.ndarray, margin: int) -> tuple[int, int]:
    h, w = fg.shape
    inner = np.zeros_like(fg)
    inner[margin : h - margin, margin : w - margin] = fg[margin : h - margin, margin : w - margin]
    ys, xs = np.nonzero(inner)
    if len(ys) == 0:
        return h // 2, w // 2
    i = rng.integers(len(ys))
    return int(ys[i]), int(xs[i])


def _cutpaste(rng: np.random.Generator, img: np.ndarray, fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    area = rng.uniform(0.005, 0.03) * h * w
    aspect = rng.uniform(0.3, 3.3)
    ph = int(np.clip(np.sqrt(area * aspect), 4, h // 3))
    pw = int(np.clip(np.sqrt(area / aspect), 4, w // 3))
    sy, sx = rng.integers(0, h - ph), rng.integers(0, w - pw)
    patch = img[sy : sy + ph, sx : sx + pw].copy()
    patch = np.rot90(patch, k=int(rng.integers(1, 4)))
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    patch = np.clip(patch.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30), 0, 255).astype(np.uint8)
    ph, pw = patch.shape[:2]
    cy, cx = _center(rng, fg, max(ph, pw) // 2 + 1)
    y0, x0 = int(np.clip(cy - ph // 2, 0, h - ph)), int(np.clip(cx - pw // 2, 0, w - pw))
    out = img.copy()
    out[y0 : y0 + ph, x0 : x0 + pw] = patch
    mask = np.zeros((h, w), bool)
    mask[y0 : y0 + ph, x0 : x0 + pw] = True
    return out, mask


def _scar(rng: np.random.Generator, img: np.ndarray, fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    length = rng.uniform(0.08, 0.2) * min(h, w)
    thick = max(2, int(rng.uniform(0.004, 0.012) * min(h, w)))
    cy, cx = _center(rng, fg, int(length / 2) + 1)
    angle = rng.uniform(0, np.pi)
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1, thick)
    mask = m.astype(bool)
    out = img.copy()
    out[mask] = rng.integers(0, 256, 3, dtype=np.uint8)
    return out, mask


def _noise_blob(rng: np.random.Generator, img: np.ndarray, fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    radius = rng.uniform(0.03, 0.08) * min(h, w)
    cy, cx = _center(rng, fg, int(radius) + 1)
    # Irregular outline: a disc whose radius is modulated by smoothed noise.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.arctan2(yy - cy, xx - cx)
    harmonics = sum(rng.uniform(0, 0.25) * np.cos(k * theta + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 5))
    mask = np.hypot(yy - cy, xx - cx) < radius * (1 + harmonics)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 2.0)
    tint = rng.uniform(0, 255, 3).astype(np.float32)
    alpha = rng.uniform(0.6, 0.9)
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * (0.5 * texture[mask] + 0.5 * tint)
    return np.clip(out, 0, 255).astype(np.uint8), mask


_GENERATORS = {"cutpaste": _cutpaste, "scar": _scar, "noise_blob": _noise_blob}


def make_validation_anomalies(images: list[np.ndarray], seed: int) -> list[SyntheticAnomaly]:
    """One synthetic anomaly per validation image, kinds cycled; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    out = []
    for i, img in enumerate(images):
        kind = KINDS[i % len(KINDS)]
        fg = foreground_mask(img)
        aug, mask = _GENERATORS[kind](rng, img, fg)
        if not mask.any():  # pragma: no cover - generators always mark at least one pixel
            raise RuntimeError(f"Empty synthetic mask ({kind})")
        out.append(SyntheticAnomaly(aug, mask, kind))
    return out
