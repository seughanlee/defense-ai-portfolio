"""Heatmap, overlay and binary-mask rendering.

The heatmap shows where the model's patch features are far from anything seen in normal training
images. It is an *estimate* of the anomalous region, not proof of a defect.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HEATMAP_DISCLAIMER = "Heatmap = model-estimated anomalous region, not a confirmed defect"


def normalize_map(amap: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Scale an anomaly map to [0, 1] with a *fixed* range so colours are comparable across images.

    Per-image min-max scaling would paint every normal image with a bright hot spot, so callers should
    pass a range tied to the model (e.g. validation min score .. 2x pixel threshold).
    """
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    return np.clip((amap - vmin) / (vmax - vmin), 0.0, 1.0)


def model_map_range(calibration: dict, pixel_threshold: float | None, fallback_max: float) -> tuple[float, float]:
    """Fixed colour range for one model: [lowest validation image score, 2 x pixel threshold].

    Regions below the least anomalous validation image stay blue; regions above the pixel threshold turn
    warm. Shared by the evaluation panels and the API so both render a given map identically.
    """
    vmax = 2 * pixel_threshold if pixel_threshold else fallback_max
    vmin = calibration.get("val_score_min", 0.0)
    if vmin >= vmax:
        vmin = 0.0
    return float(vmin), float(vmax)


def colorize(norm_map: np.ndarray) -> np.ndarray:
    """[0, 1] map -> RGB uint8 heatmap (JET colormap)."""
    bgr = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(image: np.ndarray, norm_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image; the map is resized to the image size."""
    h, w = image.shape[:2]
    heat = colorize(cv2.resize(norm_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR))
    return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def save_panel(
    path: str | Path,
    image: np.ndarray,
    amap: np.ndarray,
    *,
    score: float,
    image_threshold: float | None,
    pixel_threshold: float | None,
    map_range: tuple[float, float],
    gt_mask: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Save Original | Heatmap | Overlay | Predicted mask | (Ground truth, only if one exists)."""
    size = amap.shape[0]
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    norm = normalize_map(amap, *map_range)
    pred_label = "n/a (no threshold)" if image_threshold is None else (
        "ABNORMAL" if score > image_threshold else "NORMAL"
    )
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Original", img, None),
        ("Anomaly heatmap", colorize(norm), None),
        ("Overlay", overlay(img, norm), None),
    ]
    if pixel_threshold is not None:
        panels.append(("Predicted defect mask", mask_to_image(amap > pixel_threshold), "gray"))
    if gt_mask is not None:  # only drawn when a real ground-truth mask exists
        panels.append(("Ground truth mask", mask_to_image(gt_mask), "gray"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (name, arr, cmap) in zip(axes, panels, strict=True):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=255 if cmap else None)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    thr = "none" if image_threshold is None else f"{image_threshold:.3f}"
    fig.suptitle(f"{title}  score={score:.3f}  threshold={thr}  -> {pred_label}", fontsize=10)
    fig.text(0.5, 0.01, HEATMAP_DISCLAIMER, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
