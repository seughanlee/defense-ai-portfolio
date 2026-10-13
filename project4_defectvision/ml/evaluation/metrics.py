"""Image- and pixel-level anomaly detection metrics.

Threshold-free metrics (AUROC, AP, pixel AUROC, AUPRO) describe ranking quality.
Threshold-dependent metrics (precision, recall, F1, confusion matrix, IoU) are always reported together
with the threshold that produced them.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _both_classes(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) == 2


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} must be equal 1-D shapes")
    out: dict = {"n": int(labels.size), "n_anomalous": int(labels.sum()), "n_normal": int((labels == 0).sum())}
    if _both_classes(labels):
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["average_precision"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = out["average_precision"] = None
    if threshold is None:
        out["threshold"] = None
        return out
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out.update(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )
    return out


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float | None:
    y = np.asarray(masks, dtype=bool).ravel()
    if not _both_classes(y):
        return None
    return float(roc_auc_score(y, np.asarray(maps, dtype=np.float32).ravel()))


def mean_iou(masks: np.ndarray, maps: np.ndarray, pixel_threshold: float) -> float | None:
    """Mean IoU over images that have a non-empty ground-truth mask (i.e. defective images only)."""
    ious = []
    for gt, amap in zip(masks, maps, strict=True):
        gt = gt.astype(bool)
        if not gt.any():
            continue
        pred = amap > pixel_threshold
        union = np.logical_or(gt, pred).sum()
        ious.append(np.logical_and(gt, pred).sum() / union)
    return float(np.mean(ious)) if ious else None


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float | None:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1].

    Follows Bergmann et al. (MVTec AD, IJCV 2021) and the official MVTec evaluation code: each
    8-connected ground-truth region counts equally regardless of size, and the false-positive rate is
    measured on all defect-free pixels. Exact over every distinct score (sort + cumulative sums);
    agreement with the official implementation is checked by scripts/crosscheck_aupro.py.
    """
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float32)
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} differ")
    if not masks.any() or masks.all():
        return None

    # Per-pixel contribution to PRO when the pixel becomes positive: 1 / (size of its region).
    pro_weight = np.zeros(masks.shape, np.float64)
    n_regions = 0
    for i, m in enumerate(masks):
        labeled, n = ndimage.label(m, structure=np.ones((3, 3), int))
        if n:
            sizes = np.bincount(labeled.ravel())
            region_pixels = labeled > 0
            pro_weight[i][region_pixels] = 1.0 / sizes[labeled[region_pixels]]
        n_regions += n
    fp_weight = (~masks).astype(np.float64)

    order = np.argsort(-maps.ravel(), kind="stable")
    scores_sorted = maps.ravel()[order]
    fprs = np.cumsum(fp_weight.ravel()[order]) / fp_weight.sum()
    pros = np.cumsum(pro_weight.ravel()[order]) / n_regions
    last_of_tie = np.append(np.diff(scores_sorted) != 0, True)  # one curve point per distinct threshold
    fprs = np.concatenate([[0.0], np.minimum(fprs[last_of_tie], 1.0)])
    pros = np.concatenate([[0.0], np.minimum(pros[last_of_tie], 1.0)])

    # Integrate up to fpr_limit, interpolating the PRO value at the limit.
    ins = int(np.searchsorted(fprs, fpr_limit, side="right"))
    x, y = fprs[:ins], pros[:ins]
    if ins < len(fprs) and x[-1] < fpr_limit:
        y_lim = y[-1] + (pros[ins] - y[-1]) * (fpr_limit - x[-1]) / (fprs[ins] - x[-1])
        x, y = np.append(x, fpr_limit), np.append(y, y_lim)
    return float(np.trapezoid(y, x) / fpr_limit)


def latency_summary(times_ms: list[float]) -> dict:
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size == 0:
        return {"status": "NOT_RUN"}
    return {
        "n": int(t.size),
        "mean_ms": float(t.mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "max_ms": float(t.max()),
        "throughput_img_per_s": float(1000.0 / t.mean()),
    }
