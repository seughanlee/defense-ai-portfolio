"""Evaluate a saved PatchCore artifact on the official test split.

    python -m ml.evaluation.evaluate --artifact artifacts/models/metal_nut/patchcore/<version> \
        --data-root data/mvtec_ad

Thresholds come from the artifact (calibrated on held-out normal training images); nothing here is tuned
on the test split. Outputs: evaluation.json, per_image.csv, report.md and a few example panels.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets.mvtec import NORMAL, load_image, load_mask, official_test_split, scan_category
from ml.datasets.transforms import resize_mask
from ml.evaluation.metrics import aupro, image_metrics, latency_summary, mean_iou, pixel_auroc
from ml.inference.patchcore import METADATA_FILE, PatchCoreDetector
from ml.visualization.overlay import save_panel

logger = logging.getLogger("evaluate")


def _fmt(v: object, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def evaluate(artifact_dir: Path, data_root: Path, out_dir: Path | None = None, n_latency: int = 20,
             n_panels: int = 6) -> dict:
    t0 = time.perf_counter()
    detector, meta = PatchCoreDetector.load(artifact_dir)
    load_seconds = time.perf_counter() - t0

    category = meta["dataset"]["category"]
    is_synthetic = meta["dataset"]["name"] == "synthetic"
    out_dir = out_dir or artifact_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = detector.cfg.preprocessing.image_size

    tests = official_test_split(scan_category(data_root, category))
    scores, maps = detector.score([s.image_path for s in tests])
    labels = np.array([s.label for s in tests])
    masks = np.stack([
        resize_mask(load_mask(s.mask_path), size) if s.mask_path else np.zeros((size, size), bool) for s in tests
    ])

    img = image_metrics(labels, scores, detector.image_threshold)
    pix = {
        "evaluated_at_resolution": size,
        "pixel_threshold": detector.pixel_threshold,
        "pixel_auroc": pixel_auroc(masks, maps),
        "aupro_fpr0.3": aupro(masks, maps),
        "mean_iou_defective_images": None if detector.pixel_threshold is None
        else mean_iou(masks, maps, detector.pixel_threshold),
    }

    per_image = pd.DataFrame({
        "file": [f"{s.defect_type}/{s.image_path.name}" for s in tests],
        "defect_type": [s.defect_type for s in tests],
        "label": labels,
        "score": scores,
        "predicted_abnormal": scores > detector.image_threshold,
    })
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    by_type = (
        per_image.groupby("defect_type")
        .agg(n=("score", "size"), mean_score=("score", "mean"), flagged_rate=("predicted_abnormal", "mean"))
        .reset_index()
    )

    # Latency: single-image requests, as the API will serve them. Two warm-up calls are excluded.
    lat_items = [s.image_path for s in tests[: max(n_latency, 0) + 2]]
    times = [detector.predict(p).inference_time_ms for p in lat_items][2:] if len(lat_items) > 2 else []
    operational = {
        "device": str(detector.device),
        "num_threads": detector.cfg.runtime.num_threads,
        "model_load_seconds": round(load_seconds, 2),
        "latency_batch1_includes": "preprocess + backbone + kNN + map (image already decoded)",
        "latency_batch1": latency_summary(times),
        "peak_memory": {"status": "NOT_RUN"},
        "cpu_gpu_utilization": {"status": "NOT_RUN"},
    }

    # Example panels: highest-scoring normals (false-alarm candidates) and lowest-scoring defects (misses).
    vmin = meta["calibration"].get("val_score_min", float(maps.min()))
    vmax = 2 * detector.pixel_threshold if detector.pixel_threshold else float(maps.max())
    order_normal = [i for i in np.argsort(-scores) if labels[i] == 0][: n_panels // 2]
    order_defect = [i for i in np.argsort(scores) if labels[i] == 1][: n_panels - len(order_normal)]
    for i in order_normal + order_defect:
        s = tests[i]
        save_panel(
            out_dir / "panels" / f"{s.defect_type}_{s.image_path.stem}.png",
            load_image(s.image_path), maps[i], score=float(scores[i]),
            image_threshold=detector.image_threshold, pixel_threshold=detector.pixel_threshold,
            map_range=(vmin, vmax), gt_mask=masks[i] if s.mask_path else None,
            title=f"{category}/{s.defect_type}/{s.image_path.name}",
        )

    result = {
        "status": "SYNTHETIC_SMOKE_TEST" if is_synthetic else "COMPLETED",
        "warning": "Synthetic data: these numbers are NOT detection performance." if is_synthetic else None,
        "model_version": meta["model_version"],
        "category": category,
        "dataset_fingerprint": meta["dataset"]["fingerprint_sha256"],
        "threshold_source": meta["calibration"],
        "image_level": img,
        "pixel_level": pix,
        "per_defect_type": by_type.to_dict(orient="records"),
        "operational": operational,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    _write_markdown(out_dir / "report.md", result)

    # Record a summary in the artifact metadata (the checksummed memory bank file is not modified).
    meta["evaluation"] = {
        "status": result["status"],
        "image_auroc": img["auroc"],
        "image_f1": img.get("f1"),
        "pixel_auroc": pix["pixel_auroc"],
        "aupro": pix["aupro_fpr0.3"],
        "latency_p95_ms": operational["latency_batch1"].get("p95_ms"),
        "report": str(out_dir.relative_to(artifact_dir)) if out_dir.is_relative_to(artifact_dir) else str(out_dir),
    }
    (artifact_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def _write_markdown(path: Path, r: dict) -> None:
    img, pix, op = r["image_level"], r["pixel_level"], r["operational"]
    lat = op["latency_batch1"]
    lines = [f"# Evaluation — {r['category']} / {r['model_version']}", ""]
    if r["warning"]:
        lines += [f"> **{r['warning']}**", ""]
    cm = img.get("confusion_matrix", {})
    lines += [
        f"Status: `{r['status']}`  ·  dataset fingerprint `{(r['dataset_fingerprint'] or '')[:12]}`", "",
        "## Image level", "",
        "| metric | value |", "|---|---|",
        f"| test images (normal / anomalous) | {img['n_normal']} / {img['n_anomalous']} |",
        f"| AUROC | {_fmt(img['auroc'])} |",
        f"| Average precision | {_fmt(img['average_precision'])} |",
        f"| threshold (val-normal q={r['threshold_source'].get('image_quantile')}) | {_fmt(img.get('threshold'))} |",
        f"| precision / recall / F1 @ threshold | {_fmt(img.get('precision'))} / {_fmt(img.get('recall'))} / "
        f"{_fmt(img.get('f1'))} |",
        f"| false-positive rate on normals | {_fmt(img.get('false_positive_rate'))} |",
        f"| confusion (tn, fp, fn, tp) | {cm.get('tn')}, {cm.get('fp')}, {cm.get('fn')}, {cm.get('tp')} |",
        "", "## Pixel level", "",
        "| metric | value |", "|---|---|",
        f"| resolution | {pix['evaluated_at_resolution']} px (resized, not original resolution) |",
        f"| pixel AUROC | {_fmt(pix['pixel_auroc'])} |",
        f"| AUPRO (FPR ≤ 0.3) | {_fmt(pix['aupro_fpr0.3'])} |",
        f"| mean IoU (defective images, pixel thr {_fmt(pix['pixel_threshold'])}) | "
        f"{_fmt(pix['mean_iou_defective_images'])} |",
        "", "## Per defect type", "", "| type | n | mean score | flagged rate |", "|---|---|---|---|",
    ]
    lines += [f"| {d['defect_type']} | {d['n']} | {d['mean_score']:.4f} | {d['flagged_rate']:.2f} |"
              for d in r["per_defect_type"]]
    lines += [
        "", "## Operational", "",
        f"- device `{op['device']}`, threads {op['num_threads']}, model load {op['model_load_seconds']} s",
        f"- batch-1 latency ({op['latency_batch1_includes']}): "
        + (f"mean {lat['mean_ms']:.1f} ms, p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms, "
           f"n={lat['n']}" if "mean_ms" in lat else "NOT_RUN"),
        "- peak memory: NOT_RUN · CPU/GPU utilisation: NOT_RUN",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-latency", type=int, default=20)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    r = evaluate(args.artifact, args.data_root, args.out, args.n_latency)
    print(json.dumps({"status": r["status"], "image_auroc": r["image_level"]["auroc"],
                      "pixel_auroc": r["pixel_level"]["pixel_auroc"]}, indent=2))


if __name__ == "__main__":
    main()
