"""One ablation run = two fresh processes, so each peak-memory number belongs to one phase only.

    python -m ml.ablation.run_one fit  --exp B0 --category metal_nut --seed 0
    python -m ml.ablation.run_one eval --exp B0 --category metal_nut --seed 0

fit:  build memory bank on fit images, calibrate threshold on validation normals, save artifact.
eval: load artifact -> batch-1 latency on validation images (serving peak memory measured here)
      -> validation proxy (val normals + synthetic anomalies) -> official test metrics.
Test metrics are recorded for reporting only; nothing in this file reads them back to change a setting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.datasets.mvtec import category_dir, load_image, load_mask, official_test_split, scan_category
from ml.datasets.synthetic_anomalies import make_validation_anomalies
from ml.datasets.transforms import resize_mask
from ml.evaluation.metrics import aupro, image_metrics, latency_summary, pixel_auroc
from ml.evaluation.resources import current_rss_mb, peak_rss_mb
from ml.inference.patchcore import MEMORY_BANK_FILE, PatchCoreDetector
from ml.training.train_patchcore import train

logger = logging.getLogger("ablation.run_one")
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = PROJECT_DIR / "ml/ablation/ablation_v1.yaml"


def load_plan(path: Path) -> dict[str, Any]:
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = [e["id"] for e in plan["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate experiment ids in {path}")
    return plan


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def experiment(plan: dict, exp_id: str) -> dict:
    for e in plan["experiments"]:
        if e["id"] == exp_id:
            return e
    raise KeyError(f"Unknown experiment id: {exp_id}")


def run_overrides(plan: dict, exp_id: str, seed: int) -> dict:
    merged = _deep_merge(plan["baseline"], experiment(plan, exp_id).get("overrides") or {})
    return _deep_merge(merged, {"model": {"sampling_seed": seed}})


def run_name(exp_id: str, category: str, seed: int) -> str:
    return f"{exp_id}__{category}__s{seed}"


def _paths(out_root: Path, exp_id: str, category: str, seed: int) -> tuple[Path, Path]:
    artifact = out_root / "models" / category / "patchcore" / f"{exp_id}-s{seed}"
    stage_dir = out_root / "stages" / run_name(exp_id, category, seed)
    return artifact, stage_dir


def stage_fit(plan: dict, exp_id: str, category: str, seed: int, data_root: Path, out_root: Path) -> dict:
    artifact, stage_dir = _paths(out_root, exp_id, category, seed)
    if artifact.exists():  # leftover from an interrupted run; artifacts here are reproducible
        shutil.rmtree(artifact)
    stage_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train(data_root, category, PROJECT_DIR / plan["base_config"], artifact.parents[2], f"{exp_id}-s{seed}",
          run_overrides(plan, exp_id, seed))
    meta = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    result = {
        "fit_wall_seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_fit_mb": round(peak_rss_mb(), 1),
        "fit_stats": meta["dataset"]["fit_stats"],
        "memory_bank_mb": round((artifact / MEMORY_BANK_FILE).stat().st_size / 2**20, 1),
        "image_threshold": meta["thresholds"]["image"],
    }
    (stage_dir / "fit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _proxy_and_test_masks(masks_src: list[np.ndarray | None], size: int) -> np.ndarray:
    return np.stack([resize_mask(m, size) if m is not None else np.zeros((size, size), bool) for m in masks_src])


def stage_eval(plan: dict, exp_id: str, category: str, seed: int, data_root: Path, out_root: Path,
               keep_bank: bool) -> dict:
    artifact, stage_dir = _paths(out_root, exp_id, category, seed)
    t0 = time.perf_counter()
    det, meta = PatchCoreDetector.load(artifact)
    load_seconds = time.perf_counter() - t0
    rss_after_load = current_rss_mb()
    size = det.cfg.preprocessing.image_size
    fpr_limit = plan["validation_proxy"]["aupro_fpr_limit"]

    train_good = category_dir(data_root, category) / "train" / "good"
    val_imgs = [load_image(train_good / f) for f in meta["dataset"]["split"]["val_files"]]

    # 1) Serving: batch-1 latency on validation images. Peak memory read right after = serving footprint.
    n_lat = plan["latency"]["n_images"]
    seq = [val_imgs[i % len(val_imgs)] for i in range(n_lat + 3)]
    times = [det.predict(img).inference_time_ms for img in seq][3:]
    peak_serve = peak_rss_mb()

    # 2) Validation proxy: held-out normals + one synthetic anomaly per normal.
    synth = make_validation_anomalies(val_imgs, seed=plan["validation_proxy"]["anomaly_seed"])
    v_scores, v_maps = det.score(val_imgs + [s.image for s in synth])
    v_labels = np.array([0] * len(val_imgs) + [1] * len(synth))
    v_masks = _proxy_and_test_masks([None] * len(val_imgs) + [s.mask for s in synth], size)

    # 3) Official test split (reporting only).
    tests = official_test_split(scan_category(data_root, category))
    t_scores, t_maps = det.score([s.image_path for s in tests])
    t_labels = np.array([s.label for s in tests])
    t_masks = _proxy_and_test_masks([load_mask(s.mask_path) if s.mask_path else None for s in tests], size)
    t_img = image_metrics(t_labels, t_scores, det.image_threshold)

    lat = latency_summary(times)
    result = {
        "status": "OK",
        "plan_sha256": plan["_sha256"],
        "model_version": meta["model_version"],
        "dataset_fingerprint": meta["dataset"]["fingerprint_sha256"],
        "config": meta["config"],
        "n_fit": meta["dataset"]["fit_stats"]["n_images"],
        "n_val_normal": len(val_imgs),
        "n_test": len(tests),
        "val_image_auroc": image_metrics(v_labels, v_scores, None)["auroc"],
        "val_pixel_auroc": pixel_auroc(v_masks, v_maps),
        "val_aupro": aupro(v_masks, v_maps, fpr_limit=fpr_limit),
        "test_image_auroc": t_img["auroc"],
        "test_image_ap": t_img["average_precision"],
        "test_pixel_auroc": pixel_auroc(t_masks, t_maps),
        "test_aupro": aupro(t_masks, t_maps, fpr_limit=fpr_limit),
        "test_f1_at_val_threshold": t_img.get("f1"),
        "test_fpr_at_val_threshold": t_img.get("false_positive_rate"),
        "test_recall_at_val_threshold": t_img.get("recall"),
        "image_threshold": det.image_threshold,
        "latency_mean_ms": lat.get("mean_ms"),
        "latency_p50_ms": lat.get("p50_ms"),
        "latency_p95_ms": lat.get("p95_ms"),
        "model_load_seconds": round(load_seconds, 2),
        "rss_after_load_mb": round(rss_after_load, 1),
        "peak_rss_serve_mb": round(peak_serve, 1),
        "peak_rss_eval_mb": round(peak_rss_mb(), 1),
    }
    (stage_dir / "eval.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if not keep_bank:  # banks are reproducible from (plan, seed); keep metadata + checksum only
        (artifact / MEMORY_BANK_FILE).unlink()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["fit", "eval"])
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--exp", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data/mvtec_ad")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "artifacts/ablation")
    p.add_argument("--keep-bank", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plan = load_plan(args.plan)
    if args.stage == "fit":
        r = stage_fit(plan, args.exp, args.category, args.seed, args.data_root, args.out)
    else:
        r = stage_eval(plan, args.exp, args.category, args.seed, args.data_root, args.out, args.keep_bank)
    print(json.dumps(r, default=float))


if __name__ == "__main__":
    main()
