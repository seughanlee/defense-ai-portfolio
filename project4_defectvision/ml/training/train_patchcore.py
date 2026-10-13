"""Fit a PatchCore memory bank on normal training images and calibrate thresholds.

    python -m ml.training.train_patchcore --data-root data/mvtec_ad --category metal_nut

Only ``train/good`` is read. It is split into a memory-bank part and a held-out normal validation part
used for threshold calibration. The official test split is not touched here.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ml.config import load_config
from ml.datasets.mvtec import DatasetError, scan_category, split_train_val, validate_category
from ml.inference.patchcore import PatchCoreDetector

logger = logging.getLogger("train_patchcore")
PROJECT_DIR = Path(__file__).resolve().parents[2]


def train(
    data_root: Path,
    category: str,
    config_path: Path,
    artifacts_root: Path,
    model_version: str | None = None,
    overrides: dict | None = None,
) -> Path:
    cfg = load_config(config_path, overrides)
    report = validate_category(data_root, category)
    for w in report.warnings:
        logger.warning(w)
    if not report.ok:
        raise DatasetError("Dataset validation failed:\n  " + "\n  ".join(report.errors))

    samples = scan_category(data_root, category)
    fit, val = split_train_val(samples, cfg.data.val_fraction, cfg.data.seed)
    logger.info("category=%s fit=%d val=%d (test split untouched)", category, len(fit), len(val))

    detector = PatchCoreDetector(cfg, seed=cfg.data.seed)
    fit_stats = detector.fit([s.image_path for s in fit])
    detector.calibrate([s.image_path for s in val])

    version = model_version or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = artifacts_root / category / cfg.model.name / version
    dataset_info = {
        "name": "synthetic" if (data_root / category / "SYNTHETIC_DATA_NOT_FOR_PERFORMANCE_CLAIMS.txt").exists()
        else "mvtec_ad",
        "category": category,
        "fingerprint_sha256": report.fingerprint,
        "counts": report.counts,
        "split": {
            "seed": cfg.data.seed,
            "val_fraction": cfg.data.val_fraction,
            "fit_files": [s.image_path.name for s in fit],
            "val_files": [s.image_path.name for s in val],
        },
        "fit_stats": fit_stats,
    }
    meta = detector.save(out_dir, model_version=version, dataset=dataset_info)
    logger.info("thresholds=%s", meta["thresholds"])
    return out_dir


def _parse_overrides(pairs: list[str]) -> dict:
    """``section.key=value`` pairs; value parsed as JSON when possible (e.g. 64, false, [\"layer2\"])."""
    out: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        section, _, name = key.partition(".")
        if not (section and name and _):
            raise argparse.ArgumentTypeError(f"Override must look like section.key=value: {pair}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        out.setdefault(section, {})[name] = parsed
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--config", type=Path, default=PROJECT_DIR / "ml/configs/patchcore_cpu.yaml")
    p.add_argument("--artifacts", type=Path, default=PROJECT_DIR / "artifacts/models")
    p.add_argument("--version", default=None, help="model version label (default: UTC timestamp)")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="config overrides")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        out = train(args.data_root, args.category, args.config, args.artifacts, args.version,
                    _parse_overrides(args.set))
    except DatasetError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc
    print(out)


if __name__ == "__main__":
    main()
