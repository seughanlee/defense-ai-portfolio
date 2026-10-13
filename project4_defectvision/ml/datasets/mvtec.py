"""MVTec AD directory scanning, validation and splitting.

Expected layout for one category (official MVTec AD structure)::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png

Rules enforced here:
- Only ``train/good`` is used to build the memory bank. A held-out part of it (``val``) is used for
  threshold calibration. The official ``test`` split is used for evaluation only.
- Every defective test image must have exactly one mask; ``test/good`` has no mask files.
  For metrics, a ``test/good`` image is treated as an all-zero mask, which is MVTec's definition of
  "defect free" — it is not a mask that exists on disk.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
NORMAL = "good"


class DatasetError(RuntimeError):
    """Raised when a dataset directory is missing or structurally invalid."""


@dataclass(frozen=True)
class Sample:
    image_path: Path
    split: str  # "train" or "test"
    defect_type: str  # "good" or a defect name
    mask_path: Path | None = None

    @property
    def label(self) -> int:
        return 0 if self.defect_type == NORMAL else 1


@dataclass
class ValidationReport:
    category: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> defect_type -> n
    image_sizes: dict[str, int] = field(default_factory=dict)  # "WxH mode" -> n
    duplicates: list[list[str]] = field(default_factory=list)
    fingerprint: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": self.counts,
            "image_sizes": self.image_sizes,
            "duplicates": self.duplicates,
            "fingerprint": self.fingerprint,
        }


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def category_dir(root: str | Path, category: str) -> Path:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(
            f"Dataset root not found: {root}. Download MVTec AD (see docs/DATASET.md) "
            "or run scripts/make_synthetic.py for a smoke-test dataset."
        )
    path = root / category
    if not path.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise DatasetError(f"Category '{category}' not found under {root}. Available: {available or 'none'}")
    return path


def scan_category(root: str | Path, category: str) -> list[Sample]:
    """Return every image of a category with its split, defect type and mask path.

    Raises ``DatasetError`` if the mandatory folders are missing or empty.
    """
    base = category_dir(root, category)
    train_good = base / "train" / NORMAL
    test_dir = base / "test"
    for required in (train_good, test_dir):
        if not required.is_dir():
            raise DatasetError(f"Missing required directory: {required}")

    samples = [Sample(p, "train", NORMAL) for p in _images_in(train_good)]
    if not samples:
        raise DatasetError(f"No training images in {train_good}")

    gt_dir = base / "ground_truth"
    for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        defect = defect_dir.name
        for img in _images_in(defect_dir):
            mask = None
            if defect != NORMAL:
                candidate = gt_dir / defect / f"{img.stem}_mask.png"
                mask = candidate if candidate.is_file() else None
            samples.append(Sample(img, "test", defect, mask))

    if not any(s.split == "test" for s in samples):
        raise DatasetError(f"No test images in {test_dir}")
    return samples


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 array (H, W, 3). Grayscale images are replicated to 3 channels."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a ground-truth mask as a boolean array (H, W)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(samples: list[Sample], root: str | Path) -> str:
    """SHA-256 over the sorted (relative path, file hash) manifest — identifies the exact data used."""
    root = Path(root)
    lines = []
    for s in samples:
        for p in (s.image_path, s.mask_path):
            if p is not None:
                lines.append(f"{p.relative_to(root).as_posix()}\t{file_sha256(p)}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def validate_category(root: str | Path, category: str, check_pixels: bool = True) -> ValidationReport:
    """Check structure, file integrity, mask pairing and exact duplicates for one category.

    Structural problems that make the category unusable raise ``DatasetError``; problems with individual
    files are collected in the report so they can all be fixed at once.
    """
    report = ValidationReport(category)
    samples = scan_category(root, category)
    base = category_dir(root, category)

    counts: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        counts[s.split][s.defect_type] += 1
    report.counts = {split: dict(sorted(c.items())) for split, c in counts.items()}

    sizes: Counter = Counter()
    hashes: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        rel = s.image_path.relative_to(base).as_posix()
        try:
            with Image.open(s.image_path) as im:
                im.verify()  # detects truncated / corrupt files
            with Image.open(s.image_path) as im:
                size, mode = im.size, im.mode
                if check_pixels:
                    im.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            report.errors.append(f"Corrupt or unreadable image: {rel} ({exc})")
            continue
        sizes[f"{size[0]}x{size[1]} {mode}"] += 1
        hashes[file_sha256(s.image_path)].append(s)

        if s.split == "test" and s.defect_type != NORMAL:
            if s.mask_path is None:
                report.errors.append(f"Missing mask for defective image: {rel}")
                continue
            try:
                with Image.open(s.mask_path) as m:
                    if m.size != size:
                        report.errors.append(f"Mask size {m.size} != image size {size}: {rel}")
                    elif check_pixels:
                        values = np.unique(np.asarray(m.convert("L")))
                        if not set(values.tolist()) <= {0, 255}:
                            report.warnings.append(f"Mask is not strictly binary {values[:5].tolist()}: {rel}")
                        if values.max() == 0:
                            report.warnings.append(f"Mask is empty for a defective image: {rel}")
            except (UnidentifiedImageError, OSError) as exc:
                report.errors.append(f"Corrupt mask for {rel} ({exc})")

    report.image_sizes = dict(sizes.most_common())
    if len(sizes) > 1:
        report.warnings.append(f"Mixed image sizes/modes: {report.image_sizes}")

    # Orphan masks: mask files with no matching test image.
    used_masks = {s.mask_path for s in samples if s.mask_path is not None}
    gt_dir = base / "ground_truth"
    if gt_dir.is_dir():
        for mask in sorted(gt_dir.rglob("*_mask.png")):
            if mask not in used_masks:
                report.warnings.append(f"Orphan mask without image: {mask.relative_to(base).as_posix()}")

    # Exact duplicates. A duplicate spanning train and test is leakage and is reported as an error.
    for group in hashes.values():
        if len(group) > 1:
            names = [g.image_path.relative_to(base).as_posix() for g in group]
            report.duplicates.append(names)
            if len({g.split for g in group}) > 1:
                report.errors.append(f"Train/test leakage (identical files): {names}")
            else:
                report.warnings.append(f"Duplicate files within a split: {names}")

    if not report.errors:
        report.fingerprint = dataset_fingerprint(samples, Path(root))
    return report


def split_train_val(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Split ``train/good`` into memory-bank (fit) and threshold-calibration (val) subsets.

    Deterministic for a given seed. Test samples are never returned.
    """
    train = sorted((s for s in samples if s.split == "train"), key=lambda s: s.image_path.name)
    if len(train) < 2:
        raise DatasetError(f"Need at least 2 training images to hold out a validation set, got {len(train)}")
    n_val = max(1, round(len(train) * val_fraction))
    if n_val >= len(train):
        raise DatasetError(f"val_fraction={val_fraction} leaves no images for the memory bank")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    fit = [s for i, s in enumerate(train) if i not in val_idx]
    val = [s for i, s in enumerate(train) if i in val_idx]
    return fit, val


def official_test_split(samples: list[Sample]) -> list[Sample]:
    return [s for s in samples if s.split == "test"]
