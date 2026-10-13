"""PatchCore detector built on anomalib's ``PatchcoreModel`` (plain ``nn.Module``, no Lightning engine).

Pipeline: image -> preprocess -> backbone features (layer2+layer3, 3x3 avg-pooled) -> patch embeddings
-> coreset memory bank (fit) / nearest-neighbour distance (predict) -> image score + anomaly map
-> thresholds calibrated on held-out *normal* images -> inspection result.

Persistence avoids pickle entirely: the memory bank is stored with safetensors and every other field is
JSON. On load, the memory-bank file hash and the backbone weight hash are checked against the metadata,
so a swapped file or a different backbone weight version is rejected instead of producing silent garbage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.config import Config, config_from_dict
from ml.datasets.mvtec import file_sha256, load_image
from ml.datasets.transforms import preprocess

logger = logging.getLogger(__name__)

MEMORY_BANK_FILE = "memory_bank.safetensors"
METADATA_FILE = "metadata.json"
ARTIFACT_FORMAT = 1


class ArtifactError(RuntimeError):
    """Raised when a model artifact is missing, incomplete or fails integrity checks."""


class NotFittedError(RuntimeError):
    pass


@dataclass
class Prediction:
    score: float
    is_anomalous: bool
    anomaly_map: np.ndarray  # (S, S) float32, raw patch distances after upsampling + blur
    defect_mask: np.ndarray | None  # (S, S) bool; None when no pixel threshold is calibrated
    image_threshold: float | None
    inference_time_ms: float


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _tensor_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    import anomalib
    import timm
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "anomalib": anomalib.__version__,
        "numpy": np.__version__,
    }


ImageSource = np.ndarray | str | Path


def _as_array(item: ImageSource) -> np.ndarray:
    return item if isinstance(item, np.ndarray) else load_image(item)


class PatchCoreDetector:
    def __init__(self, cfg: Config, seed: int = 0) -> None:
        # Imported lazily: anomalib takes ~20 s to import, and API workers should only pay that once.
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.cfg = cfg
        self.init_seed = seed
        self.device = _resolve_device(cfg.runtime.device)
        torch.set_num_threads(cfg.runtime.num_threads)
        # Seeded so that pre_trained=False (smoke tests) still gives a reproducible random backbone.
        torch.manual_seed(seed)
        self.model = PatchcoreModel(
            layers=list(cfg.model.layers),
            backbone=cfg.model.backbone,
            pre_trained=cfg.model.pre_trained,
            num_neighbors=cfg.model.num_neighbors,
        ).to(self.device)
        self.model.eval()
        self.image_threshold: float | None = None
        self.pixel_threshold: float | None = None
        self.calibration: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def is_fitted(self) -> bool:
        return self.model.memory_bank.numel() > 0

    def backbone_sha256(self) -> str:
        return _tensor_dict_sha256(self.model.feature_extractor.state_dict())

    def _batches(self, items: Sequence[ImageSource]) -> Iterator[torch.Tensor]:
        bs = self.cfg.runtime.batch_size
        for i in range(0, len(items), bs):
            chunk = [preprocess(_as_array(x), self.cfg.preprocessing) for x in items[i : i + bs]]
            yield torch.stack(chunk).to(self.device)

    # ------------------------------------------------------------------ fit / score
    def fit(self, images: Sequence[ImageSource]) -> dict[str, Any]:
        """Build the memory bank from normal images only."""
        if len(images) == 0:
            raise ValueError("Cannot fit PatchCore on an empty image list")
        t0 = time.perf_counter()
        self.model.train()  # in anomalib, train mode = collect embeddings (no gradients are computed)
        self.model.embedding_store.clear()
        with torch.no_grad():
            for batch in self._batches(images):
                self.model(batch)
        embeddings = torch.vstack(self.model.embedding_store)
        self.model.embedding_store.clear()
        extract_seconds = time.perf_counter() - t0
        self.model.memory_bank = self._subsample(embeddings)
        self.model.eval()
        stats = {
            "n_images": len(images),
            "n_patches": int(embeddings.shape[0]),
            "memory_bank_shape": list(self.model.memory_bank.shape),
            "extract_seconds": round(extract_seconds, 2),
            "sampling_seconds": round(time.perf_counter() - t0 - extract_seconds, 2),
            "fit_seconds": round(time.perf_counter() - t0, 2),
        }
        logger.info("Memory bank built: %s", stats)
        return stats

    def _subsample(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reduce patch embeddings to the memory bank, deterministically for a given sampling_seed.

        anomalib's KCenterGreedy draws its start index from torch's global RNG and its random projection
        from numpy's global RNG, so both are seeded here; otherwise two runs give different banks.
        """
        mc = self.cfg.model
        n = embeddings.shape[0]
        if mc.coreset_ratio >= 1.0:
            return embeddings
        k = max(1, int(n * mc.coreset_ratio))
        if mc.sampling == "random":
            g = torch.Generator().manual_seed(mc.sampling_seed)
            return embeddings[torch.randperm(n, generator=g)[:k].to(embeddings.device)]
        from anomalib.models.components import KCenterGreedy

        torch.manual_seed(mc.sampling_seed)
        np.random.seed(mc.sampling_seed)
        return KCenterGreedy(embedding=embeddings, sampling_ratio=mc.coreset_ratio).sample_coreset()

    def score(self, images: Sequence[ImageSource]) -> tuple[np.ndarray, np.ndarray]:
        """Return (image scores (N,), anomaly maps (N, S, S)) for a list of images."""
        if not self.is_fitted:
            raise NotFittedError("Memory bank is empty: fit() or load() a model first")
        self.model.eval()
        scores, maps = [], []
        with torch.no_grad():
            for batch in self._batches(images):
                out = self.model(batch)
                scores.append(out.pred_score.detach().cpu().reshape(-1))
                maps.append(out.anomaly_map.detach().cpu()[:, 0])
        return torch.cat(scores).numpy().astype(np.float64), torch.cat(maps).numpy().astype(np.float32)

    def calibrate(self, val_images: Sequence[ImageSource]) -> dict[str, Any]:
        """Set image/pixel thresholds from held-out normal images (never from the test split)."""
        if len(val_images) == 0:
            raise ValueError("Threshold calibration needs at least one validation image")
        scores, maps = self.score(val_images)
        tc = self.cfg.threshold
        self.image_threshold = float(np.quantile(scores, tc.image_quantile))
        self.pixel_threshold = float(np.quantile(maps, tc.pixel_quantile))
        self.calibration = {
            "method": tc.method,
            "n_val_normal": len(val_images),
            "image_quantile": tc.image_quantile,
            "pixel_quantile": tc.pixel_quantile,
            "val_score_min": float(scores.min()),
            "val_score_mean": float(scores.mean()),
            "val_score_max": float(scores.max()),
            "note": (
                f"With {len(val_images)} validation images, quantile {tc.image_quantile} is close to the "
                "validation maximum; the expected false-alarm rate on normals is only roughly bounded."
            ),
        }
        logger.info("Calibrated thresholds: image=%.4f pixel=%.4f", self.image_threshold, self.pixel_threshold)
        return self.calibration

    def predict(self, image: ImageSource) -> Prediction:
        t0 = time.perf_counter()
        scores, maps = self.score([image])
        elapsed = (time.perf_counter() - t0) * 1000
        score, amap = float(scores[0]), maps[0]
        return Prediction(
            score=score,
            is_anomalous=self.image_threshold is not None and score > self.image_threshold,
            anomaly_map=amap,
            defect_mask=None if self.pixel_threshold is None else amap > self.pixel_threshold,
            image_threshold=self.image_threshold,
            inference_time_ms=elapsed,
        )

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path, *, model_version: str, dataset: dict[str, Any]) -> dict[str, Any]:
        from safetensors.torch import save_file

        if not self.is_fitted:
            raise NotFittedError("Refusing to save an unfitted model")
        out_dir = Path(out_dir)
        if (out_dir / METADATA_FILE).exists():
            raise ArtifactError(f"Artifact already exists, refusing to overwrite: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        bank_path = out_dir / MEMORY_BANK_FILE
        save_file({"memory_bank": self.model.memory_bank.detach().cpu().contiguous()}, str(bank_path))
        metadata = {
            "artifact_format": ARTIFACT_FORMAT,
            "model_name": self.cfg.model.name,
            "model_version": model_version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": self.cfg.to_dict(),
            "init_seed": self.init_seed,  # only matters when pre_trained=False (random backbone)
            "dataset": dataset,
            "thresholds": {"image": self.image_threshold, "pixel": self.pixel_threshold},
            "calibration": self.calibration,
            "checksums": {
                "memory_bank_sha256": file_sha256(bank_path),
                "backbone_sha256": self.backbone_sha256(),
            },
            "environment": _library_versions(),
            "evaluation": {"status": "NOT_RUN"},
        }
        (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved artifact to %s", out_dir)
        return metadata

    @classmethod
    def load(
        cls, artifact_dir: str | Path, *, device: str | None = None, on_progress: Callable[[str], None] | None = None
    ) -> tuple[PatchCoreDetector, dict[str, Any]]:
        from safetensors.torch import load as load_safetensors

        artifact_dir = Path(artifact_dir)
        meta_path, bank_path = artifact_dir / METADATA_FILE, artifact_dir / MEMORY_BANK_FILE
        for p in (meta_path, bank_path):
            if not p.is_file():
                raise ArtifactError(f"Missing artifact file: {p}")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Invalid metadata JSON: {meta_path} ({exc})") from exc
        if metadata.get("artifact_format") != ARTIFACT_FORMAT:
            raise ArtifactError(f"Unsupported artifact_format: {metadata.get('artifact_format')}")

        expected = metadata["checksums"]["memory_bank_sha256"]
        # Read once, verify and deserialize the same bytes: no window between check and use, and no
        # memory-mapped handle left open (Windows would otherwise lock the file).
        bank_bytes = bank_path.read_bytes()
        actual = hashlib.sha256(bank_bytes).hexdigest()
        if actual != expected:
            raise ArtifactError(f"Memory bank checksum mismatch: expected {expected[:12]}..., got {actual[:12]}...")

        raw_cfg = metadata["config"]
        if device is not None:
            raw_cfg = {**raw_cfg, "runtime": {**raw_cfg["runtime"], "device": device}}
        cfg = config_from_dict(raw_cfg)
        if on_progress:
            on_progress("building backbone")
        det = cls(cfg, seed=metadata.get("init_seed", 0))
        backbone_hash = det.backbone_sha256()
        if backbone_hash != metadata["checksums"]["backbone_sha256"]:
            raise ArtifactError(
                "Backbone weights differ from the ones used at training time "
                "(different timm weight version or pre_trained setting). Retrain or restore the weights."
            )
        bank = load_safetensors(bank_bytes)["memory_bank"].to(det.device)
        det.model.memory_bank = bank
        det.image_threshold = metadata["thresholds"]["image"]
        det.pixel_threshold = metadata["thresholds"]["pixel"]
        det.calibration = metadata.get("calibration", {})
        return det, metadata
