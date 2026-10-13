"""Preprocessing shared by training, evaluation and the API so all three see identical inputs."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from ml.config import PreprocessingConfig


def preprocess(image: np.ndarray, cfg: PreprocessingConfig) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> normalised float tensor (3, S, S)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected RGB uint8 array (H, W, 3), got shape={image.shape} dtype={image.dtype}")
    size = cfg.image_size
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(cfg.mean).view(3, 1, 1)
    std = torch.tensor(cfg.std).view(3, 1, 1)
    return (x - mean) / std


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Boolean mask -> boolean mask at (size, size). Nearest-neighbour keeps it binary."""
    out = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)
