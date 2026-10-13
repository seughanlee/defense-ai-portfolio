"""Result image storage. Paths are built only from server-generated UUIDs and a fixed set of names."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ARTIFACT_NAMES = ("original", "heatmap", "overlay", "mask")


class ResultStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, inspection_id: uuid.UUID) -> Path:
        if not isinstance(inspection_id, uuid.UUID):
            raise TypeError("inspection_id must be a UUID")
        path = (self.root / str(inspection_id)).resolve()
        if path.parent != self.root:  # defence in depth; a UUID string cannot contain separators
            raise ValueError("Resolved path escapes storage root")
        return path

    def path(self, inspection_id: uuid.UUID, name: str) -> Path:
        if name not in ARTIFACT_NAMES:
            raise ValueError(f"Unknown artifact name: {name}")
        return self._dir(inspection_id) / f"{name}.png"

    def save(self, inspection_id: uuid.UUID, images: dict[str, np.ndarray]) -> None:
        d = self._dir(inspection_id)
        d.mkdir(parents=True, exist_ok=False)
        for name, arr in images.items():
            Image.fromarray(arr).save(self.path(inspection_id, name), format="PNG")

    def delete(self, inspection_id: uuid.UUID) -> None:
        shutil.rmtree(self._dir(inspection_id), ignore_errors=True)
