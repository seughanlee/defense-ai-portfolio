"""Validate uploaded bytes as a real image of an allowed format, by content — not by extension or MIME."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.core.errors import APIError


@dataclass(frozen=True)
class ValidatedImage:
    rgb: np.ndarray  # (H, W, 3) uint8
    format: str
    width: int
    height: int
    sha256: str
    size_bytes: int


def sanitize_filename(name: str | None) -> str:
    """Display-only name: last path component, printable characters, at most 255 characters."""
    if not name:
        return "upload"
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in unicodedata.normalize("NFC", name) if unicodedata.category(ch)[0] != "C")
    name = name.strip().strip(".") or "upload"
    return name[:255]


def validate_image(data: bytes, *, max_bytes: int, max_pixels: int, allowed_formats: tuple[str, ...]) -> ValidatedImage:
    if not data:
        raise APIError(400, "EMPTY_FILE", "Uploaded file is empty")
    if len(data) > max_bytes:
        raise APIError(413, "FILE_TOO_LARGE", f"File exceeds {max_bytes} bytes")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                width, height = probe.size
                if fmt not in allowed_formats:
                    raise APIError(415, "UNSUPPORTED_IMAGE_FORMAT",
                                   f"Image format {fmt} is not allowed; use one of {list(allowed_formats)}")
                if width * height > max_pixels:
                    raise APIError(413, "IMAGE_TOO_LARGE", f"Image has {width * height} pixels; limit is {max_pixels}")
                probe.verify()  # structural check of the whole file
            with Image.open(io.BytesIO(data)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)  # full decode
    except APIError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise APIError(400, "INVALID_IMAGE", "File is not a readable image") from exc
    return ValidatedImage(rgb, fmt, width, height, hashlib.sha256(data).hexdigest(), len(data))
