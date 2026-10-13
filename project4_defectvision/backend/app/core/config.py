"""Application settings, read from environment variables with the ``DV_`` prefix (or a .env file).

Development defaults run without any setup (SQLite, no auth, open docs). Production (``DV_ENV=prod``)
refuses to start without an API key, a non-SQLite database and explicit CORS origins.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DV_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    app_version: str = "0.2.0"
    log_level: str = "INFO"

    database_url: str = f"sqlite:///{(PROJECT_DIR / 'var' / 'defectvision.db').as_posix()}"
    storage_dir: Path = PROJECT_DIR / "var" / "inspections"

    # Path to a PatchCore artifact directory (memory_bank.safetensors + metadata.json). Unset = no model.
    model_artifact_dir: Path | None = None
    model_device: str | None = None  # overrides the device stored in the artifact (e.g. "cpu")

    max_upload_mb: float = Field(10.0, gt=0, le=100)
    max_image_pixels: int = Field(40_000_000, gt=0)  # decompression-bomb guard (~6300 x 6300)
    allowed_formats: tuple[str, ...] = ("PNG", "JPEG", "BMP")

    api_key: str | None = None  # when set, required as X-API-Key on every endpoint except health
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    enable_docs: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.env == "prod":
            problems = []
            if not self.api_key or len(self.api_key) < 16:
                problems.append("DV_API_KEY must be set (>= 16 characters)")
            if self.database_url.startswith("sqlite"):
                problems.append("DV_DATABASE_URL must point to PostgreSQL")
            if "*" in self.cors_origins:
                problems.append("DV_CORS_ORIGINS must list explicit origins")
            if problems:
                raise ValueError("Unsafe production settings: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
