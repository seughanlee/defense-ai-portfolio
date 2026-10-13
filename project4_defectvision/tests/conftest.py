from __future__ import annotations

from pathlib import Path

import pytest

from ml.config import Config, load_config
from ml.datasets.synthetic import SYNTHETIC_CATEGORY, make_synthetic_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "ml/configs/patchcore_cpu.yaml"

# Small, offline settings: random backbone weights (no download), tiny images, few threads.
SMOKE_OVERRIDES = {
    "model": {"pre_trained": False, "coreset_ratio": 0.1},
    "preprocessing": {"image_size": 64},
    "runtime": {"batch_size": 4, "num_threads": 2},
}


@pytest.fixture(scope="session")
def smoke_config() -> Config:
    return load_config(DEFAULT_CONFIG, SMOKE_OVERRIDES)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    make_synthetic_dataset(root, n_train=12, n_test_good=4, n_test_defect=4, size=64, seed=0)
    return root


@pytest.fixture(scope="session")
def category() -> str:
    return SYNTHETIC_CATEGORY
