import pytest

from ml.config import load_config
from tests.conftest import DEFAULT_CONFIG


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.model.name == "patchcore"
    assert cfg.model.layers == ("layer2", "layer3")
    assert cfg.threshold.method == "val_normal_quantile"


def test_overrides_apply():
    cfg = load_config(DEFAULT_CONFIG, {"preprocessing": {"image_size": 128}})
    assert cfg.preprocessing.image_size == 128


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": {"coreset_ratio": 0}},
        {"data": {"val_fraction": 1.0}},
        {"threshold": {"image_quantile": 1.5}},
        {"threshold": {"method": "test_set_f1_max"}},  # tuning on the test split is not a supported method
        {"model": {"typo_key": 1}},
        {"unknown_section": {"a": 1}},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG, overrides)


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
