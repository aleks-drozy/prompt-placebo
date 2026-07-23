import json

import pytest

from prereg.config import build_frozen_config
from prereg.freeze import (
    FreezeNotCommittedError,
    FrozenConfigMismatchError,
    check_freeze,
    config_hash,
    write_freeze,
)


def test_build_frozen_config_is_json_serializable():
    cfg = build_frozen_config()
    json.dumps(cfg)  # must not raise


def test_config_hash_is_deterministic():
    assert config_hash() == config_hash()


def test_config_hash_differs_for_different_configs():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_write_then_check_freeze_passes(tmp_path):
    path = tmp_path / "prereg_freeze.json"
    write_freeze(path=path)
    check_freeze(path=path)  # must not raise


def test_check_freeze_raises_when_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    with pytest.raises(FreezeNotCommittedError):
        check_freeze(path=path)


def test_check_freeze_raises_on_hash_mismatch(tmp_path):
    path = tmp_path / "tampered_freeze.json"
    artifact = {"config_hash": "0" * 64, "frozen_config": build_frozen_config()}
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(FrozenConfigMismatchError):
        check_freeze(path=path)
