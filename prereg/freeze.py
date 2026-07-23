"""SHA-256 hash-freeze gate for the pre-registration.

The frozen config (prereg/config.py) is canonicalized to JSON and hashed;
the hash plus the config itself are committed to data/prereg_freeze.json.
Once that file is committed, any runner that would execute the full
experiment must call check_freeze() first and refuse to run on a mismatch --
silently editing an arm's wording or a stats constant after the freeze is
exactly the failure mode pre-registration exists to prevent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prereg.config import build_frozen_config

FREEZE_PATH = Path(__file__).resolve().parent.parent / "data" / "prereg_freeze.json"


def config_hash(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else build_frozen_config()
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_freeze(path: Path = FREEZE_PATH) -> dict:
    """Compute the current config hash and write the freeze artifact.

    Call this ONCE, deliberately, when the pre-registration is finalized --
    not from any automated pipeline.
    """
    cfg = build_frozen_config()
    artifact = {"config_hash": config_hash(cfg), "frozen_config": cfg}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


class FreezeNotCommittedError(RuntimeError):
    ...


class FrozenConfigMismatchError(RuntimeError):
    ...


def check_freeze(path: Path = FREEZE_PATH) -> None:
    """Refuse to proceed unless the current config matches the committed freeze.

    Raises FreezeNotCommittedError if no freeze artifact exists yet, or
    FrozenConfigMismatchError if the current config's hash no longer matches
    the committed one (a frozen decision was changed after the fact).
    """
    if not path.exists():
        raise FreezeNotCommittedError(
            f"no pre-registration freeze at {path} -- run "
            "prereg.freeze.write_freeze() and commit it before running the "
            "full experiment"
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    recomputed = config_hash()
    if recomputed != frozen["config_hash"]:
        raise FrozenConfigMismatchError(
            f"FROZEN CONFIG MISMATCH: recomputed {recomputed} != committed "
            f"{frozen['config_hash']} -- refusing to run (pre-registration "
            "violated)"
        )
