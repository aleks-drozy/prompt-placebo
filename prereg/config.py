"""Single source of truth for every decision this project's pre-registration
freezes before the full run. prereg/freeze.py hashes the dict this module
builds; changing anything a study should have committed to before seeing
results means the hash no longer matches and the runner refuses to proceed
(see prereg/freeze.py).
"""
from __future__ import annotations

from prereg.arms import ARMS

# Frozen statistical constants (docs/prereg.md, Statistics section).
N_BOOT = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
EQUIVALENCE_BOUND_PP = 0.02  # 2 percentage points

# Frozen model list (docs/prereg.md, Models section). "reasoning" is None for
# the main-grid cells (thinking explicitly disabled -- see harness/runner.py)
# and "adaptive" for the extended-thinking condition, which only runs the B
# and T1 arms on claude-sonnet-5.
MODELS = (
    {"model_id": "claude-haiku-4-5", "reasoning": None, "arms": tuple(a.id for a in ARMS)},
    {"model_id": "claude-sonnet-5", "reasoning": None, "arms": tuple(a.id for a in ARMS)},
    {"model_id": "claude-sonnet-5", "reasoning": "adaptive", "arms": ("B", "T1")},
)

# Frozen task sets (docs/prereg.md, Tasks section). N is filled in after the
# pilot (P2) -- None here means "not yet frozen"; the hash still covers this
# key, so a post-pilot update to N is a deliberate, visible re-freeze rather
# than a silent edit.
TASK_SETS = (
    {"name": "math", "n": None},
    {"name": "logic", "n": None},
    {"name": "procedural", "n": None},
)


def build_frozen_config() -> dict:
    """Assemble every frozen decision into one JSON-serializable dict."""
    return {
        "arms": [
            {"id": arm.id, "name": arm.name, "template": arm.template} for arm in ARMS
        ],
        "models": [dict(m) for m in MODELS],
        "task_sets": [dict(t) for t in TASK_SETS],
        "stats": {
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
            "equivalence_bound_pp": EQUIVALENCE_BOUND_PP,
        },
    }
