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
# pilot (P2) -- None means "not yet frozen"; the hash still covers this key,
# so a post-pilot update to N is a deliberate, visible re-freeze rather than
# a silent edit.
#
# math: 449, from scripts/analyze_pilot.py's power analysis on the real
# pilot (worst-case combo: claude-haiku-4-5 / T2, observed variance 0.0333).
#
# logic: 250, capped by real data availability -- the pilot's power analysis
# recommended 1795, but the BBH date_understanding test split (our only
# source for this task set) only has 250 questions total. See DECISIONS.md:
# accepted a coarser detectable effect (~4-8pp depending on arm, worst case
# on Haiku's noisiest arms) rather than diluting the task with unrelated BBH
# subtasks to inflate N.
#
# procedural: 739. The original pilot showed 100% accuracy on EVERY arm/model
# combo (a ceiling effect, zero variance); harness/task_generator.py's operand
# range and step count were hardened in response, and a fresh 30-question
# check (data/procedural_check_results.jsonl) confirmed real variance except
# for claude-sonnet-5 with reasoning on, which still hits 100% on both B and
# T1 -- a genuine, reported boundary finding (arithmetic under full reasoning
# is at ceiling for this task, not a bug). Ideal N from the worst DEFINED
# combo (claude-haiku-4-5 / T2, variance 0.1023) was 1377; capped to 739 to
# fit the remaining budget (with a 10% safety margin) after math (449) and
# logic (250, itself data-capped) are paid for -- math and logic were
# decided first and are not renegotiated just because procedural turned out
# more expensive per question than originally estimated.
TASK_SETS = (
    {"name": "math", "n": 449},
    {"name": "logic", "n": 250},
    {"name": "procedural", "n": 739},
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
