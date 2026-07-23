"""Paired bootstrap confidence intervals for technique-vs-baseline deltas.

Every reported result in this project runs through this module. It resamples
the PAIRED PER-QUESTION delta (technique_correct[i] - baseline_correct[i]),
not each arm's accuracy separately -- both arms are scored on the same
question, so pairing removes question-difficulty as a source of noise and
gives a tighter, more honest interval than resampling the two arms
independently would. Adapted from the same pattern used for
skill-vs-baseline comparisons in a sibling repo (football-trajectory).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    delta: float       # mean(technique_correct) - mean(baseline_correct), paired
    ci_lower: float
    ci_upper: float
    n_boot: int
    seed: int


def paired_bootstrap_ci(
    technique_correct: list[bool] | tuple[bool, ...],
    baseline_correct: list[bool] | tuple[bool, ...],
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap CI over technique_correct[i] - baseline_correct[i].

    Both sequences must be the same length and paired by question index.
    Resamples question indices with replacement n_boot times (vectorized:
    one rng.integers draw), takes the mean paired delta per resample, and
    returns the (alpha/2, 1-alpha/2) percentile interval alongside the
    observed (non-resampled) delta.
    """
    if len(technique_correct) != len(baseline_correct):
        raise ValueError(
            "technique_correct and baseline_correct must be the same length "
            f"(got {len(technique_correct)} and {len(baseline_correct)})"
        )

    n = len(technique_correct)
    if n < 2:
        raise ValueError(f"need at least 2 paired questions, got {n}")

    paired = np.asarray(technique_correct, dtype=float) - np.asarray(baseline_correct, dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = paired[idx].mean(axis=1)

    ci_lower, ci_upper = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta = paired.mean()

    return BootstrapResult(
        delta=float(delta),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_boot=n_boot,
        seed=seed,
    )


class Verdict(str, Enum):
    STILL_WORKS = "still_works"
    ACTIVELY_HURTS = "actively_hurts"
    PLACEBO = "placebo"
    INCONCLUSIVE = "inconclusive"


def classify_verdict(result: BootstrapResult, equivalence_bound: float = 0.02) -> Verdict:
    """Pre-registered verdict rule (docs/prereg.md), checked in this order:

    1. PLACEBO if the CI is entirely within +/- equivalence_bound -- checked
       FIRST, even when the CI also happens to exclude 0, because a technique
       whose whole plausible effect is smaller than the equivalence bound is
       placebo, not "works", regardless of sign.
    2. STILL_WORKS if the CI excludes 0 on the positive side.
    3. ACTIVELY_HURTS if the CI excludes 0 on the negative side.
    4. INCONCLUSIVE otherwise.
    """
    if -equivalence_bound <= result.ci_lower and result.ci_upper <= equivalence_bound:
        return Verdict.PLACEBO
    if result.ci_lower > 0:
        return Verdict.STILL_WORKS
    if result.ci_upper < 0:
        return Verdict.ACTIVELY_HURTS
    return Verdict.INCONCLUSIVE
