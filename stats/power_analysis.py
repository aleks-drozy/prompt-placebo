"""Pilot -> full-run sample size calculation.

The pilot (N=30 questions per task set) exists to measure real observed
variance so the full run's sample size can be chosen deliberately instead of
guessed. This module answers the load-bearing question that determines both
budget and scientific validity of the eventual full run: given the paired
technique-vs-baseline noise we actually observed, how many questions does the
full run need per task set to reliably detect a 3-percentage-point accuracy
difference?

This repo deliberately has no scipy dependency, so the inverse standard
normal CDF (needed for the sample-size formula's z-critical-values) is
implemented from scratch below rather than imported.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Rational approximation for the inverse standard normal CDF, published by
# Peter J. Acklam (https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/).
# This gives relative error <= 1.15e-9 on its own. We then take one step of
# Halley's rational method (as Acklam's own page recommends for applications
# needing full float precision) using math.erfc, which is Python standard
# library (not scipy), to push the result to within a few ULPs of the true
# value -- this also makes the result robust to a small error in any single
# coefficient transcribed above, since the refinement step corrects it.
_A1 = -3.969683028665376e01
_A2 = 2.209460984245205e02
_A3 = -2.759285104469687e02
_A4 = 1.383577518672690e02
_A5 = -3.066479806614716e01
_A6 = 2.506628277459239e00

_B1 = -5.447609879822406e01
_B2 = 1.615858368580409e02
_B3 = -1.556989798598866e02
_B4 = 6.680131188771972e01
_B5 = -1.328068155288572e01

_C1 = -7.784894002430293e-03
_C2 = -3.223964580411365e-01
_C3 = -2.400758277161838e00
_C4 = -2.549732539343734e00
_C5 = 4.374664141464968e00
_C6 = 2.938163982698783e00

_D1 = 7.784695709041462e-03
_D2 = 3.224671290700398e-01
_D3 = 2.445134137142996e00
_D4 = 3.754408661907416e00

_P_LOW = 0.02425
_P_HIGH = 1 - _P_LOW


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (probit): P(Z <= norm_ppf(p)) = p.

    Uses Acklam's rational approximation plus one Halley refinement step.
    Defined only on the open interval (0, 1); the CDF only reaches 0 and 1
    in the limit, so p=0 and p=1 have no finite z-value.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be strictly between 0 and 1, got {p}")

    if p < _P_LOW:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C1 * q + _C2) * q + _C3) * q + _C4) * q + _C5) * q + _C6) / (
            ((((_D1 * q + _D2) * q + _D3) * q + _D4) * q + 1)
        )
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A1 * r + _A2) * r + _A3) * r + _A4) * r + _A5) * r + _A6) * q
        ) / ((((((_B1 * r + _B2) * r + _B3) * r + _B4) * r + _B5) * r + 1))
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C1 * q + _C2) * q + _C3) * q + _C4) * q + _C5) * q + _C6) / (
            ((((_D1 * q + _D2) * q + _D3) * q + _D4) * q + 1)
        )

    # Halley refinement step: e is the CDF error at x, u rescales it into a
    # correction on x. This uses math.erfc (stdlib), not scipy.
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return x


def paired_delta_variance(
    technique_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
) -> float:
    """Sample variance (ddof=1) of the paired per-question deltas.

    d_i = int(technique_correct[i]) - int(baseline_correct[i]). Both
    sequences must be paired by index (same question) and the same length.
    This is the variance the sample-size formula needs: it is the noise in
    the paired difference we would actually be testing at full scale, not
    the noise in either arm's raw accuracy.
    """
    if len(technique_correct) != len(baseline_correct):
        raise ValueError(
            "technique_correct and baseline_correct must be the same length "
            f"(got {len(technique_correct)} and {len(baseline_correct)})"
        )

    n = len(technique_correct)
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")

    deltas = np.asarray(technique_correct, dtype=float) - np.asarray(baseline_correct, dtype=float)
    return float(deltas.var(ddof=1))


def required_sample_size(
    variance: float,
    effect_size: float = 0.03,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Paired observations needed to detect `effect_size` at `power`/`alpha`.

    Standard normal-approximation sample-size formula for a paired-difference
    test (equivalent to the one-sample/paired z-test case of "sample size for
    a test of two means with known variance"; see e.g. Chow, Shao & Wang,
    "Sample Size Calculations in Clinical Research", the one-sample mean-
    difference case):

        n = ceil( (z_(1-alpha/2) + z_power)^2 * variance / effect_size^2 )

    z_(1-alpha/2) = norm_ppf(1 - alpha/2) is the two-sided critical value;
    z_power = norm_ppf(power) is the one-sided quantile corresponding to the
    target power (NOT norm_ppf(1-power) -- that would invert the formula and
    silently under-provision the sample).

    `variance` must be the variance of the PAIRED per-question difference
    (see paired_delta_variance) -- not the variance of either arm's raw
    accuracy individually. Passing the wrong one silently produces a
    plausible-looking but wrong sample size.
    """
    if variance <= 0:
        raise ValueError(f"variance must be > 0, got {variance}")
    if effect_size <= 0:
        raise ValueError(f"effect_size must be > 0, got {effect_size}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be strictly between 0 and 1, got {power}")

    z_alpha2 = norm_ppf(1 - alpha / 2)
    z_power = norm_ppf(power)

    n = ((z_alpha2 + z_power) ** 2) * variance / (effect_size**2)
    return math.ceil(n)
