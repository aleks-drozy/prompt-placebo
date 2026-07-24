"""Tests for stats/power_analysis.py.

These tests exist to catch a wrong rational-approximation coefficient or a
subtly inverted sample-size formula -- either failure mode would silently
under- or over-provision the full run's sample size, so every numeric claim
here is checked against an independently hand-computed or textbook value,
never just against the module's own internal consistency.
"""
from __future__ import annotations

import pytest

from stats.power_analysis import (
    norm_ppf,
    paired_delta_variance,
    required_sample_size,
)


class TestNormPpf:
    def test_matches_well_known_reference_quantiles(self):
        # Standard textbook/table values for the inverse standard normal CDF.
        assert norm_ppf(0.975) == pytest.approx(1.960, abs=1e-3)
        assert norm_ppf(0.95) == pytest.approx(1.645, abs=1e-3)
        assert norm_ppf(0.80) == pytest.approx(0.842, abs=1e-3)
        assert norm_ppf(0.99) == pytest.approx(2.326, abs=1e-3)
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-3)

    def test_matches_reference_quantile_in_lower_tail_branch(self):
        # p < _P_LOW (0.02425) takes a different code branch than the
        # p >= 0.5 cases above -- exercise it directly, and its mirror-image
        # upper-tail case, so a sign/structural bug isolated to that branch
        # can't hide behind an all-p>=0.5 test suite.
        assert norm_ppf(0.01) == pytest.approx(-2.326, abs=1e-3)
        assert norm_ppf(0.001) == pytest.approx(-3.090, abs=1e-3)
        assert norm_ppf(0.999) == pytest.approx(3.090, abs=1e-3)

    def test_rejects_boundary_and_out_of_range_p(self):
        for bad_p in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError):
                norm_ppf(bad_p)


class TestPairedDeltaVariance:
    def test_hand_computed_example(self):
        # deltas = [1, 0, 0, 1, 0]; mean = 0.4
        # sum of squared deviations = (1-0.4)^2*2 + (0-0.4)^2*3
        #                           = 0.36*2 + 0.16*3 = 0.72 + 0.48 = 1.2
        # sample variance (ddof=1) = 1.2 / (5-1) = 0.3
        technique_correct = [True, True, False, True, False]
        baseline_correct = [False, True, False, False, False]
        assert paired_delta_variance(technique_correct, baseline_correct) == pytest.approx(0.3)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            paired_delta_variance([True, False], [True])

    def test_fewer_than_two_observations_raises(self):
        with pytest.raises(ValueError):
            paired_delta_variance([True], [False])
        with pytest.raises(ValueError):
            paired_delta_variance([], [])

    def test_identical_sequences_give_zero_variance(self):
        technique_correct = [True, False, True, True, False]
        baseline_correct = [True, False, True, True, False]
        assert paired_delta_variance(technique_correct, baseline_correct) == 0.0


class TestRequiredSampleSize:
    def test_hand_computed_example(self):
        # variance=0.25, effect_size=0.03, alpha=0.05, power=0.80
        # z_alpha2 = norm_ppf(0.975) = 1.959964 (~1.9600 to 4dp)
        # z_power  = norm_ppf(0.80)  = 0.841621 (~0.8416 to 4dp)
        # (z_alpha2 + z_power)^2 = (1.959964 + 0.841621)^2
        #                        = 2.801585^2 = 7.848880
        # n = ceil(7.848880 * 0.25 / 0.03^2)
        #   = ceil(1.962220 / 0.0009)
        #   = ceil(2180.244...)
        #   = 2181
        n = required_sample_size(variance=0.25, effect_size=0.03, alpha=0.05, power=0.80)
        assert n == 2181

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"variance": 0.0},
            {"variance": -1.0},
            {"effect_size": 0.0},
            {"effect_size": -0.01},
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"power": 0.0},
            {"power": 1.0},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        base = {"variance": 0.25, "effect_size": 0.03, "alpha": 0.05, "power": 0.80}
        base.update(kwargs)
        with pytest.raises(ValueError):
            required_sample_size(**base)

    def test_higher_variance_requires_larger_n(self):
        # Exact values (not just the inequality) so a monotonic-but-wrong
        # formula -- e.g. a missing square -- can't slip through.
        n_low = required_sample_size(variance=0.1, effect_size=0.03, alpha=0.05, power=0.80)
        n_high = required_sample_size(variance=0.4, effect_size=0.03, alpha=0.05, power=0.80)
        assert n_low == 873
        assert n_high == 3489
        assert n_high > n_low

    def test_smaller_effect_size_requires_larger_n(self):
        n_large_effect = required_sample_size(variance=0.25, effect_size=0.05, alpha=0.05, power=0.80)
        n_small_effect = required_sample_size(variance=0.25, effect_size=0.01, alpha=0.05, power=0.80)
        assert n_large_effect == 785
        assert n_small_effect == 19623
        assert n_small_effect > n_large_effect

    def test_returns_int(self):
        n = required_sample_size(variance=0.25, effect_size=0.03, alpha=0.05, power=0.80)
        assert isinstance(n, int)
