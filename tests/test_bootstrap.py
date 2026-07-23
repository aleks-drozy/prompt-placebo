import numpy as np
import pytest

from stats.bootstrap import Verdict, classify_verdict, paired_bootstrap_ci


def test_mismatched_length_raises_value_error():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([True, False, True], [True, False])


def test_n_less_than_2_raises_value_error():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([True], [False])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])


def test_determinism_same_seed_same_result():
    rng = np.random.default_rng(123)
    technique = rng.random(50) > 0.4
    baseline = rng.random(50) > 0.5

    result1 = paired_bootstrap_ci(technique.tolist(), baseline.tolist(), seed=7)
    result2 = paired_bootstrap_ci(technique.tolist(), baseline.tolist(), seed=7)

    assert result1.ci_lower == result2.ci_lower
    assert result1.ci_upper == result2.ci_upper
    assert result1.delta == result2.delta


def test_different_seeds_give_different_ci():
    rng = np.random.default_rng(999)
    technique = (rng.random(37) > 0.45).tolist()
    baseline = (rng.random(37) > 0.55).tolist()

    result_a = paired_bootstrap_ci(technique, baseline, seed=1)
    result_b = paired_bootstrap_ci(technique, baseline, seed=2)

    assert (result_a.ci_lower, result_a.ci_upper) != (result_b.ci_lower, result_b.ci_upper)


def test_technique_always_right_baseline_always_wrong_still_works():
    technique = [True] * 30
    baseline = [False] * 30

    result = paired_bootstrap_ci(technique, baseline)

    assert result.delta == 1.0
    assert result.ci_lower > 0
    assert classify_verdict(result) == Verdict.STILL_WORKS


def test_technique_always_wrong_baseline_always_right_actively_hurts():
    technique = [False] * 30
    baseline = [True] * 30

    result = paired_bootstrap_ci(technique, baseline)

    assert result.delta == -1.0
    assert result.ci_upper < 0
    assert classify_verdict(result) == Verdict.ACTIVELY_HURTS


def test_identical_mixed_sequences_are_placebo():
    pattern = [True, False, True, True, False, False, True, False, True, False]
    sequence = (pattern * 20)[:200]

    result = paired_bootstrap_ci(sequence, sequence)

    assert result.delta == 0.0
    assert classify_verdict(result) == Verdict.PLACEBO


def test_small_n_noisy_signal_is_inconclusive():
    # Small N (10) with a small true delta: technique wins 6/10, baseline wins
    # 4/10 on different questions, so the paired delta is nonzero but the
    # bootstrap CI over only 10 paired observations is wide -- too wide to
    # be placebo-narrow (bound 0.02) but also too wide to exclude 0 cleanly.
    # Confirmed empirically: this exact data yields ci_lower < 0 < ci_upper
    # with a width well beyond the equivalence bound.
    technique = [True, True, True, False, False, True, False, True, False, False]
    baseline = [False, True, False, False, True, True, False, False, True, False]

    result = paired_bootstrap_ci(technique, baseline)

    assert result.ci_lower < 0 < result.ci_upper
    assert classify_verdict(result) == Verdict.INCONCLUSIVE
