import json

import numpy as np
import pytest

from scripts import analyze_p3
from scripts.analyze_p3 import (
    analyze,
    bootstrap_p_value,
    final_verdict,
    holm_bonferroni,
    load_results,
)
from stats.bootstrap import Verdict

# ---------------------------------------------------------------------------
# Synthetic paired data construction
# ---------------------------------------------------------------------------
# All synthetic families below use n=120 paired questions with a fixed
# baseline pattern (60 correct, 60 incorrect) and a "technique" pattern
# derived by flipping some of the 60 baseline-incorrect questions to
# correct. Effect sizes below were picked empirically (see scratch
# exploration) against the FROZEN seed/n_boot (prereg.config.BOOTSTRAP_SEED
# = 42, N_BOOT = 10_000) so the resulting p-values/CIs are deterministic and
# reproducible by anyone re-running this test.

_BASELINE = [True] * 60 + [False] * 60


def _make_technique(extra_correct_from_wrong: int) -> list[bool]:
    tech = list(_BASELINE)
    for i in range(60, 60 + extra_correct_from_wrong):
        tech[i] = True
    return tech


def _row(question_id, model_id, reasoning, arm_id, is_correct):
    return {
        "custom_id": f"{question_id}-{arm_id}",
        "question_id": question_id,
        "arm_id": arm_id,
        "model_id": model_id,
        "reasoning": reasoning,
        "raw_text": "ANSWER: 1",
        "extracted_answer": "1",
        "is_correct": is_correct,
        "input_tokens": 10,
        "output_tokens": 5,
    }


def _write_family_jsonl(path, model_id="claude-haiku-4-5", reasoning=None):
    """Writes a full 6-arm-plus-baseline family for task=math, covering:

    T1 -- identical to baseline: clearly PLACEBO.
    T2 -- large, consistent improvement: clearly STILL_WORKS, and survives
          Holm-Bonferroni (p ~ 0, comfortably under the strictest rank-1
          threshold alpha/6).
    T3 -- moderate improvement: CI excludes 0 (STILL_WORKS by the CI rule
          alone, p ~ 0.034 < alpha=0.05) but its raw p fails its Holm rank
          threshold (rank 2, alpha/5=0.01), so it must be downgraded to
          INCONCLUSIVE once Holm-Bonferroni is applied across the family of
          6. This is the case that proves the correction isn't a no-op.
    T4, T5, T6 -- small noisy deltas: INCONCLUSIVE under the CI rule alone,
          unaffected by Holm (never upgraded).
    """
    baseline = _BASELINE
    arms = {
        "T1": baseline,  # placebo (identical)
        "T2": _make_technique(15),  # strong, survives Holm
        "T3": _make_technique(4),  # moderate, downgraded by Holm
        "T4": _make_technique(1),  # noise
        "T5": _make_technique(2),  # noise
        "T6": _make_technique(3),  # noise
    }

    rows = []
    for i, correct in enumerate(baseline):
        qid = f"math-{i:04d}"
        rows.append(_row(qid, model_id, reasoning, "B", correct))
    for arm_id, pattern in arms.items():
        for i, correct in enumerate(pattern):
            qid = f"math-{i:04d}"
            rows.append(_row(qid, model_id, reasoning, arm_id, correct))

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# §7 missing/empty/malformed data behavior
# ---------------------------------------------------------------------------


def test_missing_file_exits_cleanly_not_crash(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        load_results(missing)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert "P3" in err


def test_empty_file_exits_cleanly(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_results(empty)

    assert exc_info.value.code == 1
    assert "zero parseable rows" in capsys.readouterr().err


def test_whitespace_only_file_exits_cleanly(tmp_path, capsys):
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n\n   \n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        load_results(blank)

    assert exc_info.value.code == 1
    assert "zero parseable rows" in capsys.readouterr().err


def test_malformed_line_fails_loudly_with_line_number(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"question_id": "math-0001", "is_correct": true}\n'
        "{not valid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        load_results(bad)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "bad.jsonl:2" in err  # line number of the malformed row


def test_analyze_exits_cleanly_when_data_path_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(analyze_p3, "DATA_PATH", tmp_path / "nope.jsonl")

    with pytest.raises(SystemExit) as exc_info:
        analyze()

    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# bootstrap_p_value
# ---------------------------------------------------------------------------


def test_bootstrap_p_value_all_positive_draws_is_tiny():
    boot = np.full(10_000, 0.05)
    assert bootstrap_p_value(boot) == pytest.approx(0.0)


def test_bootstrap_p_value_symmetric_around_zero_is_near_one():
    boot = np.concatenate([np.full(5000, 0.01), np.full(5000, -0.01)])
    assert bootstrap_p_value(boot) == pytest.approx(1.0)


def test_bootstrap_p_value_capped_at_one():
    boot = np.zeros(100)  # every draw touches both sides
    assert bootstrap_p_value(boot) <= 1.0


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_bonferroni_all_significant():
    p_values = {"T1": 0.001, "T2": 0.002, "T3": 0.003}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result == {"T1": True, "T2": True, "T3": True}


def test_holm_bonferroni_none_significant():
    p_values = {"T1": 0.9, "T2": 0.8, "T3": 0.7}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result == {"T1": False, "T2": False, "T3": False}


def test_holm_bonferroni_step_down_blocking():
    """Once a rank fails its threshold, every later (higher-p) rank is
    blocked too -- even if a later rank's raw p would individually clear
    its own (larger, less strict) threshold in isolation.

    Family of 4, alpha=0.05: thresholds by rank are alpha/4, alpha/3,
    alpha/2, alpha/1 = 0.0125, 0.0167, 0.025, 0.05.
      rank1 p=0.01   <= 0.0125  -> pass
      rank2 p=0.02   >  0.0167  -> FAIL (blocks the rest)
      rank3 p=0.02   <= 0.025   -> would pass in isolation, but blocked
      rank4 p=0.03   <= 0.05    -> would pass in isolation, but blocked
    """
    p_values = {"A": 0.01, "B": 0.02, "C": 0.02, "D": 0.03}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result["A"] is True
    assert result["B"] is False
    assert result["C"] is False
    assert result["D"] is False


# ---------------------------------------------------------------------------
# final_verdict combination rule
# ---------------------------------------------------------------------------


def test_final_verdict_placebo_always_wins_regardless_of_holm():
    verdict, downgraded = final_verdict(Verdict.PLACEBO, holm_significant=True)
    assert verdict == Verdict.PLACEBO
    assert downgraded is False

    verdict, downgraded = final_verdict(Verdict.PLACEBO, holm_significant=False)
    assert verdict == Verdict.PLACEBO
    assert downgraded is False


def test_final_verdict_still_works_survives_with_holm_significant():
    verdict, downgraded = final_verdict(Verdict.STILL_WORKS, holm_significant=True)
    assert verdict == Verdict.STILL_WORKS
    assert downgraded is False


def test_final_verdict_still_works_downgraded_without_holm_significance():
    verdict, downgraded = final_verdict(Verdict.STILL_WORKS, holm_significant=False)
    assert verdict == Verdict.INCONCLUSIVE
    assert downgraded is True


def test_final_verdict_actively_hurts_downgraded_without_holm_significance():
    verdict, downgraded = final_verdict(Verdict.ACTIVELY_HURTS, holm_significant=False)
    assert verdict == Verdict.INCONCLUSIVE
    assert downgraded is True


def test_final_verdict_inconclusive_never_upgraded():
    verdict, downgraded = final_verdict(Verdict.INCONCLUSIVE, holm_significant=True)
    assert verdict == Verdict.INCONCLUSIVE
    assert downgraded is False


# ---------------------------------------------------------------------------
# End-to-end: analyze() over a synthetic family covering all three headline
# cases (placebo, still-works-surviving-Holm, still-works-downgraded-by-Holm)
# ---------------------------------------------------------------------------


def test_analyze_end_to_end_placebo_survivor_and_holm_downgrade(tmp_path, monkeypatch):
    data_path = tmp_path / "p3_results.jsonl"
    _write_family_jsonl(data_path)
    monkeypatch.setattr(analyze_p3, "DATA_PATH", data_path)

    records = analyze()
    by_arm = {r["arm_id"]: r for r in records if r["model_id"] == "claude-haiku-4-5" and r["reasoning"] is None}

    assert set(by_arm) == {"T1", "T2", "T3", "T4", "T5", "T6"}

    # T1: clearly placebo.
    t1 = by_arm["T1"]
    assert t1["ci_verdict"] == "placebo"
    assert t1["verdict"] == "placebo"
    assert t1["downgraded_by_holm"] is False

    # T2: clearly still-works, survives Holm-Bonferroni across the family of 6.
    t2 = by_arm["T2"]
    assert t2["ci_verdict"] == "still_works"
    assert t2["holm_significant"] is True
    assert t2["verdict"] == "still_works"
    assert t2["downgraded_by_holm"] is False

    # T3: looks still-works by the CI rule alone, but gets downgraded to
    # INCONCLUSIVE once Holm-Bonferroni correction is applied across the
    # family of 6 -- proves the correction is doing real work, not a no-op.
    t3 = by_arm["T3"]
    assert t3["ci_verdict"] == "still_works"
    assert t3["holm_significant"] is False
    assert t3["verdict"] == "inconclusive"
    assert t3["downgraded_by_holm"] is True

    # Every record in this family carries the correct family size.
    for arm_id in by_arm:
        assert by_arm[arm_id]["holm_family_size"] == 6

    # n_pairs/n_expected are both recorded (this synthetic family uses 120
    # pairs, not the frozen math N=449, so this also exercises the
    # incomplete-pairing bookkeeping from spec L4).
    assert t2["n_pairs"] == 120
    assert t2["n_expected"] == 449


def test_analyze_downgrade_count_is_visible_in_records(tmp_path, monkeypatch):
    data_path = tmp_path / "p3_results.jsonl"
    _write_family_jsonl(data_path)
    monkeypatch.setattr(analyze_p3, "DATA_PATH", data_path)

    records = analyze()
    downgraded = [r for r in records if r["downgraded_by_holm"]]
    assert len(downgraded) == 1
    assert downgraded[0]["arm_id"] == "T3"
