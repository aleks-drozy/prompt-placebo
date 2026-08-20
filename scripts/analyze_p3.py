"""Final verdict analysis for P3 (the full run).

For every (task set, model, reasoning, technique arm) comparison, runs the
frozen paired bootstrap against that family's baseline, applies the frozen
equivalence/CI verdict rule (stats.bootstrap.classify_verdict), and layers a
Holm-Bonferroni correction across the family of non-baseline arms on top of
it. See §4 and §8 (L1, L2) of the spec this script implements for exactly how
the CI rule and Holm compose -- in short: Holm can only downgrade a
STILL_WORKS/ACTIVELY_HURTS verdict to INCONCLUSIVE, never upgrade anything,
and PLACEBO (a magnitude claim) is decided by the CI rule alone.

Usage:
    python scripts/analyze_p3.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Bootstrap the repo root onto sys.path so `prereg`/`stats` are importable
# regardless of how this script is invoked (`python scripts/analyze_p3.py`
# only puts scripts/ on sys.path, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prereg.config import ALPHA, BOOTSTRAP_SEED, EQUIVALENCE_BOUND_PP, MODELS, N_BOOT, TASK_SETS
from prereg.arms import ARMS_BY_ID
from stats.bootstrap import Verdict, classify_verdict, paired_bootstrap

DATA_PATH = Path("data/p3_results.jsonl")
OUTPUT_PATH = Path("data/p3_verdicts.json")
FREEZE_PATH = Path("data/prereg_freeze.json")


def _task_type_from_question_id(question_id: str) -> str:
    """Copied from scripts/analyze_pilot.py -- see that module's docstring
    for why this is a copy rather than a shared import (repo's no-refactor
    freeze posture)."""
    if question_id.startswith("math-"):
        return "math"
    if question_id.startswith("logic-"):
        return "logic"
    if question_id.startswith("proc-"):
        return "procedural"
    raise ValueError(f"unrecognized question id prefix: {question_id!r}")


def load_results(path: Path) -> list[dict]:
    """Load data/p3_results.jsonl. Exits cleanly (no traceback) if the file
    is missing, empty, or has zero parseable rows -- P3 analysis cannot run
    on data that doesn't exist yet. Malformed individual lines fail loudly
    with the line number: a final analysis must not silently skip data."""
    if not path.exists():
        print(
            "data/p3_results.jsonl not found -- P3 (the full run) has not been "
            "executed yet. Run scripts/run_p3.py first; no verdicts exist until it "
            "completes. (Nothing was analyzed; nothing was written.)",
            file=sys.stderr,
        )
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(
                f"{path}:{lineno}: malformed JSON line -- refusing to "
                f"silently skip data. ({exc})",
                file=sys.stderr,
            )
            sys.exit(1)

    if not rows:
        print(
            "data/p3_results.jsonl exists but contains zero parseable rows -- P3 "
            "(the full run) has not produced any data yet. (Nothing was analyzed; "
            "nothing was written.)",
            file=sys.stderr,
        )
        sys.exit(1)

    return rows


def index_results(rows: list[dict]) -> dict[tuple, dict[str, bool]]:
    """(task, model_id, reasoning, arm_id) -> {question_id: is_correct}."""
    index: dict[tuple, dict[str, bool]] = {}
    for row in rows:
        task_type = _task_type_from_question_id(row["question_id"])
        key = (task_type, row["model_id"], row["reasoning"], row["arm_id"])
        index.setdefault(key, {})[row["question_id"]] = row["is_correct"]
    return index


def bootstrap_p_value(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value: the fraction of bootstrap draws that
    cross zero, doubled for two-sidedness and capped at 1.0. Counts
    boundary draws (boot == 0) on both sides -- conservative, per spec L3."""
    p_le = float(np.mean(boot <= 0))
    p_ge = float(np.mean(boot >= 0))
    return min(1.0, 2 * min(p_le, p_ge))


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Standard Holm step-down over one family {arm_id: raw p}.

    Sort ascending; arm at rank i (1-based, m = len(family)) is significant
    iff p_(j) <= alpha / (m - j + 1) for ALL j <= i -- step-down: the first
    failure blocks all later (higher-p) ranks even if their own threshold
    would otherwise be cleared.
    """
    m = len(p_values)
    ranked = sorted(p_values.items(), key=lambda kv: kv[1])
    significant: dict[str, bool] = {}
    blocked = False
    for rank, (arm_id, p) in enumerate(ranked, start=1):
        threshold = alpha / (m - rank + 1)
        if blocked or p > threshold:
            blocked = True
            significant[arm_id] = False
        else:
            significant[arm_id] = True
    return significant


def final_verdict(ci_verdict: Verdict, holm_significant: bool) -> tuple[Verdict, bool]:
    """Combine the frozen CI rule with Holm-Bonferroni. Returns
    (verdict, downgraded_by_holm). See spec §4 / §8 L2:

    1. PLACEBO from the CI rule always stays PLACEBO, Holm-independent.
    2. STILL_WORKS/ACTIVELY_HURTS + holm_significant -> keep the CI verdict.
    3. STILL_WORKS/ACTIVELY_HURTS + not holm_significant -> downgrade to
       INCONCLUSIVE (flagged).
    4. INCONCLUSIVE never gets upgraded by Holm.
    """
    if ci_verdict == Verdict.PLACEBO:
        return Verdict.PLACEBO, False
    if ci_verdict in (Verdict.STILL_WORKS, Verdict.ACTIVELY_HURTS):
        if holm_significant:
            return ci_verdict, False
        return Verdict.INCONCLUSIVE, True
    return Verdict.INCONCLUSIVE, False


def analyze() -> list[dict]:
    rows = load_results(DATA_PATH)
    index = index_results(rows)

    records: list[dict] = []
    for task_cfg in TASK_SETS:
        task = task_cfg["name"]
        n_expected = task_cfg["n"]

        for model_cfg in MODELS:
            model_id = model_cfg["model_id"]
            reasoning = model_cfg["reasoning"]
            family_desc = f"task={task} model={model_id} reasoning={reasoning}"

            baseline = index.get((task, model_id, reasoning, "B"))
            if not baseline:
                print(f"WARNING: no baseline data for {family_desc} -- skipping family", file=sys.stderr)
                continue

            non_baseline_arms = [a for a in model_cfg["arms"] if a != "B"]

            # Pass 1: bootstrap every non-baseline arm in this family.
            family: dict[str, dict] = {}
            for arm_id in non_baseline_arms:
                technique = index.get((task, model_id, reasoning, arm_id))
                if not technique:
                    print(
                        f"WARNING: no data for arm {arm_id!r} in {family_desc} -- skipping arm",
                        file=sys.stderr,
                    )
                    continue

                shared_qids = sorted(set(baseline) & set(technique))
                n_pairs = len(shared_qids)
                if n_pairs != n_expected:
                    print(
                        f"WARNING: {family_desc} arm={arm_id} has {n_pairs} paired questions, "
                        f"expected {n_expected} -- analyzing the intersection (prereg deviation)",
                        file=sys.stderr,
                    )

                baseline_correct = [baseline[q] for q in shared_qids]
                technique_correct = [technique[q] for q in shared_qids]

                result, boot = paired_bootstrap(
                    technique_correct,
                    baseline_correct,
                    n_boot=N_BOOT,
                    seed=BOOTSTRAP_SEED,
                    alpha=ALPHA,
                )
                p_raw = bootstrap_p_value(boot)
                baseline_acc = sum(baseline_correct) / n_pairs
                technique_acc = sum(technique_correct) / n_pairs

                family[arm_id] = {
                    "result": result,
                    "p_raw": p_raw,
                    "n_pairs": n_pairs,
                    "baseline_acc": baseline_acc,
                    "technique_acc": technique_acc,
                }

            if not family:
                continue

            # Pass 2: Holm-Bonferroni within the family. m = actual number
            # of non-baseline arms present in this family (see spec §8 L1:
            # the adaptive-thinking row only has arm T1, so m=1 there --
            # i.e. no correction, which is what Holm with m=1 is).
            p_values = {arm_id: data["p_raw"] for arm_id, data in family.items()}
            significant = holm_bonferroni(p_values, ALPHA)
            m = len(family)
            ranked_arms = [arm_id for arm_id, _ in sorted(p_values.items(), key=lambda kv: kv[1])]
            rank_by_arm = {arm_id: rank for rank, arm_id in enumerate(ranked_arms, start=1)}

            # Pass 3: verdicts.
            for arm_id, data in family.items():
                result = data["result"]
                ci_v = classify_verdict(result, EQUIVALENCE_BOUND_PP)
                holm_sig = significant[arm_id]
                verdict, downgraded = final_verdict(ci_v, holm_sig)
                rank = rank_by_arm[arm_id]
                holm_threshold = ALPHA / (m - rank + 1)

                records.append(
                    {
                        "model_id": model_id,
                        "reasoning": reasoning,
                        "task_set": task,
                        "arm_id": arm_id,
                        "arm_name": ARMS_BY_ID[arm_id].name,
                        "n_pairs": data["n_pairs"],
                        "n_expected": n_expected,
                        "baseline_acc": data["baseline_acc"],
                        "technique_acc": data["technique_acc"],
                        "delta": result.delta,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                        "p_raw": data["p_raw"],
                        "holm_family_size": m,
                        "holm_rank": rank,
                        "holm_threshold": holm_threshold,
                        "holm_significant": holm_sig,
                        "ci_verdict": ci_v.value,
                        "verdict": verdict.value,
                        "downgraded_by_holm": downgraded,
                    }
                )

    return records


def _fmt_ci(record: dict) -> str:
    return f"[{record['ci_lower']:+.4f}, {record['ci_upper']:+.4f}]"


def _print_table(records: list[dict]) -> None:
    print("\nLegend: ci_verdict = frozen CI/equivalence rule alone; FINAL = ci_verdict")
    print("combined with Holm-Bonferroni (see stats.bootstrap.classify_verdict and")
    print("scripts/analyze_p3.py::final_verdict). '*' marks rows downgraded by Holm.\n")

    families: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["model_id"], r["reasoning"], r["task_set"])
        families.setdefault(key, []).append(r)

    downgraded_count = 0
    for (model_id, reasoning, task), fam_records in sorted(
        families.items(), key=lambda kv: (kv[0][0], kv[0][1] or "", kv[0][2])
    ):
        print(f"=== model={model_id} reasoning={reasoning} task={task} ===")
        print(f"{'arm':6s} {'delta':>8s} {'95% CI':>20s} {'p_raw':>8s} {'holm':>6s} {'ci_verdict':>14s} {'FINAL':>14s}")
        for r in sorted(fam_records, key=lambda r: r["arm_id"]):
            flag = "*" if r["downgraded_by_holm"] else " "
            if r["downgraded_by_holm"]:
                downgraded_count += 1
            print(
                f"{r['arm_id']:6s} {r['delta']:+8.4f} {_fmt_ci(r):>20s} {r['p_raw']:8.4f} "
                f"{str(r['holm_significant']):>6s} {r['ci_verdict']:>14s} {flag}{r['verdict']:>13s}"
            )
        print()

    print(f"Total rows downgraded by Holm-Bonferroni (STILL_WORKS/ACTIVELY_HURTS -> INCONCLUSIVE): {downgraded_count}")


def main() -> None:
    records = analyze()

    _print_table(records)

    config_hash = None
    if FREEZE_PATH.exists():
        config_hash = json.loads(FREEZE_PATH.read_text(encoding="utf-8")).get("config_hash")

    output = {
        "config_hash": config_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
