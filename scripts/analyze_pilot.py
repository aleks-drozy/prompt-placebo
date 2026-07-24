"""Analyze the pilot run: recommend a per-task-set N for the full run.

Uses the pilot's own observed variance (not a guessed one) to compute how
many questions the full run needs per task set to reliably detect a
3-percentage-point accuracy difference, then cross-checks the projected
full-run cost -- using the pilot's own OBSERVED average token usage per
model, not a guess -- against the remaining €25 budget. Per the project's
own plan: if the projection would breach the cap, cut N, not the cap.

Combines two results files: the original pilot (data/pilot_results.jsonl,
math + logic + the FIRST, since-superseded procedural attempt) and the
post-hardening procedural re-check (data/procedural_check_results.jsonl).
The original pilot's procedural rows are excluded from the merge -- they
were generated at the old, too-easy difficulty and no longer describe the
task set the full run will actually use.

Usage:
    python scripts/analyze_pilot.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from harness.cost_meter import CostMeter
from prereg.config import MODELS
from stats.power_analysis import paired_delta_variance, required_sample_size

PILOT_RESULTS_PATH = Path("data/pilot_results.jsonl")
PROCEDURAL_CHECK_RESULTS_PATH = Path("data/procedural_check_results.jsonl")
EFFECT_SIZE = 0.03  # detect a 3-percentage-point accuracy difference
ALPHA = 0.05
POWER = 0.80
N_COMPARISONS = 6  # non-baseline arms compared per model x task (Holm-Bonferroni)
BUDGET_CAP_USD = 25.0

# Hard data-availability ceilings, independent of budget: BBH date_understanding's
# ENTIRE test split is 250 rows -- there is no more real data to add regardless
# of how much budget remains. procedural has no such cap (generated on demand).
DATA_CAPS = {"logic": 250}


def _task_type_from_question_id(question_id: str) -> str:
    if question_id.startswith("math-"):
        return "math"
    if question_id.startswith("logic-"):
        return "logic"
    if question_id.startswith("proc-"):
        return "procedural"
    raise ValueError(f"unrecognized question id prefix: {question_id!r}")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_results() -> list[dict]:
    """Merge the original pilot (math + logic only -- its procedural rows
    are stale) with the post-hardening procedural re-check."""
    pilot_rows = [r for r in _load_jsonl(PILOT_RESULTS_PATH) if not r["question_id"].startswith("proc-")]
    procedural_rows = _load_jsonl(PROCEDURAL_CHECK_RESULTS_PATH)
    return pilot_rows + procedural_rows


def already_spent_usd(rows: list[dict]) -> float:
    meter = CostMeter(cap_usd=float("inf"), batch=True)
    return sum(meter.estimate_cost(r["model_id"], r["input_tokens"], r["output_tokens"]) for r in rows)


def true_historical_spend_usd() -> float:
    """Total real money spent so far, including the original (now-superseded)
    procedural attempt -- that money is gone regardless of whether its data
    is still used for anything, so the budget check must count it."""
    all_pilot_rows = _load_jsonl(PILOT_RESULTS_PATH)  # includes stale procedural rows, deliberately
    all_procedural_check_rows = _load_jsonl(PROCEDURAL_CHECK_RESULTS_PATH)
    return already_spent_usd(all_pilot_rows + all_procedural_check_rows)


def recommend_n_per_task(rows: list[dict]) -> dict[str, dict]:
    """Worst-case (largest) required N across every (model, reasoning,
    technique-vs-baseline) comparison available for each task set, using a
    Holm-Bonferroni-conservative alpha (alpha / N_COMPARISONS -- the most
    stringent step of the correction the real analysis will apply, so this
    recommendation doesn't under-provision relative to that correction)."""
    index: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        task_type = _task_type_from_question_id(row["question_id"])
        key = (task_type, row["model_id"], row["reasoning"], row["arm_id"])
        index[key][row["question_id"]] = row["is_correct"]

    corrected_alpha = ALPHA / N_COMPARISONS
    recommendations = {}
    for task_type in ("math", "logic", "procedural"):
        worst = {"n": 0, "combo": None, "variance": None}
        for model_cfg in MODELS:
            model_id, reasoning = model_cfg["model_id"], model_cfg["reasoning"]
            baseline_by_qid = index.get((task_type, model_id, reasoning, "B"))
            if not baseline_by_qid:
                continue
            for arm_id in model_cfg["arms"]:
                if arm_id == "B":
                    continue
                technique_by_qid = index.get((task_type, model_id, reasoning, arm_id))
                if not technique_by_qid:
                    continue
                shared_qids = sorted(set(baseline_by_qid) & set(technique_by_qid))
                technique_correct = [technique_by_qid[q] for q in shared_qids]
                baseline_correct = [baseline_by_qid[q] for q in shared_qids]
                variance = paired_delta_variance(technique_correct, baseline_correct)
                if variance <= 0:
                    continue  # formula is undefined at 0; a worse combo elsewhere still drives the recommendation
                n = required_sample_size(variance, effect_size=EFFECT_SIZE, alpha=corrected_alpha, power=POWER)
                if n > worst["n"]:
                    worst = {"n": n, "combo": f"{model_id} / reasoning={reasoning} / {arm_id}", "variance": variance}
        recommendations[task_type] = worst
    return recommendations


def project_full_run_cost(rows: list[dict], n_per_task: dict[str, int]) -> float:
    """Project cost by scaling each (task set, model config)'s OWN OBSERVED
    average per-request token usage (measured from the pilot/check data, not
    guessed) up to that task's recommended N.

    Keyed by task set as well as model, not just model: blending token
    averages across task sets would be a bad approximation now that
    procedural's token profile (long multi-step arithmetic chains) differs
    substantially from math/logic's -- especially since a single blended
    average was previously (silently) skewed by procedural's now-superseded,
    much cheaper, trivially-easy responses.
    """
    meter = CostMeter(cap_usd=float("inf"), batch=True)
    tokens_by_config: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for row in rows:
        task_type = _task_type_from_question_id(row["question_id"])
        key = (task_type, row["model_id"], row["reasoning"])
        bucket = tokens_by_config[key]
        bucket[0] += row["input_tokens"]
        bucket[1] += row["output_tokens"]
        bucket[2] += 1

    total_cost = 0.0
    for task_type, n in n_per_task.items():
        for model_cfg in MODELS:
            key = (task_type, model_cfg["model_id"], model_cfg["reasoning"])
            sum_in, sum_out, count = tokens_by_config.get(key, (0.0, 0.0, 0))
            if count == 0:
                continue
            avg_in, avg_out = sum_in / count, sum_out / count
            per_request_cost = meter.estimate_cost(model_cfg["model_id"], avg_in, avg_out)
            n_arms = len(model_cfg["arms"])
            total_cost += n * n_arms * per_request_cost
    return total_cost


def main() -> None:
    rows = load_results()
    print(f"Loaded {len(rows)} pilot result rows.")

    spent = true_historical_spend_usd()
    remaining = BUDGET_CAP_USD - spent
    print(f"Total real spend so far (incl. the superseded procedural attempt): {spent:.4f} USD")
    print(f"Remaining budget: {remaining:.4f} USD\n")

    recommendations = recommend_n_per_task(rows)
    print("Recommended N per task set (worst-case comparison, Holm-Bonferroni-conservative alpha):")
    for task_type, rec in recommendations.items():
        if rec["combo"] is None:
            print(
                f"  {task_type:12s} UNDEFINED -- every arm/model combo showed zero paired "
                "variance vs baseline (a ceiling or floor effect: identical right/wrong "
                "pattern across every technique). The sample-size formula needs positive "
                "variance and has nothing to work with here -- this needs a human decision, "
                "not a silently-picked fallback number."
            )
            continue
        print(f"  {task_type:12s} N={rec['n']:>6d}   driven by {rec['combo']}   observed variance={rec['variance']:.4f}")

    undefined_tasks = [t for t, rec in recommendations.items() if rec["combo"] is None]
    ideal_n_per_task = {t: rec["n"] for t, rec in recommendations.items() if rec["combo"] is not None}
    if undefined_tasks:
        print(f"\n(Excluding {undefined_tasks} -- no N to project.)")

    # Split into FIXED tasks (already capped by real data availability --
    # can't be pushed higher with more budget, and shouldn't be scaled down
    # below what real data already provides) and FLEXIBLE tasks (only
    # limited by budget, so they absorb whatever cutting is needed).
    fixed_n, flexible_n = {}, {}
    for task_type, n in ideal_n_per_task.items():
        cap = DATA_CAPS.get(task_type)
        if cap is not None:
            fixed_n[task_type] = min(n, cap)
            if n > cap:
                print(f"\nNOTE: {task_type} ideal N={n} exceeds real data availability ({cap}) -- capped to {cap} regardless of budget.")
        else:
            flexible_n[task_type] = n

    fixed_cost = project_full_run_cost(rows, fixed_n)
    flexible_cost = project_full_run_cost(rows, flexible_n)
    budget_for_flexible = remaining - fixed_cost

    print(f"\nData-capped tasks {list(fixed_n)}: fixed cost {fixed_cost:.2f} USD")
    print(f"Flexible tasks {list(flexible_n)}: ideal cost {flexible_cost:.2f} USD, budget available for them {budget_for_flexible:.2f} USD")

    if fixed_cost > remaining:
        print("\nEven the data-capped tasks alone exceed the remaining budget -- this needs a human decision, not an automatic cut.")
        return

    if flexible_cost > budget_for_flexible:
        scale = budget_for_flexible / flexible_cost
        print("\nFlexible tasks' ideal cost exceeds their share of the budget -- cutting N, not the cap (per docs/prereg.md):")
        final_flexible = {}
        for task_type, n in flexible_n.items():
            capped_n = max(2, int(n * scale))
            final_flexible[task_type] = capped_n
            print(f"  {task_type:12s} {n} -> {capped_n}")
    else:
        final_flexible = flexible_n
        print("\nFlexible tasks fit within their share of the budget -- no cutting needed.")

    final_n = {**fixed_n, **final_flexible}
    final_cost = project_full_run_cost(rows, final_n)
    print(f"\nFINAL recommended N per task set: {final_n}")
    print(f"Total projected full-run cost: {final_cost:.2f} USD (remaining budget: {remaining:.2f} USD)")


if __name__ == "__main__":
    main()
