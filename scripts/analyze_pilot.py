"""Analyze the pilot run: recommend a per-task-set N for the full run.

Uses the pilot's own observed variance (not a guessed one) to compute how
many questions the full run needs per task set to reliably detect a
3-percentage-point accuracy difference, then cross-checks the projected
full-run cost -- using the pilot's own OBSERVED average token usage per
model, not a guess -- against the remaining €25 budget. Per the project's
own plan: if the projection would breach the cap, cut N, not the cap.

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

RESULTS_PATH = Path("data/pilot_results.jsonl")
EFFECT_SIZE = 0.03  # detect a 3-percentage-point accuracy difference
ALPHA = 0.05
POWER = 0.80
N_COMPARISONS = 6  # non-baseline arms compared per model x task (Holm-Bonferroni)
BUDGET_CAP_USD = 25.0


def _task_type_from_question_id(question_id: str) -> str:
    if question_id.startswith("math-"):
        return "math"
    if question_id.startswith("logic-"):
        return "logic"
    if question_id.startswith("proc-"):
        return "procedural"
    raise ValueError(f"unrecognized question id prefix: {question_id!r}")


def load_results(path: Path = RESULTS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def already_spent_usd(rows: list[dict]) -> float:
    meter = CostMeter(cap_usd=float("inf"), batch=True)
    return sum(meter.estimate_cost(r["model_id"], r["input_tokens"], r["output_tokens"]) for r in rows)


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
    """Project cost by scaling each model config's OBSERVED average
    per-request token usage (measured from the pilot, not guessed) up to
    the recommended N."""
    meter = CostMeter(cap_usd=float("inf"), batch=True)
    tokens_by_config: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for row in rows:
        key = (row["model_id"], row["reasoning"])
        bucket = tokens_by_config[key]
        bucket[0] += row["input_tokens"]
        bucket[1] += row["output_tokens"]
        bucket[2] += 1

    total_cost = 0.0
    for model_cfg in MODELS:
        key = (model_cfg["model_id"], model_cfg["reasoning"])
        sum_in, sum_out, count = tokens_by_config.get(key, (0.0, 0.0, 0))
        if count == 0:
            continue
        avg_in, avg_out = sum_in / count, sum_out / count
        per_request_cost = meter.estimate_cost(model_cfg["model_id"], avg_in, avg_out)
        n_arms = len(model_cfg["arms"])
        for n in n_per_task.values():
            total_cost += n * n_arms * per_request_cost
    return total_cost


def main() -> None:
    rows = load_results()
    print(f"Loaded {len(rows)} pilot result rows.")

    spent = already_spent_usd(rows)
    remaining = BUDGET_CAP_USD - spent
    print(f"Pilot spend so far: {spent:.4f} USD -- remaining budget: {remaining:.4f} USD\n")

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
    n_per_task = {t: rec["n"] for t, rec in recommendations.items() if rec["combo"] is not None}
    if undefined_tasks:
        print(f"\n(Excluding {undefined_tasks} from the cost projection below -- no N to project.)")

    projected_cost = project_full_run_cost(rows, n_per_task)
    print(f"\nProjected full-run cost at these Ns: {projected_cost:.2f} USD")
    print(f"Remaining budget: {remaining:.2f} USD")

    if projected_cost > remaining:
        scale = remaining / projected_cost
        print("\nProjected cost EXCEEDS remaining budget -- cutting N, not the cap (per docs/prereg.md):")
        capped = {}
        for task_type, n in n_per_task.items():
            capped_n = max(2, int(n * scale))
            capped[task_type] = capped_n
            print(f"  {task_type:12s} {n} -> {capped_n}")
        capped_cost = project_full_run_cost(rows, capped)
        print(f"Projected cost at capped Ns: {capped_cost:.2f} USD")
    else:
        print("\nFits within budget -- no capping needed.")


if __name__ == "__main__":
    main()
