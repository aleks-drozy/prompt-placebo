"""Re-check the procedural task set after hardening the generator.

The original pilot (data/pilot_results.jsonl) showed 100% accuracy on every
single arm/model combo for procedural questions -- a ceiling effect with
zero variance to compute a sample size from. harness/task_generator.py's
operand range and step count were widened in response; this script runs the
same 16 model/arm combos against a fresh batch of 30 harder procedural
questions, writing to its own results/state files so it doesn't disturb the
original pilot's historical record.

Same real-money caveat as scripts/run_pilot.py: this makes real, billed API
calls. Expected cost: proportional to run_pilot.py's $0.84 for 1,440
requests across 3 task sets -- this is 1 task set (480 requests), so
roughly a third of that.

Usage:
    python scripts/check_procedural.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from harness.anthropic_client import build_real_batch_client
from harness.cost_meter import CostMeter
from harness.runner import BatchRunner, build_run_grid
from harness.task_generator import generate_procedural_questions

CHECK_N = 30
CHECK_SEED = 42  # same seed as the original pilot; different questions anyway since the generator's ranges changed
BUDGET_CAP_USD = 25.0  # same project-wide cap; BatchRunner checks true remaining spend via this cap

RESULTS_PATH = Path("data/procedural_check_results.jsonl")
STATE_PATH = Path("data/procedural_check_batch_state.json")


def _run_is_fully_recorded(runner, cells) -> bool:
    if runner.state_path.exists():
        return False
    if not runner.results_path.exists():
        return False
    recorded = {
        json.loads(line)["custom_id"]
        for line in runner.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return all(cell.custom_id in recorded for cell in cells)


def _total_recorded_cost_usd(runner) -> float:
    if not runner.results_path.exists():
        return 0.0
    total = 0.0
    for line in runner.results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        total += runner.cost_meter.estimate_cost(
            record["model_id"], record["input_tokens"], record["output_tokens"]
        )
    return total


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to .env before running this check -- "
            "it makes real, billed API calls."
        )

    questions = generate_procedural_questions(n=CHECK_N, seed=CHECK_SEED)
    cells = build_run_grid(questions)

    cost_meter = CostMeter(cap_usd=BUDGET_CAP_USD, batch=True)
    runner = BatchRunner(build_real_batch_client(), cost_meter, RESULTS_PATH, STATE_PATH)

    print(f"Procedural check grid: {len(cells)} requests across {len(questions)} questions.")
    runner.run(cells)
    print(f"Total spent this results file: {_total_recorded_cost_usd(runner):.4f} USD (cap: {BUDGET_CAP_USD:.2f})")
    print(f"Results so far: {RESULTS_PATH}")
    if _run_is_fully_recorded(runner, cells):
        print("All results recorded -- check complete.")
    else:
        print("Batch still in flight -- re-run this script to poll and continue.")


if __name__ == "__main__":
    main()
