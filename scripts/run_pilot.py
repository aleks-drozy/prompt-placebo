"""Pilot run driver (P2): N=30/task, both models, all arms.

NOT run automatically by anything -- this script makes real, billed calls to
the Anthropic Batches API the moment it's executed. It requires a real
ANTHROPIC_API_KEY (see .env.example) and should only be run with explicit
sign-off, since it spends real money against the project's hard €25 cap.

Idempotent: re-running while a batch is still in flight just polls it (see
BatchRunner's own resumability contract); re-running after it's already
fully recorded costs nothing extra and changes nothing.

Usage:
    python scripts/run_pilot.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from harness.cost_meter import CostMeter
from harness.datasets import fetch_logic_questions, fetch_math_questions
from harness.runner import BatchRunner, build_run_grid
from harness.schema import TaskType
from harness.task_generator import generate_procedural_questions

PILOT_N = 30
PILOT_SEED = 42
BUDGET_CAP_USD = 25.0

RESULTS_PATH = Path("data/pilot_results.jsonl")
STATE_PATH = Path("data/pilot_batch_state.json")

# Fixed per task set, same for every question (see prereg/arms.py's module
# docstring) -- original questions, not copied from the real GSM8K/BBH pool,
# so there's no risk of a few-shot example leaking an actual eval question.
# Reviewable/editable up until the pre-registration is frozen.
FEW_SHOT_EXAMPLES = {
    TaskType.MATH: (
        (
            "A bakery sells 12 muffins in the morning and 18 muffins in the "
            "afternoon. How many muffins did the bakery sell in total?",
            "The bakery sold 12 muffins in the morning and 18 in the afternoon. "
            "12 + 18 = 30.\nANSWER: 30",
        ),
        (
            "Maria has 5 boxes of pencils, and each box contains 8 pencils. "
            "How many pencils does Maria have in total?",
            "Maria has 5 boxes with 8 pencils each. 5 * 8 = 40.\nANSWER: 40",
        ),
        (
            "A train travels 60 miles in the first hour and 45 miles in the "
            "second hour. How many miles did the train travel in total?",
            "The train travels 60 miles then 45 miles. 60 + 45 = 105.\nANSWER: 105",
        ),
    ),
    TaskType.LOGIC: (
        (
            "Today is the first day of March, 2021. What is the date one week "
            "from today in MM/DD/YYYY?\nOptions:\n(A) 03/08/2021\n(B) 03/01/2021\n"
            "(C) 02/08/2021\n(D) 04/08/2021",
            "Today is 03/01/2021. One week later is 7 days after March 1st, "
            "which is March 8th, 2021.\nANSWER: A",
        ),
        (
            "Yesterday was December 31, 2019. What is the date today in "
            "MM/DD/YYYY?\nOptions:\n(A) 01/01/2020\n(B) 12/31/2019\n"
            "(C) 01/01/2019\n(D) 02/01/2020",
            "Yesterday was 12/31/2019, so today is the next day, 01/01/2020.\nANSWER: A",
        ),
        (
            "Today is New Year's Eve of 1999. What is the date tomorrow in "
            "MM/DD/YYYY?\nOptions:\n(A) 01/01/2000\n(B) 12/31/1999\n"
            "(C) 01/01/1999\n(D) 12/31/2000",
            "New Year's Eve of 1999 is 12/31/1999. Tomorrow is 01/01/2000.\nANSWER: A",
        ),
    ),
    TaskType.PROCEDURAL: (
        (
            "Start with 10, then add 5, then multiply by 2. What is the final result?",
            "10 + 5 = 15. Then 15 * 2 = 30.\nANSWER: 30",
        ),
        (
            "Begin with the number 20, then subtract 8. What number do you end up with?",
            "20 - 8 = 12.\nANSWER: 12",
        ),
        (
            "Take 6 as your starting number, then multiply by 3, then subtract 4. "
            "What is the result?",
            "6 * 3 = 18. Then 18 - 4 = 14.\nANSWER: 14",
        ),
    ),
}


def _run_is_fully_recorded(runner, cells) -> bool:
    """Whether every cell this run is responsible for has already landed in
    the results file. Checked from the caller's side (rather than having
    BatchRunner.run() return a status) so the already-reviewed, tested
    runner module doesn't need a new return contract for what's purely a
    driver-script reporting concern."""
    if runner.state_path.exists():
        return False  # a batch is still submitted/in-flight
    if not runner.results_path.exists():
        return False  # nothing recorded yet at all
    recorded = {
        json.loads(line)["custom_id"]
        for line in runner.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return all(cell.custom_id in recorded for cell in cells)


def _total_recorded_cost_usd(runner) -> float:
    """True cumulative spend across every call so far, recomputed from the
    results file's own recorded token counts -- NOT cost_meter.total_spent_usd,
    which only reflects whatever this single process instance has recorded
    (a fresh CostMeter is constructed every time this script runs, so its
    in-memory total is never actually cumulative across separate runs)."""
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


def _real_batches_client():
    """Adapt anthropic.Anthropic().messages.batches to harness.runner.BatchClient."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batches = anthropic.Anthropic().messages.batches

    class _Adapter:
        def create(self, requests):
            return batches.create(
                requests=[
                    Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
                    for r in requests
                ]
            )

        def retrieve(self, batch_id):
            return batches.retrieve(batch_id)

        def results(self, batch_id):
            return batches.results(batch_id)

    return _Adapter()


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) "
            "before running the pilot -- this script makes real, billed API calls."
        )

    math_questions = fetch_math_questions(n=PILOT_N, seed=PILOT_SEED)
    logic_questions = fetch_logic_questions(n=PILOT_N, seed=PILOT_SEED)
    procedural_questions = generate_procedural_questions(n=PILOT_N, seed=PILOT_SEED)
    all_questions = math_questions + logic_questions + procedural_questions

    cells = build_run_grid(all_questions)
    cost_meter = CostMeter(cap_usd=BUDGET_CAP_USD, batch=True)
    runner = BatchRunner(_real_batches_client(), cost_meter, RESULTS_PATH, STATE_PATH)

    print(f"Pilot grid: {len(cells)} requests across {len(all_questions)} questions.")
    runner.run(cells, few_shot_examples=FEW_SHOT_EXAMPLES)
    print(f"Total spent so far: {_total_recorded_cost_usd(runner):.4f} USD (cap: {BUDGET_CAP_USD:.2f})")
    print(f"Results so far: {RESULTS_PATH}")
    if _run_is_fully_recorded(runner, cells):
        print("All results recorded -- pilot complete.")
    else:
        print("Batch still in flight -- re-run this script to poll and continue.")


if __name__ == "__main__":
    main()
