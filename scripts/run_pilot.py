"""Pilot run driver (P2): N=30/task, both models, all arms.

NOT run automatically by anything -- this script makes real, billed calls to
the Anthropic Batches API the moment it's executed. It requires a real
ANTHROPIC_API_KEY (see .env.example) and should only be run with explicit
sign-off, since it spends real money against the project's hard €25 cap.

TODO before this can run for real: FEW_SHOT_EXAMPLES below has placeholder
math examples and empty logic/procedural examples. The T6 arm needs 3 FIXED
worked examples per task set, frozen at pre-registration time (see
prereg/arms.py's module docstring: "fixed per task set, same for every
question" -- these are not meant to be authored casually, since they shape
a whole experimental arm).

Usage:
    python scripts/run_pilot.py
"""
from __future__ import annotations

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

# TODO: replace with the real, pre-registered worked examples before this
# script is run for real. These three math examples are placeholders; logic
# and procedural have none yet at all.
FEW_SHOT_EXAMPLES = {
    TaskType.MATH: (
        ("What is 15 + 27?", "ANSWER: 42"),
        ("What is 100 - 37?", "ANSWER: 63"),
        ("What is 8 * 9?", "ANSWER: 72"),
    ),
    TaskType.LOGIC: (),
    TaskType.PROCEDURAL: (),
}


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
    print(f"Spent so far: {cost_meter.total_spent_usd:.4f} / {BUDGET_CAP_USD:.2f} USD")
    print(f"Results so far: {RESULTS_PATH}")
    print("Batch may still be processing -- re-run this script to poll and continue.")


if __name__ == "__main__":
    main()
