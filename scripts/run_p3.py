"""Full pre-registered run driver (P3): frozen Ns per task set, both models,
all arms.

NOT run automatically by anything -- this script makes real, billed calls to
the Anthropic Batches API the moment it's executed. It requires a real
ANTHROPIC_API_KEY (see .env.example) and should only be run with explicit
sign-off, since it spends real money against the project's hard $25 cap.

Unlike the pilot (P2, N=30/task, a single small batch), P3 uses the frozen
per-task-set Ns from prereg/config.py (449 math / 250 logic / 739
procedural), which multiplied across 16 model/arm rows is 23,008 requests --
far too large to submit as one batch under the $25 cap (BatchRunner's
pre-submission projection would reject it outright; see
harness/runner.py's `_load_or_submit_batch`). This script instead submits in
budget-fitted chunks, one in-flight batch at a time, driving BatchRunner
through as many chunks as it takes, polling until each is done before
submitting the next.

Idempotent/resumable exactly like the pilot at the level of any single
chunk: kill this script any time; re-running it resumes an in-flight batch
(by passing the full `cells` superset, per BatchRunner's stability
contract) or submits the next not-yet-recorded chunk. A fully-recorded run
costs nothing extra to re-invoke.

IMPORTANT -- never pool results across files: data/pilot_results.jsonl and
data/p3_results.jsonl share some custom_ids (all 30 pilot math/logic
questions reappear in P3's larger samples; procedural ids
proc-42-0000..0029 collide too, though with different prompts pre- vs
post-hardening of harness/task_generator.py -- see docs/prereg.md and
DECISIONS.md). Analysis must read ONLY data/p3_results.jsonl for the
pre-registered full run.

Usage:
    python scripts/run_p3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Bootstrap the repo root onto sys.path so `harness`/`prereg` are importable
# regardless of how this script is invoked (`python scripts/run_p3.py` only
# puts scripts/ on sys.path, not the repo root -- unlike `python -m
# scripts.run_p3`, which would add the cwd automatically).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from harness.anthropic_client import build_real_batch_client
from harness.cost_meter import CostMeter
from harness.datasets import fetch_logic_questions, fetch_math_questions
from harness.runner import BatchRunner, RunCell, build_run_grid
from harness.schema import TaskType
from harness.task_generator import generate_procedural_questions
from prereg.config import TASK_SETS
from prereg.freeze import check_freeze

P3_SEED = 42  # same sampling seed as the pilot -- see module docstring re: overlap
BUDGET_CAP_USD = 25.0

RESULTS_PATH = Path("data/p3_results.jsonl")
STATE_PATH = Path("data/p3_batch_state.json")

MAX_TOKENS = 2048  # same as BatchRunner's default; kept explicit here since
                    # this script also uses it directly for cost projection
POLL_INTERVAL_S = 30
CHUNK_SAFETY_FRACTION = 0.9  # submit chunks worth at most 90% of remaining
                             # budget's worst-case projection, leaving margin
                             # for the runner's own pre-submission check

# Fixed per task set, same for every question (see prereg/arms.py's module
# docstring) -- original questions, not copied from the real GSM8K/BBH pool,
# so there's no risk of a few-shot example leaking an actual eval question.
#
# Byte-for-byte copy of run_pilot.py's FEW_SHOT_EXAMPLES. This is NOT part
# of the hashed frozen config (prereg/config.py's build_frozen_config()
# only hashes arms/models/task_sets/stats -- the T6 arm freezes just the
# template, not example text), but docs/prereg.md's prose pre-registers T6
# as reusing the "same 3 examples for every question in that task set", so
# verbatim reuse here is a methodological requirement even though the
# freeze hash gate can't catch a drift from it. See tests/test_run_p3.py
# for a guard asserting this dict matches run_pilot.py's.
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


def _frozen_n(name: str) -> int:
    """Look up a task set's frozen N from prereg/config.py -- never hardcoded
    here, so a re-freeze that changes N is automatically picked up."""
    ns = {t["name"]: t["n"] for t in TASK_SETS}
    n = ns[name]  # KeyError if the task set was renamed -- loud failure
    if n is None:
        raise SystemExit(f"task set {name!r} has n=None -- N not yet frozen (see prereg/config.py)")
    return n


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


def _replay_recorded_spend(runner: BatchRunner, meter: CostMeter) -> None:
    """Seed a freshly-constructed CostMeter with every already-recorded
    row's cost, so remaining_usd() reflects TRUE cumulative spend across
    process invocations rather than resetting to the full cap every time
    this script is re-run. If replay itself raises BudgetExceededError, the
    cap is already blown -- let it propagate."""
    if not runner.results_path.exists():
        return
    for line in runner.results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            meter.record(r["model_id"], r["input_tokens"], r["output_tokens"])


def _est_input_tokens(cell: RunCell, few_shot_examples) -> int:
    """Mirror harness/runner.py's own pre-flight estimate (~4 chars/token,
    rounded up) so chunk sizing uses the same conservative worst-case basis
    the runner's own submission gate will re-check."""
    from prereg.arms import FEW_SHOT_ARM_ID
    from harness.prompt_builder import build_prompt

    examples = few_shot_examples.get(cell.question.task_type, ()) if cell.arm_id == FEW_SHOT_ARM_ID else ()
    prompt_text = build_prompt(cell.question, cell.arm_id, few_shot_examples=examples)
    return max(1, -(-len(prompt_text) // 4))


def _next_chunk(pending: list[RunCell], meter: CostMeter, few_shot_examples) -> list[RunCell]:
    """Longest PREFIX of `pending` whose worst-case cost projection fits
    under CHUNK_SAFETY_FRACTION * meter.remaining_usd(). A prefix (rather
    than some other selection) keeps chunking deterministic given
    build_run_grid's deterministic cell order."""
    budget = CHUNK_SAFETY_FRACTION * meter.remaining_usd()
    chunk: list[RunCell] = []
    projected = 0.0
    for cell in pending:
        c = meter.estimate_cost(cell.model_id, _est_input_tokens(cell, few_shot_examples), MAX_TOKENS)
        if projected + c > budget:
            break
        chunk.append(cell)
        projected += c
    if not chunk:
        raise SystemExit("budget remaining cannot fit even one worst-case request -- stopping")
    return chunk


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) "
            "before running P3 -- this script makes real, billed API calls."
        )

    # VERY NEXT action, no try/except of any kind: FreezeNotCommittedError /
    # FrozenConfigMismatchError must propagate uncaught. This is the
    # pre-registration integrity gate -- a full run must never proceed
    # against an un-frozen or drifted config.
    check_freeze()

    math_qs = fetch_math_questions(n=_frozen_n("math"), seed=P3_SEED)
    logic_qs = fetch_logic_questions(n=_frozen_n("logic"), seed=P3_SEED)
    proc_qs = generate_procedural_questions(n=_frozen_n("procedural"), seed=P3_SEED)
    all_questions = math_qs + logic_qs + proc_qs

    cells = build_run_grid(all_questions)
    cost_meter = CostMeter(cap_usd=BUDGET_CAP_USD, batch=True)
    runner = BatchRunner(build_real_batch_client(), cost_meter, RESULTS_PATH, STATE_PATH)
    _replay_recorded_spend(runner, cost_meter)

    print(f"P3 grid: {len(cells)} requests across {len(all_questions)} questions.")

    while True:
        completed = runner._load_completed_custom_ids()
        pending = [c for c in cells if c.custom_id not in completed]
        recorded_cost = _total_recorded_cost_usd(runner)
        print(
            f"Progress: {len(cells) - len(pending)}/{len(cells)} recorded, "
            f"spend so far {recorded_cost:.4f} USD, remaining {cost_meter.remaining_usd():.4f} USD"
        )
        if not pending:
            break

        if runner.state_path.exists():
            # A batch is in flight: pass the FULL cells superset (satisfies
            # BatchRunner's stability contract even if a crash mid-recording
            # shrank what a fresh chunk computation would produce). run()
            # only polls/collects an in-flight batch, it never resubmits
            # while state_path exists.
            runner.run(cells, few_shot_examples=FEW_SHOT_EXAMPLES, max_tokens=MAX_TOKENS)
            if runner.state_path.exists():
                print(f"Batch still processing -- polling again in {POLL_INTERVAL_S}s.")
                time.sleep(POLL_INTERVAL_S)
            continue

        chunk = _next_chunk(pending, cost_meter, FEW_SHOT_EXAMPLES)
        print(f"Submitting next chunk: {len(chunk)} requests.")
        runner.run(tuple(chunk), few_shot_examples=FEW_SHOT_EXAMPLES, max_tokens=MAX_TOKENS)

    print(f"Total spent so far: {_total_recorded_cost_usd(runner):.4f} USD (cap: {BUDGET_CAP_USD:.2f})")
    print(f"Results so far: {RESULTS_PATH}")
    if _run_is_fully_recorded(runner, cells):
        print("All results recorded -- P3 complete.")
    else:
        print("Batch still in flight -- re-run this script to poll and continue.")


if __name__ == "__main__":
    main()
