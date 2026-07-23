"""Resumable, cost-metered batch runner.

Ties prompt_builder + extractors + cost_meter + prereg/config.py's frozen
model/arm grid together, driving the Anthropic Message Batches API through a
minimal injected client so this module never needs network access to test.

Resumability/idempotency in two layers:

1. A submitted-but-not-yet-`ended` batch's id is persisted to `state_path`.
   Calling run() again while the batch is still processing reads that id
   back and polls/retrieves instead of resubmitting -- resubmitting would
   double real spend for work that was already paid for.
2. Every scored result is appended (one JSON line per row) to `results_path`
   as soon as it's scored. A subsequent run() loads the set of already-
   recorded custom_ids and excludes them from both the pending-cells list
   and the results it processes from a re-fetched batch -- so a crash
   partway through result-processing never re-records or re-pays for a row
   that already made it to disk.

Stability contract: the `cells` tuple passed to run() must stay the same
(or a superset) across calls that resume the same in-flight batch. The
Batches API bills at submission time, before this code ever sees a result,
so there is no way to "un-charge" a succeeded row the caller stops asking
about -- run() raises loudly rather than silently dropping one, precisely
so that mistake surfaces instead of quietly costing money for nothing.

The budget cap is enforced in two places, not one: a conservative
pre-submission projection (using max_tokens as the worst-case output
bound) refuses to submit a batch it doesn't fit under, since the Batches
API bills the moment a batch is created, before any result comes back;
the exact per-row cost is then recorded from real usage once results
arrive, which is the number that actually matters for the running total.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from harness.cost_meter import BudgetExceededError, CostMeter
from harness.extractors import score_multiple_choice, score_numeric
from harness.prompt_builder import build_prompt
from harness.schema import AnswerFormat, Question, TaskType
from prereg.arms import FEW_SHOT_ARM_ID
from prereg.config import MODELS


@dataclass(frozen=True)
class RunCell:
    """One (model config, technique arm, question) combination to evaluate."""

    model_id: str
    reasoning: str | None  # None = thinking explicitly disabled; "adaptive" = thinking on
    arm_id: str
    question: Question

    @property
    def custom_id(self) -> str:
        return f"{self.model_id}--{self.reasoning or 'off'}--{self.arm_id}--{self.question.id}"


def build_run_grid(questions: Iterable[Question]) -> tuple[RunCell, ...]:
    """Expand prereg/config.py's MODELS (with each model's arm restriction)
    across every supplied question."""
    cells = []
    for model_cfg in MODELS:
        for arm_id in model_cfg["arms"]:
            for question in questions:
                cells.append(
                    RunCell(
                        model_id=model_cfg["model_id"],
                        reasoning=model_cfg["reasoning"],
                        arm_id=arm_id,
                        question=question,
                    )
                )
    return tuple(cells)


def build_request_params(
    cell: RunCell,
    few_shot_examples: dict[TaskType, tuple[tuple[str, str], ...]] | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Pure function: RunCell -> plain-dict Messages API request params.

    No SDK objects and no network access -- the caller's client adapter is
    responsible for turning this dict into whatever the real Batches API
    needs. `thinking` is always set EXPLICITLY (never omitted): Claude
    Sonnet 5 runs adaptive thinking by default when the parameter is left
    out, so leaving it out would silently contaminate the "reasoning off"
    main-grid cells that this whole model row exists to measure.
    """
    few_shot_examples = few_shot_examples or {}
    examples = few_shot_examples.get(cell.question.task_type, ()) if cell.arm_id == FEW_SHOT_ARM_ID else ()
    prompt_text = build_prompt(cell.question, cell.arm_id, few_shot_examples=examples)
    return {
        "model": cell.model_id,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"} if cell.reasoning == "adaptive" else {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt_text}],
    }


def _estimate_input_tokens(prompt_text: str) -> int:
    """Conservative pre-flight token estimate, without a real tokenizer or an
    extra network call (that's what the Messages API's count_tokens endpoint
    is for, in production, and it isn't worth the round trip just to gate a
    submission). ~4 characters per token is the standard rough heuristic for
    English text; rounded up so this leans toward over-estimating cost, not
    under-estimating it."""
    return max(1, -(-len(prompt_text) // 4))  # ceil division


def score_response(question: Question, raw_text: str) -> tuple[str | None, bool]:
    """Dispatch to the numeric or multiple-choice extractor based on the
    question's own answer_format -- the runner never branches on arm."""
    if question.answer_format is AnswerFormat.NUMERIC:
        return score_numeric(raw_text, question.gold_answer)
    return score_multiple_choice(raw_text, question.gold_answer, question.choices)


class BatchClient(Protocol):
    """The minimal slice of the Anthropic Batches API this runner needs.

    A production client wraps `anthropic.Anthropic().messages.batches`;
    tests use a small in-memory fake implementing the same three methods.
    """

    def create(self, requests: list[dict[str, Any]]) -> Any:
        """Submit a batch of {custom_id, params} dicts; return an object with `.id`."""

    def retrieve(self, batch_id: str) -> Any:
        """Return an object with `.processing_status` ("ended" when done)."""

    def results(self, batch_id: str) -> Iterable[Any]:
        """Yield objects with `.custom_id` and `.result` (`.type`, and on
        "succeeded": `.message.content` text blocks + `.message.usage`)."""


class BatchRunner:
    def __init__(
        self,
        client: BatchClient,
        cost_meter: CostMeter,
        results_path: Path,
        state_path: Path,
    ) -> None:
        self.client = client
        self.cost_meter = cost_meter
        self.results_path = results_path
        self.state_path = state_path

    def _load_completed_custom_ids(self) -> set[str]:
        if not self.results_path.exists():
            return set()
        completed = set()
        with self.results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(json.loads(line)["custom_id"])
        return completed

    def _append_result(self, record: dict[str, Any]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _load_or_submit_batch(
        self,
        pending_cells: list[RunCell],
        few_shot_examples: dict[TaskType, tuple[tuple[str, str], ...]] | None,
        max_tokens: int,
    ) -> str:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))["batch_id"]

        requests = [
            {"custom_id": cell.custom_id, "params": build_request_params(cell, few_shot_examples, max_tokens)}
            for cell in pending_cells
        ]

        # The Batches API bills at submission, before any result comes back,
        # so this is the only point where a pre-flight check can actually
        # stop the spend rather than merely notice it afterward.
        projected_cost = sum(
            self.cost_meter.estimate_cost(
                cell.model_id, _estimate_input_tokens(req["params"]["messages"][0]["content"]), max_tokens
            )
            for cell, req in zip(pending_cells, requests)
        )
        if projected_cost > self.cost_meter.remaining_usd():
            raise BudgetExceededError(
                f"submitting this batch of {len(pending_cells)} requests projects "
                f"{projected_cost:.6f} USD (worst case, at max_tokens={max_tokens}), "
                f"exceeding the {self.cost_meter.remaining_usd():.6f} USD remaining "
                "budget -- refusing to submit"
            )

        batch = self.client.create(requests)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"batch_id": batch.id}), encoding="utf-8")
        return batch.id

    def _clear_batch_state(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def run(
        self,
        cells: tuple[RunCell, ...],
        few_shot_examples: dict[TaskType, tuple[tuple[str, str], ...]] | None = None,
        max_tokens: int = 2048,
    ) -> None:
        """Advance this run by one step.

        Submits the pending batch if none is in flight, does nothing further
        if it's still processing (call run() again later), or scores and
        records every succeeded result once the batch has ended. Safe to
        call repeatedly -- each call only ever does the work not already
        reflected in results_path / state_path.
        """
        completed = self._load_completed_custom_ids()
        pending_cells = [cell for cell in cells if cell.custom_id not in completed]
        if not pending_cells:
            return

        batch_id = self._load_or_submit_batch(pending_cells, few_shot_examples, max_tokens)

        status = self.client.retrieve(batch_id)
        if status.processing_status != "ended":
            return

        cells_by_custom_id = {cell.custom_id: cell for cell in pending_cells}
        for result in self.client.results(batch_id):
            if result.custom_id in completed:
                continue  # already recorded in a prior partial run of this batch
            cell = cells_by_custom_id.get(result.custom_id)
            if cell is None:
                raise RuntimeError(
                    f"batch {batch_id!r} returned a result for custom_id "
                    f"{result.custom_id!r}, which is neither in the current run's "
                    "`cells` nor already recorded. See the stability contract in "
                    "this module's docstring: `cells` must stay the same (or a "
                    "superset) across calls that resume the same in-flight batch "
                    "-- refusing to silently drop what may be a succeeded, "
                    "already-paid-for result."
                )
            if result.result.type != "succeeded":
                continue  # errored/expired/canceled -- left pending, retried next run

            message = result.result.message
            raw_text = "".join(block.text for block in message.content if block.type == "text")
            extracted, is_correct = score_response(cell.question, raw_text)

            self.cost_meter.record(cell.model_id, message.usage.input_tokens, message.usage.output_tokens)

            self._append_result(
                {
                    "custom_id": cell.custom_id,
                    "question_id": cell.question.id,
                    "arm_id": cell.arm_id,
                    "model_id": cell.model_id,
                    "reasoning": cell.reasoning,
                    "raw_text": raw_text,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
            )

        self._clear_batch_state()
