"""Tests for harness/runner.py.

No network access anywhere here -- FakeBatchClient stands in for the real
Anthropic Batches API so the resumability/idempotency logic (the whole
point of this module) can be exercised deterministically.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness.cost_meter import BudgetExceededError, CostMeter
from harness.runner import BatchRunner, RunCell, build_request_params, build_run_grid, score_response
from harness.schema import AnswerFormat, Question, TaskType
from prereg.config import MODELS


def _numeric_question(qid="math-0001"):
    return Question(
        id=qid,
        task_type=TaskType.MATH,
        domain="mathematics",
        prompt="What is 2 + 2?",
        answer_format=AnswerFormat.NUMERIC,
        gold_answer="4",
    )


def _mc_question(qid="logic-0001"):
    return Question(
        id=qid,
        task_type=TaskType.LOGIC,
        domain="logical reasoning",
        prompt="Which is true?",
        answer_format=AnswerFormat.MULTIPLE_CHOICE,
        gold_answer="B",
        choices=("A", "B", "C", "D"),
    )


class FakeBatchClient:
    """In-memory stand-in for anthropic.Anthropic().messages.batches.

    `succeeded_text_by_custom_id` maps custom_id -> response text for cells
    that should come back "succeeded"; any custom_id not in that dict comes
    back "errored" instead, so tests can exercise the retry-on-next-run path.
    """

    def __init__(self, succeeded_text_by_custom_id: dict[str, str], processing_status: str = "ended"):
        self.succeeded_text_by_custom_id = succeeded_text_by_custom_id
        self.processing_status = processing_status
        self.create_calls: list[list[dict]] = []
        self._next_batch_id = 0

    def create(self, requests):
        self.create_calls.append(requests)
        self._next_batch_id += 1
        return SimpleNamespace(id=f"batch_{self._next_batch_id}")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status=self.processing_status)

    def results(self, batch_id):
        for custom_id, text in self.succeeded_text_by_custom_id.items():
            yield SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(
                    type="succeeded",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=text)],
                        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                    ),
                ),
            )
        # every requested custom_id not in succeeded_text_by_custom_id errored
        requested_ids = {req["custom_id"] for call in self.create_calls for req in call}
        for custom_id in requested_ids - set(self.succeeded_text_by_custom_id):
            yield SimpleNamespace(
                custom_id=custom_id,
                result=SimpleNamespace(type="errored"),
            )


class TestRunCell:
    def test_custom_id_is_deterministic_and_distinguishes_reasoning(self):
        q = _numeric_question()
        a = RunCell(model_id="claude-sonnet-5", reasoning=None, arm_id="B", question=q)
        b = RunCell(model_id="claude-sonnet-5", reasoning="adaptive", arm_id="B", question=q)
        assert a.custom_id != b.custom_id
        assert a.custom_id == RunCell(model_id="claude-sonnet-5", reasoning=None, arm_id="B", question=q).custom_id


class TestBuildRunGrid:
    def test_extended_thinking_condition_only_covers_b_and_t1(self):
        questions = (_numeric_question(),)
        grid = build_run_grid(questions)
        adaptive_cells = [c for c in grid if c.reasoning == "adaptive"]
        assert {c.arm_id for c in adaptive_cells} == {"B", "T1"}

    def test_main_grid_models_cover_all_seven_arms(self):
        questions = (_numeric_question(),)
        grid = build_run_grid(questions)
        off_cells = [c for c in grid if c.reasoning is None]
        arms_per_model = {}
        for cell in off_cells:
            arms_per_model.setdefault(cell.model_id, set()).add(cell.arm_id)
        for model_id, arms in arms_per_model.items():
            assert arms == {"B", "T1", "T2", "T3", "T4", "T5", "T6"}, model_id

    def test_grid_size_matches_config(self):
        questions = (_numeric_question(), _numeric_question(qid="math-0002"))
        grid = build_run_grid(questions)
        expected = sum(len(m["arms"]) for m in MODELS) * len(questions)
        assert len(grid) == expected


class TestBuildRequestParams:
    def test_main_grid_cell_disables_thinking_explicitly(self):
        cell = RunCell(model_id="claude-sonnet-5", reasoning=None, arm_id="B", question=_numeric_question())
        params = build_request_params(cell)
        assert params["thinking"] == {"type": "disabled"}

    def test_extended_thinking_cell_sets_adaptive(self):
        cell = RunCell(model_id="claude-sonnet-5", reasoning="adaptive", arm_id="T1", question=_numeric_question())
        params = build_request_params(cell)
        assert params["thinking"] == {"type": "adaptive"}

    def test_haiku_cell_still_disables_thinking_explicitly(self):
        # Even though thinking behavior differences are Sonnet-5-specific,
        # every main-grid cell must set thinking explicitly, never omit it.
        cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=_numeric_question())
        params = build_request_params(cell)
        assert params["thinking"] == {"type": "disabled"}

    def test_few_shot_examples_only_applied_to_t6(self):
        q = _numeric_question()
        examples = {TaskType.MATH: (("What is 1 + 1?", "ANSWER: 2"),)}

        t6_cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="T6", question=q)
        b_cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q)

        t6_prompt = build_request_params(t6_cell, few_shot_examples=examples)["messages"][0]["content"]
        b_prompt = build_request_params(b_cell, few_shot_examples=examples)["messages"][0]["content"]

        assert "What is 1 + 1?" in t6_prompt
        assert "What is 1 + 1?" not in b_prompt


class TestScoreResponse:
    def test_numeric_question_uses_numeric_extractor(self):
        extracted, correct = score_response(_numeric_question(), "Reasoning.\nANSWER: 4")
        assert extracted == "4"
        assert correct is True

    def test_multiple_choice_question_uses_choice_extractor(self):
        extracted, correct = score_response(_mc_question(), "Reasoning.\nANSWER: B")
        assert extracted == "B"
        assert correct is True


class TestBatchRunner:
    def test_first_call_submits_batch_and_records_nothing_while_processing(self, tmp_path):
        cells = (RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=_numeric_question()),)
        client = FakeBatchClient(succeeded_text_by_custom_id={}, processing_status="in_progress")
        meter = CostMeter(cap_usd=100.0)
        results_path = tmp_path / "results.jsonl"
        state_path = tmp_path / "state.json"
        runner = BatchRunner(client, meter, results_path, state_path)

        runner.run(cells)

        assert len(client.create_calls) == 1
        assert state_path.exists()
        assert not results_path.exists()
        assert meter.total_spent_usd == 0.0

    def test_second_call_does_not_resubmit_while_still_processing(self, tmp_path):
        cells = (RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=_numeric_question()),)
        client = FakeBatchClient(succeeded_text_by_custom_id={}, processing_status="in_progress")
        meter = CostMeter(cap_usd=100.0)
        runner = BatchRunner(client, meter, tmp_path / "results.jsonl", tmp_path / "state.json")

        runner.run(cells)
        runner.run(cells)

        assert len(client.create_calls) == 1  # never resubmitted

    def test_processes_and_records_results_once_batch_ends(self, tmp_path):
        question = _numeric_question()
        cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=question)
        client = FakeBatchClient(succeeded_text_by_custom_id={cell.custom_id: "Reasoning.\nANSWER: 4"})
        meter = CostMeter(cap_usd=100.0)
        results_path = tmp_path / "results.jsonl"
        state_path = tmp_path / "state.json"
        runner = BatchRunner(client, meter, results_path, state_path)

        runner.run((cell,))

        assert results_path.exists()
        lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["custom_id"] == cell.custom_id
        assert record["is_correct"] is True
        assert record["extracted_answer"] == "4"
        assert meter.total_spent_usd > 0.0
        assert not state_path.exists()  # cleared once the batch is fully processed

    def test_resume_skips_already_completed_custom_ids(self, tmp_path):
        q1 = _numeric_question(qid="math-0001")
        q2 = _numeric_question(qid="math-0002")
        cell1 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q1)
        cell2 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q2)

        results_path = tmp_path / "results.jsonl"
        results_path.write_text(
            json.dumps(
                {
                    "custom_id": cell1.custom_id,
                    "question_id": q1.id,
                    "arm_id": "B",
                    "model_id": "claude-haiku-4-5",
                    "reasoning": None,
                    "raw_text": "ANSWER: 4",
                    "extracted_answer": "4",
                    "is_correct": True,
                    "input_tokens": 10,
                    "output_tokens": 5,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        client = FakeBatchClient(succeeded_text_by_custom_id={cell2.custom_id: "ANSWER: 4"})
        meter = CostMeter(cap_usd=100.0)
        runner = BatchRunner(client, meter, results_path, tmp_path / "state.json")

        runner.run((cell1, cell2))

        assert len(client.create_calls) == 1
        submitted_ids = {req["custom_id"] for req in client.create_calls[0]}
        assert submitted_ids == {cell2.custom_id}  # cell1 was already recorded, never resubmitted

        lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_errored_results_are_left_unrecorded_for_retry(self, tmp_path):
        question = _numeric_question()
        cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=question)
        client = FakeBatchClient(succeeded_text_by_custom_id={})  # nothing succeeds -> all errored
        meter = CostMeter(cap_usd=100.0)
        results_path = tmp_path / "results.jsonl"
        runner = BatchRunner(client, meter, results_path, tmp_path / "state.json")

        runner.run((cell,))

        assert not results_path.exists()
        assert meter.total_spent_usd == 0.0

    def test_budget_exceeded_mid_loop_stops_processing_without_partial_or_duplicate_record(self, tmp_path):
        q1 = _numeric_question(qid="math-0001")
        q2 = _numeric_question(qid="math-0002")
        cell1 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q1)
        cell2 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q2)

        client = FakeBatchClient(
            succeeded_text_by_custom_id={cell1.custom_id: "ANSWER: 4", cell2.custom_id: "ANSWER: 4"}
        )
        # FakeBatchClient always reports usage of 100 input / 20 output tokens,
        # so each haiku batch call actually costs (100/1e6*1.00 + 20/1e6*5.00)*0.5
        # = 0.0001 USD. Cap sits strictly between one call's cost and two, so
        # the first result records fine and recording the second must raise.
        meter = CostMeter(cap_usd=0.00015)
        results_path = tmp_path / "results.jsonl"
        runner = BatchRunner(client, meter, results_path, tmp_path / "state.json")

        # A small max_tokens keeps the pre-submission projection (which uses
        # max_tokens as its worst-case output bound) comfortably under cap,
        # so this exercises the mid-loop *actual-usage* budget check, not the
        # pre-flight one covered by test_pre_submission_budget_gate_blocks_before_any_api_call.
        with pytest.raises(BudgetExceededError):
            runner.run((cell1, cell2), max_tokens=20)

        lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["custom_id"] == cell1.custom_id
        assert meter.total_spent_usd == pytest.approx(0.0001)

    def test_pre_submission_budget_gate_blocks_before_any_api_call(self, tmp_path):
        cell = RunCell(model_id="claude-sonnet-5", reasoning=None, arm_id="B", question=_numeric_question())
        client = FakeBatchClient(succeeded_text_by_custom_id={cell.custom_id: "ANSWER: 4"})
        meter = CostMeter(cap_usd=0.0000001)  # any real submission blows this
        state_path = tmp_path / "state.json"
        runner = BatchRunner(client, meter, tmp_path / "results.jsonl", state_path)

        with pytest.raises(BudgetExceededError):
            runner.run((cell,), max_tokens=2048)

        assert client.create_calls == []  # never even attempted to submit
        assert not state_path.exists()

    def test_resume_after_crash_mid_loop_does_not_double_charge_or_duplicate(self, tmp_path):
        q1 = _numeric_question(qid="math-0001")
        q2 = _numeric_question(qid="math-0002")
        cell1 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q1)
        cell2 = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=q2)
        cells = (cell1, cell2)

        results_path = tmp_path / "results.jsonl"
        state_path = tmp_path / "state.json"

        # Simulate: a batch was already submitted (state persisted) and
        # cell1's result was already scored and appended by a prior call
        # that then crashed before it got to cell2.
        state_path.write_text(json.dumps({"batch_id": "batch_1"}), encoding="utf-8")
        results_path.write_text(
            json.dumps(
                {
                    "custom_id": cell1.custom_id,
                    "question_id": q1.id,
                    "arm_id": "B",
                    "model_id": "claude-haiku-4-5",
                    "reasoning": None,
                    "raw_text": "ANSWER: 4",
                    "extracted_answer": "4",
                    "is_correct": True,
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        client = FakeBatchClient(
            succeeded_text_by_custom_id={cell1.custom_id: "ANSWER: 4", cell2.custom_id: "ANSWER: 4"}
        )
        meter = CostMeter(cap_usd=100.0)
        runner = BatchRunner(client, meter, results_path, state_path)

        runner.run(cells)

        assert client.create_calls == []  # reused the persisted batch id, never resubmitted
        lines = results_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        ids = {json.loads(line)["custom_id"] for line in lines}
        assert ids == {cell1.custom_id, cell2.custom_id}
        # Only cell2 was actually recorded THIS call -- cell1's cost was
        # already accounted for (and, in a real crash, recorded) before.
        assert meter.total_spent_usd == pytest.approx(0.0001)
        assert not state_path.exists()

    def test_unknown_custom_id_in_results_raises_instead_of_silently_dropping(self, tmp_path):
        cell = RunCell(model_id="claude-haiku-4-5", reasoning=None, arm_id="B", question=_numeric_question())
        client = FakeBatchClient(
            succeeded_text_by_custom_id={
                cell.custom_id: "ANSWER: 4",
                "some-other-unexpected-custom-id": "ANSWER: 4",
            }
        )
        meter = CostMeter(cap_usd=100.0)
        runner = BatchRunner(client, meter, tmp_path / "results.jsonl", tmp_path / "state.json")

        with pytest.raises(RuntimeError, match="some-other-unexpected-custom-id"):
            runner.run((cell,))
