from __future__ import annotations

from harness.schema import AnswerFormat, TaskType
from harness.task_generator import _evaluate_steps, generate_procedural_questions


def test_evaluate_steps_add_then_multiply():
    assert _evaluate_steps(10, [("+", 5), ("*", 2)]) == 30


def test_evaluate_steps_subtract():
    assert _evaluate_steps(20, [("-", 8)]) == 12


def test_evaluate_steps_multiply_then_subtract():
    assert _evaluate_steps(5, [("*", 3), ("-", 1)]) == 14


def test_zero_n_returns_empty_tuple():
    assert generate_procedural_questions(n=0, seed=1) == ()


def test_deterministic_same_seed():
    batch1 = generate_procedural_questions(n=10, seed=42)
    batch2 = generate_procedural_questions(n=10, seed=42)
    assert len(batch1) == len(batch2)
    for q1, q2 in zip(batch1, batch2):
        assert q1.prompt == q2.prompt
        assert q1.gold_answer == q2.gold_answer
        assert q1.id == q2.id


def test_different_seeds_produce_different_answers():
    batch1 = generate_procedural_questions(n=10, seed=1)
    batch2 = generate_procedural_questions(n=10, seed=2)
    answers1 = tuple(q.gold_answer for q in batch1)
    answers2 = tuple(q.gold_answer for q in batch2)
    assert answers1 != answers2


def test_question_fields_valid_and_unique_ids():
    batch = generate_procedural_questions(n=20, seed=7)
    seen_ids = set()
    for q in batch:
        assert q.task_type == TaskType.PROCEDURAL
        assert q.domain == "arithmetic"
        assert q.answer_format == AnswerFormat.NUMERIC
        assert q.id not in seen_ids
        seen_ids.add(q.id)
        int(q.gold_answer)  # must not raise


def test_multiple_templates_present():
    batch = generate_procedural_questions(n=20, seed=7)
    prompts = [q.prompt for q in batch]
    # Distinct template trigger phrases we expect to use in the generator.
    trigger_phrases = ["Start with", "Begin with", "You start with", "Take"]
    found = [phrase for phrase in trigger_phrases if any(phrase in p for p in prompts)]
    assert len(found) >= 2
