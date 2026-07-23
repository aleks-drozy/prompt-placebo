import dataclasses

import pytest

from harness.schema import AnswerFormat, ArmResponse, Question, TaskType


def test_numeric_question_constructs_without_choices():
    q = Question(
        id="math-001",
        task_type=TaskType.MATH,
        domain="mathematics",
        prompt="What is 2 + 2?",
        answer_format=AnswerFormat.NUMERIC,
        gold_answer="4",
    )
    assert q.choices == ()


def test_multiple_choice_question_requires_choices():
    with pytest.raises(ValueError, match="requires choices"):
        Question(
            id="logic-001",
            task_type=TaskType.LOGIC,
            domain="logical reasoning",
            prompt="Which is true?",
            answer_format=AnswerFormat.MULTIPLE_CHOICE,
            gold_answer="A",
            choices=(),
        )


def test_numeric_question_rejects_choices():
    with pytest.raises(ValueError, match="must not have choices"):
        Question(
            id="math-002",
            task_type=TaskType.MATH,
            domain="mathematics",
            prompt="What is 2 + 2?",
            answer_format=AnswerFormat.NUMERIC,
            gold_answer="4",
            choices=("A", "B"),
        )


def test_multiple_choice_question_constructs_with_choices():
    q = Question(
        id="logic-002",
        task_type=TaskType.LOGIC,
        domain="logical reasoning",
        prompt="Which is true?",
        answer_format=AnswerFormat.MULTIPLE_CHOICE,
        gold_answer="B",
        choices=("A", "B", "C", "D"),
    )
    assert q.choices == ("A", "B", "C", "D")


def test_question_is_frozen():
    q = Question(
        id="math-003",
        task_type=TaskType.MATH,
        domain="mathematics",
        prompt="What is 2 + 2?",
        answer_format=AnswerFormat.NUMERIC,
        gold_answer="4",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.gold_answer = "5"  # type: ignore[misc]


def test_arm_response_constructs():
    r = ArmResponse(
        question_id="math-001",
        arm_id="B",
        model_id="claude-haiku-4-5",
        raw_text="ANSWER: 4",
        extracted_answer="4",
        is_correct=True,
        input_tokens=42,
        output_tokens=8,
    )
    assert r.is_correct is True
