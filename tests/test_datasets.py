"""Tests for harness/datasets.py.

No network access here: real HTTP fetching is exercised once, manually,
outside the test suite (see the module docstring in harness/datasets.py).
These tests cover:

  1. the pure raw-row -> Question parsing functions (parse_gsm8k_row,
     parse_bbh_row), against hand-written fake rows shaped like what a
     live fetch actually returned, and
  2. the cache/sample logic (fetch_math_questions / fetch_logic_questions),
     against small fake pool JSON files written to tmp_path, with the
     module's cache-path constants monkeypatched so the real
     data/tasksets/ directory and the network are never touched.
"""
from __future__ import annotations

import json

import pytest

from harness import datasets as ds
from harness.schema import AnswerFormat, Question, TaskType

# ---------------------------------------------------------------------------
# parse_gsm8k_row
# ---------------------------------------------------------------------------


def test_parse_gsm8k_row_extracts_final_numeric_answer():
    row = {
        "question": "If a train travels 60 miles in 2 hours, what is its speed in mph?",
        "answer": "Speed = distance / time.\n60 / 2 = <<60/2=30>>30\n#### 30",
    }

    question = ds.parse_gsm8k_row(row, index=7)

    assert question.id == "math-0007"
    assert question.task_type is TaskType.MATH
    assert question.domain == "mathematics"
    assert question.prompt == row["question"]
    assert question.answer_format is AnswerFormat.NUMERIC
    assert question.gold_answer == "30"
    assert question.choices == ()


def test_parse_gsm8k_row_strips_comma_thousands_separator():
    row = {
        "question": "A factory produced widgets over a year, how many in total?",
        "answer": "Summing every month's production gives the total.\n#### 1,234",
    }

    question = ds.parse_gsm8k_row(row, index=0)

    assert question.gold_answer == "1234"
    assert "," not in question.gold_answer


def test_parse_gsm8k_row_raises_when_no_marker_present():
    row = {"question": "What is 2 + 2?", "answer": "It is four, obviously."}

    with pytest.raises(ValueError):
        ds.parse_gsm8k_row(row, index=0)


# ---------------------------------------------------------------------------
# parse_bbh_row
# ---------------------------------------------------------------------------


def test_parse_bbh_row_extracts_choices_and_gold_letter():
    # Hand-crafted fixture in the same shape as the real lukaemon/bbh
    # date_understanding rows (not copied verbatim from the dataset).
    row = {
        "input": (
            "If today is March 3, 1999, what was the date 10 days ago?\n"
            "Options:\n"
            "(A) 02/21/1999\n"
            "(B) 03/01/1999\n"
            "(C) 02/21/1999\n"
            "(D) 01/21/1999"
        ),
        "target": "(C)",
    }

    question = ds.parse_bbh_row(row, index=3)

    assert question.id == "logic-0003"
    assert question.task_type is TaskType.LOGIC
    assert question.domain == "logical reasoning"
    assert question.prompt == row["input"]
    assert "Options:" in question.prompt
    assert question.answer_format is AnswerFormat.MULTIPLE_CHOICE
    assert question.choices == ("A", "B", "C", "D")
    assert question.gold_answer == "C"


def test_parse_bbh_row_derives_choice_count_from_row_not_hardcoded():
    row = {
        "input": "Pick the odd one out.\nOptions:\n(A) cat\n(B) dog\n(C) car",
        "target": "(C)",
    }

    question = ds.parse_bbh_row(row, index=0)

    assert question.choices == ("A", "B", "C")


def test_parse_bbh_row_raises_when_no_options_found():
    row = {"input": "There are no lettered options here at all.", "target": "(A)"}

    with pytest.raises(ValueError):
        ds.parse_bbh_row(row, index=0)


# ---------------------------------------------------------------------------
# cache + sampling: fetch_math_questions / fetch_logic_questions
# ---------------------------------------------------------------------------


def _fake_math_pool(n: int) -> list[dict]:
    return [
        {
            "id": f"math-{i:04d}",
            "task_type": "math",
            "domain": "mathematics",
            "prompt": f"What is {i} + 1?",
            "answer_format": "numeric",
            "gold_answer": str(i + 1),
            "choices": [],
        }
        for i in range(n)
    ]


def _fake_logic_pool(n: int) -> list[dict]:
    letters = ["A", "B", "C", "D"]
    return [
        {
            "id": f"logic-{i:04d}",
            "task_type": "logic",
            "domain": "logical reasoning",
            "prompt": f"Question {i}?\nOptions:\n(A) x\n(B) y\n(C) z\n(D) w",
            "answer_format": "multiple_choice",
            "gold_answer": letters[i % 4],
            "choices": letters,
        }
        for i in range(n)
    ]


@pytest.fixture
def fake_math_pool(tmp_path, monkeypatch):
    pool_path = tmp_path / "math_pool.json"
    pool_path.write_text(json.dumps(_fake_math_pool(20)), encoding="utf-8")
    monkeypatch.setattr(ds, "MATH_POOL_PATH", pool_path)
    return pool_path


@pytest.fixture
def fake_logic_pool(tmp_path, monkeypatch):
    pool_path = tmp_path / "logic_pool.json"
    pool_path.write_text(json.dumps(_fake_logic_pool(20)), encoding="utf-8")
    monkeypatch.setattr(ds, "LOGIC_POOL_PATH", pool_path)
    return pool_path


def test_fetch_math_questions_same_seed_is_deterministic(fake_math_pool):
    first = ds.fetch_math_questions(n=5, seed=1)
    second = ds.fetch_math_questions(n=5, seed=1)

    assert first == second
    assert len(first) == 5


def test_fetch_math_questions_different_seed_gives_different_sample(fake_math_pool):
    a = ds.fetch_math_questions(n=5, seed=1)
    b = ds.fetch_math_questions(n=5, seed=2)

    assert a != b


def test_fetch_math_questions_n_larger_than_pool_raises_value_error(fake_math_pool):
    with pytest.raises(ValueError):
        ds.fetch_math_questions(n=21, seed=1)


def test_fetch_math_questions_round_trips_enums(fake_math_pool):
    questions = ds.fetch_math_questions(n=3, seed=1)

    for question in questions:
        assert isinstance(question, Question)
        assert isinstance(question.task_type, TaskType)
        assert isinstance(question.answer_format, AnswerFormat)
        assert question.task_type is TaskType.MATH
        assert question.answer_format is AnswerFormat.NUMERIC


def test_fetch_logic_questions_same_seed_is_deterministic(fake_logic_pool):
    first = ds.fetch_logic_questions(n=5, seed=1)
    second = ds.fetch_logic_questions(n=5, seed=1)

    assert first == second


def test_fetch_logic_questions_n_larger_than_pool_raises_value_error(fake_logic_pool):
    with pytest.raises(ValueError):
        ds.fetch_logic_questions(n=21, seed=1)


def test_fetch_logic_questions_round_trips_enums_and_choices(fake_logic_pool):
    questions = ds.fetch_logic_questions(n=3, seed=1)

    for question in questions:
        assert isinstance(question.task_type, TaskType)
        assert isinstance(question.answer_format, AnswerFormat)
        assert question.task_type is TaskType.LOGIC
        assert question.answer_format is AnswerFormat.MULTIPLE_CHOICE
        assert question.choices == ("A", "B", "C", "D")
