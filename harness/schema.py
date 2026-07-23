"""Shared data contract for the prompt-audit harness.

Every module (prompt builder, extractors, cost meter, runner) reads and
writes these types, so this is the one place their shapes are allowed to
change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    MATH = "math"
    LOGIC = "logic"
    PROCEDURAL = "procedural"


class AnswerFormat(str, Enum):
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"


@dataclass(frozen=True)
class Question:
    id: str
    task_type: TaskType
    domain: str
    prompt: str
    answer_format: AnswerFormat
    gold_answer: str
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.answer_format is AnswerFormat.MULTIPLE_CHOICE and not self.choices:
            raise ValueError(f"question {self.id}: multiple_choice requires choices")
        if self.answer_format is AnswerFormat.NUMERIC and self.choices:
            raise ValueError(f"question {self.id}: numeric question must not have choices")


@dataclass(frozen=True)
class ArmResponse:
    question_id: str
    arm_id: str
    model_id: str
    raw_text: str
    extracted_answer: str | None
    is_correct: bool
    input_tokens: int
    output_tokens: int
