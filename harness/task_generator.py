"""Seeded, fully deterministic procedural arithmetic word-problem generator.

This is "task set 3" in the project design: a contamination-free control.
Problems are generated fresh from a seeded PRNG, so by construction they
cannot appear in any model's training data.

Operand range and step count were widened after the pilot (P2) showed 100%
accuracy on every single arm/model combo at the original difficulty
(2-digit operands, 2-4 steps) -- a ceiling effect with zero variance to
measure anything against. See DECISIONS.md.
"""
from __future__ import annotations

import random

from harness.schema import AnswerFormat, Question, TaskType

_OPERATORS = ("+", "-", "*")
_MIN_OPERAND = 10
_MAX_OPERAND = 999
_MIN_STEPS = 5
_MAX_STEPS = 7

_OP_WORDS = {
    "+": "add",
    "-": "subtract",
    "*": "multiply by",
}


def _evaluate_steps(start: int, steps: list[tuple[str, int]]) -> int:
    """Apply (op, operand) pairs to start, left to right, and return the result."""
    result = start
    for op, operand in steps:
        if op == "+":
            result = result + operand
        elif op == "-":
            result = result - operand
        elif op == "*":
            result = result * operand
        else:
            raise ValueError(f"unknown operator: {op!r}")
    return result


def _steps_to_clause(steps: list[tuple[str, int]]) -> str:
    """Render the steps as an unambiguous, left-to-right list of clauses."""
    parts = []
    for op, operand in steps:
        parts.append(f"then {_OP_WORDS[op]} {operand}")
    return ", ".join(parts)


def _render_prompt(rng: random.Random, start: int, steps: list[tuple[str, int]]) -> str:
    clause = _steps_to_clause(steps)
    templates = (
        "Start with {start}, {clause}. What is the final result?",
        "Begin with the number {start}, {clause}. What number do you end up with?",
        "You start with {start}, {clause}. Compute the final value.",
        "Take {start} as your starting number, {clause}. What is the result?",
    )
    template = rng.choice(templates)
    return template.format(start=start, clause=clause)


def generate_procedural_questions(n: int, seed: int) -> tuple[Question, ...]:
    """Deterministically generate n procedural arithmetic Questions from seed.

    Calling this twice with the same (n, seed) returns questions with
    identical prompt, gold_answer, and id values across the whole batch.
    n=0 returns an empty tuple. Different seeds should, in general,
    produce different question sets.
    """
    rng = random.Random(seed)
    questions = []
    for index in range(n):
        start = rng.randint(_MIN_OPERAND, _MAX_OPERAND)
        num_steps = rng.randint(_MIN_STEPS, _MAX_STEPS)
        steps = [
            (rng.choice(_OPERATORS), rng.randint(_MIN_OPERAND, _MAX_OPERAND))
            for _ in range(num_steps)
        ]
        prompt = _render_prompt(rng, start, steps)
        gold_answer = str(_evaluate_steps(start, steps))
        questions.append(
            Question(
                id=f"proc-{seed}-{index:04d}",
                task_type=TaskType.PROCEDURAL,
                domain="arithmetic",
                prompt=prompt,
                answer_format=AnswerFormat.NUMERIC,
                gold_answer=gold_answer,
            )
        )
    return tuple(questions)
