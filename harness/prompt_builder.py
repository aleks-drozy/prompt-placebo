"""Compose the final prompt sent to the model.

This is the one place an arm template gets filled in with a real question —
the runner and any analysis code should never format an arm template
directly, so the identical-format-instruction invariant can't drift.
"""
from __future__ import annotations

from harness.schema import Question
from prereg.arms import ARMS_BY_ID, FEW_SHOT_ARM_ID, FORMAT_INSTRUCTION, Arm


def format_few_shot_block(examples: tuple[tuple[str, str], ...]) -> str:
    """Render fixed worked examples for the T6 few-shot arm.

    `examples` is (question_text, formatted_answer_line) pairs — frozen per
    task set at pre-registration, never generated per-question (see
    prereg/arms.py module docstring).
    """
    if not examples:
        return ""
    blocks = [
        f"Example {i}:\n{question}\n{answer}\n"
        for i, (question, answer) in enumerate(examples, start=1)
    ]
    return "\n".join(blocks) + "\n"


def build_prompt(
    question: Question,
    arm_id: str,
    few_shot_examples: tuple[tuple[str, str], ...] = (),
) -> str:
    """Compose the full user-turn text for one (question, arm) pair.

    `few_shot_examples` is only consumed by the T6 arm; every other arm
    ignores it.
    """
    arm: Arm = ARMS_BY_ID[arm_id]
    return arm.template.format(
        question=question.prompt,
        format_instruction=FORMAT_INSTRUCTION[question.answer_format],
        domain=question.domain,
        examples=format_few_shot_block(few_shot_examples) if arm_id == FEW_SHOT_ARM_ID else "",
    )
