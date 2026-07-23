"""Frozen technique-arm definitions for the prompt-audit experiment.

Every arm template is verbatim, pre-registered text: this file (plus
prereg/config.py) is what prereg/freeze.py hashes. Changing a template after
the freeze is committed invalidates the experiment (see prereg/freeze.py).

Design invariant (see docs/prereg.md): the format instruction and the answer
parser are IDENTICAL across every arm. The chain-of-thought trigger in T1 is
appended AFTER the format instruction, not before — this follows the original
zero-shot-CoT construction (Kojima et al. 2022, "Let's think step by step"
appended at the very end before generation). The format instruction asks the
model to END its response with a specific final line rather than respond with
ONLY that line, so preceding reasoning text is permitted in every arm without
contradicting the instruction — T1 only differs from B by explicitly inviting
that reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.schema import AnswerFormat

FORMAT_INSTRUCTION: dict[AnswerFormat, str] = {
    AnswerFormat.NUMERIC: (
        "End your response with a final line in exactly this form: "
        "ANSWER: <number>"
    ),
    AnswerFormat.MULTIPLE_CHOICE: (
        "End your response with a final line in exactly this form: "
        "ANSWER: <letter>"
    ),
}


@dataclass(frozen=True)
class Arm:
    id: str
    name: str
    template: str  # str.format() placeholders: question, format_instruction, domain, examples


ARMS: tuple[Arm, ...] = (
    Arm(id="B", name="baseline", template="{question}\n\n{format_instruction}"),
    Arm(
        id="T1",
        name="zero_shot_cot",
        template=(
            "{question}\n\n{format_instruction}\n\n"
            "Think step by step before answering."
        ),
    ),
    Arm(
        id="T2",
        name="role_prompt",
        template=(
            "You are a world-class expert in {domain}.\n\n"
            "{question}\n\n{format_instruction}"
        ),
    ),
    Arm(
        id="T3",
        name="emotional_stakes",
        template=(
            "{question}\n\n{format_instruction}\n\n"
            "This is very important to my career."
        ),
    ),
    Arm(
        id="T4",
        name="incentive",
        template=(
            "{question}\n\n{format_instruction}\n\n"
            "I'll tip $200 for a correct answer."
        ),
    ),
    Arm(
        id="T5",
        name="politeness",
        template=(
            "Could you please help me with the following question?\n\n"
            "{question}\n\n{format_instruction}\n\n"
            "Thank you so much for your help!"
        ),
    ),
    Arm(
        id="T6",
        name="few_shot",
        template="{examples}{question}\n\n{format_instruction}",
    ),
)

ARMS_BY_ID: dict[str, Arm] = {arm.id: arm for arm in ARMS}

# The one arm whose prompt depends on injected content (fixed worked
# examples) rather than being fully static text. Referenced by id here,
# rather than hardcoding the "T6" string at each call site, so a future
# rename of the arm can't silently desync prompt_builder.py from runner.py.
FEW_SHOT_ARM_ID = "T6"
