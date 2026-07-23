"""Extract and score final answers from raw model text.

This is the single arm-agnostic answer extractor for the whole experiment
(see docs/prereg.md and the project risk register: "parser bias between
arms" is risk #1). Every arm's format instruction asks the model to end its
response with a line of the form ``ANSWER: <value>``, but models will not
always comply exactly -- they may add commentary before or after the tag,
use different case, or restate the answer more than once.

To keep the extractor identical no matter which arm produced the text, this
module applies exactly ONE fixed, testable strategy and nothing else:

1. Search the entire text for the word ANSWER (case-insensitive) followed by
   a colon and a value, matching only up to the end of that line (so trailing
   prose on a later line is never swallowed into the match).
2. If there is more than one such occurrence, take the LAST one.
3. Strip whitespace from the captured value and return it as-is -- callers
   normalize it further depending on whether the question is numeric or
   multiple-choice.

This file must never import anything related to prompt arms, and must never
branch on which arm produced the text.
"""
from __future__ import annotations

import re

_ANSWER_PATTERN = re.compile(r"(?<![a-z-])answer\s*:\s*(.*)", re.IGNORECASE)


def extract_raw_answer(text: str) -> str | None:
    matches = _ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_numeric(value: str) -> float | None:
    cleaned = value.strip()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def score_numeric(raw_text: str, gold_answer: str) -> tuple[str | None, bool]:
    extracted = extract_raw_answer(raw_text)
    if extracted is None:
        return None, False

    extracted_value = normalize_numeric(extracted)
    gold_value = normalize_numeric(gold_answer)
    if extracted_value is None or gold_value is None:
        return extracted, False

    return extracted, abs(extracted_value - gold_value) <= 1e-6


def normalize_choice(value: str) -> str | None:
    cleaned = value.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.upper()
    if len(cleaned) == 1 and "A" <= cleaned <= "Z":
        return cleaned
    return None


def score_multiple_choice(
    raw_text: str, gold_answer: str, choices: tuple[str, ...]
) -> tuple[str | None, bool]:
    extracted = extract_raw_answer(raw_text)
    if extracted is None:
        return None, False

    return extracted, normalize_choice(extracted) == normalize_choice(gold_answer)
