"""Tests for harness/extractors.py.

The extractor is the single highest-stakes piece of this project (risk #1:
parser bias between arms). These tests verify one fixed, arm-agnostic
strategy for pulling an ANSWER tag out of raw model text, plus numeric and
multiple-choice normalization/scoring built on top of it.
"""
from __future__ import annotations

from harness.extractors import (
    extract_raw_answer,
    normalize_choice,
    normalize_numeric,
    score_multiple_choice,
    score_numeric,
)


class TestExtractRawAnswer:
    def test_simple_two_line_answer(self):
        text = "Some reasoning here.\nANSWER: 42"
        assert extract_raw_answer(text) == "42"

    def test_last_occurrence_wins(self):
        text = "First I said ANSWER: 42\nBut actually ANSWER: 43"
        assert extract_raw_answer(text) == "43"

    def test_no_tag_returns_none(self):
        text = "I have no idea what the answer is."
        assert extract_raw_answer(text) is None

    def test_case_insensitive_on_word_answer(self):
        text = "Some reasoning.\nanswer: 7"
        assert extract_raw_answer(text) == "7"

    def test_trailing_commentary_on_later_line_not_swallowed(self):
        text = "ANSWER: 42\nHope that helps! Let me know if you have questions."
        assert extract_raw_answer(text) == "42"

    def test_strips_whitespace_around_value(self):
        text = "ANSWER:    42   "
        assert extract_raw_answer(text) == "42"

    def test_word_boundary_rejects_unanswer(self):
        """A substring like 'unanswer:' must not be mistaken for the ANSWER
        tag -- otherwise the extractor could latch onto the wrong text."""
        text = "My unanswer: 5 is a trick.\nANSWER: 9"
        assert extract_raw_answer(text) == "9"

    def test_word_boundary_rejects_no_real_tag(self):
        text = "This is a sub-answer: not the real one."
        assert extract_raw_answer(text) is None


class TestNormalizeNumeric:
    def test_dollar_amount_with_thousands_comma(self):
        assert normalize_numeric("$1,234.50") == 1234.50

    def test_surrounding_whitespace(self):
        assert normalize_numeric("  42  ") == 42.0

    def test_non_numeric_text_returns_none(self):
        assert normalize_numeric("not a number") is None

    def test_percent_sign_stripped(self):
        assert normalize_numeric("12.5%") == 12.5

    def test_empty_string_returns_none(self):
        assert normalize_numeric("") is None


class TestScoreNumeric:
    def test_matching_within_tolerance(self):
        raw_text = "Some reasoning.\nANSWER: 7.0"
        extracted, correct = score_numeric(raw_text, "7")
        assert extracted == "7.0"
        assert correct is True

    def test_mismatch(self):
        raw_text = "Some reasoning.\nANSWER: 8"
        extracted, correct = score_numeric(raw_text, "7")
        assert extracted == "8"
        assert correct is False

    def test_no_tag_at_all(self):
        raw_text = "I really don't know."
        extracted, correct = score_numeric(raw_text, "7")
        assert extracted is None
        assert correct is False


class TestNormalizeChoice:
    def test_lowercase_parenthesized_padded(self):
        assert normalize_choice(" (b). ") == "B"

    def test_two_letter_string_returns_none(self):
        assert normalize_choice("AB") is None

    def test_prose_returns_none(self):
        assert normalize_choice("the first one") is None

    def test_empty_string_returns_none(self):
        assert normalize_choice("") is None


class TestScoreMultipleChoice:
    def test_matching_letter(self):
        raw_text = "Some reasoning.\nANSWER: (B)"
        extracted, correct = score_multiple_choice(raw_text, "B", ("A", "B", "C", "D"))
        assert extracted == "(B)"
        assert correct is True

    def test_mismatch(self):
        raw_text = "Some reasoning.\nANSWER: C"
        extracted, correct = score_multiple_choice(raw_text, "B", ("A", "B", "C", "D"))
        assert extracted == "C"
        assert correct is False
