import pytest

from harness.prompt_builder import build_prompt, format_few_shot_block
from harness.schema import AnswerFormat, Question, TaskType
from prereg.arms import ARMS, FORMAT_INSTRUCTION


@pytest.fixture
def numeric_question():
    return Question(
        id="math-001",
        task_type=TaskType.MATH,
        domain="mathematics",
        prompt="What is 12 + 30?",
        answer_format=AnswerFormat.NUMERIC,
        gold_answer="42",
    )


@pytest.fixture
def mc_question():
    return Question(
        id="logic-001",
        task_type=TaskType.LOGIC,
        domain="logical reasoning",
        prompt="If all bloops are razzies, which must be true?",
        answer_format=AnswerFormat.MULTIPLE_CHOICE,
        gold_answer="B",
        choices=("A", "B", "C", "D"),
    )


def test_baseline_contains_question_and_format_instruction(numeric_question):
    prompt = build_prompt(numeric_question, "B")
    assert numeric_question.prompt in prompt
    assert FORMAT_INSTRUCTION[AnswerFormat.NUMERIC] in prompt


def test_format_instruction_identical_across_every_arm(numeric_question):
    """The whole experiment rests on this: only the technique text may vary."""
    expected = FORMAT_INSTRUCTION[AnswerFormat.NUMERIC]
    for arm in ARMS:
        prompt = build_prompt(numeric_question, arm.id)
        assert expected in prompt, f"arm {arm.id} dropped or altered the format instruction"


def test_role_prompt_fills_in_domain(mc_question):
    prompt = build_prompt(mc_question, "T2")
    assert "world-class expert in logical reasoning" in prompt


def test_politeness_arm_is_wrapped_not_substituted(numeric_question):
    prompt = build_prompt(numeric_question, "T5")
    assert "please" in prompt.lower()
    assert "thank you" in prompt.lower()
    assert numeric_question.prompt in prompt
    assert FORMAT_INSTRUCTION[AnswerFormat.NUMERIC] in prompt


def test_cot_trigger_appended_after_format_instruction(numeric_question):
    prompt = build_prompt(numeric_question, "T1")
    format_idx = prompt.index(FORMAT_INSTRUCTION[AnswerFormat.NUMERIC])
    cot_idx = prompt.index("Think step by step")
    assert cot_idx > format_idx


def test_few_shot_block_renders_fixed_examples():
    examples = (("What is 1 + 1?", "ANSWER: 2"), ("What is 3 + 3?", "ANSWER: 6"))
    block = format_few_shot_block(examples)
    assert "Example 1:" in block
    assert "What is 1 + 1?" in block
    assert "Example 2:" in block
    assert "What is 3 + 3?" in block


def test_few_shot_arm_includes_examples_before_question(numeric_question):
    examples = (("What is 1 + 1?", "ANSWER: 2"),)
    prompt = build_prompt(numeric_question, "T6", few_shot_examples=examples)
    assert prompt.index("Example 1:") < prompt.index(numeric_question.prompt)


def test_other_arms_ignore_few_shot_examples(numeric_question):
    examples = (("What is 1 + 1?", "ANSWER: 2"),)
    prompt = build_prompt(numeric_question, "B", few_shot_examples=examples)
    assert "Example 1:" not in prompt
