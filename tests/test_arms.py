from prereg.arms import ARMS, ARMS_BY_ID, FORMAT_INSTRUCTION
from harness.schema import AnswerFormat


def test_seven_arms_frozen():
    assert len(ARMS) == 7
    assert [arm.id for arm in ARMS] == ["B", "T1", "T2", "T3", "T4", "T5", "T6"]


def test_arms_by_id_covers_every_arm():
    assert set(ARMS_BY_ID) == {"B", "T1", "T2", "T3", "T4", "T5", "T6"}


def test_format_instruction_defined_for_every_answer_format():
    assert set(FORMAT_INSTRUCTION) == set(AnswerFormat)
