"""Tests for harness.cost_meter — the hard-cap USD spend guard."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import harness.cost_meter as cost_meter
from harness.cost_meter import (
    BATCH_DISCOUNT,
    SONNET_5_INTRO_PRICING_EXPIRES,
    BudgetExceededError,
    CostMeter,
    ModelPricing,
    PRICING,
)


@pytest.fixture(autouse=True)
def _reset_expiry_warning_state():
    """The warn-once flag is module-level state; isolate each test from it."""
    cost_meter._intro_pricing_expiry_warned = False
    yield
    cost_meter._intro_pricing_expiry_warned = False


def test_pricing_table_has_expected_models() -> None:
    assert PRICING["claude-haiku-4-5"] == ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00)
    assert PRICING["claude-sonnet-5"] == ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00)


def test_estimate_cost_haiku_batch_halves_standard_rate() -> None:
    meter = CostMeter(cap_usd=1000.0, batch=True)
    expected = (1000 / 1_000_000 * 1.00 + 1000 / 1_000_000 * 5.00) * BATCH_DISCOUNT
    assert meter.estimate_cost("claude-haiku-4-5", 1000, 1000) == pytest.approx(expected)
    assert meter.estimate_cost("claude-haiku-4-5", 1000, 1000) == pytest.approx(0.003)


def test_estimate_cost_sonnet_batch_halves_standard_rate() -> None:
    meter = CostMeter(cap_usd=1000.0, batch=True)
    expected = (1000 / 1_000_000 * 2.00 + 1000 / 1_000_000 * 10.00) * BATCH_DISCOUNT
    assert meter.estimate_cost("claude-sonnet-5", 1000, 1000) == pytest.approx(expected)
    assert meter.estimate_cost("claude-sonnet-5", 1000, 1000) == pytest.approx(0.006)


def test_estimate_cost_no_batch_uses_full_standard_rate() -> None:
    meter = CostMeter(cap_usd=1000.0, batch=False)
    expected = 1000 / 1_000_000 * 1.00 + 1000 / 1_000_000 * 5.00
    assert meter.estimate_cost("claude-haiku-4-5", 1000, 1000) == pytest.approx(expected)
    assert meter.estimate_cost("claude-haiku-4-5", 1000, 1000) == pytest.approx(0.006)


def test_record_accumulates_total_spent() -> None:
    meter = CostMeter(cap_usd=1000.0, batch=True)
    meter.record("claude-haiku-4-5", 1000, 1000)  # 0.003
    meter.record("claude-sonnet-5", 1000, 1000)  # 0.006
    meter.record("claude-haiku-4-5", 2000, 2000)  # 0.006
    assert meter.total_spent_usd == pytest.approx(0.003 + 0.006 + 0.006)


def test_record_raises_budget_exceeded_and_does_not_partially_record() -> None:
    # Each haiku batch call costs 0.003 usd. Cap allows exactly one call.
    meter = CostMeter(cap_usd=0.003, batch=True)
    meter.record("claude-haiku-4-5", 1000, 1000)
    assert meter.total_spent_usd == pytest.approx(0.003)

    with pytest.raises(BudgetExceededError):
        meter.record("claude-haiku-4-5", 1000, 1000)

    # Rejected call must not be partially recorded.
    assert meter.total_spent_usd == pytest.approx(0.003)


def test_remaining_usd_tracks_cap_minus_spent_at_multiple_points() -> None:
    meter = CostMeter(cap_usd=1.0, batch=True)
    assert meter.remaining_usd() == pytest.approx(1.0)

    meter.record("claude-haiku-4-5", 1000, 1000)  # 0.003
    assert meter.remaining_usd() == pytest.approx(1.0 - 0.003)

    meter.record("claude-sonnet-5", 1000, 1000)  # 0.006
    assert meter.remaining_usd() == pytest.approx(1.0 - 0.003 - 0.006)


def test_estimate_cost_unknown_model_raises_key_error() -> None:
    meter = CostMeter(cap_usd=1000.0)
    with pytest.raises(KeyError):
        meter.estimate_cost("gpt-4", 1000, 1000)


def test_record_unknown_model_raises_key_error() -> None:
    meter = CostMeter(cap_usd=1000.0)
    with pytest.raises(KeyError):
        meter.record("gpt-4", 1000, 1000)


def test_warns_once_if_sonnet_intro_pricing_has_expired(monkeypatch, recwarn) -> None:
    monkeypatch.setattr(
        cost_meter, "_today", lambda: SONNET_5_INTRO_PRICING_EXPIRES + timedelta(days=1)
    )
    meter = CostMeter(cap_usd=1000.0)

    meter.estimate_cost("claude-sonnet-5", 1000, 1000)
    assert len(recwarn) == 1
    assert "introductory pricing expired" in str(recwarn[0].message)

    # Second call must not warn again (warn-once behavior).
    meter.estimate_cost("claude-sonnet-5", 1000, 1000)
    assert len(recwarn) == 1


def test_no_warning_before_expiry(monkeypatch, recwarn) -> None:
    monkeypatch.setattr(cost_meter, "_today", lambda: date(2026, 7, 23))
    meter = CostMeter(cap_usd=1000.0)

    meter.estimate_cost("claude-sonnet-5", 1000, 1000)
    assert len(recwarn) == 0
