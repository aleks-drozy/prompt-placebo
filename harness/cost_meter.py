"""Hard-cap USD spend tracker for Anthropic Message Batches API calls.

This project runs a fixed-budget experiment against the Anthropic API. A
single bug in a retry loop or a mis-estimated token count could otherwise
burn through the entire research budget before anyone notices. This module
is the one place that turns "estimated tokens" into "estimated USD" and
refuses to let cumulative spend cross a cap -- it halts BEFORE the spend
happens, not after, so the real bill can never exceed the cap the caller
configured. It does not do any EUR/USD currency conversion; the caller is
responsible for supplying cap_usd already converted from their EUR budget.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float   # USD per 1,000,000 input tokens, STANDARD (non-batch) rate
    output_per_mtok: float  # USD per 1,000,000 output tokens, STANDARD rate


# Exact current Anthropic pricing, verified 2026-07-23. Sonnet 5 is running
# introductory pricing through 2026-08-31 (reverts to 3.00 / 15.00 after --
# this project is expected to finish its full run well before that date, so
# use the intro rate as the constant; leave a comment noting the revert date).
PRICING = {
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-sonnet-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
}

SONNET_5_INTRO_PRICING_EXPIRES = date(2026, 8, 31)

BATCH_DISCOUNT = 0.5  # Anthropic Message Batches API: 50 percent off standard per-token rates

_intro_pricing_expiry_warned = False


def _today() -> date:
    """Indirection point so tests can monkeypatch "today" without touching
    the real system clock."""
    return date.today()


def _warn_if_intro_pricing_expired() -> None:
    """If the project's run slips past the Sonnet 5 intro-pricing window,
    PRICING silently under-estimates real cost -- which would quietly weaken
    the one guarantee this whole module exists to provide (spend can never
    exceed cap_usd). Warn once, rather than let that happen silently."""
    global _intro_pricing_expiry_warned
    if _intro_pricing_expiry_warned:
        return
    if _today() > SONNET_5_INTRO_PRICING_EXPIRES:
        warnings.warn(
            "claude-sonnet-5 introductory pricing expired on "
            f"{SONNET_5_INTRO_PRICING_EXPIRES}; PRICING still uses the intro "
            "rate (2.00/10.00), so cost estimates are now too low. Update "
            "PRICING to the standard rate (3.00/15.00).",
            stacklevel=3,
        )
        _intro_pricing_expiry_warned = True


class BudgetExceededError(RuntimeError):
    """Raised when recording a call would push cumulative spend over the cap."""


class CostMeter:
    def __init__(self, cap_usd: float, batch: bool = True):
        self.cap_usd = cap_usd
        self.batch = batch
        self._spent_usd = 0.0

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        if model_id == "claude-sonnet-5":
            _warn_if_intro_pricing_expired()
        pricing = PRICING[model_id]
        cost = (
            input_tokens / 1_000_000 * pricing.input_per_mtok
            + output_tokens / 1_000_000 * pricing.output_per_mtok
        )
        if self.batch:
            cost *= BATCH_DISCOUNT
        return cost

    def record(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        cost = self.estimate_cost(model_id, input_tokens, output_tokens)
        if self._spent_usd + cost > self.cap_usd:
            raise BudgetExceededError(
                f"recording this call ({cost:.6f} USD) would push cumulative spend "
                f"({self._spent_usd:.6f} USD) over the cap ({self.cap_usd:.6f} USD)"
            )
        self._spent_usd += cost

    @property
    def total_spent_usd(self) -> float:
        return self._spent_usd

    def remaining_usd(self) -> float:
        return self.cap_usd - self._spent_usd
