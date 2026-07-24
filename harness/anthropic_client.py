"""Production adapter: anthropic.Anthropic().messages.batches -> BatchClient.

The one place any real (non-test) driver script constructs a client against
the live Anthropic API, so the Batches API wire-up exists in exactly one
spot rather than being copy-pasted into every driver script that needs it.
"""
from __future__ import annotations

from typing import Any


def build_real_batch_client():
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batches = anthropic.Anthropic().messages.batches

    class _Adapter:
        def create(self, requests: list[dict[str, Any]]):
            return batches.create(
                requests=[
                    Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
                    for r in requests
                ]
            )

        def retrieve(self, batch_id: str):
            return batches.retrieve(batch_id)

        def results(self, batch_id: str):
            return batches.results(batch_id)

    return _Adapter()
