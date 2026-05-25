"""Tests for observability: tracing decorator + LLM telemetry (SPEC §6.9)."""

import pytest

from common.types import LLMCallRecord
from harness.observability import compute_cost, get_logger, instrumented_call, traced


def test_get_logger_returns_bound_logger():
    log = get_logger("test")
    # should not raise; structured event call returns None
    assert log.info("hello", key="value") is None


def test_traced_sync_preserves_return_and_metadata():
    @traced(span_name="unit.sync")
    def add(a, b):
        return a + b

    assert add.__name__ == "add"
    assert add(2, 3) == 5


@pytest.mark.asyncio
async def test_traced_async_returns_value():
    @traced()
    async def double(x):
        return x * 2

    assert await double(21) == 42


def test_traced_reraises_exceptions():
    @traced(span_name="unit.boom")
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()


def test_compute_cost_known_model():
    # 1M input + 1M output of sonnet = 3 + 15 USD
    cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_compute_cost_cached_discount():
    full = compute_cost("claude-sonnet-4-6", 1_000_000, 0)
    cached = compute_cost("claude-sonnet-4-6", 1_000_000, 0, cached_tokens_in=1_000_000)
    assert cached < full
    assert cached == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_instrumented_call_records_usage():
    class FakeResp:
        usage = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 40}

        def __str__(self):
            return "answer"

    async def call():
        return FakeResp()

    resp = await instrumented_call("claude-haiku-4-5-20251001", call, extract_text=str)
    assert resp.text == "answer"
    assert isinstance(resp.record, LLMCallRecord)
    assert resp.record.tokens_in == 100
    assert resp.record.tokens_out == 50
    assert resp.record.cache_hit is True
    assert resp.record.cost_usd > 0
