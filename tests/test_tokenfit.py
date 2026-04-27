"""Tests for tokenfit."""
from __future__ import annotations

import pytest

from tokenfit import (
    __version__,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    fits_in_context_messages,
    list_models,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty_string():
    assert estimate_tokens("", "gpt-4") == 0


def test_estimate_tokens_none_input():
    assert estimate_tokens(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_tokens_proportional_to_length():
    a = estimate_tokens("a" * 40, "gpt-4")
    b = estimate_tokens("a" * 80, "gpt-4")
    assert abs(b - 2 * a) <= 1


def test_estimate_tokens_rounds_up():
    # "hello" = 5 chars, gpt-4 ratio 4.0 → ceil(1.25) = 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    assert estimate_tokens("hello world", "totally-fake-model") == \
        estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio is 3.5 vs gpt-4 ratio of 4.0 → more tokens estimated
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_returns_integer():
    result = estimate_tokens("some text", "gpt-4")
    assert isinstance(result, int)


def test_estimate_tokens_gpt4o_ratio():
    # gpt-4o ratio is 3.8 (slightly tighter than gpt-4's 4.0)
    text = "a" * 38
    assert estimate_tokens(text, "gpt-4o") == estimate_tokens(text, "gpt-4o")
    assert estimate_tokens(text, "gpt-4o") >= estimate_tokens(text, "gpt-4")


def test_estimate_tokens_versioned_model_name():
    # Versioned variants should match the base family via prefix matching
    assert estimate_tokens("hello", "claude-3.5-sonnet-20241022") == \
        estimate_tokens("hello", "claude-3.5-sonnet")


def test_estimate_tokens_claude_haiku():
    assert estimate_tokens("hello", "claude-3-haiku") > 0


def test_estimate_tokens_claude_opus4():
    assert estimate_tokens("hello", "claude-opus-4") > 0


# ---------------------------------------------------------------------------
# estimate_messages
# ---------------------------------------------------------------------------

def test_estimate_messages_empty():
    assert estimate_messages([], "gpt-4") == 0


def test_estimate_messages_none_input():
    assert estimate_messages(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_messages_basic():
    msgs = [{"role": "user", "content": "Hello world"}]
    n = estimate_messages(msgs, "gpt-4")
    assert 4 < n < 30


def test_estimate_messages_returns_integer():
    msgs = [{"role": "user", "content": "test"}]
    assert isinstance(estimate_messages(msgs, "gpt-4"), int)


def test_estimate_messages_more_messages_means_more_tokens():
    one = [{"role": "user", "content": "Hi"}]
    two = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    assert estimate_messages(two, "gpt-4") > estimate_messages(one, "gpt-4")


def test_estimate_messages_skips_non_mappings():
    valid = [{"role": "user", "content": "Hi"}]
    mixed = [{"role": "user", "content": "Hi"}, "not a message"]  # type: ignore[list-item]
    assert estimate_messages(mixed, "gpt-4") == estimate_messages(valid, "gpt-4")


def test_estimate_messages_handles_missing_role():
    msgs = [{"content": "just content"}]
    n = estimate_messages(msgs, "gpt-4")
    assert n > 0


def test_estimate_messages_handles_missing_content():
    msgs = [{"role": "user"}]
    n = estimate_messages(msgs, "gpt-4")
    assert n > 0  # overhead tokens are still counted


def test_estimate_messages_claude_overhead():
    msgs = [{"role": "user", "content": "Hi"}]
    # Claude has overhead of 5 per message vs GPT-4's 4
    assert estimate_messages(msgs, "claude-3-opus") >= estimate_messages(msgs, "gpt-4")


def test_estimate_messages_gpt4o_lower_overhead():
    # With empty role and content only overhead tokens remain,
    # showing gpt-4o (3) vs gpt-4 (4) per-message overhead difference.
    msgs = [{"role": "", "content": ""}]
    assert estimate_messages(msgs, "gpt-4o") == 3
    assert estimate_messages(msgs, "gpt-4") == 4


# ---------------------------------------------------------------------------
# fits_in_context
# ---------------------------------------------------------------------------

def test_fits_in_context_short_text_fits():
    assert fits_in_context("hello", "gpt-4") is True


def test_fits_in_context_returns_bool():
    assert isinstance(fits_in_context("hello", "gpt-4"), bool)


def test_fits_in_context_huge_text_overflows_small_window():
    huge = "a" * 100_000
    assert fits_in_context(huge, "gpt-4") is False


def test_fits_in_context_huge_text_fits_large_window():
    text = "a" * 100_000
    assert fits_in_context(text, "gpt-4-turbo") is True


def test_fits_in_context_headroom_reduces_capacity():
    assert fits_in_context("hi", "gpt-4", headroom=10_000) is False


def test_fits_in_context_negative_headroom_clamps_to_zero():
    assert fits_in_context("hi", "gpt-4", headroom=-1000) is True


def test_fits_in_context_none_text():
    assert fits_in_context(None, "gpt-4") is True  # type: ignore[arg-type]


def test_fits_in_context_unknown_model_uses_default():
    # Unknown model falls back to gpt-4 (8,192 token window)
    result = fits_in_context("hi", "unknown-model-xyz")
    assert result is True


# ---------------------------------------------------------------------------
# fits_in_context_messages
# ---------------------------------------------------------------------------

def test_fits_in_context_messages_empty():
    assert fits_in_context_messages([], "gpt-4") is True


def test_fits_in_context_messages_returns_bool():
    msgs = [{"role": "user", "content": "hi"}]
    assert isinstance(fits_in_context_messages(msgs, "gpt-4"), bool)


def test_fits_in_context_messages_short_fits():
    msgs = [{"role": "user", "content": "Hello!"}]
    assert fits_in_context_messages(msgs, "gpt-4") is True


def test_fits_in_context_messages_overflow():
    msgs = [{"role": "user", "content": "a" * 100_000}]
    assert fits_in_context_messages(msgs, "gpt-4") is False


def test_fits_in_context_messages_large_window():
    msgs = [{"role": "user", "content": "a" * 100_000}]
    assert fits_in_context_messages(msgs, "claude-3-opus") is True


def test_fits_in_context_messages_headroom():
    msgs = [{"role": "user", "content": "hi"}]
    assert fits_in_context_messages(msgs, "gpt-4", headroom=10_000) is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_returns_list():
    models = list_models()
    assert isinstance(models, list)


def test_list_models_nonempty():
    assert len(list_models()) > 0


def test_list_models_is_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_known_models():
    models = list_models()
    for name in ("gpt-4", "gpt-4o", "claude-3-opus", "gemini-1.5-pro"):
        assert name in models, f"{name!r} missing from list_models()"


def test_list_models_new_models_present():
    models = list_models()
    for name in ("claude-3-haiku", "claude-3.5-haiku", "claude-opus-4"):
        assert name in models, f"{name!r} missing from list_models()"


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
