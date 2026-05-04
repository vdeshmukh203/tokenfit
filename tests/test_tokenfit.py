"""Tests for tokenfit."""
from __future__ import annotations

import pytest

from tokenfit import (
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    get_context_window,
    list_models,
)


# ---- estimate_tokens ----

def test_estimate_tokens_empty_string():
    assert estimate_tokens("", "gpt-4") == 0


def test_estimate_tokens_none_input():
    assert estimate_tokens(None, "gpt-4") == 0


def test_estimate_tokens_proportional_to_length():
    a = estimate_tokens("a" * 40, "gpt-4")
    b = estimate_tokens("a" * 80, "gpt-4")
    assert abs(b - 2 * a) <= 1


def test_estimate_tokens_rounds_up():
    # "hello" = 5 chars / 4.0 ratio = 1.25 → ceil → 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    assert estimate_tokens("hello world", "totally-fake-model") == \
        estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio 3.5 < GPT-4 ratio 4.0, so same text → more Claude tokens.
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


@pytest.mark.parametrize("model", [
    "gpt-3.5",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
    "o1",
    "o3",
    "o4-mini",
    "claude-3.5-haiku",
    "claude-3.5-sonnet",
    "claude-3-5-haiku",
    "claude-3-5-sonnet",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-3-opus",
    "claude-3-haiku",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-pro",
])
def test_estimate_tokens_all_supported_models_return_positive(model: str):
    assert estimate_tokens("Hello, world!", model) > 0


def test_estimate_tokens_prefix_match_claude_api_format():
    # "claude-3-5-sonnet-20241022" should prefix-match "claude-3-5-sonnet".
    result = estimate_tokens("test", "claude-3-5-sonnet-20241022")
    assert result == estimate_tokens("test", "claude-3-5-sonnet")


# ---- estimate_messages ----

def test_estimate_messages_empty():
    assert estimate_messages([], "gpt-4") == 0


def test_estimate_messages_basic():
    msgs = [{"role": "user", "content": "Hello world"}]
    n = estimate_messages(msgs, "gpt-4")
    assert 4 < n < 30


def test_estimate_messages_more_messages_means_more_tokens():
    one = [{"role": "user", "content": "Hi"}]
    two = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert estimate_messages(two, "gpt-4") > estimate_messages(one, "gpt-4")


def test_estimate_messages_skips_non_mappings():
    valid = [{"role": "user", "content": "Hi"}]
    mixed = [{"role": "user", "content": "Hi"}, "not a message"]  # type: ignore[list-item]
    assert estimate_messages(mixed, "gpt-4") == estimate_messages(valid, "gpt-4")


def test_estimate_messages_handles_missing_role():
    msgs = [{"content": "just content"}]
    assert estimate_messages(msgs, "gpt-4") > 0


def test_estimate_messages_claude_overhead_higher_than_gpt():
    # Claude adds 5 tokens/message vs 4 for GPT-4.
    msgs = [{"role": "user", "content": "Hi"}] * 10
    assert estimate_messages(msgs, "claude-3-opus") > estimate_messages(msgs, "gpt-4")


def test_estimate_messages_none_iterable():
    assert estimate_messages(None, "gpt-4") == 0  # type: ignore[arg-type]


# ---- fits_in_context ----

def test_fits_in_context_short_text_fits():
    assert fits_in_context("hello", "gpt-4") is True


def test_fits_in_context_huge_text_overflows_small_window():
    huge = "a" * 100_000
    assert fits_in_context(huge, "gpt-4") is False


def test_fits_in_context_huge_text_fits_large_window():
    assert fits_in_context("a" * 100_000, "gpt-4-turbo") is True


def test_fits_in_context_headroom_reduces_capacity():
    assert fits_in_context("hi", "gpt-4", headroom=10_000) is False


def test_fits_in_context_negative_headroom_clamps_to_zero():
    assert fits_in_context("hi", "gpt-4", headroom=-1000) is True


def test_fits_in_context_none_text_treated_as_empty():
    assert fits_in_context(None, "gpt-4") is True


def test_fits_in_context_large_model_accepts_big_text():
    text = "word " * 50_000   # ~250 000 chars → ~62 500 tokens
    assert fits_in_context(text, "claude-3-opus") is True
    assert fits_in_context(text, "gemini-2.0-flash") is True


# ---- get_context_window ----

def test_get_context_window_gpt4():
    assert get_context_window("gpt-4") == 8_192


def test_get_context_window_gpt4_turbo():
    assert get_context_window("gpt-4-turbo") == 128_000


def test_get_context_window_gpt41():
    assert get_context_window("gpt-4.1") == 1_047_576


def test_get_context_window_claude():
    assert get_context_window("claude-3-opus") == 200_000


def test_get_context_window_gemini_flash():
    assert get_context_window("gemini-2.0-flash") == 1_048_576


def test_get_context_window_unknown_falls_back_to_gpt4():
    assert get_context_window("mystery-model") == get_context_window("gpt-4")


def test_get_context_window_prefix_match():
    # Full API model ID should resolve to the same window as the family key.
    assert get_context_window("claude-3-opus-20240229") == 200_000


# ---- list_models ----

def test_list_models_returns_non_empty_list():
    models = list_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_list_models_is_alphabetically_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_key_families():
    models = list_models()
    expected = {"gpt-4", "gpt-4o", "claude-3-opus", "gemini-pro", "o1", "o3"}
    assert expected.issubset(set(models))


def test_list_models_all_have_context_window():
    for model in list_models():
        assert get_context_window(model) > 0
