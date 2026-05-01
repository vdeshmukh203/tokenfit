"""Tests for tokenfit."""
from __future__ import annotations

import math

import pytest

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    list_models,
    token_budget,
)


# ============================================================
# estimate_tokens
# ============================================================

def test_estimate_tokens_empty_string():
    assert estimate_tokens("", "gpt-4") == 0


def test_estimate_tokens_none_input():
    assert estimate_tokens(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_tokens_proportional_to_length():
    a = estimate_tokens("a" * 40, "gpt-4")
    b = estimate_tokens("a" * 80, "gpt-4")
    assert abs(b - 2 * a) <= 1


def test_estimate_tokens_rounds_up():
    # "hello" = 5 chars, ratio 4.0 → ceil(5/4) = 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    assert estimate_tokens("hello world", "totally-fake-model") == \
        estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio 3.5 < gpt-4 ratio 4.0 → more tokens for same text
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_always_positive_for_nonempty():
    assert estimate_tokens("x", "gpt-4") >= 1


def test_estimate_tokens_never_underreports():
    # The estimate must be >= the true mathematical ceiling
    text = "a" * 13
    ratio = 4.0  # gpt-4
    assert estimate_tokens(text, "gpt-4") == math.ceil(len(text) / ratio)


@pytest.mark.parametrize("model,ratio", [
    ("gpt-4", 4.0),
    ("gpt-4-turbo", 4.0),
    ("gpt-4o", 3.8),
    ("gpt-3.5", 4.0),
    ("claude-3-opus", 3.5),
    ("claude-3.5-sonnet", 3.5),
    ("claude-sonnet-4", 3.5),
    ("gemini-pro", 4.0),
    ("gemini-1.5-pro", 4.0),
    ("gemini-2.0-flash", 4.0),
])
def test_estimate_tokens_matches_ratio(model: str, ratio: float):
    text = "a" * 100
    expected = math.ceil(100 / ratio)
    assert estimate_tokens(text, model) == expected


def test_estimate_tokens_gpt4o_uses_lower_ratio():
    # gpt-4o ratio (3.8) is lower than gpt-4 (4.0), so more tokens for same text
    text = "a" * 100
    assert estimate_tokens(text, "gpt-4o") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_prefix_matching():
    # "gpt-4-turbo-preview" should match "gpt-4-turbo" prefix
    assert estimate_tokens("hello", "gpt-4-turbo-preview") == \
        estimate_tokens("hello", "gpt-4-turbo")

    # "claude-3-opus-20240229" should match "claude-3-opus" prefix
    assert estimate_tokens("hello", "claude-3-opus-20240229") == \
        estimate_tokens("hello", "claude-3-opus")


# ============================================================
# estimate_messages
# ============================================================

def test_estimate_messages_empty():
    assert estimate_messages([], "gpt-4") == 0


def test_estimate_messages_none_input():
    assert estimate_messages(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_messages_basic():
    msgs = [{"role": "user", "content": "Hello world"}]
    n = estimate_messages(msgs, "gpt-4")
    assert 4 < n < 30


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


def test_estimate_messages_overhead_applied_per_message():
    # Each message adds per-message overhead; two identical messages cost more than one
    msg = {"role": "user", "content": "test"}
    single = estimate_messages([msg], "gpt-4")
    double = estimate_messages([msg, msg], "gpt-4")
    assert double == 2 * single


def test_estimate_messages_claude_overhead():
    # Claude overhead is 5 tokens per message (vs 4 for gpt-4)
    msgs = [{"role": "user", "content": ""}]
    gpt_count = estimate_messages(msgs, "gpt-4")
    claude_count = estimate_messages(msgs, "claude-3-opus")
    assert claude_count > gpt_count


def test_estimate_messages_generator_input():
    def _gen():
        yield {"role": "user", "content": "hello"}
        yield {"role": "assistant", "content": "world"}

    assert estimate_messages(_gen(), "gpt-4") > 0


# ============================================================
# fits_in_context
# ============================================================

def test_fits_in_context_short_text_fits():
    assert fits_in_context("hello", "gpt-4") is True


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


def test_fits_in_context_exactly_at_window_boundary():
    # Craft a text whose estimate equals exactly the window size
    win = context_window("gpt-4")  # 8192
    ratio = 4.0
    # 8192 * 4 = 32768 chars → estimate = ceil(32768/4) = 8192 = window
    text = "a" * (win * int(ratio))
    assert fits_in_context(text, "gpt-4") is True


def test_fits_in_context_one_over_boundary():
    win = context_window("gpt-4")
    ratio = 4.0
    # One extra token's worth of characters over the limit
    text = "a" * (win * int(ratio) + 1)
    assert fits_in_context(text, "gpt-4") is False


# ============================================================
# context_window
# ============================================================

@pytest.mark.parametrize("model,expected", [
    ("gpt-4", 8_192),
    ("gpt-4-turbo", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-3.5", 16_385),
    ("claude-3-opus", 200_000),
    ("claude-3.5-sonnet", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3", 200_000),
    ("claude", 200_000),
    ("gemini-pro", 32_768),
    ("gemini-1.5-pro", 1_048_576),
    ("gemini-2.0-flash", 1_048_576),
])
def test_context_window_known_models(model: str, expected: int):
    assert context_window(model) == expected


def test_context_window_unknown_falls_back_to_gpt4():
    assert context_window("unknown-model-xyz") == context_window("gpt-4")


def test_context_window_prefix_matching():
    # Versioned variants resolve to the correct family
    assert context_window("gpt-4-turbo-preview") == 128_000
    assert context_window("claude-3-opus-20240229") == 200_000


# ============================================================
# token_budget
# ============================================================

def test_token_budget_positive_for_short_text():
    budget = token_budget("hi", "gpt-4")
    assert 0 < budget < context_window("gpt-4")


def test_token_budget_sums_to_window():
    text = "hello world"
    model = "gpt-4"
    used = estimate_tokens(text, model)
    remaining = token_budget(text, model)
    assert used + remaining == context_window(model)


def test_token_budget_negative_on_overflow():
    huge = "a" * 200_000
    assert token_budget(huge, "gpt-4") < 0


def test_token_budget_headroom_reduces_budget():
    text = "hi"
    model = "gpt-4"
    budget_no_head = token_budget(text, model, headroom=0)
    budget_with_head = token_budget(text, model, headroom=1000)
    assert budget_no_head - budget_with_head == 1000


def test_token_budget_negative_headroom_clamped():
    text = "hi"
    model = "gpt-4"
    assert token_budget(text, model, headroom=-500) == token_budget(text, model, headroom=0)


# ============================================================
# list_models
# ============================================================

def test_list_models_returns_list():
    assert isinstance(list_models(), list)


def test_list_models_is_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_core_families():
    models = list_models()
    for expected in ("gpt-4", "gpt-4o", "claude-3-opus", "gemini-pro"):
        assert expected in models, f"{expected!r} missing from list_models()"


def test_list_models_nonempty():
    assert len(list_models()) > 0


def test_list_models_all_accepted_by_estimate_tokens():
    for model in list_models():
        result = estimate_tokens("test text", model)
        assert isinstance(result, int) and result >= 0


# ============================================================
# __version__
# ============================================================

def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_has_major_minor_patch():
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
