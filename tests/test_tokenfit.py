"""Tests for tokenfit."""
from __future__ import annotations

import pytest

from tokenfit import (
    __version__,
    chars_per_token,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    list_models,
)


# ---- estimate_tokens ------------------------------------------------- #

def test_estimate_tokens_empty_string():
    assert estimate_tokens("", "gpt-4") == 0


def test_estimate_tokens_none_input():
    assert estimate_tokens(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_tokens_proportional_to_length():
    a = estimate_tokens("a" * 40, "gpt-4")
    b = estimate_tokens("a" * 80, "gpt-4")
    assert abs(b - 2 * a) <= 1


def test_estimate_tokens_rounds_up():
    # "hello" is 5 chars, gpt-4 ratio is 4.0 → ceil(1.25) == 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    assert estimate_tokens("hello world", "totally-fake-model") == \
        estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio (3.5) < GPT-4 ratio (4.0) → more tokens per character
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_returns_int():
    result = estimate_tokens("some text", "gpt-4")
    assert isinstance(result, int)


def test_estimate_tokens_new_models_recognised():
    for model in ("gpt-4o-mini", "o1", "o3-mini", "claude-3.5-haiku", "claude-opus-4",
                  "gemini-2.5-pro", "gemini-1.5-flash"):
        n = estimate_tokens("hello world", model)
        assert n > 0, f"expected positive count for model {model!r}"


# ---- estimate_messages ----------------------------------------------- #

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
    assert estimate_messages(msgs, "gpt-4") > 0


def test_estimate_messages_overhead_differs_by_family():
    msgs = [{"role": "user", "content": "a" * 100}]
    # Claude overhead (5) > GPT-4o overhead (3)
    n_claude = estimate_messages(msgs, "claude-3-opus")
    n_gpt4o = estimate_messages(msgs, "gpt-4o")
    assert n_claude != n_gpt4o


# ---- fits_in_context ------------------------------------------------- #

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


def test_fits_in_context_returns_bool():
    result = fits_in_context("hello", "gpt-4")
    assert isinstance(result, bool)


# ---- context_window -------------------------------------------------- #

def test_context_window_gpt4():
    assert context_window("gpt-4") == 8_192


def test_context_window_gpt4_turbo():
    assert context_window("gpt-4-turbo") == 128_000


def test_context_window_claude():
    assert context_window("claude-3-opus") == 200_000


def test_context_window_gemini_15_pro():
    assert context_window("gemini-1.5-pro") == 1_048_576


def test_context_window_unknown_falls_back():
    assert context_window("totally-unknown-model") == context_window("gpt-4")


def test_context_window_case_insensitive():
    assert context_window("GPT-4") == context_window("gpt-4")


def test_context_window_returns_int():
    assert isinstance(context_window("gpt-4"), int)


# ---- chars_per_token ------------------------------------------------- #

def test_chars_per_token_gpt4():
    assert chars_per_token("gpt-4") == 4.0


def test_chars_per_token_claude():
    assert chars_per_token("claude-3-opus") == 3.5


def test_chars_per_token_returns_float():
    assert isinstance(chars_per_token("gpt-4"), float)


def test_chars_per_token_consistent_with_estimate():
    model = "gpt-4"
    ratio = chars_per_token(model)
    text = "x" * 100
    n = estimate_tokens(text, model)
    # ceil(100 / ratio) must equal the returned estimate
    import math
    assert n == math.ceil(len(text) / ratio)


# ---- list_models ----------------------------------------------------- #

def test_list_models_returns_list():
    assert isinstance(list_models(), list)


def test_list_models_non_empty():
    assert len(list_models()) > 0


def test_list_models_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_known_entries():
    models = list_models()
    for expected in ("gpt-4", "claude-3-opus", "gemini-1.5-pro"):
        assert expected in models, f"{expected!r} missing from list_models()"


def test_list_models_no_duplicates():
    models = list_models()
    assert len(models) == len(set(models))


# ---- __version__ ----------------------------------------------------- #

def test_version_string():
    assert isinstance(__version__, str)
    assert __version__  # non-empty
