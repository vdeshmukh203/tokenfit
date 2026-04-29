"""Tests for tokenfit."""
from __future__ import annotations

import math
import warnings

import pytest

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    list_models,
)


# ── estimate_tokens ──────────────────────────────────────────────────────────


def test_estimate_tokens_empty_string():
    assert estimate_tokens("", "gpt-4") == 0


def test_estimate_tokens_none_input():
    assert estimate_tokens(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_tokens_proportional_to_length():
    a = estimate_tokens("a" * 40, "gpt-4")
    b = estimate_tokens("a" * 80, "gpt-4")
    assert abs(b - 2 * a) <= 1


def test_estimate_tokens_rounds_up():
    # "hello" is 5 chars; gpt-4 ratio is 4 chars/token → ceil(5/4) = 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimate_tokens("hello world", "totally-fake-model")
    assert result == estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_unknown_model_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimate_tokens("hello", "totally-fake-model")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "totally-fake-model" in str(w[0].message)


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio (3.5) is lower than gpt-4 (4.0) → more tokens for same text
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


@pytest.mark.parametrize(
    "model, ratio",
    [
        ("gpt-4", 4.0),
        ("gpt-4o", 3.8),
        ("gpt-4-turbo", 4.0),
        ("gpt-3.5", 4.0),
        ("claude-3-opus", 3.5),
        ("claude-3.5-sonnet", 3.5),
        ("claude-sonnet-4", 3.5),
        ("gemini-2.0-flash", 4.0),
        ("gemini-1.5-pro", 4.0),
        ("gemini-pro", 4.0),
    ],
)
def test_estimate_tokens_per_model_ratio(model: str, ratio: float):
    """Token count must equal ceil(len / ratio) for each supported family."""
    text = "a" * 100
    assert estimate_tokens(text, model) == math.ceil(100 / ratio)


@pytest.mark.parametrize(
    "prefix, base_model",
    [
        ("gpt-4-preview", "gpt-4"),
        ("gpt-4o-mini", "gpt-4o"),
        ("claude-3-haiku", "claude-3"),
        ("gemini-ultra", "gemini"),
    ],
)
def test_estimate_tokens_prefix_matching(prefix: str, base_model: str):
    """Model names that start with a known prefix use that family's ratio."""
    text = "a" * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert estimate_tokens(text, prefix) == estimate_tokens(text, base_model)


# ── estimate_messages ────────────────────────────────────────────────────────


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


def test_estimate_messages_overhead_varies_by_family():
    # Claude overhead (5) > gpt-4o overhead (3), so Claude total is higher
    msgs = [{"role": "user", "content": "a" * 100}]
    assert estimate_messages(msgs, "claude-3-opus") > estimate_messages(
        msgs, "gpt-4o"
    )


@pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5",
                                    "claude-3-opus", "claude-3.5-sonnet",
                                    "gemini-2.0-flash", "gemini-1.5-pro"])
def test_estimate_messages_all_models_positive(model: str):
    msgs = [{"role": "user", "content": "Hello"}]
    assert estimate_messages(msgs, model) > 0


# ── fits_in_context ───────────────────────────────────────────────────────────


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


def test_fits_in_context_exact_boundary():
    # gpt-4 window is 8192; craft text that uses exactly the window
    from tokenfit import _WINDOWS, _RATIOS
    win = _WINDOWS["gpt-4"]
    ratio = _RATIOS["gpt-4"]
    # len such that ceil(len/ratio) == win
    length = int(win * ratio)
    text = "a" * length
    assert estimate_tokens(text, "gpt-4") == win
    assert fits_in_context(text, "gpt-4") is True
    assert fits_in_context(text, "gpt-4", headroom=1) is False


# ── list_models ───────────────────────────────────────────────────────────────


def test_list_models_returns_list():
    assert isinstance(list_models(), list)


def test_list_models_is_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_known_families():
    models = list_models()
    for name in ("gpt-4", "gpt-4o", "claude-3-opus", "gemini-1.5-pro"):
        assert name in models


def test_list_models_all_have_positive_context_window():
    for m in list_models():
        assert context_window(m) > 0


# ── context_window ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-4", 8_192),
        ("gpt-4o", 128_000),
        ("gpt-4-turbo", 128_000),
        ("gpt-3.5", 16_385),
        ("claude-3-opus", 200_000),
        ("gemini-2.0-flash", 1_048_576),
        ("gemini-1.5-pro", 1_048_576),
        ("gemini-pro", 32_768),
    ],
)
def test_context_window_known_models(model: str, expected: int):
    assert context_window(model) == expected


def test_context_window_unknown_model_warns_and_falls_back():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = context_window("totally-unknown-model")
    assert result == context_window("gpt-4")
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "totally-unknown-model" in str(w[0].message)


# ── version ───────────────────────────────────────────────────────────────────


def test_version_string_is_present():
    assert isinstance(__version__, str)
    assert __version__  # non-empty
