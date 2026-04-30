"""Tests for tokenfit."""
from __future__ import annotations

import pytest

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
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
    # "hello" is 5 chars; gpt-4 ratio is 4.0 → ceil(5/4) = 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_falls_back():
    assert estimate_tokens("hello world", "totally-fake-model") == \
        estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio (3.5) < GPT-4 ratio (4.0) → more tokens per char
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_model_case_insensitive():
    assert estimate_tokens("hello", "GPT-4") == estimate_tokens("hello", "gpt-4")


def test_estimate_tokens_gpt4o_variant():
    # gpt-4o has ratio 3.8, so tokens > gpt-4 (ratio 4.0) for same text
    text = "a" * 100
    assert estimate_tokens(text, "gpt-4o") >= estimate_tokens(text, "gpt-4")


def test_estimate_tokens_o_series_model():
    # o1/o3/o4 use the gpt-4o ratio (3.8)
    text = "hello world"
    assert estimate_tokens(text, "o1") == estimate_tokens(text, "o3")


def test_estimate_tokens_llama_model():
    # llama ratio is 3.7, produces more tokens than gpt-4 (4.0)
    text = "a" * 100
    assert estimate_tokens(text, "llama-3") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_new_claude_models():
    # All Claude families share the same ratio (3.5)
    text = "some sample text for testing"
    assert estimate_tokens(text, "claude-opus-4") == \
        estimate_tokens(text, "claude-3-opus")
    assert estimate_tokens(text, "claude-3.5-haiku") == \
        estimate_tokens(text, "claude-3-haiku")


def test_estimate_tokens_gemini_25_pro():
    text = "hello gemini"
    # gemini-2.5-pro uses same ratio as generic gemini
    assert estimate_tokens(text, "gemini-2.5-pro") == \
        estimate_tokens(text, "gemini")


def test_estimate_tokens_model_with_suffix():
    # Version-dated variants should fall through to the correct family
    assert estimate_tokens("hi", "claude-3.5-sonnet-20241022") == \
        estimate_tokens("hi", "claude-3.5-sonnet")


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
    msgs = [{"role": "system"}]
    n = estimate_messages(msgs, "gpt-4")
    # should still account for the role + overhead, not crash
    assert n > 0


def test_estimate_messages_claude_overhead():
    msgs = [{"role": "user", "content": "test"}]
    gpt = estimate_messages(msgs, "gpt-4")
    claude = estimate_messages(msgs, "claude-3-opus")
    # Claude has higher per-message overhead (5 vs 4) AND higher per-char ratio
    assert claude > gpt


def test_estimate_messages_system_prompt_counts():
    no_sys = [{"role": "user", "content": "Hello"}]
    with_sys = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    assert estimate_messages(with_sys, "gpt-4") > estimate_messages(no_sys, "gpt-4")


# ---------------------------------------------------------------------------
# fits_in_context
# ---------------------------------------------------------------------------

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


def test_fits_in_context_claude_large_window():
    # Claude has 200k context; 100k chars at 3.5 ratio ≈ 28k tokens → fits
    text = "a" * 100_000
    assert fits_in_context(text, "claude-3-opus") is True


def test_fits_in_context_gemini_huge_window():
    # Gemini 1.5 Pro has 1M token context
    text = "a" * 500_000
    assert fits_in_context(text, "gemini-1.5-pro") is True


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------

def test_context_window_gpt4():
    assert context_window("gpt-4") == 8_192


def test_context_window_gpt4_turbo():
    assert context_window("gpt-4-turbo") == 128_000


def test_context_window_gpt4o():
    assert context_window("gpt-4o") == 128_000


def test_context_window_claude():
    assert context_window("claude-3-opus") == 200_000


def test_context_window_gemini_15_pro():
    assert context_window("gemini-1.5-pro") == 1_048_576


def test_context_window_gemini_pro():
    # gemini-pro (without version) has the 32k window
    assert context_window("gemini-pro") == 32_768


def test_context_window_unknown_model_fallback():
    assert context_window("no-such-model-xyz") == context_window("gpt-4")


def test_context_window_o1():
    assert context_window("o1") == 200_000


def test_context_window_llama3():
    assert context_window("llama-3") == 128_000


def test_context_window_returns_positive_int():
    for model in list_models():
        w = context_window(model)
        assert isinstance(w, int) and w > 0, f"Bad window for {model!r}: {w}"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_returns_list():
    assert isinstance(list_models(), list)


def test_list_models_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_contains_known_families():
    models = list_models()
    for expected in ("gpt-4", "claude-3-opus", "gemini", "llama-3", "mistral"):
        assert expected in models, f"{expected!r} missing from list_models()"


def test_list_models_non_empty():
    assert len(list_models()) > 0


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

def test_version_string():
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts)
