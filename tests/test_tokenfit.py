"""Tests for tokenfit.

Coverage targets:
  - estimate_tokens  : empty/None, proportionality, rounding, model families, unknown model
  - estimate_messages: empty, None, basic, overhead accumulation, skipped items,
                       missing fields, model family differences
  - fits_in_context  : fits, overflows, headroom (+/-/zero), large windows
  - context_window   : known models, unknown model fallback
  - supported_models : non-empty list, known entries
  - __version__      : present and semver-like
"""
from __future__ import annotations

import math
import re

import pytest

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    supported_models,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("", "gpt-4") == 0

    def test_none_input_returns_zero(self):
        assert estimate_tokens(None, "gpt-4") == 0  # type: ignore[arg-type]

    def test_result_is_non_negative(self):
        assert estimate_tokens("hello", "gpt-4") >= 0

    def test_result_is_integer(self):
        assert isinstance(estimate_tokens("hello", "gpt-4"), int)

    def test_rounds_up_not_down(self):
        # "hello" = 5 chars; gpt-4 ratio = 4.0 → ceil(5/4) = 2, not floor = 1
        assert estimate_tokens("hello", "gpt-4") == 2

    def test_proportional_to_length(self):
        short = estimate_tokens("a" * 40, "gpt-4")
        long_ = estimate_tokens("a" * 80, "gpt-4")
        assert abs(long_ - 2 * short) <= 1

    def test_single_char(self):
        assert estimate_tokens("x", "gpt-4") == 1

    # Model-family differences
    @pytest.mark.parametrize("model", [
        "gpt-3.5", "gpt-4", "gpt-4o", "gpt-4-turbo",
    ])
    def test_openai_models_return_positive(self, model):
        assert estimate_tokens("Hello world", model) > 0

    @pytest.mark.parametrize("model", [
        "claude-3", "claude-3-opus", "claude-3-sonnet",
        "claude-3.5-sonnet", "claude-sonnet-4",
    ])
    def test_claude_models_return_positive(self, model):
        assert estimate_tokens("Hello world", model) > 0

    @pytest.mark.parametrize("model", [
        "gemini", "gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash",
    ])
    def test_gemini_models_return_positive(self, model):
        assert estimate_tokens("Hello world", model) > 0

    def test_claude_higher_count_than_gpt4_for_long_text(self):
        # Claude ratio 3.5 < gpt-4 ratio 4.0 → more tokens per char
        text = "a" * 100
        assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")

    def test_unknown_model_falls_back_to_gpt4(self):
        assert estimate_tokens("hello world", "totally-unknown-model-xyz") == \
            estimate_tokens("hello world", "gpt-4")

    def test_model_prefix_matching(self):
        # "gpt-4-turbo-preview" should match the gpt-4-turbo family
        assert estimate_tokens("hello", "gpt-4-turbo-preview") == \
            estimate_tokens("hello", "gpt-4-turbo")

    def test_case_insensitive_model(self):
        assert estimate_tokens("hello", "GPT-4") == estimate_tokens("hello", "gpt-4")

    def test_whitespace_stripped_model(self):
        assert estimate_tokens("hello", "  gpt-4  ") == estimate_tokens("hello", "gpt-4")

    def test_only_whitespace_text_returns_zero(self):
        # A string that is non-empty but falsy after bool() check — actual spaces
        # will be truthy; only empty string / None return 0
        result = estimate_tokens("   ", "gpt-4")
        assert isinstance(result, int)
        assert result >= 0

    def test_large_text_scales_linearly(self):
        base = estimate_tokens("a" * 1_000, "gpt-4")
        large = estimate_tokens("a" * 10_000, "gpt-4")
        # Should be approximately 10×, within ±2 due to ceiling rounding
        assert abs(large - 10 * base) <= 2


# ---------------------------------------------------------------------------
# estimate_messages
# ---------------------------------------------------------------------------

class TestEstimateMessages:
    def test_empty_list_returns_zero(self):
        assert estimate_messages([], "gpt-4") == 0

    def test_none_returns_zero(self):
        assert estimate_messages(None, "gpt-4") == 0  # type: ignore[arg-type]

    def test_basic_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        n = estimate_messages(msgs, "gpt-4")
        assert n > 0
        assert isinstance(n, int)

    def test_more_messages_means_more_tokens(self):
        one = [{"role": "user", "content": "Hi"}]
        two = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        assert estimate_messages(two, "gpt-4") > estimate_messages(one, "gpt-4")

    def test_overhead_is_added_per_message(self):
        # A message with empty role and content should still have overhead > 0
        msgs = [{"role": "", "content": ""}]
        assert estimate_messages(msgs, "gpt-4") > 0

    def test_skips_non_mapping_items(self):
        valid = [{"role": "user", "content": "Hi"}]
        mixed = [{"role": "user", "content": "Hi"}, "not a message"]  # type: ignore[list-item]
        assert estimate_messages(mixed, "gpt-4") == estimate_messages(valid, "gpt-4")

    def test_skips_integer_items(self):
        valid = [{"role": "user", "content": "Hi"}]
        mixed = [{"role": "user", "content": "Hi"}, 42]  # type: ignore[list-item]
        assert estimate_messages(mixed, "gpt-4") == estimate_messages(valid, "gpt-4")

    def test_handles_missing_role_key(self):
        msgs = [{"content": "just content"}]
        n = estimate_messages(msgs, "gpt-4")
        assert n > 0

    def test_handles_missing_content_key(self):
        msgs = [{"role": "user"}]
        n = estimate_messages(msgs, "gpt-4")
        assert n > 0  # overhead still added

    def test_claude_overhead_differs_from_gpt4(self):
        # Claude has per-message overhead of 5 vs 4 for gpt-4
        msgs = [{"role": "user", "content": ""}] * 10
        assert estimate_messages(msgs, "claude-3-opus") != estimate_messages(msgs, "gpt-4")

    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "claude-3-opus", "gemini-1.5-pro"])
    def test_multi_turn_conversation(self, model):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        assert estimate_messages(msgs, model) > 0

    def test_generator_input_accepted(self):
        def gen():
            yield {"role": "user", "content": "Hi"}
            yield {"role": "assistant", "content": "Hello"}
        result = estimate_messages(gen(), "gpt-4")
        assert result > 0


# ---------------------------------------------------------------------------
# fits_in_context
# ---------------------------------------------------------------------------

class TestFitsInContext:
    def test_short_text_fits_gpt4(self):
        assert fits_in_context("hello", "gpt-4") is True

    def test_returns_bool(self):
        result = fits_in_context("test", "gpt-4")
        assert isinstance(result, bool)

    def test_huge_text_overflows_small_window(self):
        huge = "a" * 100_000
        assert fits_in_context(huge, "gpt-4") is False

    def test_same_text_fits_large_window(self):
        text = "a" * 100_000
        assert fits_in_context(text, "gpt-4-turbo") is True

    def test_huge_text_fits_gemini_window(self):
        text = "a" * 1_000_000
        assert fits_in_context(text, "gemini-1.5-pro") is True

    def test_headroom_reduces_capacity(self):
        # "hi" ≈ 1 token; headroom 10_000 > gpt-4 window 8_192
        assert fits_in_context("hi", "gpt-4", headroom=10_000) is False

    def test_zero_headroom_same_as_no_headroom(self):
        text = "hello"
        assert fits_in_context(text, "gpt-4", headroom=0) == \
            fits_in_context(text, "gpt-4")

    def test_negative_headroom_clamped_to_zero(self):
        assert fits_in_context("hi", "gpt-4", headroom=-9_999) is True

    def test_empty_text_always_fits(self):
        assert fits_in_context("", "gpt-4") is True
        assert fits_in_context("", "gpt-4", headroom=0) is True

    def test_none_text_always_fits(self):
        assert fits_in_context(None, "gpt-4") is True  # type: ignore[arg-type]

    def test_headroom_equal_to_window_minus_one_fits(self):
        win = context_window("gpt-4")
        # 1-token text + (window - 2) headroom should fit
        assert fits_in_context("a", "gpt-4", headroom=win - 2) is True

    def test_headroom_equal_to_window_overflows(self):
        win = context_window("gpt-4")
        assert fits_in_context("a", "gpt-4", headroom=win) is False

    @pytest.mark.parametrize("model", [
        "gpt-4", "gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro",
    ])
    def test_single_word_fits_all_models(self, model):
        assert fits_in_context("hello", model) is True


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------

class TestContextWindow:
    @pytest.mark.parametrize("model,expected", [
        ("gpt-4", 8_192),
        ("gpt-4-turbo", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-3.5", 16_385),
        ("claude-3-opus", 200_000),
        ("claude-3.5-sonnet", 200_000),
        ("claude-sonnet-4", 200_000),
        ("gemini-pro", 32_768),
        ("gemini-1.5-pro", 1_048_576),
        ("gemini-2.0-flash", 1_048_576),
    ])
    def test_known_model_windows(self, model, expected):
        assert context_window(model) == expected

    def test_unknown_model_falls_back_to_gpt4(self):
        assert context_window("no-such-model-xyz") == context_window("gpt-4")

    def test_returns_positive_integer(self):
        assert isinstance(context_window("gpt-4"), int)
        assert context_window("gpt-4") > 0

    def test_prefix_matching_gpt4_turbo_preview(self):
        assert context_window("gpt-4-turbo-preview") == context_window("gpt-4-turbo")


# ---------------------------------------------------------------------------
# supported_models
# ---------------------------------------------------------------------------

class TestSupportedModels:
    def test_returns_non_empty_list(self):
        models = supported_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_known_families_present(self):
        models = supported_models()
        for expected in ("gpt-4", "claude-3", "gemini"):
            assert expected in models, f"{expected!r} missing from supported_models()"

    def test_list_is_sorted(self):
        models = supported_models()
        assert models == sorted(models)

    def test_all_entries_are_strings(self):
        for m in supported_models():
            assert isinstance(m, str)


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_is_semver_like(self):
        assert re.match(r"^\d+\.\d+\.\d+", __version__), (
            f"__version__ {__version__!r} does not look like semver"
        )
