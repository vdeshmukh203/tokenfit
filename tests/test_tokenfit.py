"""Tests for tokenfit."""
from __future__ import annotations

import pytest

import tokenfit
from tokenfit import (
    TokenEstimate,
    __version__,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    list_models,
    remaining_tokens,
    token_summary,
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
    # Claude ratio 3.5 vs GPT-4 ratio 4.0 → more tokens per char
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


@pytest.mark.parametrize("model,expected_gt", [
    ("gpt-4o", 0),
    ("gpt-4-turbo", 0),
    ("gpt-4.1", 0),
    ("claude-3.5-sonnet", 0),
    ("claude-3.5-haiku", 0),
    ("claude-sonnet-4", 0),
    ("claude-opus-4", 0),
    ("gemini-1.5-pro", 0),
    ("gemini-2.0-flash", 0),
    ("llama-3.1", 0),
    ("mistral-large", 0),
    ("o1", 0),
    ("o3", 0),
])
def test_estimate_tokens_known_models_return_positive(model: str, expected_gt: int):
    assert estimate_tokens("hello world", model) > expected_gt


def test_estimate_tokens_gpt4o_uses_different_ratio():
    # gpt-4o ratio 3.8, gpt-4 ratio 4.0 → slightly more tokens for gpt-4o
    text = "a" * 100
    assert estimate_tokens(text, "gpt-4o") >= estimate_tokens(text, "gpt-4")


# ---------------------------------------------------------------------------
# estimate_messages
# ---------------------------------------------------------------------------

def test_estimate_messages_empty():
    assert estimate_messages([], "gpt-4") == 0


def test_estimate_messages_none():
    assert estimate_messages(None, "gpt-4") == 0  # type: ignore[arg-type]


def test_estimate_messages_basic():
    msgs = [{"role": "user", "content": "Hello world"}]
    n = estimate_messages(msgs, "gpt-4")
    assert 4 < n < 30


def test_estimate_messages_exact_value():
    # "user" = 4 chars → 1 token; "Hello" = 5 chars → 2 tokens; overhead = 4
    msgs = [{"role": "user", "content": "Hello"}]
    assert estimate_messages(msgs, "gpt-4") == 7


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


def test_estimate_messages_claude_higher_overhead():
    # Claude overhead = 5, gpt-4 overhead = 4
    msgs = [{"role": "user", "content": "Hi"}]
    assert estimate_messages(msgs, "claude-3-opus") >= estimate_messages(msgs, "gpt-4")


def test_estimate_messages_generator_input():
    def gen():
        yield {"role": "user", "content": "Hello"}
    n = estimate_messages(gen(), "gpt-4")
    assert n > 0


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


# ---------------------------------------------------------------------------
# remaining_tokens
# ---------------------------------------------------------------------------

def test_remaining_tokens_basic():
    # "hi" = 2 chars, gpt-4 ratio 4.0 → 1 token; window = 8192
    assert remaining_tokens("hi", "gpt-4") == 8191


def test_remaining_tokens_empty_text():
    from tokenfit import _WINDOWS
    assert remaining_tokens("", "gpt-4") == _WINDOWS["gpt-4"]


def test_remaining_tokens_negative_on_overflow():
    huge = "a" * 100_000
    assert remaining_tokens(huge, "gpt-4") < 0


def test_remaining_tokens_headroom_reduces_result():
    base = remaining_tokens("hi", "gpt-4")
    with_headroom = remaining_tokens("hi", "gpt-4", headroom=100)
    assert with_headroom == base - 100


def test_remaining_tokens_negative_headroom_clamps():
    base = remaining_tokens("hi", "gpt-4")
    with_neg = remaining_tokens("hi", "gpt-4", headroom=-999)
    assert with_neg == base


def test_remaining_tokens_consistent_with_fits():
    text = "hello world " * 50
    model = "gpt-4"
    rem = remaining_tokens(text, model)
    fits = fits_in_context(text, model)
    assert (rem >= 0) == fits


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_returns_list():
    assert isinstance(list_models(), list)


def test_list_models_non_empty():
    assert len(list_models()) > 0


def test_list_models_contains_core_families():
    models = list_models()
    for expected in ("gpt-4", "claude", "gemini", "llama", "mistral"):
        assert expected in models, f"Expected {expected!r} in list_models()"


def test_list_models_is_sorted():
    models = list_models()
    assert models == sorted(models)


def test_list_models_all_usable():
    for model in list_models():
        result = estimate_tokens("hello", model)
        assert isinstance(result, int) and result >= 0


# ---------------------------------------------------------------------------
# token_summary
# ---------------------------------------------------------------------------

def test_token_summary_returns_token_estimate():
    result = token_summary("hello", "gpt-4")
    assert isinstance(result, TokenEstimate)


def test_token_summary_fields_consistent():
    text = "hello world"
    model = "gpt-4"
    est = token_summary(text, model)
    assert est.tokens == estimate_tokens(text, model)
    assert est.fits == fits_in_context(text, model)
    assert est.remaining == remaining_tokens(text, model)


def test_token_summary_model_family_resolved():
    est = token_summary("hi", "gpt-4o-mini")
    assert est.model_family == "gpt-4o-mini"


def test_token_summary_headroom_applied():
    est_no_h = token_summary("hi", "gpt-4")
    est_h = token_summary("hi", "gpt-4", headroom=500)
    assert est_h.remaining == est_no_h.remaining - 500


def test_token_summary_overflow_case():
    est = token_summary("x" * 500_000, "gpt-4")
    assert not est.fits
    assert est.remaining < 0


def test_token_summary_large_window():
    est = token_summary("hello", "gemini-1.5-pro")
    assert est.window == 2_097_152
    assert est.fits is True


# ---------------------------------------------------------------------------
# __version__
# ---------------------------------------------------------------------------

def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------

def test_all_exports_importable():
    for name in tokenfit.__all__:
        assert hasattr(tokenfit, name), f"tokenfit.__all__ lists {name!r} but it is not importable"
