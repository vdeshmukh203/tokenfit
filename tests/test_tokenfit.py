"""Tests for tokenfit."""
from __future__ import annotations

import json
import warnings

import pytest

import tokenfit
from tokenfit import (
    TokenEstimate,
    __version__,
    context_window_size,
    estimate_messages,
    estimate_messages_detailed,
    estimate_tokens,
    estimate_tokens_detailed,
    fits_in_context,
)
from tokenfit.cli import main as cli_main


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
    # 5 chars / 4.0 chars-per-token = 1.25 → ceil = 2
    assert estimate_tokens("hello", "gpt-4") == 2


def test_estimate_tokens_unknown_model_warns_and_falls_back():
    with pytest.warns(UserWarning, match="not recognised"):
        result = estimate_tokens("hello world", "totally-fake-model")
    assert result == estimate_tokens("hello world", "gpt-4")


def test_estimate_tokens_claude_higher_count_than_gpt4():
    # Claude ratio 3.5 < GPT-4 ratio 4.0 → more tokens for same text
    text = "a" * 70
    assert estimate_tokens(text, "claude-3-opus") > estimate_tokens(text, "gpt-4")


def test_estimate_tokens_gpt4o_mini():
    # gpt-4o-mini must NOT fall back to gpt-4 (8k window); it has 128k
    est = estimate_tokens_detailed("hello", "gpt-4o-mini")
    assert est.window_size == 128_000


def test_estimate_tokens_o1_window():
    est = estimate_tokens_detailed("test", "o1")
    assert est.window_size == 200_000


def test_estimate_tokens_o3_window():
    est = estimate_tokens_detailed("test", "o3")
    assert est.window_size == 200_000


def test_estimate_tokens_versioned_model_name():
    # Versioned names like claude-3.5-sonnet-20241022 should resolve via prefix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokens = estimate_tokens("hello", "claude-3.5-sonnet-20241022")
    assert tokens == estimate_tokens("hello", "claude-3.5-sonnet")


# ============================================================
# TokenEstimate dataclass
# ============================================================


def test_token_estimate_fields():
    est = estimate_tokens_detailed("hello world", "gpt-4")
    assert isinstance(est, TokenEstimate)
    assert est.tokens > 0
    assert est.model == "gpt-4"
    assert est.family == "gpt-4"
    assert est.window_size == 8_192


def test_token_estimate_utilization():
    est = estimate_tokens_detailed("a" * 8_192, "gpt-4")
    assert 0.0 < est.utilization <= 1.0


def test_token_estimate_fits_no_headroom():
    est = TokenEstimate(tokens=100, model="gpt-4", family="gpt-4", window_size=8_192)
    assert est.fits() is True


def test_token_estimate_fits_with_headroom():
    est = TokenEstimate(tokens=100, model="gpt-4", family="gpt-4", window_size=8_192)
    assert est.fits(headroom=8_092) is True
    assert est.fits(headroom=8_093) is False


def test_token_estimate_remaining():
    est = TokenEstimate(tokens=100, model="gpt-4", family="gpt-4", window_size=8_192)
    assert est.remaining() == 8_092
    assert est.remaining(headroom=500) == 7_592
    assert est.remaining(headroom=9_000) == 0  # clamped


def test_token_estimate_is_frozen():
    est = estimate_tokens_detailed("hello", "gpt-4")
    with pytest.raises((AttributeError, TypeError)):
        est.tokens = 0  # type: ignore[misc]


# ============================================================
# estimate_tokens_detailed
# ============================================================


def test_estimate_tokens_detailed_empty():
    est = estimate_tokens_detailed("", "gpt-4")
    assert est.tokens == 0
    assert est.window_size == 8_192


def test_estimate_tokens_detailed_claude():
    est = estimate_tokens_detailed("hello", "claude-3.5-sonnet")
    assert est.family == "claude-3.5-sonnet"
    assert est.window_size == 200_000


# ============================================================
# estimate_messages
# ============================================================


def test_estimate_messages_empty():
    assert estimate_messages([], "gpt-4") == 0


def test_estimate_messages_none():
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


def test_estimate_messages_claude_overhead():
    # Claude adds 5 tokens per message vs GPT-4's 4
    msg = [{"role": "user", "content": "Hi"}]
    assert estimate_messages(msg, "claude-3-opus") >= estimate_messages(msg, "gpt-4")


# ============================================================
# estimate_messages_detailed
# ============================================================


def test_estimate_messages_detailed_returns_token_estimate():
    msgs = [{"role": "user", "content": "hello"}]
    est = estimate_messages_detailed(msgs, "gpt-4")
    assert isinstance(est, TokenEstimate)
    assert est.tokens > 0
    assert est.window_size == 8_192


def test_estimate_messages_detailed_claude_window():
    msgs = [{"role": "user", "content": "hello"}]
    est = estimate_messages_detailed(msgs, "claude-3.5-sonnet")
    assert est.window_size == 200_000


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


# ============================================================
# context_window_size
# ============================================================


def test_context_window_size_gpt4():
    assert context_window_size("gpt-4") == 8_192


def test_context_window_size_gpt4o():
    assert context_window_size("gpt-4o") == 128_000


def test_context_window_size_claude():
    assert context_window_size("claude-3.5-sonnet") == 200_000


def test_context_window_size_gemini_flash():
    assert context_window_size("gemini-2.0-flash") == 1_048_576


def test_context_window_size_llama3():
    assert context_window_size("llama-3") == 128_000


def test_context_window_size_unknown_warns_falls_back():
    with pytest.warns(UserWarning, match="not recognised"):
        size = context_window_size("brand-new-model-xyz")
    assert size == 8_192


# ============================================================
# __version__
# ============================================================


def test_version_string_exists():
    assert isinstance(__version__, str)
    assert __version__


# ============================================================
# CLI
# ============================================================


def test_cli_estimate_plain(capsys):
    rc = cli_main(["estimate", "Hello, world!", "--model", "gpt-4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "token" in out.lower()


def test_cli_estimate_json(capsys):
    rc = cli_main(["estimate", "Hello", "--model", "gpt-4", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "tokens" in data
    assert data["model"] == "gpt-4"


def test_cli_fits_yes(capsys):
    rc = cli_main(["fits", "hi", "--model", "gpt-4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "YES" in out


def test_cli_fits_no(capsys):
    rc = cli_main(["fits", "a" * 100_000, "--model", "gpt-4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NO" in out


def test_cli_fits_json(capsys):
    rc = cli_main(["fits", "hello", "--model", "gpt-4", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "fits" in data
    assert data["fits"] is True
    assert rc == 0


def test_cli_info_plain(capsys):
    rc = cli_main(["info", "gpt-4o"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "128" in out  # 128,000


def test_cli_info_json(capsys):
    rc = cli_main(["info", "claude-3.5-sonnet", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["window_size"] == 200_000


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
