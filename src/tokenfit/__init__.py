"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate. These are estimates -- exact counts require the real
# tokenizer for each model.
_RATIOS = {
    "gpt-3.5": 4.0,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    "claude-3.5-sonnet": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3": 3.5,
    "claude": 3.5,
    "gemini-2.0-flash": 4.0,
    "gemini-1.5-pro": 4.0,
    "gemini-pro": 4.0,
    "gemini": 4.0,
}

# Approximate context-window sizes in tokens.
_WINDOWS = {
    "gpt-3.5": 16_385,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "claude-3.5-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
}

# Per-message overhead in tokens (role markers, separators).
_MESSAGE_OVERHEAD = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"


def _family(model: str) -> str:
    """Return the canonical family key for the given model name.

    Uses longest-prefix match against the known model keys, falling back
    to the default family (gpt-4) when nothing matches.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Pick a coarser family bucket for per-message overhead lookups."""
    m = (model or "").lower().strip()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4"):
        return "gpt-4"
    return "gpt-4"


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in text for the given model.

    Uses a per-family character-to-token ratio. Empty or None input
    returns 0. The estimate rounds up so the count never under-reports.
    """
    if not text:
        return 0
    fam = _family(model)
    return math.ceil(len(text) / _RATIOS[fam])


def estimate_messages(messages: Iterable[Mapping[str, str]], model: str = "gpt-4") -> int:
    """Estimate the token count of a chat-style messages list.

    Each message is expected to be a mapping with at least a content key
    (and typically a role key). A small per-message overhead is added to
    account for role markers and message separators. Items that are not
    mappings are skipped.
    """
    overhead_fam = _overhead_family(model)
    per_msg = _MESSAGE_OVERHEAD.get(overhead_fam, 4)
    total = 0
    for msg in messages or ():
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        total += estimate_tokens(role, model)
        total += estimate_tokens(content, model)
        total += per_msg
    return total


def fits_in_context(text: str, model: str, headroom: int = 0) -> bool:
    """Return True if text fits in the model's context window.

    headroom reserves that many tokens for the model's response, so the
    estimate of text plus headroom must be no greater than the window
    size. Negative headroom is clamped to 0.
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


__all__ = ["estimate_tokens", "estimate_messages", "fits_in_context"]
