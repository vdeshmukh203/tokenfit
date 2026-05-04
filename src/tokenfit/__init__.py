"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

__version__ = "0.2.0"

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate significantly. These are estimates — exact counts require
# the real tokenizer for each model.
_RATIOS: Dict[str, float] = {
    # OpenAI — GPT-3.5
    "gpt-3.5": 4.0,
    # OpenAI — GPT-4o
    "gpt-4o-mini": 3.8,
    "gpt-4o": 3.8,
    # OpenAI — GPT-4.1 family (1 M-token context)
    "gpt-4.1-mini": 3.8,
    "gpt-4.1-nano": 3.8,
    "gpt-4.1": 3.8,
    # OpenAI — GPT-4 (legacy)
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    # OpenAI — reasoning models
    "o1-mini": 3.8,
    "o1": 3.8,
    "o3-mini": 3.8,
    "o3": 3.8,
    "o4-mini": 3.8,
    # Anthropic — Claude 3.5 (dot notation, as in Anthropic docs)
    "claude-3.5-haiku": 3.5,
    "claude-3.5-sonnet": 3.5,
    # Anthropic — Claude 3.5 (dash notation, as in API model IDs)
    "claude-3-5-haiku": 3.5,
    "claude-3-5-sonnet": 3.5,
    # Anthropic — Claude 4 family
    "claude-opus-4": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-haiku-4": 3.5,
    # Anthropic — Claude 3
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3-haiku": 3.5,
    "claude-3": 3.5,
    # Anthropic — generic fallback
    "claude": 3.5,
    # Google — Gemini 2.x
    "gemini-2.5-pro": 4.0,
    "gemini-2.5-flash": 4.0,
    "gemini-2.0-flash": 4.0,
    # Google — Gemini 1.5
    "gemini-1.5-pro": 4.0,
    "gemini-1.5-flash": 4.0,
    # Google — legacy / generic
    "gemini-pro": 4.0,
    "gemini": 4.0,
}

# Context-window sizes in tokens.
_WINDOWS: Dict[str, int] = {
    "gpt-3.5": 16_385,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1": 1_047_576,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
}

# Per-message overhead in tokens (role markers and message separators).
_MESSAGE_OVERHEAD: Dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "o1": 3,
    "o3": 3,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"

# Pre-sorted for O(len(keys)) longest-prefix matching — avoids re-sorting on every call.
_SORTED_RATIO_KEYS: List[str] = sorted(_RATIOS, key=len, reverse=True)


def _family(model: str) -> str:
    """Return the canonical family key for *model* using longest-prefix match.

    Falls back to ``_DEFAULT_FAMILY`` (``"gpt-4"``) when nothing matches.
    """
    m = (model or "").lower().strip()
    for key in _SORTED_RATIO_KEYS:
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return the coarse family bucket used for per-message overhead lookups."""
    m = (model or "").lower().strip()
    for prefix, family in (
        ("gpt-4o", "gpt-4o"),       # must precede "gpt-4"
        ("gpt-4-turbo", "gpt-4-turbo"),
        ("gpt-4", "gpt-4"),
        ("gpt-3.5", "gpt-3.5"),
        ("o1", "o1"),
        ("o3", "o3"),
        ("o4", "o3"),               # treat o4 same as o3
        ("claude", "claude"),
        ("gemini", "gemini"),
    ):
        if m.startswith(prefix):
            return family
    return _DEFAULT_FAMILY


def estimate_tokens(text: str | None, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for the given *model*.

    Uses a per-family character-to-token ratio calibrated against English
    prose.  The estimate rounds up so the count never under-reports.
    ``None`` or empty *text* returns ``0``.

    Parameters
    ----------
    text:
        The text to estimate.  ``None`` is treated as an empty string.
    model:
        The LLM model name.  Unknown names fall back to ``gpt-4`` ratios.
        Call :func:`list_models` to see all explicitly supported names.

    Returns
    -------
    int
        Estimated token count (>= 0).
    """
    if not text:
        return 0
    fam = _family(model)
    return math.ceil(len(text) / _RATIOS[fam])


def estimate_messages(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
) -> int:
    """Estimate the token count of a chat-style messages list.

    Each message must be a :class:`~collections.abc.Mapping` with at least
    a ``"content"`` key and optionally a ``"role"`` key.  A small
    per-message overhead is added to account for role markers and
    separators.  Non-mapping items are silently skipped.

    Parameters
    ----------
    messages:
        Iterable of message dicts, e.g.
        ``[{"role": "user", "content": "Hello"}]``.
    model:
        LLM model name.

    Returns
    -------
    int
        Estimated total token count for the conversation.
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


def fits_in_context(text: str | None, model: str, headroom: int = 0) -> bool:
    """Return ``True`` if *text* fits within the model's context window.

    *headroom* reserves that many tokens for the model's response, so the
    sum of the estimated token count and *headroom* must not exceed the
    window size.  Negative headroom is clamped to zero.

    Parameters
    ----------
    text:
        Text to evaluate.  ``None`` is treated as an empty string.
    model:
        LLM model name.
    headroom:
        Tokens to reserve for the model response (default ``0``).

    Returns
    -------
    bool
        ``True`` when the text (plus headroom) fits in the context window.
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def get_context_window(model: str) -> int:
    """Return the context window size in tokens for *model*.

    Unknown model names fall back to the ``gpt-4`` window (8 192 tokens).

    Parameters
    ----------
    model:
        LLM model name.

    Returns
    -------
    int
        Context window size in tokens.
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


def list_models() -> List[str]:
    """Return an alphabetically sorted list of explicitly supported model names.

    Each name can be passed to any tokenfit function as the *model* argument
    for the best accuracy.  Other strings are also accepted and fall back to
    longest-prefix matching or ``gpt-4`` defaults.

    Returns
    -------
    list[str]
        Sorted list of model name strings.
    """
    return sorted(_RATIOS)


__all__ = [
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "get_context_window",
    "list_models",
]
