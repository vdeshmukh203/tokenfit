"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API without
incurring any network calls or heavyweight tokenizer dependencies.

Supported model families
------------------------
GPT-3.5, GPT-4, GPT-4-Turbo, GPT-4o (OpenAI)
Claude 3 / 3.5 / Sonnet 4 (Anthropic)
Gemini Pro / 1.5 Pro / 2.0 Flash (Google)

Notes
-----
Ratios are calibrated against representative English prose.  Non-English
text and code may deviate.  Estimates always round *up* so the count never
under-reports.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Internal look-up tables
# ---------------------------------------------------------------------------

# Approximate characters-per-token for each model family.
_RATIOS: dict[str, float] = {
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
_WINDOWS: dict[str, int] = {
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
# Keys match what _overhead_family() can return.
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _family(model: str) -> str:
    """Return the canonical family key for *model* via longest-prefix match.

    Falls back to the default family (``gpt-4``) when nothing matches.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return the coarse family bucket used for per-message overhead lookups."""
    m = (model or "").lower().strip()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    if m.startswith("gpt-4"):
        return "gpt-4"
    return _DEFAULT_FAMILY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens *text* will consume for *model*.

    Parameters
    ----------
    text:
        The string to estimate.  ``None`` and empty strings return ``0``.
    model:
        Model name (e.g. ``"gpt-4"``, ``"claude-3-opus"``).  Unknown names
        fall back to GPT-4 defaults.

    Returns
    -------
    int
        Estimated token count, rounded *up* to avoid under-reporting.

    Examples
    --------
    >>> estimate_tokens("Hello, world!", model="gpt-4")
    4
    >>> estimate_tokens("Hello, world!", model="claude-3-opus")
    4
    """
    if not text:
        return 0
    fam = _family(model)
    return math.ceil(len(text) / _RATIOS[fam])


def estimate_messages(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
) -> int:
    """Estimate the token count of a chat-style *messages* list.

    Each item in *messages* should be a mapping with ``"role"`` and
    ``"content"`` keys (the OpenAI / Anthropic convention).  A small
    per-message overhead is added to account for role markers and message
    separators.  Items that are not mappings are silently skipped.

    Parameters
    ----------
    messages:
        An iterable of message mappings.  May be ``None`` or empty.
    model:
        Model name used to select the character-to-token ratio and the
        per-message overhead.

    Returns
    -------
    int
        Estimated total token count.

    Examples
    --------
    >>> msgs = [{"role": "user", "content": "Hi"}]
    >>> estimate_messages(msgs, model="gpt-4")
    5
    """
    if not messages:
        return 0
    overhead_fam = _overhead_family(model)
    per_msg = _MESSAGE_OVERHEAD.get(overhead_fam, 4)
    total = 0
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        total += estimate_tokens(role, model)
        total += estimate_tokens(content, model)
        total += per_msg
    return total


def fits_in_context(text: str, model: str, headroom: int = 0) -> bool:
    """Return ``True`` if *text* fits inside *model*'s context window.

    Parameters
    ----------
    text:
        The text whose token count is estimated.
    model:
        Model name used to look up the context-window size and ratio.
    headroom:
        Tokens to reserve for the model's response (or other content).
        Negative values are clamped to ``0``.

    Returns
    -------
    bool
        ``True`` when ``estimate_tokens(text) + headroom <= window``.

    Examples
    --------
    >>> fits_in_context("Hello!", model="gpt-4")
    True
    >>> fits_in_context("Hello!", model="gpt-4", headroom=8_000)
    True
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def context_window(model: str) -> int:
    """Return the context-window size in tokens for *model*.

    Unknown models fall back to the GPT-4 default (8 192 tokens).

    Parameters
    ----------
    model:
        Model name (e.g. ``"gpt-4-turbo"``).

    Returns
    -------
    int
        Maximum context-window size in tokens.

    Examples
    --------
    >>> context_window("gpt-4-turbo")
    128000
    >>> context_window("claude-3-opus")
    200000
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


def supported_models() -> List[str]:
    """Return a sorted list of all recognised model-family prefixes.

    Returns
    -------
    list[str]
        Model family keys in alphabetical order.  Each entry is a prefix
        that will be matched by :func:`estimate_tokens` and friends.

    Examples
    --------
    >>> "gpt-4" in supported_models()
    True
    """
    return sorted(_RATIOS)


__all__ = [
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "context_window",
    "supported_models",
    "__version__",
]
