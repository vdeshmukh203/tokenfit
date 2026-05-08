"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios
calibrated against English prose.  Useful for budget checks before sending
requests to an LLM API.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping

__version__ = "0.2.0"

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate substantially.  Exact counts require the real tokenizer.
_RATIOS: dict[str, float] = {
    # OpenAI GPT / reasoning models
    "gpt-3.5": 4.0,
    "gpt-4o-mini": 3.8,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    "o1-mini": 3.8,
    "o1": 3.8,
    "o3-mini": 3.8,
    "o3": 3.8,
    # Anthropic Claude
    "claude-3.5-sonnet": 3.5,
    "claude-3.5-haiku": 3.5,
    "claude-3-haiku": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-opus-4": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3": 3.5,
    "claude": 3.5,
    # Google Gemini
    "gemini-2.5-pro": 4.0,
    "gemini-2.0-flash": 4.0,
    "gemini-1.5-flash": 4.0,
    "gemini-1.5-pro": 4.0,
    "gemini-pro": 4.0,
    "gemini": 4.0,
}

# Context-window sizes in tokens.
_WINDOWS: dict[str, int] = {
    "gpt-3.5": 16_385,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1-mini": 128_000,
    "o1": 128_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
}

# Per-message overhead (role markers, separators, special tokens).
# Keyed by the coarsest prefix that unambiguously identifies the family.
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o-mini": 3,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "o1-mini": 4,
    "o1": 4,
    "o3-mini": 4,
    "o3": 4,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"


def _family(model: str) -> str:
    """Return the canonical key for *model* using longest-prefix matching.

    Falls back to ``gpt-4`` when nothing in ``_RATIOS`` matches.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return the coarser overhead-bucket key using longest-prefix matching."""
    m = (model or "").lower().strip()
    for key in sorted(_MESSAGE_OVERHEAD, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for *model*.

    Uses a per-family character-to-token ratio calibrated on English prose.
    Empty or ``None`` input returns 0.  The estimate rounds **up** so the
    count never under-reports.

    Parameters
    ----------
    text:
        The text to estimate.  ``None`` is treated as an empty string.
    model:
        Target model name (e.g. ``"gpt-4"``, ``"claude-3-opus"``).
        Unrecognised names fall back to the ``gpt-4`` ratio (4.0 chars/token).

    Returns
    -------
    int
        Non-negative integer token estimate.

    Examples
    --------
    >>> estimate_tokens("Hello, world!", model="gpt-4")
    4
    >>> estimate_tokens("", model="gpt-4")
    0
    """
    if not text:
        return 0
    fam = _family(model)
    return math.ceil(len(text) / _RATIOS[fam])


def estimate_messages(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
) -> int:
    """Estimate the token count of a chat-style message list.

    Each message should be a mapping with at least a ``content`` key (and
    typically a ``role`` key).  A small per-message overhead is added to
    account for role markers and separator tokens.  Items that are not
    mappings are silently skipped.

    Parameters
    ----------
    messages:
        An iterable of message dicts, e.g.
        ``[{"role": "user", "content": "Hi"}]``.  ``None`` is treated as
        an empty sequence.
    model:
        Target model name.

    Returns
    -------
    int
        Non-negative integer token estimate.

    Examples
    --------
    >>> estimate_messages([{"role": "user", "content": "Hi"}], model="gpt-4")
    6
    """
    overhead_key = _overhead_family(model)
    per_msg = _MESSAGE_OVERHEAD.get(overhead_key, 4)
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
    """Return ``True`` if *text* fits within *model*'s context window.

    *headroom* reserves that many tokens for the model's response, so the
    sum of the token estimate and headroom must be no greater than the
    context-window size.  Negative headroom is clamped to zero.

    Parameters
    ----------
    text:
        The text to check.
    model:
        Target model name.
    headroom:
        Tokens to reserve for the model's own output.  Negative values are
        treated as zero.

    Returns
    -------
    bool

    Examples
    --------
    >>> fits_in_context("hello", "gpt-4")
    True
    >>> fits_in_context("hello", "gpt-4", headroom=8_192)
    False
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def context_window(model: str) -> int:
    """Return the context-window size in tokens for *model*.

    Unrecognised model names fall back to the ``gpt-4`` window (8 192 tokens).

    Parameters
    ----------
    model:
        Target model name.

    Returns
    -------
    int
        Context-window size in tokens.

    Examples
    --------
    >>> context_window("gpt-4")
    8192
    >>> context_window("claude-3-opus")
    200000
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


def chars_per_token(model: str) -> float:
    """Return the characters-per-token ratio used internally for *model*.

    A higher value means each token covers more characters (i.e. fewer
    tokens are needed for the same amount of text).

    Parameters
    ----------
    model:
        Target model name.

    Returns
    -------
    float
        Characters per token.

    Examples
    --------
    >>> chars_per_token("gpt-4")
    4.0
    >>> chars_per_token("claude-3-opus")
    3.5
    """
    return _RATIOS[_family(model)]


def list_models() -> List[str]:
    """Return a sorted list of all explicitly recognised model names.

    Returns
    -------
    list[str]
        Alphabetically sorted list of model identifier strings.

    Examples
    --------
    >>> "gpt-4" in list_models()
    True
    """
    return sorted(_RATIOS)


__all__ = [
    "__version__",
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "context_window",
    "chars_per_token",
    "list_models",
]
