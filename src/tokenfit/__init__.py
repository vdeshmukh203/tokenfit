"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.

All estimates round **up** so they never under-report the true count.
Exact counts require the real tokenizer for each model.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Optional

__version__ = "0.1.0"

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate significantly. Values are intentionally conservative
# (lower ratio → more tokens estimated) to stay safe near context limits.
_RATIOS: dict[str, float] = {
    "gpt-3.5": 4.0,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    "claude-3.5-sonnet": 3.5,
    "claude-3.5-haiku": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-opus-4": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3-haiku": 3.5,
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
    "claude-3.5-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
}

# Per-message overhead in tokens (role markers, separators).
# Keys are coarse family buckets; "gpt-4-turbo" maps to "gpt-4" via
# _overhead_family so no separate entry is needed here.
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4": 4,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"


def _family(model: str) -> str:
    """Return the canonical family key for *model*.

    Uses longest-prefix match against the known model keys so that versioned
    variants such as ``claude-3.5-sonnet-20241022`` are handled correctly.
    Falls back to ``gpt-4`` when nothing matches.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Map *model* to a coarse bucket key used by ``_MESSAGE_OVERHEAD``."""
    m = (model or "").lower().strip()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    # gpt-4, gpt-4-turbo, and unknown GPT variants all share overhead=4
    return "gpt-4"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_tokens(text: Optional[str], model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for *model*.

    Uses a per-family character-to-token ratio. Empty or ``None`` input
    returns 0. The result is rounded **up** so the count never under-reports.

    Parameters
    ----------
    text:
        The text to estimate. ``None`` or the empty string returns 0.
    model:
        A model name string (e.g. ``"gpt-4"`` or ``"claude-3-opus"``).
        Unrecognised names fall back to ``gpt-4`` ratios.

    Returns
    -------
    int
        Estimated token count, always ``>= 0``.

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

    Each item should be a ``Mapping`` with at least a ``"content"`` key and
    optionally a ``"role"`` key.  A small per-message overhead is added to
    account for role markers and message separators.  Non-mapping items are
    silently skipped.

    Parameters
    ----------
    messages:
        An iterable of message mappings.  ``None`` or an empty iterable
        returns 0.
    model:
        A model name string.

    Returns
    -------
    int
        Estimated token count for the full conversation.

    Examples
    --------
    >>> msgs = [{"role": "user", "content": "Hi!"}]
    >>> estimate_messages(msgs, model="gpt-4")
    6
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


def fits_in_context(
    text: Optional[str],
    model: str,
    headroom: int = 0,
) -> bool:
    """Return ``True`` if *text* fits within *model*'s context window.

    *headroom* reserves that many tokens for the model's reply, so the
    check is ``estimate_tokens(text) + headroom <= window_size``.
    Negative *headroom* is clamped to 0.

    Parameters
    ----------
    text:
        The text to check.
    model:
        The target model name.
    headroom:
        Tokens to reserve for the model's response.  Clamped to ``>= 0``.

    Returns
    -------
    bool
        ``True`` when the text (plus headroom) fits.

    Examples
    --------
    >>> fits_in_context("Hello!", model="gpt-4")
    True
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def fits_in_context_messages(
    messages: Iterable[Mapping[str, str]],
    model: str,
    headroom: int = 0,
) -> bool:
    """Return ``True`` if *messages* fit within *model*'s context window.

    Convenience wrapper around :func:`estimate_messages` for callers that
    work with chat-style message lists instead of raw text strings.

    Parameters
    ----------
    messages:
        An iterable of message mappings (same format as
        :func:`estimate_messages`).
    model:
        The target model name.
    headroom:
        Tokens to reserve for the model's response.  Clamped to ``>= 0``.

    Returns
    -------
    bool
        ``True`` when the messages (plus headroom) fit.
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_messages(messages, model) + max(0, int(headroom))
    return used <= window


def list_models() -> List[str]:
    """Return a sorted list of model names with built-in support.

    Returns
    -------
    list[str]
        Model identifiers recognised by :func:`estimate_tokens` and
        :func:`fits_in_context`.

    Examples
    --------
    >>> "gpt-4" in list_models()
    True
    """
    return sorted(_WINDOWS)


__all__ = [
    "__version__",
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "fits_in_context_messages",
    "list_models",
]
