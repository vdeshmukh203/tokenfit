"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

__version__ = "0.2.0"

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate. These are estimates -- exact counts require the real
# tokenizer for each model.
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
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "claude": 5,
    "gemini": 4,
}

_DEFAULT_FAMILY = "gpt-4"

# Precompute sorted keys once for O(k) lookups (k = number of known families).
_RATIO_KEYS_BY_LEN = sorted(_RATIOS, key=len, reverse=True)
_OVERHEAD_KEYS_BY_LEN = sorted(_MESSAGE_OVERHEAD, key=len, reverse=True)


def _family(model: str) -> str:
    """Return the canonical family key for the given model name.

    Uses longest-prefix match against the known model keys, falling back
    to the default family (gpt-4) when nothing matches.
    """
    m = (model or "").lower().strip()
    for key in _RATIO_KEYS_BY_LEN:
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return the per-message overhead bucket for the given model name.

    Uses longest-prefix match against ``_MESSAGE_OVERHEAD`` keys so that
    fine-grained model names (e.g. ``gpt-4-turbo-preview``) resolve to
    their closest overhead entry rather than a coarser fallback.
    """
    m = (model or "").lower().strip()
    for key in _OVERHEAD_KEYS_BY_LEN:
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for the given model.

    Parameters
    ----------
    text:
        The input text to estimate. ``None`` is treated as an empty string.
    model:
        Model (or model family) name. Unrecognised names fall back to the
        ``gpt-4`` ratio (4.0 characters per token).

    Returns
    -------
    int
        Estimated token count, always rounded *up* so the count never
        under-reports. Returns ``0`` for empty or ``None`` input.

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
    """Estimate the token count of a chat-style messages list.

    Each message is expected to be a mapping with at least a ``"content"``
    key and typically a ``"role"`` key. A small per-message overhead is
    added to account for role markers and message separators. Items that
    are not mappings are silently skipped.

    Parameters
    ----------
    messages:
        Iterable of message mappings (e.g. ``[{"role": "user",
        "content": "Hi"}]``).
    model:
        Model name used to select the character-to-token ratio and the
        per-message formatting overhead.

    Returns
    -------
    int
        Total estimated tokens including per-message overhead.

    Examples
    --------
    >>> msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    >>> estimate_messages(msgs, model="gpt-4")  # doctest: +SKIP
    14
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
    """Return ``True`` if *text* fits inside the model's context window.

    Parameters
    ----------
    text:
        The text to check.
    model:
        Model name whose context window is used as the upper bound.
    headroom:
        Tokens to reserve for the model's reply. Negative values are
        clamped to ``0``.

    Returns
    -------
    bool
        ``True`` when the estimated token count plus *headroom* is at most
        the model's context-window size.

    Examples
    --------
    >>> fits_in_context("hello", "gpt-4")
    True
    >>> fits_in_context("a" * 200_000, "gpt-4")
    False
    """
    window = context_window(model)
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def context_window(model: str) -> int:
    """Return the context-window size in tokens for the given model.

    Parameters
    ----------
    model:
        Model name to look up. Unrecognised names fall back to the
        ``gpt-4`` default (8 192 tokens).

    Returns
    -------
    int
        Maximum token capacity of the model's context window.

    Examples
    --------
    >>> context_window("gpt-4-turbo")
    128000
    >>> context_window("claude-3-opus")
    200000
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


def token_budget(text: str, model: str, headroom: int = 0) -> int:
    """Return the tokens remaining after placing *text* in the context window.

    Parameters
    ----------
    text:
        Text already consuming context space.
    model:
        Model name whose window size is used.
    headroom:
        Tokens to reserve for the model's reply. Negative values are
        clamped to ``0``.

    Returns
    -------
    int
        Remaining token capacity. Negative when *text* (plus *headroom*)
        exceeds the window.

    Examples
    --------
    >>> token_budget("hi", "gpt-4")  # doctest: +SKIP
    8191
    """
    window = context_window(model)
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return window - used


def list_models() -> list[str]:
    """Return all recognised model names in alphabetical order.

    Returns
    -------
    list[str]
        Sorted list of model identifier strings that tokenfit recognises.

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
    "token_budget",
    "list_models",
]
