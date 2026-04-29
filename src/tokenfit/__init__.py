"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.

The estimation method is intentionally conservative: counts always round up
so that callers never under-estimate cost or overflow a context window.
Exact token counts require the real tokenizer for each model family.

References
----------
Ratios were calibrated against representative English prose using the
publicly documented byte-pair encoding families employed by OpenAI (GPT),
Anthropic (Claude), and Google (Gemini) models.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable, List, Mapping

__version__ = "0.2.0"
__all__ = [
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "list_models",
    "context_window",
]

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
# Coarser groupings: gpt-4-turbo is handled separately from gpt-4 so that
# the "gpt-4-turbo" key is reachable via _overhead_family().
_MESSAGE_OVERHEAD: dict[str, int] = {
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
    to the default family (gpt-4) when nothing matches. Emits a
    ``UserWarning`` for unrecognised model names so callers are aware of
    the fallback.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    warnings.warn(
        f"Unknown model {model!r}; falling back to {_DEFAULT_FAMILY!r}. "
        "Call list_models() to see supported names.",
        UserWarning,
        stacklevel=3,
    )
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Pick the coarse family bucket used for per-message overhead lookups.

    The order of checks matters: more-specific prefixes (gpt-4-turbo,
    gpt-4o) must come before the generic gpt-4 prefix so that they are
    not swallowed by the shorter match.
    """
    m = (model or "").lower().strip()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4-turbo"):   # must come before plain "gpt-4"
        return "gpt-4-turbo"
    if m.startswith("gpt-4"):
        return "gpt-4"
    return "gpt-4"


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for the given *model*.

    Uses a per-family character-to-token ratio calibrated against English
    prose. The estimate rounds up so the count never under-reports.

    Parameters
    ----------
    text:
        The input text whose token count is to be estimated.
    model:
        The target LLM model name (e.g. ``"gpt-4"``, ``"claude-3-opus"``).
        Unknown names fall back to ``"gpt-4"`` with a ``UserWarning``.

    Returns
    -------
    int
        Estimated token count, or ``0`` for empty or ``None`` input.
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

    Each message should be a mapping with at least a ``"content"`` key and
    typically a ``"role"`` key.  A small per-message overhead is added to
    account for role markers and message separators used by the respective
    model family.  Non-mapping items in the iterable are silently skipped.

    Parameters
    ----------
    messages:
        An iterable of chat messages, each a mapping with ``"role"`` and
        ``"content"`` string fields.
    model:
        The target LLM model name.  Unknown names fall back to ``"gpt-4"``
        with a ``UserWarning``.

    Returns
    -------
    int
        Total estimated token count across all messages, including the
        per-message overhead for role markers and separators.
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

    *headroom* reserves that many tokens for the model's response; the
    estimate of text tokens plus headroom must not exceed the window size.
    Negative headroom is clamped to zero.

    Parameters
    ----------
    text:
        The input text to check.
    model:
        The target LLM model name.  Unknown names fall back to ``"gpt-4"``
        with a ``UserWarning``.
    headroom:
        Tokens to reserve for the model's response (default ``0``).

    Returns
    -------
    bool
        ``True`` if the text (plus headroom) fits within the context window.
    """
    fam = _family(model)
    window = _WINDOWS[fam]
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def list_models() -> List[str]:
    """Return the sorted list of model-family prefix strings known to tokenfit.

    Any model name that *starts with* one of the returned strings will be
    matched to the corresponding family by the estimation functions.

    Returns
    -------
    list of str
        Alphabetically sorted list of supported model-family keys.

    Examples
    --------
    >>> "gpt-4" in list_models()
    True
    """
    return sorted(_RATIOS)


def context_window(model: str) -> int:
    """Return the approximate context-window size in tokens for *model*.

    Parameters
    ----------
    model:
        The target LLM model name.  Unknown names fall back to ``"gpt-4"``
        with a ``UserWarning``.

    Returns
    -------
    int
        Approximate context-window size in tokens.

    Examples
    --------
    >>> context_window("gpt-4o")
    128000
    """
    fam = _family(model)
    return _WINDOWS[fam]
