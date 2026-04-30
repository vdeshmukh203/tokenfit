"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.

Public API
----------
estimate_tokens(text, model)
    Estimate token count for a plain string.
estimate_messages(messages, model)
    Estimate token count for a chat-style message list.
fits_in_context(text, model, headroom)
    Check whether text fits in a model's context window.
context_window(model)
    Return the context-window size in tokens for a model.
list_models()
    Return a sorted list of all recognised model name prefixes.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Model tables
# ---------------------------------------------------------------------------

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate. These are estimates — exact counts require the real
# tokeniser for each model.
_RATIOS: dict[str, float] = {
    # OpenAI GPT series
    "gpt-3.5": 4.0,
    "gpt-4o-mini": 3.8,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    # OpenAI reasoning models (o-series use the same BPE tokeniser as GPT-4o)
    "o4-mini": 3.8,
    "o3-mini": 3.8,
    "o3": 3.8,
    "o1-mini": 3.8,
    "o1": 3.8,
    # Anthropic Claude 4 series
    "claude-opus-4": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-haiku-4": 3.5,
    # Anthropic Claude 3.x series
    "claude-3.5-sonnet": 3.5,
    "claude-3.5-haiku": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3-haiku": 3.5,
    # Generic Claude fallback
    "claude-3": 3.5,
    "claude": 3.5,
    # Google Gemini series
    "gemini-2.5-pro": 4.0,
    "gemini-2.5-flash": 4.0,
    "gemini-2.0-flash": 4.0,
    "gemini-1.5-pro": 4.0,
    "gemini-1.5-flash": 4.0,
    "gemini-pro": 4.0,
    "gemini": 4.0,
    # Meta Llama series
    "llama-3": 3.7,
    "llama-2": 3.7,
    "llama": 3.7,
    # Mistral series
    "mistral-large": 3.8,
    "mistral": 3.8,
    # Cohere Command series
    "command-r-plus": 3.8,
    "command-r": 3.8,
    "command": 3.8,
}

# Approximate context-window sizes in tokens.
_WINDOWS: dict[str, int] = {
    # OpenAI GPT series
    "gpt-3.5": 16_385,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    # OpenAI reasoning models
    "o4-mini": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o1-mini": 128_000,
    "o1": 200_000,
    # Anthropic Claude 4 series
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    # Anthropic Claude 3.x series
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    # Google Gemini series
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
    # Meta Llama series
    "llama-3": 128_000,
    "llama-2": 4_096,
    "llama": 128_000,
    # Mistral series
    "mistral-large": 131_072,
    "mistral": 32_768,
    # Cohere Command series
    "command-r-plus": 128_000,
    "command-r": 128_000,
    "command": 4_096,
}

# Per-message overhead in tokens (role markers, separators, special tokens).
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "o1": 3,
    "o3": 3,
    "claude": 5,
    "gemini": 4,
    "llama": 4,
    "mistral": 3,
    "command": 4,
}

_DEFAULT_FAMILY = "gpt-4"

# Pre-sort once at module load for O(n) worst-case family lookup.
_SORTED_RATIO_KEYS: list[str] = sorted(_RATIOS, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _family(model: str) -> str:
    """Return the canonical family key for *model*.

    Applies a longest-prefix match against the known model keys, falling back
    to ``_DEFAULT_FAMILY`` (``"gpt-4"``) when nothing matches.

    Parameters
    ----------
    model:
        Raw model name as supplied by the caller (case-insensitive).

    Returns
    -------
    str
        A key present in ``_RATIOS`` and ``_WINDOWS``.
    """
    m = (model or "").lower().strip()
    for key in _SORTED_RATIO_KEYS:
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return a coarser family key for per-message overhead lookups.

    Parameters
    ----------
    model:
        Raw model name (case-insensitive).

    Returns
    -------
    str
        A key present in ``_MESSAGE_OVERHEAD``.
    """
    m = (model or "").lower().strip()
    if m.startswith("claude"):
        return "claude"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("llama",)):
        return "llama"
    if m.startswith("mistral"):
        return "mistral"
    if m.startswith("command"):
        return "command"
    if m.startswith(("o1", "o3", "o4")):
        return "o1"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    if m.startswith("gpt-4"):
        return "gpt-4"
    return "gpt-4"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for *model*.

    Uses a per-family character-to-token ratio derived from English prose.
    Non-English text and source code may produce larger deviations from
    exact tokeniser output. The estimate always rounds up so the reported
    count never under-reports.

    Parameters
    ----------
    text:
        The string to estimate. ``None`` is treated as an empty string.
    model:
        Model name or prefix (case-insensitive). Unknown models fall back
        to the ``gpt-4`` family ratio.

    Returns
    -------
    int
        Estimated token count (>= 0).

    Examples
    --------
    >>> estimate_tokens("Hello, world!", model="gpt-4")
    4
    >>> estimate_tokens("", model="claude-3-opus")
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
    """Estimate the token count of a chat-style *messages* list.

    Each message is expected to be a mapping with ``"role"`` and
    ``"content"`` keys, matching the OpenAI / Anthropic chat format.
    A per-message overhead is added to account for role markers and
    message separators. Items that are not mappings are silently skipped.

    Parameters
    ----------
    messages:
        An iterable of message mappings. May be ``None`` or empty.
    model:
        Model name or prefix (case-insensitive).

    Returns
    -------
    int
        Estimated total token count (>= 0).

    Examples
    --------
    >>> msgs = [{"role": "user", "content": "Hello!"}]
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
    text: str,
    model: str,
    headroom: int = 0,
) -> bool:
    """Return ``True`` if *text* fits within *model*'s context window.

    *headroom* reserves that many tokens for the model's reply, so the
    sum of the token estimate and headroom must not exceed the window
    size. Negative headroom is clamped to zero.

    Parameters
    ----------
    text:
        The string to check.
    model:
        Model name or prefix (case-insensitive).
    headroom:
        Tokens to reserve for the model's output. Negative values are
        treated as zero.

    Returns
    -------
    bool
        ``True`` if ``estimate_tokens(text, model) + max(0, headroom)``
        does not exceed the model's context window.

    Examples
    --------
    >>> fits_in_context("hi", "gpt-4")
    True
    >>> fits_in_context("hi", "gpt-4", headroom=10_000)
    False
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def context_window(model: str) -> int:
    """Return the context-window size in tokens for *model*.

    Parameters
    ----------
    model:
        Model name or prefix (case-insensitive). Unknown models fall back
        to the ``gpt-4`` window (8 192 tokens).

    Returns
    -------
    int
        Number of tokens in the model's context window.

    Examples
    --------
    >>> context_window("gpt-4")
    8192
    >>> context_window("claude-3-opus")
    200000
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


def list_models() -> List[str]:
    """Return a sorted list of recognised model name prefixes.

    The returned prefixes can be passed directly to :func:`estimate_tokens`,
    :func:`estimate_messages`, :func:`fits_in_context`, and
    :func:`context_window`.

    Returns
    -------
    list[str]
        Alphabetically sorted model-family keys.

    Examples
    --------
    >>> "gpt-4" in list_models()
    True
    """
    return sorted(_RATIOS)


__all__ = [
    "estimate_tokens",
    "estimate_messages",
    "fits_in_context",
    "context_window",
    "list_models",
    "__version__",
]
