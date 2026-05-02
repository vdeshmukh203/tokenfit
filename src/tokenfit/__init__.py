"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or
chat-message list will consume, based on per-family character-to-token
ratios calibrated against representative English prose.  Useful for
budget checks before sending requests to an LLM API.

Supported model families
------------------------
- OpenAI:    gpt-3.5, gpt-4, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4.1,
             o1, o3
- Anthropic: claude-3 (haiku / sonnet / opus), claude-3.5-sonnet,
             claude-3.5-haiku, claude-sonnet-4, claude-opus-4
- Google:    gemini-pro, gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
- Meta:      llama-3, llama-3.1, llama-3.2, llama-3.3
- Mistral:   mistral-7b, mistral-nemo, mistral-large

All estimates round **up** so they never under-report.  Exact counts
require the real tokenizer for each model (e.g. ``tiktoken`` for OpenAI).

Notes on accuracy
-----------------
Ratios are calibrated on English prose.  Code, markup, and non-Latin
scripts may deviate significantly; treat results as upper-bound estimates
rather than ground truth.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, NamedTuple

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

# Approximate characters-per-token for each model family.  Keys are used
# as longest-prefix match targets (see _family()).
_RATIOS: dict[str, float] = {
    # OpenAI
    "gpt-3.5": 4.0,
    "gpt-4o-mini": 3.8,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4.1": 3.9,
    "gpt-4": 4.0,
    "o1": 3.8,
    "o3": 3.8,
    # Anthropic
    "claude-3.5-haiku": 3.5,
    "claude-3.5-sonnet": 3.5,
    "claude-3.5": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-opus-4": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3-haiku": 3.5,
    "claude-3": 3.5,
    "claude": 3.5,
    # Google
    "gemini-2.0-flash": 4.0,
    "gemini-1.5-pro": 4.0,
    "gemini-1.5-flash": 4.0,
    "gemini-pro": 4.0,
    "gemini": 4.0,
    # Meta / Llama
    "llama-3.3": 3.8,
    "llama-3.2": 3.8,
    "llama-3.1": 3.8,
    "llama-3": 3.8,
    "llama": 3.8,
    # Mistral
    "mistral-large": 4.0,
    "mistral-nemo": 4.0,
    "mistral-7b": 4.0,
    "mistral": 4.0,
}

# Approximate context-window sizes in tokens.
_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-3.5": 16_385,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o3": 200_000,
    # Anthropic
    "claude-3.5-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    # Google
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
    # Meta / Llama
    "llama-3.3": 128_000,
    "llama-3.2": 128_000,
    "llama-3.1": 128_000,
    "llama-3": 8_192,
    "llama": 8_192,
    # Mistral
    "mistral-large": 128_000,
    "mistral-nemo": 128_000,
    "mistral-7b": 32_768,
    "mistral": 32_768,
}

# Per-message overhead in tokens (role markers + message separators).
# Coarser than _RATIOS — many model variants share the same chat format.
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o-mini": 3,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4.1": 4,
    "gpt-4": 4,
    "o1": 4,
    "o3": 4,
    "claude": 5,
    "gemini": 4,
    "llama": 4,
    "mistral": 4,
}

_DEFAULT_FAMILY = "gpt-4"

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class TokenEstimate(NamedTuple):
    """Structured result returned by :func:`token_summary`.

    Attributes
    ----------
    tokens:
        Estimated token count for the supplied text (rounded up).
    window:
        Total context-window size for the resolved model family.
    remaining:
        ``window - tokens - headroom``.  Negative values indicate overflow.
    fits:
        ``True`` when *remaining* is non-negative.
    model_family:
        The canonical family key that was resolved from the model name.
    """

    tokens: int
    window: int
    remaining: int
    fits: bool
    model_family: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _family(model: str) -> str:
    """Return the canonical ratio-table key for *model* (longest-prefix match).

    Falls back to ``_DEFAULT_FAMILY`` when no key matches.
    """
    m = (model or "").lower().strip()
    for key in sorted(_RATIOS, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Return the overhead-table key for *model* (longest-prefix match)."""
    m = (model or "").lower().strip()
    for key in sorted(_MESSAGE_OVERHEAD, key=len, reverse=True):
        if m.startswith(key):
            return key
    return _DEFAULT_FAMILY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """Return the recognised model-family prefix strings, sorted.

    Any model name that *starts with* one of these prefixes will use the
    associated character-to-token ratio and context-window size.  Names
    that match nothing fall back to ``gpt-4`` defaults.

    Returns
    -------
    list[str]
        Sorted list of supported model-family prefix strings.

    Examples
    --------
    >>> "gpt-4" in list_models()
    True
    >>> "claude" in list_models()
    True
    """
    return sorted(_RATIOS)


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for *model*.

    Uses a per-family character-to-token ratio.  Empty or ``None`` input
    returns 0.  The result is always rounded **up** to avoid
    under-reporting.

    Parameters
    ----------
    text:
        The text to measure.  Accepts ``None`` (treated as empty).
    model:
        Model name or prefix (e.g. ``"gpt-4o"``, ``"claude-3-opus"``).
        Unrecognised names fall back to ``gpt-4`` defaults.

    Returns
    -------
    int
        Estimated token count.

    Examples
    --------
    >>> estimate_tokens("Hello, world!", model="gpt-4")
    4
    >>> estimate_tokens("", model="gpt-4")
    0
    >>> estimate_tokens(None, model="gpt-4")
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
    """Estimate the token count of a chat-style message list for *model*.

    Each message mapping should contain at least a ``"content"`` key and
    optionally a ``"role"`` key.  A small per-message overhead is added to
    account for role markers and message separators.  Non-mapping items in
    the iterable are silently skipped.

    Parameters
    ----------
    messages:
        Iterable of message mappings.  Pass ``None`` or an empty list for
        a zero result.
    model:
        Model name used to look up the ratio and per-message overhead.

    Returns
    -------
    int
        Total estimated token count including per-message overhead.

    Examples
    --------
    >>> msgs = [{"role": "user", "content": "Hello"}]
    >>> estimate_messages(msgs, model="gpt-4")
    7
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


def remaining_tokens(text: str, model: str, headroom: int = 0) -> int:
    """Return the tokens still available after placing *text* in *model*'s window.

    Parameters
    ----------
    text:
        Text whose token estimate is subtracted from the window.
    model:
        Target model name.
    headroom:
        Additional tokens to reserve (e.g. for a model response).
        Negative values are clamped to 0.

    Returns
    -------
    int
        ``window - estimate_tokens(text) - headroom``.  A negative value
        means the text overflows the context window.

    Examples
    --------
    >>> remaining_tokens("hi", "gpt-4")
    8191
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return window - used


def fits_in_context(text: str, model: str, headroom: int = 0) -> bool:
    """Return ``True`` if *text* fits within *model*'s context window.

    Parameters
    ----------
    text:
        Text to check.
    model:
        Target model name.
    headroom:
        Tokens to reserve for the model's response.  Negative values are
        clamped to 0.

    Returns
    -------
    bool
        ``True`` when ``estimate_tokens(text) + headroom <= window``.

    Examples
    --------
    >>> fits_in_context("Hello!", model="gpt-4")
    True
    >>> fits_in_context("x" * 500_000, model="gpt-4")
    False
    """
    return remaining_tokens(text, model, headroom) >= 0


def token_summary(text: str, model: str, headroom: int = 0) -> TokenEstimate:
    """Return a :class:`TokenEstimate` with full context-window diagnostics.

    Parameters
    ----------
    text:
        Text to estimate.
    model:
        Target model name.
    headroom:
        Tokens to reserve for the model's response.

    Returns
    -------
    TokenEstimate
        Named-tuple with ``tokens``, ``window``, ``remaining``, ``fits``,
        and ``model_family`` fields.

    Examples
    --------
    >>> r = token_summary("Hello!", model="gpt-4")
    >>> r.fits
    True
    >>> r.model_family
    'gpt-4'
    """
    fam = _family(model)
    tokens = estimate_tokens(text, model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    rem = window - tokens - max(0, int(headroom))
    return TokenEstimate(
        tokens=tokens,
        window=window,
        remaining=rem,
        fits=rem >= 0,
        model_family=fam,
    )


__all__ = [
    "TokenEstimate",
    "__version__",
    "estimate_messages",
    "estimate_tokens",
    "fits_in_context",
    "list_models",
    "remaining_tokens",
    "token_summary",
]
