"""tokenfit: heuristic token-count estimation for LLM context windows.

Provides cheap, offline estimates of how many tokens a given text or chat
message list will consume, based on per-family character-to-token ratios.
Useful for budget checks before sending requests to an LLM API.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Mapping

__version__ = "0.2.0"

# Approximate characters-per-token for each model family.
# Calibrated against representative English prose; non-English text and
# code may deviate. These are *estimates* — exact counts require the real
# tokenizer for each model.
_RATIOS: dict[str, float] = {
    # OpenAI GPT series
    "gpt-3.5": 4.0,
    "gpt-4o-mini": 3.8,
    "gpt-4o": 3.8,
    "gpt-4-turbo": 4.0,
    "gpt-4": 4.0,
    # OpenAI reasoning models
    "o1-mini": 3.8,
    "o1-pro": 3.8,
    "o1": 3.8,
    "o3-mini": 3.8,
    "o3": 3.8,
    # Anthropic Claude
    "claude-3.7-sonnet": 3.5,
    "claude-3.5-sonnet": 3.5,
    "claude-3.5-haiku": 3.5,
    "claude-sonnet-4": 3.5,
    "claude-opus-4": 3.5,
    "claude-haiku-4": 3.5,
    "claude-3-opus": 3.5,
    "claude-3-sonnet": 3.5,
    "claude-3-haiku": 3.5,
    "claude-3": 3.5,
    "claude": 3.5,
    # Google Gemini
    "gemini-2.5": 4.0,
    "gemini-2.0-flash": 4.0,
    "gemini-1.5-flash": 4.0,
    "gemini-1.5-pro": 4.0,
    "gemini-pro": 4.0,
    "gemini": 4.0,
    # Meta Llama
    "llama-3": 3.6,
    "llama-2": 4.0,
    "llama": 3.6,
    # Mistral AI
    "mistral-large": 3.8,
    "mistral-7b": 4.0,
    "mistral": 3.8,
    # DeepSeek
    "deepseek-r1": 3.8,
    "deepseek": 3.8,
    # Cohere
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
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    # Anthropic Claude
    "claude-3.7-sonnet": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3": 200_000,
    "claude": 200_000,
    # Google Gemini
    "gemini-2.5": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-pro": 32_768,
    "gemini": 1_048_576,
    # Meta Llama
    "llama-3": 128_000,
    "llama-2": 4_096,
    "llama": 128_000,
    # Mistral AI
    "mistral-large": 128_000,
    "mistral-7b": 32_768,
    "mistral": 32_768,
    # DeepSeek
    "deepseek-r1": 128_000,
    "deepseek": 128_000,
    # Cohere
    "command-r": 128_000,
    "command": 128_000,
}

# Per-message overhead in tokens (role markers, separators).
_MESSAGE_OVERHEAD: dict[str, int] = {
    "gpt-3.5": 4,
    "gpt-4o": 3,
    "gpt-4-turbo": 4,
    "gpt-4": 4,
    "claude": 5,
    "gemini": 4,
    "llama": 4,
    "mistral": 3,
}

_DEFAULT_FAMILY = "gpt-4"

# Pre-sorted at import time so _family() does no repeated sorting.
_SORTED_RATIO_KEYS: list[str] = sorted(_RATIOS, key=len, reverse=True)


@dataclass(frozen=True)
class TokenEstimate:
    """Immutable result of a token count estimation.

    Parameters
    ----------
    tokens:
        Estimated number of tokens consumed by the input.
    model:
        The model name provided by the caller.
    family:
        The resolved model family used for ratio and window lookups.
    window_size:
        Context-window size (tokens) for the resolved model family.
    """

    tokens: int
    model: str
    family: str
    window_size: int

    @property
    def utilization(self) -> float:
        """Fraction of the context window consumed (may exceed 1.0)."""
        return self.tokens / self.window_size if self.window_size > 0 else 1.0

    def fits(self, headroom: int = 0) -> bool:
        """Return True if *tokens* (plus *headroom*) fit in the window."""
        return self.tokens + max(0, headroom) <= self.window_size

    def remaining(self, headroom: int = 0) -> int:
        """Tokens remaining in the window after subtracting optional headroom."""
        return max(0, self.window_size - self.tokens - max(0, headroom))


def _family(model: str) -> str:
    """Return the canonical family key for *model* via longest-prefix match.

    Emits a :class:`UserWarning` and falls back to ``gpt-4`` ratios when no
    known key is a prefix of the supplied model name.
    """
    m = (model or "").lower().strip()
    for key in _SORTED_RATIO_KEYS:
        if m.startswith(key):
            return key
    warnings.warn(
        f"Model {model!r} is not recognised; using {_DEFAULT_FAMILY!r} ratios.",
        UserWarning,
        # stacklevel=1 → all unrecognised-model warnings share the same source
        # location inside _family, so Python's "once" filter deduplicates them.
        stacklevel=1,
    )
    return _DEFAULT_FAMILY


def _overhead_family(model: str) -> str:
    """Map *model* to the coarser overhead-bucket key."""
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
    if m.startswith("gpt-4") or m.startswith("o1") or m.startswith("o3"):
        return "gpt-4"
    if m.startswith("llama"):
        return "llama"
    if m.startswith("mistral"):
        return "mistral"
    return "gpt-4"


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text* for the given *model*.

    Uses a per-family character-to-token ratio calibrated on English prose.
    Empty or *None* input returns 0.  The estimate is always rounded up
    (ceiling) so the count never under-reports.

    Parameters
    ----------
    text:
        The input text to measure.
    model:
        Target model name.  Unrecognised names fall back to ``gpt-4``
        ratios and emit a :class:`UserWarning`.

    Returns
    -------
    int
        Estimated token count (≥ 0).
    """
    if not text:
        return 0
    fam = _family(model)
    return math.ceil(len(text) / _RATIOS[fam])


def estimate_tokens_detailed(text: str, model: str = "gpt-4") -> TokenEstimate:
    """Estimate tokens and return a :class:`TokenEstimate` with window metadata.

    Parameters
    ----------
    text:
        The input text to measure.
    model:
        Target model name.

    Returns
    -------
    TokenEstimate
        Frozen dataclass with ``tokens``, ``model``, ``family``,
        ``window_size``, and derived helpers ``utilization``, ``fits()``,
        ``remaining()``.
    """
    fam = _family(model)
    tokens = math.ceil(len(text) / _RATIOS[fam]) if text else 0
    return TokenEstimate(
        tokens=tokens,
        model=model,
        family=fam,
        window_size=_WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY]),
    )


def estimate_messages(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
) -> int:
    """Estimate the token count of a chat-style *messages* list.

    Each message should be a mapping with ``role`` and ``content`` keys.
    A small per-message overhead is added to account for role markers and
    message separators.  Items that are not mappings are silently skipped.

    Parameters
    ----------
    messages:
        Iterable of message mappings.  *None* is treated as an empty list.
    model:
        Target model name.

    Returns
    -------
    int
        Estimated total token count (≥ 0).
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


def estimate_messages_detailed(
    messages: Iterable[Mapping[str, str]],
    model: str = "gpt-4",
) -> TokenEstimate:
    """Estimate message-list tokens and return a :class:`TokenEstimate`.

    Parameters
    ----------
    messages:
        Iterable of message mappings.
    model:
        Target model name.

    Returns
    -------
    TokenEstimate
        Frozen dataclass with token count and context-window metadata.
    """
    fam = _family(model)
    tokens = estimate_messages(messages, model)
    return TokenEstimate(
        tokens=tokens,
        model=model,
        family=fam,
        window_size=_WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY]),
    )


def fits_in_context(text: str, model: str, headroom: int = 0) -> bool:
    """Return True if *text* fits within the model's context window.

    *headroom* reserves that many extra tokens (e.g. for the model
    response), so ``estimate_tokens(text) + headroom`` must be ≤ the
    window size.  Negative headroom is clamped to zero.

    Parameters
    ----------
    text:
        The input text to check.
    model:
        Target model name.
    headroom:
        Tokens to reserve for the model's response (default 0).

    Returns
    -------
    bool
        True if the text (plus headroom) fits.
    """
    fam = _family(model)
    window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
    used = estimate_tokens(text, model) + max(0, int(headroom))
    return used <= window


def context_window_size(model: str) -> int:
    """Return the context-window size in tokens for the given *model*.

    Parameters
    ----------
    model:
        The model name to look up.

    Returns
    -------
    int
        Context-window size in tokens.  Falls back to the ``gpt-4``
        window (8 192) for unrecognised model names.
    """
    fam = _family(model)
    return _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])


__all__ = [
    "TokenEstimate",
    "__version__",
    "context_window_size",
    "estimate_messages",
    "estimate_messages_detailed",
    "estimate_tokens",
    "estimate_tokens_detailed",
    "fits_in_context",
]
