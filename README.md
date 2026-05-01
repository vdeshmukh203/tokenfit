# tokenfit

[![CI](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenfit)](https://pypi.org/project/tokenfit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/tokenfit)](https://pypi.org/project/tokenfit/)

**Fast, offline token-count estimation for LLM context windows.**

`tokenfit` estimates how many tokens a piece of text will consume for a given
large language model (LLM) — without making any API calls and without loading
a tokenizer library. It uses calibrated character-to-token ratios for each
model family, making it ideal for pre-flight budget checks, routing logic,
and interactive tooling.

---

## Statement of Need

LLM APIs charge by the token and reject requests that exceed the model's
context window. Counting tokens precisely requires running a model-specific
tokenizer (e.g. `tiktoken` for OpenAI models) which is a heavy dependency
not always available in CI pipelines, edge functions, or notebooks. At the
same time, a rough upper-bound estimate is sufficient for most guard-rail
checks: *will this prompt overflow the context window?*

`tokenfit` fills this niche: it is a pure-Python, zero-dependency library
that provides conservative (rounding-up) token estimates in microseconds.
It supports the three most widely-used LLM families — OpenAI GPT, Anthropic
Claude, and Google Gemini — and exposes a minimal, easy-to-learn API of five
functions.

---

## Installation

```bash
pip install tokenfit
```

Requires Python ≥ 3.9. No additional dependencies.

---

## Quick Start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# How many tokens will this prompt use?
estimate_tokens("Explain quantum entanglement in simple terms.", model="gpt-4")
# → 9

# Will a long document fit in the GPT-4 context window?
with open("chapter.txt") as f:
    text = f.read()
fits_in_context(text, model="gpt-4", headroom=500)
# → True / False

# Estimate a full chat conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Summarise the following text: ..."},
]
estimate_messages(messages, model="claude-3-opus")
# → 22 (example)
```

---

## API Reference

### `estimate_tokens(text, model="gpt-4") → int`

Estimate the number of tokens in a plain-text string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text. `None` is treated as `""`. |
| `model` | `str` | Model or family name (default `"gpt-4"`). Unknown names fall back to the GPT-4 ratio. |

Returns an integer rounded *up* so the estimate never under-reports.

---

### `estimate_messages(messages, model="gpt-4") → int`

Estimate the token count of a chat-style message list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `Iterable[Mapping[str, str]]` | List of dicts with `"role"` and `"content"` keys. Non-mapping items are skipped. |
| `model` | `str` | Model name used for both the token ratio and per-message overhead. |

Per-message overhead (role markers, separators) is added automatically:
4 tokens for GPT-4, 3 for GPT-4o, 5 for Claude.

---

### `fits_in_context(text, model, headroom=0) → bool`

Check whether a text fits within the model's context window.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to test. |
| `model` | `str` | Model name whose window size is used. |
| `headroom` | `int` | Tokens to reserve for the model's reply (clamped to ≥ 0). |

---

### `context_window(model) → int`

Return the context-window size in tokens for a given model.

```python
from tokenfit import context_window
context_window("gpt-4-turbo")   # 128000
context_window("claude-3-opus") # 200000
context_window("gemini-2.0-flash")  # 1048576
```

---

### `token_budget(text, model, headroom=0) → int`

Return the number of tokens *remaining* in the context window after
placing `text` (and reserving `headroom` tokens). Can be negative when
the text exceeds the window.

```python
from tokenfit import token_budget
token_budget("hello world", "gpt-4")  # e.g. 8189
```

---

### `list_models() → list[str]`

Return all recognised model identifiers in alphabetical order.

```python
from tokenfit import list_models
list_models()
# ['claude', 'claude-3', 'claude-3-opus', 'claude-3-sonnet',
#  'claude-3.5-sonnet', 'claude-sonnet-4', 'gemini', ...]
```

---

## Supported Models

| Family | Models | Ratio | Window |
|--------|--------|-------|--------|
| OpenAI GPT-3.5 | `gpt-3.5` | 4.0 ch/tok | 16 385 |
| OpenAI GPT-4 | `gpt-4` | 4.0 ch/tok | 8 192 |
| OpenAI GPT-4o | `gpt-4o` | 3.8 ch/tok | 128 000 |
| OpenAI GPT-4 Turbo | `gpt-4-turbo` | 4.0 ch/tok | 128 000 |
| Anthropic Claude | `claude`, `claude-3`, `claude-3-opus`, `claude-3-sonnet`, `claude-3.5-sonnet`, `claude-sonnet-4` | 3.5 ch/tok | 200 000 |
| Google Gemini Pro | `gemini-pro` | 4.0 ch/tok | 32 768 |
| Google Gemini 1.5 Pro | `gemini-1.5-pro` | 4.0 ch/tok | 1 048 576 |
| Google Gemini 2.0 Flash | `gemini-2.0-flash` | 4.0 ch/tok | 1 048 576 |

Model names are matched by **longest prefix**, so versioned names like
`gpt-4-turbo-preview` or `claude-3-opus-20240229` resolve automatically.

---

## Graphical Interface

A desktop GUI is included and can be launched from the command line:

```bash
tokenfit-gui
```

Features:
- Real-time token count as you type
- Context-window usage bar (colour-coded: green / amber / red)
- Plain-text and chat-message modes
- Configurable headroom for model replies

---

## Accuracy

Estimates are based on empirical character-to-token ratios for English
prose. Code, non-Latin scripts, and highly structured data may deviate.
The estimates always round **up** to avoid under-reporting — the actual
token count for a given text will be at most a few percent lower.

For exact counts, use the model's native tokenizer
(e.g. [`tiktoken`](https://github.com/openai/tiktoken) for OpenAI models).

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on reporting issues, suggesting new model ratios, and submitting
pull requests.

---

## Citation

If you use `tokenfit` in academic work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).
