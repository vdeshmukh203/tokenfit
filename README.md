# tokenfit

[![CI](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**tokenfit** estimates LLM token counts offline — no API calls, no external dependencies, pure Python.

Use it as a cheap pre-flight check before sending text to an LLM API to avoid hitting context-window limits or paying for tokens you didn't budget.

Estimates always **round up** so they never under-report the true count.

---

## Features

- Estimates tokens for **GPT-3.5 / GPT-4 / GPT-4o / GPT-4-turbo**, **Claude 3 / Claude 3.5 / Claude 4**, and **Gemini 1.5 / 2.0** families
- Handles both raw text and chat-style message lists
- `fits_in_context()` — boolean context-window check with optional headroom reservation
- `list_models()` — enumerate all supported model names
- Interactive **GUI** included (`tokenfit-gui` command)
- Zero runtime dependencies — standard library only
- Python 3.9+

---

## Installation

```bash
pip install tokenfit
```

---

## Quick Start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# Plain text
estimate_tokens("Hello, world!", model="gpt-4")         # → 4
estimate_tokens("Hello, world!", model="claude-3-opus") # → 4

# Chat messages (includes per-message overhead)
msgs = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Summarise the following article..."},
]
estimate_messages(msgs, model="gpt-4o")  # → integer token count

# Context-window check
fits_in_context("a" * 100_000, model="gpt-4")         # → False (8 k window)
fits_in_context("a" * 100_000, model="gpt-4-turbo")   # → True  (128 k window)

# Reserve headroom for the model's response
fits_in_context(prompt, model="gpt-4o", headroom=2_000)
```

---

## API Reference

### `estimate_tokens(text, model="gpt-4") -> int`

Estimate the number of tokens in `text` for the given model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text`    | `str \| None` | Input text. `None` or empty string returns 0. |
| `model`   | `str` | Model name. Unknown names fall back to `gpt-4`. |

Returns `int` — estimated token count, rounded up.

---

### `estimate_messages(messages, model="gpt-4") -> int`

Estimate the total token count of a chat-style message list, including per-message overhead tokens for role markers and separators.

| Parameter  | Type | Description |
|------------|------|-------------|
| `messages` | `Iterable[Mapping[str, str]]` | List of `{"role": ..., "content": ...}` dicts. Non-mapping items are skipped. |
| `model`    | `str` | Model name. |

Returns `int` — estimated token count for the full conversation.

---

### `fits_in_context(text, model, headroom=0) -> bool`

Return `True` if `text` fits within the model's context window.

| Parameter  | Type  | Description |
|------------|-------|-------------|
| `text`     | `str \| None` | Input text. |
| `model`    | `str` | Model name. |
| `headroom` | `int` | Tokens to reserve for the model's response (clamped to ≥ 0). |

---

### `fits_in_context_messages(messages, model, headroom=0) -> bool`

Equivalent to `fits_in_context` but accepts a message list (same format as `estimate_messages`).

---

### `list_models() -> list[str]`

Return a sorted list of all model names with built-in support.

```python
from tokenfit import list_models
print(list_models())
# ['claude', 'claude-3', 'claude-3-haiku', 'claude-3-opus', ...]
```

---

## Supported Models

| Model | Context window | Chars/token ratio |
|-------|---------------|-------------------|
| `gpt-3.5` | 16,385 | 4.0 |
| `gpt-4` | 8,192 | 4.0 |
| `gpt-4-turbo` | 128,000 | 4.0 |
| `gpt-4o` | 128,000 | 3.8 |
| `claude-3-haiku` | 200,000 | 3.5 |
| `claude-3-sonnet` | 200,000 | 3.5 |
| `claude-3-opus` | 200,000 | 3.5 |
| `claude-3.5-haiku` | 200,000 | 3.5 |
| `claude-3.5-sonnet` | 200,000 | 3.5 |
| `claude-sonnet-4` | 200,000 | 3.5 |
| `claude-opus-4` | 200,000 | 3.5 |
| `gemini-pro` | 32,768 | 4.0 |
| `gemini-1.5-pro` | 1,048,576 | 4.0 |
| `gemini-2.0-flash` | 1,048,576 | 4.0 |

Versioned variants (e.g. `claude-3.5-sonnet-20241022`) are matched by longest-prefix, so they resolve to the correct family automatically.

---

## GUI

Launch the interactive estimator:

```bash
tokenfit-gui                 # installed via pip
python -m tokenfit           # from a source checkout
```

The GUI provides:
- Model selector and headroom spinner
- **Plain Text** tab — paste any text and see token count in real time
- **Chat Messages** tab — build a message list row-by-row
- Progress bar and colour-coded fit indicator (green / amber / red)

---

## Accuracy

tokenfit uses fixed character-to-token ratios calibrated on representative English prose. Accuracy degrades for:

- **Code** — typically 2–3 chars/token
- **Non-English text** — can be 1–2 chars/token (CJK, Arabic, etc.)
- **Highly structured data** (JSON, YAML, XML)

For exact counts, use the model vendor's tokenizer library (e.g. `tiktoken` for OpenAI, `sentencepiece` for Google).

---

## Development

```bash
git clone https://github.com/vdeshmukh203/tokenfit
cd tokenfit
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
