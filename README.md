# tokenfit

**Estimate token counts for LLM context windows — no API calls, no external dependencies.**

[![CI](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Statement of need

Every call to a large-language-model (LLM) API is gated by a context-window limit measured in tokens.
Developers routinely need to check whether a prompt or conversation will fit before dispatching the request — but running the real tokenizer (e.g. `tiktoken` for OpenAI or the Anthropic SDK) adds a dependency, requires network-accessible model files, and can be slow in tight loops.

**tokenfit** fills this gap with a pure-Python, dependency-free heuristic estimator.
It uses pre-calibrated character-to-token ratios per model family to produce conservative (always rounded-up) estimates in O(n) time.
This makes it suitable for budget checks, streaming pre-flight guards, and any situation where a near-correct answer in microseconds is preferable to an exact answer in milliseconds.

## Install

```bash
pip install tokenfit
```

Requires Python 3.9 or later.  No external dependencies — only the standard library.

## Quick start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# How many tokens will this text use?
estimate_tokens("Hello, world!", model="gpt-4")          # → 4

# Will a 500-token response still fit after this prompt?
fits_in_context(long_prompt, model="gpt-4o", headroom=500)

# Estimate a full multi-turn conversation
msgs = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Summarise the French Revolution."},
]
estimate_messages(msgs, model="claude-3-opus")
```

## GUI

A Tkinter desktop application is included.  Launch it with:

```bash
python -m tokenfit
# or, after pip install:
tokenfit-gui
```

The GUI provides two tabs:

| Tab | Description |
|-----|-------------|
| **Text** | Paste or type any text; see token count, window size, percentage used, and whether it fits with optional headroom. Updates live as you type. |
| **Chat messages** | Build a multi-turn conversation turn by turn and see the cumulative token estimate update in real time. |

## API reference

### `estimate_tokens(text, model="gpt-4") → int`

Estimate the number of tokens in `text` for `model`.

- `text` — the string to estimate.  `None` and empty strings return `0`.
- `model` — model name string (case-insensitive prefix match).  Unknown names fall back to GPT-4 defaults.
- Returns an `int` rounded **up** so the count never under-reports.

### `estimate_messages(messages, model="gpt-4") → int`

Estimate the total token count of a chat-style messages list.

- `messages` — iterable of `{"role": ..., "content": ...}` mappings (OpenAI / Anthropic convention).  Non-mapping items are silently skipped.  `None` and empty iterables return `0`.
- `model` — model name string.
- Adds a small per-message overhead for role markers and separators.

### `fits_in_context(text, model, headroom=0) → bool`

Return `True` when `estimate_tokens(text, model) + headroom ≤ context_window(model)`.

- `headroom` — tokens to reserve for the model's response.  Negative values are clamped to `0`.

### `context_window(model) → int`

Return the context-window size in tokens for `model`.  Unknown models fall back to the GPT-4 default (8 192 tokens).

### `supported_models() → list[str]`

Return a sorted list of all recognised model-family prefixes.

### `__version__`

Package version string (e.g. `"0.1.0"`).

## Supported models

| Family | Example names | Ratio (chars/token) | Window (tokens) |
|--------|---------------|---------------------|-----------------|
| GPT-3.5 | `gpt-3.5`, `gpt-3.5-turbo` | 4.0 | 16 385 |
| GPT-4 | `gpt-4`, `gpt-4-32k` | 4.0 | 8 192 |
| GPT-4-Turbo | `gpt-4-turbo`, `gpt-4-turbo-preview` | 4.0 | 128 000 |
| GPT-4o | `gpt-4o`, `gpt-4o-mini` | 3.8 | 128 000 |
| Claude 3 | `claude-3`, `claude-3-haiku` | 3.5 | 200 000 |
| Claude 3 Sonnet | `claude-3-sonnet` | 3.5 | 200 000 |
| Claude 3 Opus | `claude-3-opus` | 3.5 | 200 000 |
| Claude 3.5 Sonnet | `claude-3.5-sonnet` | 3.5 | 200 000 |
| Claude Sonnet 4 | `claude-sonnet-4` | 3.5 | 200 000 |
| Gemini Pro | `gemini-pro` | 4.0 | 32 768 |
| Gemini 1.5 Pro | `gemini-1.5-pro` | 4.0 | 1 048 576 |
| Gemini 2.0 Flash | `gemini-2.0-flash` | 4.0 | 1 048 576 |

Model names are matched by **longest prefix**, case-insensitively.  Any name not matching a known family falls back to GPT-4 defaults.

## Design notes

- **Estimates round up** — `math.ceil` is used throughout so the library never under-reports token counts.
- **Calibration** — ratios are calibrated against representative English prose.  Non-English text and code may deviate.
- **Exact counts** — require the real tokenizer for each model (e.g. `tiktoken` for OpenAI models).
- **No external dependencies** — the library imports only `math` and `typing` from the standard library.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, requesting features, and submitting pull requests.

## Citation

If you use tokenfit in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
