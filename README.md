# tokenfit

**Estimate token counts for LLM context windows — no API calls, no heavy
dependencies.**

[![CI](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/tokenfit/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`tokenfit` gives you a fast, offline estimate of how many tokens a given text
or chat-message list will consume for any major LLM family.  It works anywhere
Python does — no compiled extensions, no model downloads.

---

## Install

```bash
pip install tokenfit
```

## Quick start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# Plain-text estimate
estimate_tokens("Hello, world!", model="gpt-4")           # → 4

# Chat-style message list (includes per-message overhead)
msgs = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Summarise the French Revolution."},
]
estimate_messages(msgs, model="claude-3-opus")            # → ~20

# Context-window check
fits_in_context("..." * 10_000, model="gpt-4")            # → False
fits_in_context("..." * 10_000, model="gpt-4-turbo")      # → True
```

## API reference

### `estimate_tokens(text, model="gpt-4") → int`

Estimate the token count of a plain string.  Returns 0 for empty / `None`
input.  Always rounds up so the count never under-reports.

### `estimate_messages(messages, model="gpt-4") → int`

Estimate the token count of a chat-style message list.  Each message should be
a mapping with `"role"` and `"content"` keys.  A per-message overhead is
added for role markers and separators.  Non-mapping items are silently skipped.

### `fits_in_context(text, model, headroom=0) → bool`

Return `True` if `estimate_tokens(text, model) + max(0, headroom)` fits within
the model's context window.  `headroom` reserves tokens for the model's reply.

### `context_window(model) → int`

Return the context-window size in tokens for the given model (falls back to the
GPT-4 window for unknown models).

### `list_models() → list[str]`

Return a sorted list of all recognised model-name prefixes.

## Supported models

| Family | Example names | Ratio | Context window |
|--------|--------------|-------|----------------|
| GPT-3.5 | `gpt-3.5-turbo` | 4.0 | 16 385 |
| GPT-4 | `gpt-4` | 4.0 | 8 192 |
| GPT-4-Turbo | `gpt-4-turbo` | 4.0 | 128 000 |
| GPT-4o | `gpt-4o`, `gpt-4o-mini` | 3.8 | 128 000 |
| o-series | `o1`, `o3`, `o4-mini` | 3.8 | 128 000 – 200 000 |
| Claude 3 | `claude-3-haiku/sonnet/opus` | 3.5 | 200 000 |
| Claude 3.5 | `claude-3.5-sonnet/haiku` | 3.5 | 200 000 |
| Claude 4 | `claude-sonnet-4`, `claude-opus-4` | 3.5 | 200 000 |
| Gemini | `gemini-pro`, `gemini-1.5-pro`, `gemini-2.0-flash`, `gemini-2.5-pro` | 4.0 | 32 768 – 1 048 576 |
| Llama | `llama-2`, `llama-3` | 3.7 | 4 096 – 128 000 |
| Mistral | `mistral`, `mistral-large` | 3.8 | 32 768 – 131 072 |
| Cohere | `command`, `command-r`, `command-r-plus` | 3.8 | 4 096 – 128 000 |

Unknown model names fall back to the GPT-4 ratio (4.0) and window (8 192
tokens).

## GUI

Launch the interactive desktop app with:

```bash
tokenfit-gui
```

Features:

- **Text tab** — paste or type text and see a live token count plus a
  colour-coded context-window utilisation bar.
- **Messages tab** — build a multi-turn chat history and estimate the combined
  token count including per-message overhead.
- Real-time updates as you type.
- Works with all built-in model families via a dropdown selector.

The GUI uses Python's standard-library `tkinter` — no extra installation
required on most systems.  If your distribution ships a minimal Python build
without `tkinter`, install it with your package manager (e.g.
`sudo apt install python3-tk`).

## Design notes

Estimates use per-family character-to-token ratios calibrated against English
prose.  Non-English text and source code may deviate.  The counts always round
up so they never under-report — useful for conservative budget checks before
dispatching an API request.  Exact counts require the model's own tokeniser
(e.g. `tiktoken` for OpenAI models).

Model names are resolved via longest-prefix matching, so version-dated variants
such as `claude-3.5-sonnet-20241022` automatically resolve to the correct
family without needing an explicit entry in the table.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use `tokenfit` in published research, please cite it using the metadata
in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
