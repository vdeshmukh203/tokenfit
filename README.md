# tokenfit

[![CI](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/tokenfit/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/tokenfit.svg)](https://pypi.org/project/tokenfit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**tokenfit** estimates how many tokens a text will consume in a given LLM's context window — with no API calls, no external dependencies, and pure Python.

## Statement of need

Modern LLMs expose finite context windows (from 8 K to over 1 M tokens). Before sending a request you often need to know: *will this text fit?* Exact counts require the model's own tokenizer (e.g. `tiktoken` for OpenAI, a Rust library for Claude), which may be heavy, version-sensitive, or simply unavailable offline.

**tokenfit** fills this gap with calibrated character-to-token ratios for the major model families. The estimates round up (conservative) and are accurate to within a few percent for typical English prose — sufficient for budget checks, routing logic, and UI feedback.

## Install

```bash
pip install tokenfit
```

No extra dependencies. tokenfit uses only the Python standard library.

## Quick start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# Plain text
estimate_tokens("Hello, world!", model="gpt-4")          # → 4
estimate_tokens("Hello, world!", model="claude-3-opus")  # → 4

# Chat messages
msgs = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Summarise the attached document."},
]
estimate_messages(msgs, model="gpt-4o")  # → 17

# Budget check
fits_in_context("..." * 1000, model="gpt-4", headroom=500)  # → False
```

## API reference

### `estimate_tokens(text, model="gpt-4") → int`

Estimate the number of tokens in `text` for the given `model`.

- `text` — string to evaluate; `None` or empty string returns `0`.
- `model` — LLM model name (see [Supported models](#supported-models)). Unknown names fall back to `gpt-4` ratios.
- Returns an integer that is **never lower** than the true count for typical English prose.

### `estimate_messages(messages, model="gpt-4") → int`

Estimate the token count for a chat-style list of messages.

- `messages` — iterable of dicts with `"role"` and `"content"` keys. Non-dict items are skipped.
- Adds a small per-message overhead to account for role markers and separators.

### `fits_in_context(text, model, headroom=0) → bool`

Return `True` if `text` fits within the model's context window.

- `headroom` — tokens reserved for the model's response (clamped to 0 if negative).

### `get_context_window(model) → int`

Return the context window size in tokens for `model`. Unknown names fall back to `gpt-4` (8 192 tokens).

### `list_models() → list[str]`

Return an alphabetically sorted list of all explicitly supported model names.

## CLI

tokenfit ships a command-line interface:

```
python -m tokenfit "Hello, world!" --model gpt-4
```

```
model      gpt-4
tokens     4
window     8,192
usage      0.0%
fits       yes
```

**Options**

| Flag | Description |
|------|-------------|
| `--model MODEL` | LLM model name (default: `gpt-4`) |
| `--headroom N` | Reserve *N* tokens for the response |
| `--list-models` | Print all supported models with their context window sizes |
| `--gui` | Launch the graphical interface |
| `--version` | Print the version and exit |

The CLI exits with code `0` when the text fits, `1` when it does not — making it composable in shell pipelines:

```bash
cat document.txt | python -m tokenfit --model claude-3-opus || echo "Too large!"
```

## GUI

Launch the interactive graphical interface:

```bash
python -m tokenfit --gui
# or
python -m tokenfit.gui
```

The GUI provides:
- **Model selector** — choose any supported model or type a custom name.
- **Headroom spinner** — reserve tokens for the model's response.
- **Live text area** — estimates update in real time as you type.
- **Colour-coded progress bar** — green (< 70 %), amber (70–90 %), red (> 90 %).
- **Status line** — shows tokens remaining or how far over the limit you are.

> The GUI requires `tkinter`, which is included with most Python distributions.
> On Debian/Ubuntu install it with `sudo apt install python3-tk` if missing.

## Supported models

| Model | Context window |
|-------|---------------|
| `gpt-3.5` | 16,385 |
| `gpt-4` | 8,192 |
| `gpt-4-turbo` | 128,000 |
| `gpt-4o` | 128,000 |
| `gpt-4o-mini` | 128,000 |
| `gpt-4.1` | 1,047,576 |
| `gpt-4.1-mini` | 1,047,576 |
| `gpt-4.1-nano` | 1,047,576 |
| `o1` | 200,000 |
| `o1-mini` | 128,000 |
| `o3` | 200,000 |
| `o3-mini` | 200,000 |
| `o4-mini` | 200,000 |
| `claude-3-haiku` | 200,000 |
| `claude-3-sonnet` | 200,000 |
| `claude-3-opus` | 200,000 |
| `claude-3.5-haiku` | 200,000 |
| `claude-3.5-sonnet` | 200,000 |
| `claude-3-5-haiku` | 200,000 |
| `claude-3-5-sonnet` | 200,000 |
| `claude-haiku-4` | 200,000 |
| `claude-sonnet-4` | 200,000 |
| `claude-opus-4` | 200,000 |
| `gemini-pro` | 32,768 |
| `gemini-1.5-flash` | 1,048,576 |
| `gemini-1.5-pro` | 1,048,576 |
| `gemini-2.0-flash` | 1,048,576 |
| `gemini-2.5-flash` | 1,048,576 |
| `gemini-2.5-pro` | 1,048,576 |

Any model name not in this table is matched by longest prefix (e.g. `claude-3-5-sonnet-20241022` matches `claude-3-5-sonnet`), and falls back to `gpt-4` ratios when no prefix matches.

## Limitations

- **Estimates only.** Exact token counts require the model's native tokenizer.
- **Calibrated for English prose.** Code and non-Latin scripts may deviate by ±20 %.
- **Ratios are static.** Model updates that change tokenization are not automatically reflected.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
