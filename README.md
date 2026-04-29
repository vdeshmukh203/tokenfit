# tokenfit

**Lightweight, offline token-count estimation for LLM context windows.**

tokenfit provides heuristic estimates of how many tokens a text or chat-message list will consume for a given model family.  It requires no network access, no API keys, and no external tokenizer libraries — pure Python, standard library only.

> **Note:** Estimates are intentionally conservative (always rounded up).  
> Exact counts require the real tokenizer for each model (e.g. tiktoken for OpenAI models).

## Supported models

| Family prefix | Example models | Chars/token | Context window |
|---|---|---|---|
| `gpt-4o` | gpt-4o, gpt-4o-mini | 3.8 | 128 K |
| `gpt-4-turbo` | gpt-4-turbo-preview | 4.0 | 128 K |
| `gpt-4` | gpt-4, gpt-4-32k | 4.0 | 8 K |
| `gpt-3.5` | gpt-3.5-turbo | 4.0 | 16 K |
| `claude-3.5-sonnet` | claude-3-5-sonnet-20241022 | 3.5 | 200 K |
| `claude-sonnet-4` | claude-sonnet-4 | 3.5 | 200 K |
| `claude-3-opus` | claude-3-opus-20240229 | 3.5 | 200 K |
| `claude-3-sonnet` | claude-3-sonnet-20240229 | 3.5 | 200 K |
| `gemini-2.0-flash` | gemini-2.0-flash | 4.0 | 1 M |
| `gemini-1.5-pro` | gemini-1.5-pro | 4.0 | 1 M |
| `gemini-pro` | gemini-pro | 4.0 | 32 K |

Matching uses the longest known prefix, so `"gpt-4-turbo-preview"` resolves to the `gpt-4-turbo` family.  Unknown model names fall back to `gpt-4` with a `UserWarning`.

## Install

```bash
pip install tokenfit
```

## Usage

### Estimate tokens in plain text

```python
from tokenfit import estimate_tokens

n = estimate_tokens("Hello, world!", model="gpt-4")
print(n)  # 4
```

### Estimate tokens in a chat conversation

```python
from tokenfit import estimate_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Summarise this article in 3 bullets."},
]
n = estimate_messages(messages, model="claude-3-opus")
print(n)  # includes per-message overhead
```

### Check whether text fits in the context window

```python
from tokenfit import fits_in_context

ok = fits_in_context(long_text, model="gpt-4", headroom=500)
if not ok:
    print("Text too long — truncate before sending.")
```

### Inspect supported models and window sizes

```python
from tokenfit import list_models, context_window

for m in list_models():
    print(m, context_window(m))
```

## Graphical user interface

tokenfit ships with an optional Tkinter GUI (requires `python3-tk`):

```bash
# From the command line:
tokenfit-gui

# Or with Python's module runner:
python -m tokenfit
```

The GUI provides:
- Real-time token estimation as you type
- Visual progress bar showing context-window utilisation
- Headroom control for reserving response tokens
- Side-by-side comparison of all supported models

## Development

```bash
git clone https://github.com/vdeshmukh203/tokenfit
cd tokenfit
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
