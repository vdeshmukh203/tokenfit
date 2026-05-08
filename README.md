# tokenfit

**Estimate token counts for LLM models without API calls.**

tokenfit is a lightweight, zero-dependency Python library that gives fast,
offline estimates of how many tokens a piece of text (or a chat message list)
will consume for a given model.  It is designed for budget checks, context-window
guards, and pre-flight validation before sending requests to an LLM API.

## Statement of Need

Most production LLM applications must enforce context-window limits, track
token budgets, and decide whether to truncate or paginate inputs.  The only
fully accurate way to count tokens is to call each model's proprietary
tokenizer, which requires network access or large binary dependencies (e.g.
`tiktoken`, `sentencepiece`).  For lightweight tooling, CI pipelines, and
server-side pre-checks, a fast heuristic is often sufficient.

tokenfit fills that gap: a single pure-Python file, no external dependencies,
and estimates that deliberately round **up** so they never under-report usage.

## Supported Models

| Family | Models |
|--------|--------|
| OpenAI GPT | `gpt-3.5`, `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini` |
| OpenAI Reasoning | `o1`, `o1-mini`, `o3`, `o3-mini` |
| Anthropic Claude 3 | `claude-3` (haiku/sonnet/opus) |
| Anthropic Claude 3.5 | `claude-3.5-haiku`, `claude-3.5-sonnet` |
| Anthropic Claude 4 | `claude-sonnet-4`, `claude-opus-4` |
| Google Gemini | `gemini-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`, `gemini-2.5-pro` |

Any model name that is not explicitly listed falls back to the `gpt-4` ratio
(4.0 chars/token, 8 192-token window) by longest-prefix matching.

## Install

```bash
pip install tokenfit
```

Requires Python ≥ 3.9.  No external dependencies.

## Quick Start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# Single string
n = estimate_tokens("Hello, world!", model="gpt-4")   # → 4

# Chat message list
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Summarise the paper for me."},
]
n = estimate_messages(messages, model="claude-3-opus")  # → ~18

# Context-window guard (reserves 512 tokens for the response)
ok = fits_in_context(long_document, model="gpt-4-turbo", headroom=512)
```

## API Reference

### `estimate_tokens(text, model="gpt-4") → int`

Returns the estimated number of tokens in `text` for the given `model`.
`None` or empty strings return `0`.  The result is rounded **up**.

### `estimate_messages(messages, model="gpt-4") → int`

Returns the estimated token count for a chat-style list of message dicts.
Each dict should have `"role"` and `"content"` keys.  A small per-message
overhead is added to account for role markers and separator tokens.
Non-mapping items in the list are silently skipped.

### `fits_in_context(text, model, headroom=0) → bool`

Returns `True` if `text` fits within the model's context window, optionally
reserving `headroom` tokens for the model's own output.  Negative headroom
is clamped to zero.

### `context_window(model) → int`

Returns the context-window size in tokens for the given model.

### `chars_per_token(model) → float`

Returns the characters-per-token ratio used internally for the given model.

### `list_models() → list[str]`

Returns a sorted list of all explicitly recognised model identifiers.

## Desktop GUI

tokenfit ships with a tkinter-based desktop interface.  Launch it with:

```bash
# Via the installed console script
tokenfit-gui

# Or as a Python module
python -m tokenfit
```

The GUI provides two tabs:

- **Plain Text** – paste any text, pick a model, and see the live token count
  and context-window utilisation.
- **Chat Messages** – edit a JSON array of `{"role": …, "content": …}` message
  objects and get an instant estimate including per-message overhead.

Both tabs show a headroom field so you can reserve tokens for the model's reply.

## Accuracy and Limitations

tokenfit uses character-to-token ratios calibrated on English prose:

- 4.0 chars/token for GPT and most OpenAI models
- 3.5 chars/token for all Claude models
- 4.0 chars/token for all Gemini models

These ratios are **heuristics**.  Estimates will deviate from exact counts when
text contains significant amounts of:

- Non-English or non-Latin script
- Source code or structured data (JSON, YAML, …)
- Unusual whitespace or punctuation patterns

For exact counts, use the model's own tokenizer (e.g. `tiktoken` for OpenAI
models).  tokenfit is intentionally conservative: it rounds up so the estimate
always meets or exceeds the true token count for typical English text.

## Contributing

Bug reports and pull requests are welcome on the
[issue tracker](https://github.com/vdeshmukh203/tokenfit/issues).

Please ensure all existing tests pass (`pytest`) and add tests for any new
functionality before submitting a pull request.

## License

MIT – see [LICENSE](LICENSE).
