# tokenfit

**Estimate LLM token counts offline — no API calls, no heavy dependencies.**

`tokenfit` provides fast heuristic token-count estimates for all major LLM
families.  Use it to check whether text fits in a model's context window,
to budget token usage before sending requests, or to monitor utilisation
in automated pipelines.

## Install

```bash
pip install tokenfit
```

No extra dependencies — pure Python, standard library only.

## Quick start

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

# Plain text
estimate_tokens("Hello, world!", model="gpt-4")          # → 4
estimate_tokens("Hello, world!", model="claude-3.5-sonnet")  # → 4

# Detailed result with context-window metadata
from tokenfit import estimate_tokens_detailed
est = estimate_tokens_detailed("Hello, world!", model="gpt-4o")
print(est.tokens)       # 4
print(est.window_size)  # 128000
print(est.utilization)  # 3.125e-05
print(est.fits())       # True
print(est.remaining())  # 127996

# Chat messages
msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]
estimate_messages(msgs, model="gpt-4o")   # → ~22 tokens

# Context-window check
fits_in_context("..." * 5000, model="gpt-4", headroom=500)  # → False
fits_in_context("..." * 5000, model="gpt-4-turbo")          # → True
```

## Supported models

| Family | Example names | Window |
|--------|---------------|-------:|
| GPT-4 | `gpt-4`, `gpt-4-turbo` | 8 192 / 128 000 |
| GPT-4o | `gpt-4o`, `gpt-4o-mini` | 128 000 |
| OpenAI reasoning | `o1`, `o1-mini`, `o3`, `o3-mini` | 128 000–200 000 |
| Claude 3 / 3.5 / 3.7 | `claude-3.5-sonnet`, `claude-3.5-haiku`, `claude-3.7-sonnet` | 200 000 |
| Claude 4 | `claude-sonnet-4`, `claude-opus-4`, `claude-haiku-4` | 200 000 |
| Gemini | `gemini-2.5`, `gemini-2.0-flash`, `gemini-1.5-pro` | 32 768–1 048 576 |
| Llama | `llama-3`, `llama-2` | 4 096–128 000 |
| Mistral | `mistral-large`, `mistral-7b` | 32 768–128 000 |
| DeepSeek | `deepseek-r1`, `deepseek` | 128 000 |
| Cohere | `command-r`, `command` | 128 000 |

Any versioned suffix is handled automatically via longest-prefix matching
(e.g. `claude-3.5-sonnet-20241022` resolves to the `claude-3.5-sonnet` family).

## CLI

```bash
# Estimate token count
tokenfit estimate "My document text" --model gpt-4

# Check whether text fits (exit code 0 = fits, 1 = exceeds window)
tokenfit fits "My document text" --model gpt-4 --headroom 500

# Show model info
tokenfit info gpt-4o

# JSON output
tokenfit estimate "hello" --model claude-3.5-sonnet --json

# Read from stdin
cat big_file.txt | tokenfit estimate - --model gpt-4-turbo

# Launch the GUI
tokenfit gui
```

## GUI

```bash
tokenfit gui
# or
python -m tokenfit gui
```

The GUI provides a **Plain Text** tab for single-text estimation and a
**Chat Messages** tab for building and estimating multi-turn conversations.
A live results panel shows the token count, a context-window utilisation bar,
and a colour-coded fit indicator.

## API reference

| Function | Returns | Description |
|----------|---------|-------------|
| `estimate_tokens(text, model)` | `int` | Estimated token count |
| `estimate_tokens_detailed(text, model)` | `TokenEstimate` | Token count + window metadata |
| `estimate_messages(messages, model)` | `int` | Estimated token count for chat list |
| `estimate_messages_detailed(messages, model)` | `TokenEstimate` | Token count + window metadata |
| `fits_in_context(text, model, headroom=0)` | `bool` | True if text fits |
| `context_window_size(model)` | `int` | Context window size in tokens |

### `TokenEstimate` fields and helpers

```python
est.tokens       # int  — estimated token count
est.model        # str  — model name as supplied
est.family       # str  — resolved family key
est.window_size  # int  — context window size
est.utilization  # float — tokens / window_size
est.fits(headroom=0)      # bool  — fits within window
est.remaining(headroom=0) # int   — tokens left
```

## Design notes

- **Estimates round up.** Ceiling division ensures the count never
  under-reports, which prevents silent context-window violations.
- **Unrecognised model names** emit a `UserWarning` and fall back to
  GPT-4 ratios (4.0 chars/token, 8 192-token window).
- **Non-English text and code** will deviate from the English-prose
  calibration — treat estimates as upper bounds in those cases.
- Exact counts require the model's actual tokenizer (`tiktoken` for
  OpenAI, `sentencepiece` for most open-source models).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
