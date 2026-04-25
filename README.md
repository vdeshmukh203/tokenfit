# tokenfit

Estimate token counts for LLM models without API calls.

## Install

```bash
pip install tokenfit
```

## Usage

```python
from tokenfit import estimate_tokens, estimate_messages, fits_in_context

estimate_tokens("Hello, world!", model="gpt-4")
estimate_messages([{"role": "user", "content": "Hi"}], model="claude-3-opus")
fits_in_context("..." * 1000, model="gpt-4", headroom=500)
```

Built-in ratios for GPT-3.5/4/4o/4-turbo, Claude 3 family, Gemini family. No external API calls -- pure Python, standard library only.

Estimates round up so they never under-report. Exact counts require the real tokenizer for each model.

## License

MIT -- see LICENSE.
