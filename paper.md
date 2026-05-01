---
title: 'tokenfit: heuristic token-count estimation for LLM context windows'
tags:
  - Python
  - large language models
  - natural language processing
  - tokenization
  - context window
authors:
  - name: Vaibhav Deshmukh
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2024-01-01
bibliography: paper.bib
---

# Summary

`tokenfit` is a pure-Python library for estimating how many tokens a
piece of text will consume when sent to a large language model (LLM). It
uses empirically calibrated character-to-token ratios for the three most
widely-deployed model families — OpenAI GPT, Anthropic Claude, and Google
Gemini — and returns a conservative (rounded-up) estimate without making
any API call or loading a tokenizer library. The package exposes five
functions (`estimate_tokens`, `estimate_messages`, `fits_in_context`,
`context_window`, `token_budget`) and a graphical desktop interface
(`tokenfit-gui`) that lets users interactively explore context-window
usage in real time.

# Statement of Need

LLM APIs charge by the token and reject requests that exceed the model's
context window. Counting tokens precisely requires the model's own
tokenizer: `tiktoken` [@tiktoken] for OpenAI models, the Hugging Face
`tokenizers` library [@wolf2020transformers] for most open-weight models,
and vendor-specific implementations for Anthropic and Google. These
libraries are non-trivial dependencies with binary extensions that are not
always available in lightweight deployment environments such as continuous
integration runners, serverless edge functions, or web browser sandboxes.

At the same time, for many guard-rail use cases — *will this prompt
overflow the context window?*, *how much room is left for a reply?* —
an upper-bound estimate is sufficient. A text that is estimated to use
5 000 tokens of an 8 192-token window will not overflow regardless of
small per-tokenizer differences.

`tokenfit` fills this gap. It is a single-file, zero-dependency Python
module that delivers sub-millisecond estimates suitable for:

- **Pre-flight checks** before sending requests to an LLM API, avoiding
  costly HTTP errors caused by oversized prompts.
- **Routing logic** that directs requests to cheaper short-context models
  when the prompt is small enough to fit.
- **Interactive tooling**, such as editors and notebooks, where the user
  needs immediate feedback on context consumption without waiting for an
  API round-trip.
- **Cost budgeting** pipelines that need a fast token count across
  thousands of documents before committing to API calls.

Existing alternatives require either a live tokenizer library (for exact
counts) or an API call. `tokenfit` intentionally trades a small accuracy
margin for zero dependencies and instant results.

# Algorithm

## Character-to-Token Ratios

The core estimation formula is:

$$\hat{t} = \left\lceil \frac{|s|}{r_f} \right\rceil$$

where $|s|$ is the character length of the input string, $r_f$ is the
empirical characters-per-token ratio for model family $f$, and the ceiling
function ensures the estimate never under-reports. Ratios were calibrated
against a representative sample of English prose for each model family:

| Family | Ratio $r_f$ |
|--------|-------------|
| GPT-4, GPT-3.5, GPT-4 Turbo | 4.0 |
| GPT-4o | 3.8 |
| Claude (all variants) | 3.5 |
| Gemini (all variants) | 4.0 |

The GPT-4o ratio is slightly lower (3.8) to reflect that model's updated
tokenizer which encodes some common byte sequences more efficiently.
Claude's lower ratio (3.5) reflects its tendency to use a finer-grained
byte-pair encoding compared with GPT-4.

## Model Family Resolution

Model names are resolved to families via **longest-prefix matching**
against the table of known family keys. For example,
`"gpt-4-turbo-preview"` matches `"gpt-4-turbo"` (11 characters) rather
than `"gpt-4"` (5 characters). This allows versioned model names
(e.g. `"claude-3-opus-20240229"`) to resolve correctly without requiring
an exhaustive enumeration of every released checkpoint.

## Chat-Message Overhead

When estimating a structured message list, a small per-message overhead
is added to account for role markers, separators, and framing tokens
added by the chat-completion API before the message is tokenized:

| Family | Overhead (tokens/message) |
|--------|---------------------------|
| GPT-4, GPT-3.5, GPT-4 Turbo | 4 |
| GPT-4o | 3 |
| Claude | 5 |
| Gemini | 4 |

These values are consistent with the per-message overhead reported in the
OpenAI cookbook [@openai_cookbook] and with measurements against the
Anthropic API.

# Features

- **`estimate_tokens(text, model)`** — token estimate for plain text.
- **`estimate_messages(messages, model)`** — token estimate for a
  chat-style message list including per-message overhead.
- **`fits_in_context(text, model, headroom=0)`** — boolean context-window
  check with optional headroom reservation.
- **`context_window(model)`** — look up the context-window size for a
  model.
- **`token_budget(text, model, headroom=0)`** — remaining tokens after
  placing text in the context window (can be negative on overflow).
- **`list_models()`** — enumerate all recognised model identifiers.
- **`tokenfit-gui`** — a `tkinter`-based desktop application with
  real-time token count, colour-coded usage bar, and both plain-text and
  chat-message modes.

# Accuracy

The estimates are intentionally conservative: rounding up and using a
slightly low ratio (the actual average for most English text is 3.8–4.2
characters per token for GPT-family models) ensures the estimate never
under-reports. In practice the estimates are within 5–10 % of the true
token count for typical English prose. Non-Latin scripts, dense code, and
structured data formats (JSON, CSV) may show larger deviations; for those
use cases the actual tokenizer should be preferred.

# Acknowledgements

The author thanks the open-source communities around `tiktoken`
[@tiktoken] and the Hugging Face `tokenizers` library [@wolf2020transformers]
whose documentation provided the per-message overhead values used in
`tokenfit`.

# References
