---
title: 'tokenfit: heuristic token-count estimation for LLM context windows'
tags:
  - Python
  - large language models
  - natural language processing
  - token estimation
  - context window
authors:
  - name: Vaibhav Deshmukh
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-05-05
bibliography: paper.bib
---

# Summary

`tokenfit` is a lightweight Python library that estimates the number of tokens a
text or chat conversation will consume when processed by a large language model
(LLM), without making any API calls and without requiring heavyweight external
dependencies.  It uses pre-calibrated character-to-token ratios per model family
to return conservative (always rounded-up) estimates in O(n) time, where n is
the character count of the input.

# Statement of need

Modern LLM APIs—including those provided by OpenAI, Anthropic, and Google—impose
hard limits on the number of tokens that can appear in a single request.
Exceeding these limits results in a rejected call, so developers frequently need
to check whether a prompt or conversation *fits* in the context window before
dispatching the request.

The canonical solution is to run the real tokenizer bundled with each model
family.  OpenAI's `tiktoken` [@tiktoken] and Hugging Face's `tokenizers`
[@tokenizers] library give exact counts, but they:

- require installing non-trivial binary wheels,
- may download model-specific vocabulary files from a remote server, and
- add measurable latency in tight loops (e.g. streaming pipelines that inspect
  every incremental chunk).

For many practical use cases—budget checks, pre-flight guards, logging, UI
indicators—a fast *approximation* is sufficient.  `tokenfit` fills this niche:
it is pure Python, has zero runtime dependencies, and produces estimates in
microseconds.  Estimates always round up so the library never under-reports,
making it conservative by design.

A Tkinter desktop GUI ships with the package, enabling interactive exploration of
token counts and context-window utilisation without writing any code.

# Design and implementation

## Character-to-token ratios

English prose encoded with modern byte-pair-encoding (BPE) tokenizers yields
roughly 3.5–4.0 characters per token, depending on vocabulary size.  `tokenfit`
stores one empirically calibrated ratio per model family in the `_RATIOS` table:

| Family | Ratio (chars/token) |
|--------|--------------------:|
| GPT-3.5, GPT-4, GPT-4-Turbo | 4.0 |
| GPT-4o | 3.8 |
| Claude 3 / 3.5 / Sonnet 4 | 3.5 |
| Gemini family | 4.0 |

A ratio of 3.5 is used for all Claude models because Anthropic's tokenizer
compresses text more aggressively than GPT-4's, producing more tokens per
character of English prose in the tested sample [@anthropictokenizer].

## Model-name resolution

Model names are resolved to a family via longest-prefix matching (case-insensitive)
against the keys of `_RATIOS`.  This means that version-specific names such as
`"gpt-4-turbo-preview"` automatically resolve to the `gpt-4-turbo` family
without requiring an exhaustive enumeration of every release name.  Unknown names
fall back to GPT-4 defaults.

## Context-window table

The `_WINDOWS` table maps each family to its published maximum context-window
size in tokens.  `fits_in_context` uses this table to determine whether an
estimated token count (plus an optional response-headroom argument) exceeds the
limit.

## Per-message overhead

Chat-completion APIs prepend role markers and separators to each message.
`estimate_messages` adds a small constant overhead per message (3–5 tokens,
depending on the API family) to account for this framing cost.

## Public API

```python
from tokenfit import (
    estimate_tokens,    # int – tokens in a string
    estimate_messages,  # int – tokens in a messages list
    fits_in_context,    # bool – does text fit with optional headroom?
    context_window,     # int – window size for a model
    supported_models,   # list[str] – all recognised family prefixes
)
```

All public functions are documented with NumPy-style docstrings and include
executable examples in the doctest format.

## GUI

The bundled Tkinter application (`tokenfit-gui` or `python -m tokenfit`) exposes
a two-tab interface:

- **Text tab** — live token count, window utilisation, and a colour-coded
  "fits in context" indicator that updates as the user types.
- **Chat messages tab** — a turn-by-turn conversation builder with cumulative
  token estimates.

# Limitations

- Ratios are calibrated on English prose.  Code and non-English text may yield
  estimates that differ from the true token count by 10–30%.
- The library does not account for special tokens, tool-call markup, or
  model-specific formatting that some APIs inject automatically.
- Exact counts always require the real tokenizer for each model.

# Acknowledgements

The author thanks the open-source community for releasing `tiktoken`
[@tiktoken] and the Hugging Face `tokenizers` library [@tokenizers], whose
documentation and token statistics informed the calibration of the ratios used
in `tokenfit`.

# References
