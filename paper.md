---
title: 'tokenfit: offline heuristic token-count estimation for LLM context windows'
tags:
  - Python
  - large language models
  - tokenization
  - natural language processing
  - developer tools
authors:
  - name: Vaibhav Deshmukh
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-04-27
bibliography: paper.bib
---

# Summary

Large language models (LLMs) such as GPT-4 [@openai2023gpt4], Claude
[@anthropic2024claude], and Gemini [@google2024gemini] impose hard limits on
the number of tokens that can be processed in a single request, commonly called
the *context window*.  Developers who build applications on top of these models
must therefore manage token budgets carefully: submitting a request that exceeds
the context window results in an API error, truncated output, or unnecessary
cost.

`tokenfit` is a pure-Python library that provides **offline, heuristic token
count estimates** without requiring API calls or loading model-specific
tokenizer binaries.  It uses calibrated character-to-token ratios for each LLM
family to produce estimates that are guaranteed to round up, never
under-reporting the true count.  The library exposes a minimal public API of
five functions and includes an interactive GUI for exploratory use.

# Statement of Need

Exact token counts require the tokenizer for a specific model.  For OpenAI
models this means loading `tiktoken` [@tiktoken], which in turn requires
downloading model-specific byte-pair-encoding (BPE) merge tables.  For
Anthropic and Google models there is no publicly distributed tokenizer at all.
In practice, developers working across multiple providers must either call a
count-tokens API endpoint (incurring latency and cost) or integrate multiple
vendor-specific libraries.

`tokenfit` fills the gap between "no check at all" and "exact count": it
provides a fast, dependency-free upper bound that is suitable for pre-flight
budget checks.  Typical use cases include:

- **Retrieval-augmented generation (RAG)** pipelines that must decide how many
  retrieved chunks can be injected into a prompt before dispatching the request.
- **Batch processing** jobs that classify documents by estimated token length
  to route them to appropriately sized models.
- **Interactive development** where a developer wants immediate feedback on
  whether a prompt draft fits within a target window without waiting for an API
  round-trip.
- **Cost estimation** before committing to an LLM call.

# Implementation

`tokenfit` is implemented as a single Python module (`src/tokenfit/__init__.py`)
with no runtime dependencies beyond the standard library.  The core algorithm
is intentionally simple:

$$\hat{t}(s, m) = \left\lceil \frac{|s|}{r_m} \right\rceil$$

where $|s|$ is the length of input string $s$ in Unicode code points and $r_m$
is a per-model-family ratio (characters per token) determined empirically from
representative English prose.

| Model family | $r_m$ | Context window |
|---|---|---|
| GPT-3.5 | 4.0 | 16,385 |
| GPT-4, GPT-4-turbo | 4.0 | 8,192 / 128,000 |
| GPT-4o | 3.8 | 128,000 |
| Claude 3 / 3.5 / 4 | 3.5 | 200,000 |
| Gemini 1.5-pro, 2.0-flash | 4.0 | 1,048,576 |

Model variants are matched by **longest-prefix**, so versioned names such as
`claude-3.5-sonnet-20241022` resolve to the correct family automatically.

For chat-style message lists the function `estimate_messages` adds a small
per-message overhead (3–5 tokens depending on family) to account for role
markers and message separators, mirroring the documented overhead for OpenAI
chat completions [@openai2023chatformat].

The library also ships a `tkinter` GUI (`tokenfit-gui`) that provides real-time
estimation as the user types, with a colour-coded progress bar and an overflow
indicator.

# Limitations

Character-to-token ratios are calibrated for English prose.  Accuracy degrades
for:

- **Code** — typically 2–3 characters per token, so estimates will be
  conservative (too low a token count).
- **Non-Latin scripts** (CJK, Arabic, Hebrew) — often 1–2 bytes per character
  but multiple bytes per Unicode code point, leading to large underestimates.
- **Highly structured data** (JSON, YAML, XML) — the ratio depends heavily on
  field names and nesting depth.

Users who need exact counts for these content types should use the vendor
tokenizer (`tiktoken` for OpenAI, the Anthropic Tokenizer API, or the Vertex
AI tokenizer for Google models).

# Related Software

- **tiktoken** [@tiktoken] — OpenAI's exact BPE tokenizer; requires model
  downloads and is OpenAI-specific.
- **transformers** [@wolf2020transformers] — includes tokenizers for many
  open-source models but is a large dependency.
- **anthropic-tokenizer** — Anthropic's unofficial community tokenizer; depends
  on `sentencepiece` [@kudo2018sentencepiece].

`tokenfit` occupies a complementary niche: it is the only cross-vendor,
zero-dependency solution for token *estimation* rather than exact counting.

# Acknowledgements

The author thanks the open-source community for feedback and the maintainers of
the model documentation used to calibrate the token ratios.

# References
