---
title: 'tokenfit: Heuristic token-count estimation for LLM context windows'
tags:
  - Python
  - large language models
  - natural language processing
  - tokenisation
  - developer tools
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 30 April 2026
bibliography: paper.bib
---

# Summary

Large language models (LLMs) such as GPT-4 [@openai2023gpt4], Claude
[@anthropic2024claude], and Gemini [@google2023gemini] impose hard limits on the
number of tokens a single request may contain.  Developers who build on top of
these models must therefore track token usage continuously — for cost budgeting,
context management, and preventing request failures.

`tokenfit` is a lightweight, pure-Python library that estimates token counts
for text and chat-style message lists without making any API calls or loading
any model artefact.  It uses per-family character-to-token ratios derived from
representative English prose and always rounds up, so reported counts never
under-report.  The library ships with built-in tables covering all major
commercial model families (OpenAI GPT, Claude, Gemini, Llama, Mistral, Cohere
Command) and falls back gracefully for unknown models.

An interactive desktop GUI built with Python's standard-library `tkinter`
toolkit is included, providing real-time token estimation, a visual context-
window utilisation bar, and a message-builder panel for chat workloads — all
without any external runtime dependencies.

# Statement of need

Accurate token counting in production LLM applications typically relies on the
model vendor's official tokeniser (e.g. `tiktoken` [@tiktoken] for OpenAI
models or the SentencePiece [@kudo2018sentencepiece] vocabulary files used by
many open-weight models).  While exact, these solutions carry significant
drawbacks:

1. **Dependency weight.** Tokeniser libraries such as `tiktoken` require native
   compiled extensions and model-specific vocabulary files that can be tens of
   megabytes in size.
2. **Network round-trips.** Vocabulary files are often downloaded at first use,
   which is unsuitable for air-gapped, embedded, or latency-sensitive
   environments.
3. **Multi-model complexity.** Applications that target several model families
   must integrate multiple tokeniser implementations.

`tokenfit` addresses all three problems by replacing exact tokenisation with a
fast heuristic.  For the vast majority of practical use cases — prompt
pre-screening, context-window budgeting, and user-facing display — estimates
that are accurate to within a few percent are sufficient.  The library requires
only the Python standard library, installs in milliseconds, and covers every
major commercial model family through a single unified API.

Existing alternatives such as `tokencost` [@tokencost] couple token estimation
tightly to cost computation and introduce third-party dependencies.
`tokenfit` focuses exclusively on count estimation and preserves the zero-
dependency constraint.

# Implementation

## Character-to-token ratios

Token counts are estimated by dividing the UTF-8 character count of the input
by a per-family ratio and rounding up:

$$\hat{n} = \left\lceil \frac{|s|}{r_f} \right\rceil$$

where $s$ is the input string and $r_f$ is the characters-per-token ratio for
model family $f$.  Ratios are calibrated against representative English prose
and are held constant within a model family (Table 1).

| Model family | Ratio $r_f$ | Representative models |
|---|---|---|
| GPT-3.5 | 4.0 | `gpt-3.5-turbo` |
| GPT-4 | 4.0 | `gpt-4`, `gpt-4-turbo` |
| GPT-4o / o-series | 3.8 | `gpt-4o`, `o1`, `o3`, `o4-mini` |
| Claude | 3.5 | All Claude 3/4 variants |
| Gemini | 4.0 | All Gemini variants |
| Llama | 3.7 | `llama-2`, `llama-3` |
| Mistral | 3.8 | `mistral`, `mistral-large` |
| Command | 3.8 | `command-r`, `command-r-plus` |

Table 1: Per-family character-to-token ratios used by `tokenfit`.

## Model family resolution

Model names are resolved to family keys via longest-prefix matching against the
built-in table.  This means that version-dated model identifiers (e.g.
`claude-3.5-sonnet-20241022`) resolve correctly to their family without
requiring a full list of every released model variant.

## Chat-format overhead

For chat-style workloads, `estimate_messages` adds a per-message overhead to
account for role markers and separator tokens introduced by the model's chat
template.  Overhead values are sourced from each vendor's published token-
accounting documentation and the `openai-python` library source code
[@openai2023cookbook].

## Graphical user interface

The optional `tokenfit-gui` desktop application is built with `tkinter`, which
ships with all standard CPython distributions.  The GUI provides:

- a freeform text pane with live token and character count updates;
- a context-window utilisation progress bar that turns amber above 80 % and red
  on overflow;
- a message-builder panel for constructing multi-turn chat histories; and
- a model selector covering all built-in families.

No additional dependencies are required beyond a working `tkinter` installation,
which is present in the vast majority of Python environments.

# Quality assurance

`tokenfit` ships with a pytest [@pytest] test suite covering all public API
functions.  The continuous integration workflow (GitHub Actions) runs the suite
on Python 3.9, 3.10, 3.11, and 3.12 to ensure compatibility across all
currently supported interpreter versions.

# Acknowledgements

The author thanks the open-source NLP community for maintaining the tokeniser
implementations and published documentation that informed the calibration of the
ratio table.

# References
