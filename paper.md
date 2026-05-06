---
title: 'tokenfit: heuristic token-count estimation for LLM context windows'
tags:
  - Python
  - large language models
  - natural language processing
  - token counting
  - context window management
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-05-06
bibliography: paper.bib
---

# Summary

Large language models (LLMs) process text as sequences of *tokens*—
sub-word units produced by a model-specific tokenizer.  Every model
imposes a hard limit on the total number of tokens it can process per
request (the *context window*), and most commercial API providers charge
per token consumed.  Knowing how many tokens a piece of text will use
before submitting a request is therefore essential for both correctness
(avoiding context-window overflows) and cost management.

`tokenfit` is a lightweight, pure-Python library for *offline* token-count
estimation.  It covers the major commercial and open-source LLM families
(OpenAI GPT-3.5/4/4o, reasoning models o1/o3, Anthropic Claude 3/3.5/3.7,
Google Gemini 1.5/2.0/2.5, Meta Llama 2/3, Mistral, DeepSeek, and Cohere
Command), requires no external dependencies, and makes no network calls.
The library ships with a command-line interface (CLI) and a cross-platform
graphical user interface (GUI) built on the Python standard library.

# Statement of Need

Exact token counts require the same tokenizer used during model training.
For OpenAI models this is `tiktoken` [@tiktoken], which is a compiled
C extension.  For Anthropic Claude and most open-source models a
SentencePiece model [@sentencepiece] must be loaded at runtime.
Loading these tokenizers imposes a startup cost (100 ms–2 s), adds
non-trivial dependencies (especially in constrained environments such as
serverless functions, WebAssembly, or embedded Python runtimes), and
requires a separate library for each model family.

When exact counts are not critical—for instance, when deciding whether a
document exceeds 80% of a context window before fetching it—a fast
heuristic estimate is sufficient.  `tokenfit` targets this use-case:

* **No external dependencies.** The entire library is standard-library Python.
* **Zero network calls.** Estimates are computed locally from character counts.
* **Broad model coverage.** A single API covers 30+ named model variants across
  seven provider families.
* **Conservative by design.** Estimates always round up (ceiling division) so
  context-window limits are never silently exceeded.

Related tools either require a specific tokenizer to be installed
(e.g., `tiktoken`, `transformers`) or are narrowly scoped to a single
provider.  `tokenfit` complements these tools by serving as a fast,
provider-agnostic pre-check.

# Methods

## Character-to-token ratios

For English prose, the mapping from characters to tokens is approximately
linear.  The slope (characters per token) is determined by the tokenizer
vocabulary: larger vocabularies tend to merge more characters per token.
Based on publicly available benchmarks and tokenizer documentation, the
following family-level ratios are used:

| Family              | Chars / token |
|---------------------|:-------------:|
| GPT-3.5, GPT-4, o1 | 4.0           |
| GPT-4o, Mistral     | 3.8           |
| Claude (all)        | 3.5           |
| Gemini (all)        | 4.0           |
| Llama 3             | 3.6           |

Estimates are produced by ceiling division:

$$\hat{n} = \left\lceil \frac{|\text{text}|_{\text{chars}}}{r_{\text{family}}} \right\rceil$$

where $r_{\text{family}}$ is the family ratio.  The ceiling guarantees
$\hat{n} \geq n_{\text{true}}$ for typical English inputs, so the library
never under-reports token consumption.

## Chat-message overhead

Chat-format APIs wrap each turn with role markers and separator tokens.
`tokenfit` adds a per-message constant (3–5 tokens depending on the model
family) when estimating token counts for lists of chat messages.

## Model family resolution

Model names are resolved to families using longest-prefix matching against
a table of known model identifiers.  This allows the library to handle
versioned names (e.g., `claude-3.5-sonnet-20241022`) without requiring
exact matches.  Unrecognised model names fall back to GPT-4 ratios and
emit a `UserWarning` so callers are alerted.

# Graphical and Command-line Interfaces

`tokenfit` ships two user-facing interfaces in addition to its Python API,
making it accessible to researchers who do not write code.

The **CLI** (`tokenfit estimate`, `tokenfit fits`, `tokenfit info`)
supports plain-text output and machine-readable JSON, accepts input from
stdin or files, and returns a Unix exit code of 0/1 for the `fits`
sub-command to enable shell script integration.

The **GUI** (launched with `tokenfit gui` or `python -m tokenfit gui`)
is a Tkinter window with two tabs—*Plain Text* and *Chat Messages*—and
a live results panel showing the estimated token count, a context-window
utilisation progress bar, and a fit/overflow indicator.  Live estimation
is triggered 300 ms after the last keystroke to avoid unnecessary
redraws.

# Acknowledgements

The character-to-token ratios were calibrated against publicly available
tokenizer documentation from OpenAI [@openai_tokenizer], Anthropic, and
Google DeepMind.

# References
