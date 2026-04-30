# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-30

### Added

- **New public functions** `context_window(model)` and `list_models()`.
- **`__version__`** string exported from the package root.
- **GUI** (`tokenfit-gui` entry point) — a `tkinter`-based desktop application
  with live token estimation, context-window utilisation bar, and a
  chat-message builder panel.
- Extended model support:
  - OpenAI reasoning models: `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`.
  - OpenAI: `gpt-4o-mini`.
  - Anthropic Claude 4: `claude-opus-4`, `claude-sonnet-4`, `claude-haiku-4`.
  - Anthropic Claude 3.5: `claude-3.5-haiku`.
  - Anthropic Claude 3: `claude-3-haiku`.
  - Google Gemini 2.5: `gemini-2.5-pro`, `gemini-2.5-flash`.
  - Google Gemini 1.5: `gemini-1.5-flash`.
  - Meta Llama: `llama-2`, `llama-3`.
  - Mistral: `mistral`, `mistral-large`.
  - Cohere: `command`, `command-r`, `command-r-plus`.
- JOSS paper (`paper.md` + `paper.bib`).
- `CITATION.cff` for software citation.
- `CONTRIBUTING.md` community guidelines.
- Extended test suite covering all new public functions and additional edge
  cases (55 tests total).
- Model lookup table pre-sorted at import time for consistent O(n) worst-case
  performance.

### Changed

- Docstrings rewritten to Google / NumPy-compatible format with `Parameters`,
  `Returns`, and `Examples` sections throughout.
- `_SORTED_RATIO_KEYS` cached at module load instead of re-sorted on every
  `_family()` call.
- `_overhead_family` extended to cover Llama, Mistral, Command, and o-series
  models.

## [0.1.0] - 2026-03-01

### Added

- Initial release with `estimate_tokens`, `estimate_messages`, and
  `fits_in_context`.
- Built-in model tables for GPT-3.5/4/4o/4-turbo, Claude 3, and Gemini families.
- CI workflow (GitHub Actions) for Python 3.9 – 3.12.
