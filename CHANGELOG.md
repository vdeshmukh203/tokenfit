# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-04

### Added

- **GUI** — interactive token counter (`python -m tokenfit --gui` or `python -m tokenfit.gui`).
  - Real-time estimates as you type.
  - Colour-coded progress bar (green / amber / red by usage percentage).
  - Model selector and headroom spinner.
- **CLI** — `python -m tokenfit` entry point with `--model`, `--headroom`, `--list-models`, `--gui`, and `--version` flags.
- `get_context_window(model)` — new public function returning context window size in tokens.
- `list_models()` — new public function returning a sorted list of supported model names.
- `__version__` module attribute.
- New model support:
  - OpenAI: `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
  - OpenAI reasoning: `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`
  - Anthropic: `claude-3.5-haiku`, `claude-3-haiku`, `claude-3-5-haiku`, `claude-3-5-sonnet`, `claude-opus-4`, `claude-haiku-4`
  - Google: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-flash`
- Pre-sorted key list (`_SORTED_RATIO_KEYS`) — avoids re-sorting on every `_family()` call.
- `[project.scripts]` entry in `pyproject.toml` so `tokenfit` is available as a shell command after `pip install`.
- Expanded test suite: 53 tests covering all new functions and models.

### Changed

- `estimate_tokens` type annotation updated to `str | None` to reflect accepted `None` input.
- `pyproject.toml` classifiers expanded with `Development Status`, `Intended Audience`, and `Topic` entries.

### Fixed

- `_family()` no longer re-sorts the key list on every invocation.

## [0.1.0] - 2025

### Added

- Initial release.
- `estimate_tokens`, `estimate_messages`, `fits_in_context`.
- Built-in ratios for GPT-3.5/4/4o/4-turbo, Claude 3 family, Gemini family.
- Pure Python, no external dependencies.
- MIT licence.
