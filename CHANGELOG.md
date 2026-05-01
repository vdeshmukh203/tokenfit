# Changelog

All notable changes to `tokenfit` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] - 2024-01-01

### Added
- `context_window(model)` — look up a model's context-window size in tokens.
- `token_budget(text, model, headroom=0)` — remaining tokens after placing
  text in the context (negative on overflow).
- `list_models()` — sorted list of all recognised model identifiers.
- `__version__` module attribute (`"0.2.0"`).
- `tokenfit-gui` console script — a `tkinter`-based desktop application with
  real-time token count, colour-coded context-window usage bar, and
  plain-text / chat-message modes.
- JOSS submission files: `paper.md`, `paper.bib`, `CITATION.cff`.
- `CONTRIBUTING.md` with guidelines for adding new model ratios.
- `CHANGELOG.md`.
- Extended test suite: 63 tests covering all public functions, parametrized
  model table, boundary conditions, and prefix-matching behaviour.

### Fixed
- `_overhead_family` now uses longest-prefix matching (consistent with
  `_family`), so model names like `gpt-4-turbo-preview` correctly resolve to
  the `gpt-4-turbo` overhead bucket instead of the coarser `gpt-4` bucket.
- `fits_in_context` delegates window lookup to `context_window()`, removing
  duplicated fallback logic.

### Changed
- `pyproject.toml`: added `[project.optional-dependencies] dev`, console
  script entry point, `Bug Tracker` URL, and `[tool.pytest.ini_options]`.
- `README.md`: expanded with statement of need, full API reference, supported
  models table, GUI section, and accuracy notes.

---

## [0.1.0] - 2024-01-01

### Added
- Initial release with `estimate_tokens`, `estimate_messages`, and
  `fits_in_context`.
- Character-to-token ratios and context-window sizes for GPT-3.5, GPT-4,
  GPT-4o, GPT-4 Turbo, Claude 3 family, and Gemini family.
- 16-test suite.
- MIT license.
