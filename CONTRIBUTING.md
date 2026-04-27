# Contributing to tokenfit

Thank you for considering a contribution!  This document covers the development
workflow, coding conventions, and how to submit changes.

---

## Development setup

```bash
# Clone the repository
git clone https://github.com/vdeshmukh203/tokenfit
cd tokenfit

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest
```

---

## Project layout

```
src/tokenfit/
    __init__.py   — public API and model tables
    gui.py        — tkinter GUI
    __main__.py   — entry point for `python -m tokenfit`
tests/
    test_tokenfit.py
```

The entire library is intentionally small and dependency-free.  Please keep
new code within these constraints.

---

## Coding conventions

- Python 3.9+ compatible syntax
- `from __future__ import annotations` at the top of every module
- Type annotations on all public functions
- No runtime dependencies beyond the standard library
- All estimates must round **up** (never under-report)
- Comments only when the *why* is non-obvious

---

## Adding a new model

1. Add an entry to `_RATIOS` in `src/tokenfit/__init__.py` with the
   approximate characters-per-token ratio (calibrated on English prose).
2. Add a corresponding entry to `_WINDOWS` with the published context-window
   size in tokens.
3. If the model belongs to a new vendor family, add a case to
   `_overhead_family` and a matching entry to `_MESSAGE_OVERHEAD`.
4. Add the model name to the `test_list_models_contains_known_models` test.
5. Cite the official documentation as a comment when the window size is
   non-obvious.

---

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes and ensure `pytest` passes.
3. Open a pull request with a clear description of what changed and why.

Bug reports and feature requests are welcome via
[GitHub Issues](https://github.com/vdeshmukh203/tokenfit/issues).

---

## Code of conduct

Be respectful and constructive.  Harassment of any kind will not be tolerated.
