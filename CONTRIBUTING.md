# Contributing to tokenfit

Thank you for your interest in contributing!  This document describes how to
set up a development environment, run the tests, and submit changes.

## Development setup

```bash
git clone https://github.com/vdeshmukh203/tokenfit.git
cd tokenfit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The test suite requires only `pytest`.  All tests run in under a second on a
modern machine — no network access or compiled extensions are needed.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep the zero-runtime-dependency constraint.  Do **not** add imports outside
  the Python standard library to `src/tokenfit/__init__.py` or
  `src/tokenfit/gui.py`.
- Add type hints for all new public functions.
- Write docstrings in Google style with `Parameters`, `Returns`, and (where
  helpful) `Examples` sections.
- Default to writing no inline comments; add one only when the *why* is
  non-obvious.

## Adding a new model

1. Add an entry to `_RATIOS` in `src/tokenfit/__init__.py` with the
   approximate characters-per-token ratio.
2. Add a matching entry to `_WINDOWS` with the context-window size in tokens.
3. Add the model to `_MESSAGE_OVERHEAD` (or verify the existing coarse-family
   bucket covers it) in `_overhead_family`.
4. Add at least one test to `tests/test_tokenfit.py` that verifies both the
   ratio and window values.
5. Document the change in `CHANGELOG.md` under an `[Unreleased]` section.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, atomic commits.
3. Ensure `pytest` passes with no failures or warnings.
4. Open a pull request against `main` with a short description of the change
   and the motivation for it.

## Reporting bugs

Please open a GitHub issue and include:

- Python version (`python --version`)
- `tokenfit` version (`python -c "import tokenfit; print(tokenfit.__version__)"`)
- A minimal reproducer (input text and model name that triggers the issue)
- The observed output vs. the expected output

## Code of conduct

This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.
Be respectful, constructive, and inclusive in all interactions.
