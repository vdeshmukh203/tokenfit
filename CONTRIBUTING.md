# Contributing to tokenfit

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/vdeshmukh203/tokenfit.git
cd tokenfit
pip install -e ".[dev]"
```

Install test dependencies:

```bash
pip install pytest
```

## Running tests

```bash
pytest
```

All tests must pass before a pull request can be merged.

## Adding a new model

1. Add an entry to `_RATIOS` in `src/tokenfit/__init__.py` with the
   approximate characters-per-token ratio for English prose.
2. Add a matching entry to `_WINDOWS` with the context-window size in tokens.
3. If the new model has a distinct per-message overhead, update
   `_MESSAGE_OVERHEAD` and `_overhead_family`.
4. Add the model name to the `_KNOWN_MODELS` list in `src/tokenfit/gui.py`
   so it appears in the GUI dropdown.
5. Write a test in `tests/test_tokenfit.py` that checks the window size via
   `estimate_tokens_detailed`.

## Code style

- Pure Python, no external runtime dependencies.
- Follow PEP 8; type-annotate all public functions.
- Docstrings use NumPy parameter-style (Parameters / Returns sections).
- Default to writing no comments — only add one when the *why* is
  non-obvious.

## Pull requests

- Fork the repository and create a feature branch.
- Keep commits focused; one logical change per commit.
- Update `CHANGELOG.md` with a brief description of the change.
- Open a pull request against `main`; CI will run the test matrix
  (Python 3.9–3.12) automatically.

## Reporting issues

Please open an issue at <https://github.com/vdeshmukh203/tokenfit/issues>
and include:

- tokenfit version (`tokenfit --version`)
- Python version and OS
- Minimal reproducible example
