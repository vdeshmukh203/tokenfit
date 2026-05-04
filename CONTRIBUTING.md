# Contributing to tokenfit

Thank you for taking the time to contribute!

## Development setup

```bash
git clone https://github.com/vdeshmukh203/tokenfit.git
cd tokenfit
pip install -e ".[dev]"   # install in editable mode
pip install pytest         # test runner
```

## Running the tests

```bash
pytest
```

All tests live in `tests/`. The suite must pass on Python 3.9–3.12 before a pull request can be merged.

## Adding a new model

1. Open `src/tokenfit/__init__.py`.
2. Add an entry to **`_RATIOS`** with the model's character-to-token ratio (calibrate against a representative English sample using the model's official tokenizer).
3. Add a matching entry to **`_WINDOWS`** with the context window size from the model's official documentation.
4. If the model family has a distinct per-message overhead, add it to **`_MESSAGE_OVERHEAD`** and update `_overhead_family()` accordingly.
5. Add at least one test in `tests/test_tokenfit.py` that covers the new model key.

## Code style

- Pure Python, standard library only (no new runtime dependencies).
- Type annotations on all public functions using `from __future__ import annotations`.
- No inline comments unless the *why* is non-obvious.
- One-line docstring maximum for private helpers; full NumPy-style docstrings for public API.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Commit your changes with a concise message describing *why*, not just *what*.
3. Open a pull request against `main`. The CI will run the full test matrix automatically.

## Reporting bugs

Please open an issue at <https://github.com/vdeshmukh203/tokenfit/issues> and include:
- Python version and OS
- The model name and text that produced the unexpected result
- The estimated vs. expected token count (if known)

## Code of Conduct

Be respectful and constructive. Harassment of any kind will not be tolerated.
