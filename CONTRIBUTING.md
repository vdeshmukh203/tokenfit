# Contributing to tokenfit

Thank you for your interest in contributing!  The sections below explain how to report bugs, request features, and submit code changes.

## Reporting bugs

Open an issue at <https://github.com/vdeshmukh203/tokenfit/issues> and include:

- Python version (`python --version`)
- tokenfit version (`python -c "import tokenfit; print(tokenfit.__version__)"`)
- A minimal, self-contained code snippet that reproduces the problem
- The output you expected and what you actually got

## Requesting features

Open an issue describing the use case, the proposed API (if applicable), and any alternatives you considered.

## Setting up for development

```bash
git clone https://github.com/vdeshmukh203/tokenfit.git
cd tokenfit
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The CI matrix tests Python 3.9 through 3.12; please ensure your changes pass on all versions.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes.  Keep commits focused and atomic.
3. Add or update tests so the new behaviour is covered.
4. Ensure all tests pass: `pytest`.
5. Open a pull request against `main` with a clear description of the change and the motivation.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use `from __future__ import annotations` for forward-compatible type hints.
- Write docstrings in NumPy style for public functions.
- Keep the library dependency-free (standard library only) unless there is a strong reason.
- Comments should explain *why*, not *what*.

## Adding a new model

1. Add the model's character-to-token ratio to `_RATIOS` in `src/tokenfit/__init__.py`.
2. Add the context-window size to `_WINDOWS`.
3. Update `_overhead_family` if the model needs a distinct per-message overhead bucket.
4. Add the model to the supported-models table in `README.md`.
5. Add at least one parametrised test case in `tests/test_tokenfit.py`.

## License

By submitting a pull request you agree that your contribution will be licensed under the MIT License.
