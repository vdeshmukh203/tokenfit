# Contributing to tokenfit

Thank you for your interest in contributing! Contributions of all kinds are
welcome: bug reports, new model ratios, documentation improvements, and code.

---

## Reporting Issues

Open an issue on GitHub with:

- **Bug reports**: a minimal reproducible example and the actual vs. expected
  output.
- **New model ratios**: the model name and a source for the context-window
  size and character-to-token ratio (e.g. the provider's documentation or a
  measurement against a reference corpus).
- **Feature requests**: a clear description of the use case and why the
  existing API cannot address it.

---

## Development Setup

```bash
git clone https://github.com/vdeshmukh203/tokenfit.git
cd tokenfit
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use `from __future__ import annotations` for forward-compatible type hints.
- Public functions must have NumPy-style docstrings with `Parameters`,
  `Returns`, and at least one `Examples` entry.
- Keep runtime dependencies at zero — `tokenfit` must work with the Python
  standard library alone.

---

## Adding a New Model

1. Add an entry to `_RATIOS` in `src/tokenfit/__init__.py` with the
   characters-per-token ratio (calibrate against ≥ 1 000 English sentences).
2. Add a matching entry to `_WINDOWS` with the context-window size from the
   provider's official documentation.
3. If the model has non-standard per-message overhead, add an entry to
   `_MESSAGE_OVERHEAD`.
4. Add parametrized test cases in `tests/test_tokenfit.py` for the new model,
   covering `estimate_tokens`, `context_window`, and `fits_in_context`.
5. Update the supported-models table in `README.md`.

---

## Pull Request Guidelines

- Branch from `main`; use a descriptive branch name
  (e.g. `add-llama3-ratios` or `fix-overhead-lookup`).
- One logical change per PR.
- All CI checks must pass before merge.
- New functionality must include tests; aim for 100 % coverage of new code.

---

## Code of Conduct

Be respectful and constructive. This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/) v2.1.
