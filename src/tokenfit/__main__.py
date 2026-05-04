"""Entry point for ``python -m tokenfit``.

Examples::

    # estimate tokens for a string
    python -m tokenfit "Hello, world!" --model gpt-4

    # pipe text from stdin
    cat document.txt | python -m tokenfit --model claude-3-opus

    # list supported models
    python -m tokenfit --list-models

    # launch the GUI
    python -m tokenfit --gui
"""
from __future__ import annotations

import argparse
import sys

import tokenfit
from tokenfit import (
    estimate_tokens,
    fits_in_context,
    get_context_window,
    list_models,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenfit",
        description="Estimate token counts for LLM context windows without API calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit code is 0 when the text fits in the context window, 1 when it\n"
            "does not (or on error). This makes tokenfit composable in shell scripts."
        ),
    )
    p.add_argument(
        "text",
        nargs="?",
        help="Text to estimate (reads from stdin when omitted).",
    )
    p.add_argument(
        "--model",
        default="gpt-4",
        metavar="MODEL",
        help="LLM model name (default: gpt-4). Use --list-models to see all options.",
    )
    p.add_argument(
        "--headroom",
        type=int,
        default=0,
        metavar="N",
        help="Reserve N tokens for the model response (default: 0).",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print all supported model names with context window sizes and exit.",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {tokenfit.__version__}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the tokenfit CLI. Returns an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        header = f"{'Model':<34}  {'Context window':>14}"
        print(header)
        print("-" * len(header))
        for name in list_models():
            win = get_context_window(name)
            print(f"  {name:<32}  {win:>12,}")
        return 0

    if args.gui:
        try:
            from tokenfit.gui import main as gui_main
        except ImportError as exc:
            print(f"error: GUI unavailable ({exc})", file=sys.stderr)
            return 1
        gui_main()
        return 0

    text = args.text if args.text is not None else sys.stdin.read()
    model = args.model
    tokens = estimate_tokens(text, model)
    window = get_context_window(model)
    fits = fits_in_context(text, model, args.headroom)
    pct = tokens / window * 100 if window > 0 else float("inf")

    print(f"model      {model}")
    print(f"tokens     {tokens:,}")
    print(f"window     {window:,}")
    print(f"usage      {pct:.1f}%")
    if args.headroom:
        print(f"headroom   {args.headroom:,}")
    print(f"fits       {'yes' if fits else 'no'}")

    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
