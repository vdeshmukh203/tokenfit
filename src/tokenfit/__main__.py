"""Entry point for ``python -m tokenfit``.

With no arguments the graphical interface is launched.  With arguments a
quick CLI estimate is printed instead::

    # launch GUI
    python -m tokenfit

    # estimate tokens for a string
    python -m tokenfit "Hello, world!" --model gpt-4

    # check whether text read from a file fits in context
    python -m tokenfit --file prompt.txt --model claude-3-opus --headroom 1000

    # list supported model families
    python -m tokenfit --list-models
"""
from __future__ import annotations

import argparse
import sys


def _cli(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokenfit",
        description="Estimate token counts for LLM models without API calls.",
    )
    parser.add_argument("text", nargs="?", help="Text to estimate (optional).")
    parser.add_argument(
        "--model", "-m",
        default="gpt-4",
        metavar="MODEL",
        help="Model name or family prefix (default: gpt-4).",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Read text from FILE instead of the positional argument.",
    )
    parser.add_argument(
        "--headroom",
        type=int,
        default=0,
        metavar="N",
        help="Tokens to reserve for the model response (default: 0).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model-family prefixes and exit.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force-launch the graphical interface.",
    )

    ns = parser.parse_args(args)

    if ns.list_models:
        from tokenfit import list_models
        for m in list_models():
            print(m)
        return 0

    if ns.gui or (ns.text is None and ns.file is None):
        try:
            from tokenfit.gui import main as gui_main
            gui_main()
        except Exception as exc:  # pragma: no cover
            print(f"Could not launch GUI: {exc}", file=sys.stderr)
            print("Run with --help for CLI usage.", file=sys.stderr)
            return 1
        return 0

    # Resolve text
    if ns.file:
        try:
            with open(ns.file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(f"Cannot read file: {exc}", file=sys.stderr)
            return 1
    else:
        text = ns.text or ""

    from tokenfit import token_summary
    est = token_summary(text, ns.model, ns.headroom)

    print(f"Model family : {est.model_family}")
    print(f"Characters   : {len(text):,}")
    print(f"Tokens       : ~{est.tokens:,}")
    print(f"Window       : {est.window:,}")
    print(f"Remaining    : {est.remaining:,}")
    print(f"Fits         : {'yes' if est.fits else 'NO — overflows by ' + f'{-est.remaining:,}'}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
