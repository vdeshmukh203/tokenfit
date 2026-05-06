"""Command-line interface for tokenfit."""
from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import (
    __version__,
    context_window_size,
    estimate_messages_detailed,
    estimate_tokens_detailed,
    fits_in_context,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenfit",
        description="Estimate token counts for LLM models without API calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  tokenfit estimate 'Hello, world!' --model gpt-4\n"
            "  echo 'my doc' | tokenfit estimate - --model claude-3.5-sonnet\n"
            "  tokenfit fits myfile.txt --model gpt-4 --headroom 1000\n"
            "  tokenfit info gpt-4o\n"
            "  tokenfit gui\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ---- estimate --------------------------------------------------------
    est = sub.add_parser(
        "estimate",
        help="estimate token count for text",
        description="Print the estimated token count for the given text.",
    )
    est.add_argument(
        "text",
        help="text to estimate; use '-' to read from stdin, or a file path",
    )
    est.add_argument("--model", default="gpt-4", metavar="MODEL", help="model name (default: gpt-4)")
    est.add_argument("--json", action="store_true", help="output as JSON")

    # ---- fits ------------------------------------------------------------
    fts = sub.add_parser(
        "fits",
        help="check whether text fits in the model's context window",
        description="Exit 0 if text fits, exit 1 if it exceeds the context window.",
    )
    fts.add_argument(
        "text",
        help="text to check; use '-' to read from stdin, or a file path",
    )
    fts.add_argument("--model", default="gpt-4", metavar="MODEL", help="model name (default: gpt-4)")
    fts.add_argument(
        "--headroom",
        type=int,
        default=0,
        metavar="N",
        help="reserve N tokens for the model's response (default: 0)",
    )
    fts.add_argument("--json", action="store_true", help="output as JSON")

    # ---- info ------------------------------------------------------------
    info = sub.add_parser(
        "info",
        help="show context window size and ratio for a model",
    )
    info.add_argument("model", help="model name")
    info.add_argument("--json", action="store_true", help="output as JSON")

    # ---- gui -------------------------------------------------------------
    sub.add_parser(
        "gui",
        help="launch the graphical user interface",
    )

    return parser


def _read_text(source: str) -> str:
    """Read text from a literal string, stdin ('-'), or a file path."""
    if source == "-":
        return sys.stdin.read()
    # If the string looks like a path that exists, read it as a file.
    import os

    if os.path.exists(source):
        with open(source, encoding="utf-8") as fh:
            return fh.read()
    return source


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("always")

        if args.command == "gui":
            from .gui import main as _gui_main

            _gui_main()
            return 0

        if args.command == "info":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                window = context_window_size(args.model)
                from . import _family, _RATIOS

                fam = _family(args.model)
                ratio = _RATIOS.get(fam, 4.0)
            for w in caught:
                print(f"warning: {w.message}", file=sys.stderr)
            if args.json:
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "family": fam,
                            "window_size": window,
                            "chars_per_token": ratio,
                        }
                    )
                )
            else:
                print(f"model        : {args.model}")
                print(f"family       : {fam}")
                print(f"window       : {window:,} tokens")
                print(f"ratio        : {ratio} chars/token")
            return 0

        text = _read_text(args.text)

        if args.command == "estimate":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                est = estimate_tokens_detailed(text, args.model)
            for w in caught:
                print(f"warning: {w.message}", file=sys.stderr)
            if args.json:
                print(
                    json.dumps(
                        {
                            "tokens": est.tokens,
                            "model": est.model,
                            "family": est.family,
                            "window_size": est.window_size,
                            "utilization": round(est.utilization, 4),
                        }
                    )
                )
            else:
                print(
                    f"{est.tokens:,} tokens  "
                    f"({est.utilization * 100:.1f}% of {est.window_size:,}-token window)"
                )
            return 0

        if args.command == "fits":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ok = fits_in_context(text, args.model, headroom=args.headroom)
                est = estimate_tokens_detailed(text, args.model)
            for w in caught:
                print(f"warning: {w.message}", file=sys.stderr)
            total = est.tokens + max(0, args.headroom)
            if args.json:
                print(
                    json.dumps(
                        {
                            "fits": ok,
                            "tokens": est.tokens,
                            "headroom": args.headroom,
                            "total_used": total,
                            "window_size": est.window_size,
                            "model": est.model,
                        }
                    )
                )
            else:
                status = "YES" if ok else "NO"
                print(
                    f"Fits: {status}  "
                    f"({total:,} / {est.window_size:,} tokens used"
                    + (f", {args.headroom:,} headroom)" if args.headroom else ")")
                )
            return 0 if ok else 1

    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
