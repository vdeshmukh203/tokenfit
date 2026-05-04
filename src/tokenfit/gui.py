"""Graphical user interface for tokenfit.

Launch with::

    python -m tokenfit --gui
    # or directly:
    python -m tokenfit.gui
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from tokenfit import (
    estimate_tokens,
    fits_in_context,
    get_context_window,
    list_models,
)


class _App(tk.Tk):
    """Main tokenfit GUI window."""

    _PAD = 8

    def __init__(self) -> None:
        super().__init__()
        self.title("tokenfit — Token Counter")
        self.minsize(640, 500)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._setup_styles()
        self._build_controls()
        self._build_text_area()
        self._build_results()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        # Use a cross-platform theme that supports colour overrides.
        for t in ("clam", "alt", "default"):
            if t in style.theme_names():
                style.theme_use(t)
                break
        style.configure("Low.Horizontal.TProgressbar", background="#4caf50")
        style.configure("Mid.Horizontal.TProgressbar", background="#f9a825")
        style.configure("High.Horizontal.TProgressbar", background="#e53935")

    def _build_controls(self) -> None:
        p = self._PAD
        ctrl = ttk.Frame(self, padding=p)
        ctrl.grid(row=0, column=0, sticky="ew")

        ttk.Label(ctrl, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            ctrl,
            textvariable=self._model_var,
            values=list_models(),
            width=26,
        )
        model_cb.grid(row=0, column=1, sticky="w", padx=(0, p * 2))

        ttk.Label(ctrl, text="Headroom (tokens):").grid(
            row=0, column=2, sticky="w", padx=(0, 4)
        )
        self._headroom_var = tk.StringVar(value="0")
        ttk.Spinbox(
            ctrl,
            textvariable=self._headroom_var,
            from_=0,
            to=2_000_000,
            increment=100,
            width=10,
        ).grid(row=0, column=3, sticky="w")

        self._model_var.trace_add("write", lambda *_: self._refresh())
        self._headroom_var.trace_add("write", lambda *_: self._refresh())

    def _build_text_area(self) -> None:
        p = self._PAD
        tf = ttk.LabelFrame(self, text="Input Text", padding=4)
        tf.grid(row=1, column=0, sticky="nsew", padx=p, pady=(4, 4))
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self._text = tk.Text(
            tf, wrap="word", undo=True, font=("TkFixedFont", 11)
        )
        sb = ttk.Scrollbar(tf, command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)
        self._text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self._text.bind("<<Modified>>", self._on_modified)

    def _build_results(self) -> None:
        p = self._PAD
        rf = ttk.LabelFrame(self, text="Results", padding=p)
        rf.grid(row=2, column=0, sticky="ew", padx=p, pady=(0, p))
        rf.columnconfigure(1, weight=1)

        rows = [
            "Characters:",
            "Estimated tokens:",
            "Context window:",
            "Usage:",
            "Status:",
        ]
        for i, label in enumerate(rows):
            ttk.Label(rf, text=label).grid(
                row=i, column=0, sticky="w", padx=(0, p), pady=2
            )

        self._char_lbl = ttk.Label(rf, text="0")
        self._char_lbl.grid(row=0, column=1, sticky="w")

        self._tok_lbl = ttk.Label(
            rf, text="0", font=("TkDefaultFont", 10, "bold")
        )
        self._tok_lbl.grid(row=1, column=1, sticky="w")

        self._win_lbl = ttk.Label(rf, text="—")
        self._win_lbl.grid(row=2, column=1, sticky="w")

        pf = ttk.Frame(rf)
        pf.grid(row=3, column=1, sticky="ew")
        pf.columnconfigure(0, weight=1)
        self._pbar = ttk.Progressbar(
            pf, maximum=100, mode="determinate", length=300
        )
        self._pbar.grid(row=0, column=0, sticky="ew")
        self._pct_lbl = ttk.Label(pf, text="0.0%", width=7, anchor="e")
        self._pct_lbl.grid(row=0, column=1, padx=(p, 0))

        self._status_lbl = ttk.Label(
            rf, text="—", font=("TkDefaultFont", 10, "bold")
        )
        self._status_lbl.grid(row=4, column=1, sticky="w")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_modified(self, _event: Any) -> None:
        if self._text.edit_modified():
            self._refresh()
            self._text.edit_modified(False)

    def _refresh(self) -> None:
        """Recompute all estimates and update the results panel."""
        text = self._text.get("1.0", "end-1c")
        model = self._model_var.get().strip() or "gpt-4"
        try:
            headroom = max(0, int(self._headroom_var.get() or 0))
        except ValueError:
            headroom = 0

        chars = len(text)
        tokens = estimate_tokens(text, model)
        window = get_context_window(model)
        used = tokens + headroom
        fits = used <= window
        remaining = window - used
        pct = min(100.0, used / window * 100.0) if window > 0 else 100.0

        # Labels
        self._char_lbl.config(text=f"{chars:,}")
        self._tok_lbl.config(text=f"{tokens:,}")
        self._win_lbl.config(text=f"{window:,} tokens")
        self._pct_lbl.config(text=f"{pct:.1f}%")

        # Progress bar with colour coding
        self._pbar["value"] = pct
        if pct < 70:
            bar_style = "Low.Horizontal.TProgressbar"
        elif pct < 90:
            bar_style = "Mid.Horizontal.TProgressbar"
        else:
            bar_style = "High.Horizontal.TProgressbar"
        self._pbar.configure(style=bar_style)

        # Status
        if fits:
            self._status_lbl.config(
                text=f"✓ Fits in context  ({remaining:,} tokens remaining)",
                foreground="#2e7d32",
            )
        else:
            over = -remaining
            self._status_lbl.config(
                text=f"✗ Exceeds context window by {over:,} tokens",
                foreground="#c62828",
            )


def main() -> None:
    """Launch the tokenfit GUI application."""
    app = _App()
    app.mainloop()


if __name__ == "__main__":
    main()
