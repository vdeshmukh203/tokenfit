"""tokenfit GUI – interactive token-count estimator.

Launch with ``python -m tokenfit`` or the ``tokenfit-gui`` command after
installation.  Requires Tkinter, which is bundled with most Python
distributions (package ``python3-tk`` on Debian/Ubuntu).
"""
from __future__ import annotations

import json
import tkinter as tk
import warnings
from tkinter import scrolledtext, ttk

from tokenfit import (
    context_window,
    estimate_messages,
    estimate_tokens,
)

# ── model catalogue ─────────────────────────────────────────────────────────

_ORDERED_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5",
    "claude-3.5-sonnet",
    "claude-sonnet-4",
    "claude-3-opus",
    "claude-3-sonnet",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

_FRIENDLY: dict[str, str] = {
    "gpt-4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "gpt-3.5": "GPT-3.5",
    "claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "claude-sonnet-4": "Claude Sonnet 4",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-pro": "Gemini Pro",
}

_MSG_PLACEHOLDER = (
    '[\n'
    '  {"role": "user", "content": "Hello, how are you?"},\n'
    '  {"role": "assistant", "content": "I\'m doing well, thanks!"}\n'
    ']'
)


def _make_label(key: str) -> str:
    """Build a display string like 'GPT-4o (128K)'."""
    win = context_window(key)
    if win >= 1_000_000:
        tag = f"{round(win / 1_048_576)}M"
    elif win >= 1_000:
        tag = f"{round(win / 1_000)}K"
    else:
        tag = str(win)
    return f"{_FRIENDLY.get(key, key)} ({tag})"


def _validate_int(value: str) -> bool:
    """Tkinter validation callback: accept empty strings and integers."""
    return value == "" or value.lstrip("-").isdigit()


# Pre-build label ↔ key mappings once at import time.
_LABELS: list[str] = [_make_label(k) for k in _ORDERED_MODELS]
_LABEL_TO_KEY: dict[str, str] = dict(zip(_LABELS, _ORDERED_MODELS))
_DEFAULT_LABEL: str = _make_label("gpt-4")


# ── application ──────────────────────────────────────────────────────────────


class TokenFitApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("tokenfit – Token Count Estimator")
        self.minsize(660, 540)
        self.resizable(True, True)
        self._build_ui()
        # Defer first update until widgets have rendered sizes.
        self.after(50, self._update)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # notebook row expands

        # ── top bar: model selector + headroom ──────────────────────────────
        bar = ttk.Frame(self, padding=(8, 8, 8, 4))
        bar.grid(row=0, column=0, sticky="ew")

        ttk.Label(bar, text="Model:").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value=_DEFAULT_LABEL)
        cb = ttk.Combobox(
            bar,
            textvariable=self._model_var,
            values=_LABELS,
            state="readonly",
            width=26,
        )
        cb.grid(row=0, column=1, sticky="w", padx=(4, 24))
        cb.bind("<<ComboboxSelected>>", lambda _: self._update())

        ttk.Label(bar, text="Headroom (tokens):").grid(row=0, column=2, sticky="w")
        self._headroom_var = tk.StringVar(value="0")
        vcmd = (self.register(_validate_int), "%P")
        ttk.Entry(
            bar,
            textvariable=self._headroom_var,
            validate="key",
            validatecommand=vcmd,
            width=9,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        self._headroom_var.trace_add("write", lambda *_: self._update())

        # ── notebook: text input / messages JSON ─────────────────────────────
        self._nb = ttk.Notebook(self, padding=4)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        text_frame = ttk.Frame(self._nb)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self._nb.add(text_frame, text="  Text  ")
        self._text_box = scrolledtext.ScrolledText(
            text_frame,
            wrap="word",
            font=("Courier", 10),
            undo=True,
            relief="sunken",
        )
        self._text_box.grid(row=0, column=0, sticky="nsew")
        self._text_box.bind("<<Modified>>", self._on_text_change)

        msg_frame = ttk.Frame(self._nb)
        msg_frame.rowconfigure(0, weight=1)
        msg_frame.columnconfigure(0, weight=1)
        self._nb.add(msg_frame, text="  Messages (JSON)  ")
        self._msg_box = scrolledtext.ScrolledText(
            msg_frame,
            wrap="word",
            font=("Courier", 10),
            undo=True,
            relief="sunken",
        )
        self._msg_box.grid(row=0, column=0, sticky="nsew")
        self._msg_box.insert("1.0", _MSG_PLACEHOLDER)
        self._msg_box.bind("<<Modified>>", self._on_msg_change)

        self._nb.bind("<<NotebookTabChanged>>", lambda _: self._update())

        # ── results panel ────────────────────────────────────────────────────
        res = ttk.LabelFrame(self, text=" Results ", padding=(12, 6, 12, 8))
        res.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        res.columnconfigure(1, weight=1)

        self._tokens_var = tk.StringVar(value="—")
        self._window_var = tk.StringVar(value="—")
        self._fits_var = tk.StringVar(value="—")
        self._pct_var = tk.StringVar(value="0.0 %")

        ttk.Label(res, text="Tokens estimated:").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Label(
            res,
            textvariable=self._tokens_var,
            font=("", 12, "bold"),
            foreground="#0055aa",
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(res, text="Context window:").grid(
            row=1, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Label(res, textvariable=self._window_var).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(res, text="Fits in context:").grid(
            row=2, column=0, sticky="w", padx=(0, 12)
        )
        self._fits_label = ttk.Label(
            res, textvariable=self._fits_var, font=("", 11, "bold")
        )
        self._fits_label.grid(row=2, column=1, sticky="w")

        ttk.Label(res, text="Utilisation:").grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=(6, 0)
        )
        util_row = ttk.Frame(res)
        util_row.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        util_row.columnconfigure(0, weight=1)
        self._bar = ttk.Progressbar(util_row, maximum=100, length=300)
        self._bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(util_row, textvariable=self._pct_var, width=8).grid(
            row=0, column=1, padx=(8, 0)
        )

        # ── button row + status bar ──────────────────────────────────────────
        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        ttk.Button(bottom, text="Clear", command=self._clear).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(
            bottom, text="Compare all models", command=self._compare
        ).grid(row=0, column=1, sticky="w")

        self._status_var = tk.StringVar(value="")
        ttk.Label(
            bottom,
            textvariable=self._status_var,
            foreground="#888888",
        ).grid(row=0, column=2, sticky="e")

    # ── event callbacks ──────────────────────────────────────────────────────

    def _on_text_change(self, _event: object) -> None:
        self._text_box.edit_modified(False)
        self._update()

    def _on_msg_change(self, _event: object) -> None:
        self._msg_box.edit_modified(False)
        self._update()

    def _current_key(self) -> str:
        return _LABEL_TO_KEY.get(self._model_var.get(), "gpt-4")

    def _current_tab(self) -> str:
        return self._nb.tab(self._nb.select(), "text").strip()

    # ── core update logic ────────────────────────────────────────────────────

    def _update(self) -> None:
        model = self._current_key()
        try:
            headroom = int(self._headroom_var.get() or "0")
        except ValueError:
            headroom = 0

        status = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._current_tab() == "Messages (JSON)":
                raw = self._msg_box.get("1.0", "end-1c")
                try:
                    data = json.loads(raw) if raw.strip() else []
                    if not isinstance(data, list):
                        data = []
                        status = "Warning: expected a JSON array"
                except json.JSONDecodeError as exc:
                    data = []
                    status = f"JSON error: {exc.msg}"
                tokens = estimate_messages(data, model)
            else:
                text = self._text_box.get("1.0", "end-1c")
                tokens = estimate_tokens(text, model)

        win = context_window(model)
        fits = (tokens + max(0, headroom)) <= win
        pct = min(100.0, 100.0 * tokens / win) if win else 0.0

        self._tokens_var.set(f"{tokens:,}")
        self._window_var.set(f"{win:,}")
        self._fits_var.set("✓  Yes" if fits else "✗  No")
        self._fits_label.configure(
            foreground="#228833" if fits else "#cc3311"
        )
        self._bar["value"] = pct
        self._pct_var.set(f"{pct:.1f} %")
        self._status_var.set(status)

    # ── button actions ───────────────────────────────────────────────────────

    def _clear(self) -> None:
        if self._current_tab() == "Messages (JSON)":
            self._msg_box.delete("1.0", "end")
        else:
            self._text_box.delete("1.0", "end")
        self._update()

    def _compare(self) -> None:
        """Open a popup table comparing token counts across all models."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._current_tab() == "Messages (JSON)":
                raw = self._msg_box.get("1.0", "end-1c")
                try:
                    data = json.loads(raw) if raw.strip() else []
                    if not isinstance(data, list):
                        data = []
                except Exception:
                    data = []
                rows = [
                    (m, estimate_messages(data, m), context_window(m))
                    for m in _ORDERED_MODELS
                ]
            else:
                text = self._text_box.get("1.0", "end-1c")
                rows = [
                    (m, estimate_tokens(text, m), context_window(m))
                    for m in _ORDERED_MODELS
                ]

        dlg = tk.Toplevel(self)
        dlg.title("Model comparison – tokenfit")
        dlg.resizable(True, False)
        dlg.transient(self)
        dlg.grab_set()

        cols = ("Model", "Est. tokens", "Window", "% used", "Fits?")
        col_widths = (210, 110, 120, 80, 60)
        col_anchors = ("w", "e", "e", "e", "center")

        tv = ttk.Treeview(
            dlg, columns=cols, show="headings", height=len(rows)
        )
        for col, w, anchor in zip(cols, col_widths, col_anchors):
            tv.heading(col, text=col)
            tv.column(col, width=w, anchor=anchor)

        for key, toks, win in rows:
            pct = 100.0 * toks / win if win else 0.0
            tv.insert(
                "",
                "end",
                values=(
                    _make_label(key),
                    f"{toks:,}",
                    f"{win:,}",
                    f"{pct:.1f} %",
                    "✓" if toks <= win else "✗",
                ),
            )

        tv.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(pady=(0, 8))


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Launch the tokenfit GUI application."""
    app = TokenFitApp()
    app.mainloop()


if __name__ == "__main__":
    main()
