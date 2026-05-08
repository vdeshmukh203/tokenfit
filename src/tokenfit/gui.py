"""tokenfit GUI – interactive, offline token estimator.

Launch with::

    python -m tokenfit

or via the installed console script::

    tokenfit-gui
"""
from __future__ import annotations

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except ImportError as exc:
    raise ImportError(
        "The tokenfit GUI requires tkinter, which ships with the standard "
        "Python distribution but may need a separate package on some systems.\n"
        "  Debian/Ubuntu : sudo apt-get install python3-tk\n"
        "  Fedora/RHEL   : sudo dnf install python3-tkinter\n"
        "  macOS (brew)  : brew install python-tk"
    ) from exc

import json
from tokenfit import (
    chars_per_token,
    context_window,
    estimate_messages,
    estimate_tokens,
    list_models,
)

_FITS_FG = "#1b5e20"    # dark green
_OVER_FG = "#b71c1c"    # dark red
_NEUTRAL_FG = "#555555"


class _TokenfitApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("tokenfit – Token Estimator")
        self.geometry("820x580")
        self.minsize(620, 440)
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._build_text_tab()
        self._build_chat_tab()
        self._notebook.bind("<<NotebookTabChanged>>", self._refresh)
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 6, 8, 4))
        bar.pack(fill="x")

        ttk.Label(bar, text="Model:").pack(side="left")
        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            bar,
            textvariable=self._model_var,
            values=list_models(),
            state="readonly",
            width=24,
        )
        model_cb.pack(side="left", padx=(4, 14))
        model_cb.bind("<<ComboboxSelected>>", self._refresh)

        ttk.Label(bar, text="Headroom (tokens):").pack(side="left")
        self._headroom_var = tk.StringVar(value="0")
        hentry = ttk.Entry(bar, textvariable=self._headroom_var, width=9)
        hentry.pack(side="left", padx=(4, 14))
        hentry.bind("<KeyRelease>", self._refresh)

        self._window_label = ttk.Label(bar, text="", foreground=_NEUTRAL_FG)
        self._window_label.pack(side="left")

    def _build_text_tab(self) -> None:
        frame = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(frame, text="Plain Text")

        self._text_area = scrolledtext.ScrolledText(
            frame, wrap="word", font=("Courier", 11), undo=True
        )
        self._text_area.pack(fill="both", expand=True, pady=(0, 4))
        self._text_area.bind("<KeyRelease>", self._refresh)
        self._text_area.bind("<<Paste>>", lambda _e: self.after(10, self._refresh))

        foot = ttk.Frame(frame)
        foot.pack(fill="x")
        self._text_count_lbl = ttk.Label(foot, text="Tokens: 0  |  Chars: 0", font=("", 10))
        self._text_count_lbl.pack(side="left")
        self._text_fits_lbl = ttk.Label(foot, text="", font=("", 10, "bold"))
        self._text_fits_lbl.pack(side="left", padx=20)

    def _build_chat_tab(self) -> None:
        frame = ttk.Frame(self._notebook, padding=6)
        self._notebook.add(frame, text="Chat Messages")

        # Left pane – JSON editor
        left = ttk.LabelFrame(frame, text="Messages  (JSON array)", padding=4)
        left.pack(side="left", fill="both", expand=True)

        self._chat_area = scrolledtext.ScrolledText(
            left, wrap="word", font=("Courier", 10), undo=True
        )
        self._chat_area.pack(fill="both", expand=True)
        self._chat_area.insert(
            "1.0",
            '[\n'
            '  {"role": "system", "content": "You are a helpful assistant."},\n'
            '  {"role": "user",   "content": "Hello, how are you?"},\n'
            '  {"role": "assistant", "content": "I\'m doing well, thanks!"}\n'
            ']',
        )
        self._chat_area.bind("<KeyRelease>", self._refresh)
        self._chat_area.bind("<<Paste>>", lambda _e: self.after(10, self._refresh))

        # Right pane – results
        right = ttk.Frame(frame, padding=(14, 0, 0, 0))
        right.pack(side="left", fill="y")

        ttk.Label(right, text="Estimated tokens", font=("", 10)).pack(anchor="w")
        self._chat_count_lbl = ttk.Label(right, text="—", font=("", 26, "bold"))
        self._chat_count_lbl.pack(anchor="w", pady=(0, 8))

        self._chat_fits_lbl = ttk.Label(
            right, text="", font=("", 10, "bold"), wraplength=180
        )
        self._chat_fits_lbl.pack(anchor="w")

        self._chat_error_lbl = ttk.Label(
            right, text="", foreground=_OVER_FG, wraplength=180, font=("", 9)
        )
        self._chat_error_lbl.pack(anchor="w", pady=(8, 0))

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _model(self) -> str:
        return self._model_var.get() or "gpt-4"

    def _headroom(self) -> int:
        try:
            return max(0, int(self._headroom_var.get() or 0))
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    # Refresh callbacks
    # ------------------------------------------------------------------

    def _refresh(self, _event: object = None) -> None:
        model = self._model()
        win = context_window(model)
        ratio = chars_per_token(model)
        self._window_label.config(
            text=f"Context: {win:,} tokens  |  ~{ratio:.1f} chars/token"
        )
        tab_idx = self._notebook.index("current")
        if tab_idx == 0:
            self._refresh_text(model, win)
        else:
            self._refresh_chat(model, win)

    def _refresh_text(self, model: str, win: int) -> None:
        text = self._text_area.get("1.0", "end-1c")
        headroom = self._headroom()
        n = estimate_tokens(text, model)
        used = n + headroom
        chars = len(text)
        self._text_count_lbl.config(text=f"Tokens: {n:,}  |  Chars: {chars:,}")
        self._set_fits_label(self._text_fits_lbl, used, win)
        self._status_var.set(
            f"Model: {model}  |  {used:,} / {win:,} tokens  "
            f"({'%.1f' % (used / win * 100)}% of window)"
        )

    def _refresh_chat(self, model: str, win: int) -> None:
        raw = self._chat_area.get("1.0", "end-1c").strip()
        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                raise ValueError("Top-level value must be a JSON array")
            n = estimate_messages(messages, model)
            headroom = self._headroom()
            used = n + headroom
            self._chat_count_lbl.config(text=f"{n:,}")
            self._chat_error_lbl.config(text="")
            self._set_fits_label(self._chat_fits_lbl, used, win)
            self._status_var.set(
                f"Model: {model}  |  {used:,} / {win:,} tokens  "
                f"({'%.1f' % (used / win * 100)}% of window)"
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self._chat_count_lbl.config(text="—")
            self._chat_fits_lbl.config(text="", foreground=_NEUTRAL_FG)
            self._chat_error_lbl.config(text=f"⚠ {exc}")
            self._status_var.set("Invalid JSON — enter a valid array of message objects")

    @staticmethod
    def _set_fits_label(label: ttk.Label, used: int, win: int) -> None:
        pct = used / win * 100 if win else 0.0
        if used <= win:
            label.config(
                text=f"✓ Fits  ({pct:.1f}% of window)",
                foreground=_FITS_FG,
            )
        else:
            over = used - win
            label.config(
                text=f"✗ Exceeds by {over:,} tokens",
                foreground=_OVER_FG,
            )


def main() -> None:
    """Launch the tokenfit desktop GUI."""
    app = _TokenfitApp()
    app.mainloop()


if __name__ == "__main__":
    main()
