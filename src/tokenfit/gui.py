"""tokenfit GUI: interactive token-count explorer.

Launch with::

    tokenfit-gui

or programmatically::

    from tokenfit.gui import main
    main()
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    list_models,
    token_budget,
)

# ---------------------------------------------------------------------------
# Colour palette (light theme)
# ---------------------------------------------------------------------------
_CLR_BG = "#f5f5f5"
_CLR_PANEL = "#ffffff"
_CLR_ACCENT = "#4a90d9"
_CLR_OK = "#27ae60"
_CLR_WARN = "#e67e22"
_CLR_ERR = "#e74c3c"
_CLR_TEXT = "#2c3e50"
_CLR_MUTED = "#7f8c8d"

_TITLE = f"tokenfit {__version__}  —  Token Estimator"
_MIN_W, _MIN_H = 680, 560


def _usage_colour(pct: float) -> str:
    if pct >= 90:
        return _CLR_ERR
    if pct >= 70:
        return _CLR_WARN
    return _CLR_OK


class _ProgressCanvas(tk.Canvas):
    """Simple custom progress bar that supports dynamic fill colour."""

    _BAR_H = 18
    _RADIUS = 6

    def __init__(self, parent: tk.Widget, **kwargs: object) -> None:
        super().__init__(
            parent,
            height=self._BAR_H,
            bg=_CLR_BG,
            highlightthickness=0,
            **kwargs,  # type: ignore[arg-type]
        )
        self._pct = 0.0
        self._colour = _CLR_OK
        self.bind("<Configure>", lambda _: self._draw())

    def set(self, pct: float) -> None:
        self._pct = max(0.0, min(pct, 100.0))
        self._colour = _usage_colour(self._pct)
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = self.winfo_width() or 1
        h = self._BAR_H
        r = self._RADIUS

        # Track (grey pill)
        self._pill(0, 0, w, h, r, "#d9dde1")

        # Fill
        fill_w = max(0, int(w * self._pct / 100))
        if fill_w > 0:
            self._pill(0, 0, fill_w, h, r, self._colour)

    def _pill(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        r: int,
        colour: str,
    ) -> None:
        """Draw a rounded rectangle."""
        self.create_arc(x0, y0, x0 + 2 * r, y1, start=90, extent=180, fill=colour, outline=colour)
        self.create_arc(x1 - 2 * r, y0, x1, y1, start=270, extent=180, fill=colour, outline=colour)
        self.create_rectangle(x0 + r, y0, x1 - r, y1, fill=colour, outline=colour)


class TokenFitApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(_TITLE)
        self.minsize(_MIN_W, _MIN_H)
        self.configure(bg=_CLR_BG)
        self._configure_styles()
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # Style / theme
    # ------------------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        # Use a clean base theme
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background=_CLR_BG)
        style.configure("Panel.TFrame", background=_CLR_PANEL)
        style.configure("TLabel", background=_CLR_BG, foreground=_CLR_TEXT)
        style.configure("Panel.TLabel", background=_CLR_PANEL, foreground=_CLR_TEXT)
        style.configure(
            "Big.TLabel",
            background=_CLR_PANEL,
            foreground=_CLR_TEXT,
            font=("", 18, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background=_CLR_PANEL,
            foreground=_CLR_MUTED,
            font=("", 9),
        )
        style.configure(
            "Heading.TLabel",
            background=_CLR_BG,
            foreground=_CLR_TEXT,
            font=("", 10, "bold"),
        )
        style.configure("TNotebook", background=_CLR_BG)
        style.configure("TNotebook.Tab", padding=(10, 4))
        style.configure("TCombobox", padding=2)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_notebook()
        self._build_results_panel()

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(12, 8, 12, 4))
        bar.pack(fill="x")

        ttk.Label(bar, text="Model:", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value="gpt-4")
        cb = ttk.Combobox(
            bar,
            textvariable=self._model_var,
            values=list_models(),
            width=22,
            state="readonly",
        )
        cb.grid(row=0, column=1, padx=(6, 24), sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda _: self._refresh())

        ttk.Label(bar, text="Headroom (tokens):", style="Heading.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        self._headroom_var = tk.StringVar(value="0")
        spin = ttk.Spinbox(
            bar,
            from_=0,
            to=1_000_000,
            increment=500,
            textvariable=self._headroom_var,
            width=10,
        )
        spin.grid(row=0, column=3, padx=(6, 0), sticky="w")
        spin.bind("<KeyRelease>", lambda _: self._refresh())
        spin.bind("<<Increment>>", lambda _: self._refresh())
        spin.bind("<<Decrement>>", lambda _: self._refresh())

        # Clear button
        ttk.Button(bar, text="Clear", command=self._clear_input).grid(
            row=0, column=4, padx=(20, 0), sticky="e"
        )
        bar.columnconfigure(4, weight=1)

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self, padding=(8, 4, 8, 0))
        self._nb.pack(fill="both", expand=True)
        self._nb.bind("<<NotebookTabChanged>>", lambda _: self._refresh())

        # Tab 1 — plain text
        tab1 = ttk.Frame(self._nb)
        self._nb.add(tab1, text="  Plain Text  ")
        ttk.Label(
            tab1,
            text="Paste or type text to estimate its token count:",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=4, pady=(6, 2))
        mono = tkfont.Font(family="Courier", size=10)
        self._plain_text = scrolledtext.ScrolledText(
            tab1, wrap="word", font=mono, relief="flat", borderwidth=1
        )
        self._plain_text.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self._plain_text.bind("<KeyRelease>", lambda _: self._refresh())

        # Tab 2 — chat messages
        tab2 = ttk.Frame(self._nb)
        self._nb.add(tab2, text="  Chat Messages  ")
        ttk.Label(
            tab2,
            text="One message per line — format:  role: message content",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=4, pady=(6, 2))
        ttk.Label(
            tab2,
            text='Example:  user: Hello!    assistant: Hi there.',
            style="Muted.TLabel",
        ).pack(anchor="w", padx=4, pady=(0, 4))
        self._chat_text = scrolledtext.ScrolledText(
            tab2, wrap="word", font=mono, relief="flat", borderwidth=1
        )
        self._chat_text.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self._chat_text.bind("<KeyRelease>", lambda _: self._refresh())

    def _build_results_panel(self) -> None:
        outer = ttk.Frame(self, padding=(12, 4, 12, 12))
        outer.pack(fill="x")

        panel = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        panel.pack(fill="x")

        # ---- stat cards row ----
        cards = ttk.Frame(panel, style="Panel.TFrame")
        cards.pack(fill="x", pady=(0, 10))

        self._tokens_label = self._stat_card(cards, "Tokens Used", "0", col=0)
        self._window_label = self._stat_card(cards, "Context Window", "8,192", col=1)
        self._remaining_label = self._stat_card(cards, "Remaining", "8,192", col=2)
        self._fits_label = self._stat_card(cards, "Fits in Window", "Yes", col=3, ok=True)

        for c in range(4):
            cards.columnconfigure(c, weight=1, uniform="card")

        # ---- progress bar ----
        self._bar = _ProgressCanvas(panel)
        self._bar.pack(fill="x", pady=(0, 4))

        self._pct_label = ttk.Label(
            panel, text="0.0 % of context window used", style="Muted.TLabel"
        )
        self._pct_label.pack(anchor="e")

    def _stat_card(
        self, parent: ttk.Frame, title: str, value: str, col: int, ok: bool = False
    ) -> ttk.Label:
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 0, 16, 0))
        frame.grid(row=0, column=col, sticky="w")
        ttk.Label(frame, text=title, style="Muted.TLabel").pack(anchor="w")
        lbl = ttk.Label(frame, text=value, style="Big.TLabel")
        if ok:
            lbl.configure(foreground=_CLR_OK)
        lbl.pack(anchor="w")
        return lbl

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        model = self._model_var.get()
        try:
            headroom = max(0, int(self._headroom_var.get()))
        except (ValueError, tk.TclError):
            headroom = 0

        tab = self._nb.index("current")
        if tab == 0:
            text = self._plain_text.get("1.0", "end-1c")
            tokens = estimate_tokens(text, model)
        else:
            raw = self._chat_text.get("1.0", "end-1c")
            messages = _parse_chat(raw)
            tokens = estimate_messages(messages, model)

        win = context_window(model)
        used = tokens + headroom
        remaining = win - used
        pct = min(used / win * 100.0, 100.0) if win > 0 else 0.0
        fits = used <= win
        colour = _usage_colour(pct)

        self._tokens_label.configure(text=f"{tokens:,}", foreground=colour)
        self._window_label.configure(text=f"{win:,}", foreground=_CLR_TEXT)
        self._remaining_label.configure(
            text=f"{remaining:,}",
            foreground=_CLR_OK if remaining >= 0 else _CLR_ERR,
        )
        self._fits_label.configure(
            text="Yes" if fits else "No",
            foreground=_CLR_OK if fits else _CLR_ERR,
        )
        self._bar.set(pct)
        self._pct_label.configure(text=f"{pct:.1f} % of context window used")

    def _clear_input(self) -> None:
        self._plain_text.delete("1.0", "end")
        self._chat_text.delete("1.0", "end")
        self._refresh()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_chat(raw: str) -> list[dict[str, str]]:
    """Parse ``role: content`` lines into message dicts."""
    messages: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            role, _, content = line.partition(":")
            messages.append({"role": role.strip(), "content": content.strip()})
        else:
            messages.append({"role": "user", "content": line})
    return messages


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the tokenfit graphical interface."""
    try:
        app = TokenFitApp()
        app.mainloop()
    except tk.TclError as exc:
        print(f"tokenfit-gui: cannot open display — {exc}", file=sys.stderr)
        print("Run in a graphical environment or use the Python API instead.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
