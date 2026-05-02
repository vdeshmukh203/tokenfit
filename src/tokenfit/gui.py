"""Graphical interface for tokenfit.

Launch with::

    python -m tokenfit

or, if the package was installed with the optional ``[gui]`` extra::

    tokenfit-gui

The window contains three tabs:

* **Text** — live token estimation for free-form text.
* **Messages** — build a chat-style message list and estimate its total.
* **Compare** — display token estimates side-by-side across all model families.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any

from tokenfit import (
    TokenEstimate,
    estimate_messages,
    estimate_tokens,
    list_models,
    remaining_tokens,
    token_summary,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_CLR_OK = "#27ae60"       # green  — fits
_CLR_WARN = "#e67e22"     # orange — >80 % used
_CLR_ERR = "#e74c3c"      # red    — overflows
_CLR_FG = "#2c3e50"       # dark slate — primary text
_CLR_BG = "#f5f6fa"       # near-white background
_CLR_PANEL = "#ecf0f1"    # slightly darker panel
_CLR_BAR = "#3498db"      # blue   — progress bar fill
_CLR_BAR_BG = "#bdc3c7"   # grey   — progress bar background
_FONT_MONO = ("Courier", 10)
_FONT_LABEL = ("TkDefaultFont", 10)
_FONT_TITLE = ("TkDefaultFont", 10, "bold")

_MODELS = list_models()   # sorted list of family prefix strings
_DEFAULT_MODEL = "gpt-4"


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class _StatusBar(tk.Frame):
    """One-line status label with colour coding."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, bg=_CLR_BG, **kwargs)
        self._var = tk.StringVar(value="Ready.")
        self._lbl = tk.Label(
            self,
            textvariable=self._var,
            anchor="w",
            bg=_CLR_BG,
            fg=_CLR_FG,
            font=_FONT_LABEL,
        )
        self._lbl.pack(fill="x", padx=4)

    def set(self, text: str, colour: str = _CLR_FG) -> None:
        self._var.set(text)
        self._lbl.config(fg=colour)


class _UsageBar(tk.Canvas):
    """Horizontal bar showing % of context window used."""

    _HEIGHT = 18

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(
            parent,
            height=self._HEIGHT,
            bg=_CLR_BAR_BG,
            highlightthickness=0,
            **kwargs,
        )
        self._bar = self.create_rectangle(0, 0, 0, self._HEIGHT, fill=_CLR_BAR, width=0)
        self._text = self.create_text(4, self._HEIGHT // 2, anchor="w", text="0.00 %", fill="white", font=_FONT_LABEL)
        self.bind("<Configure>", lambda _: self._redraw())
        self._pct: float = 0.0

    def update_pct(self, used: int, total: int) -> None:
        self._pct = min(1.0, used / total) if total > 0 else 0.0
        self._redraw()

    def _redraw(self) -> None:
        w = self.winfo_width() or 1
        fill_w = int(w * self._pct)
        colour = _CLR_BAR if self._pct < 0.8 else (_CLR_WARN if self._pct < 1.0 else _CLR_ERR)
        self.coords(self._bar, 0, 0, fill_w, self._HEIGHT)
        self.itemconfig(self._bar, fill=colour)
        pct_str = f"{self._pct * 100:.1f} % of context used"
        self.itemconfig(self._text, text=pct_str)
        text_colour = "white" if fill_w > 60 else _CLR_FG
        self.itemconfig(self._text, fill=text_colour)


class _StatsRow(tk.Frame):
    """Row of labelled counters (chars / tokens / window / remaining)."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, bg=_CLR_BG, **kwargs)
        self._vars: dict[str, tk.StringVar] = {}
        for label in ("Characters", "Tokens", "Window", "Remaining"):
            var = tk.StringVar(value="—")
            self._vars[label] = var
            col = tk.Frame(self, bg=_CLR_BG)
            col.pack(side="left", padx=12, pady=2)
            tk.Label(col, text=label, font=_FONT_TITLE, bg=_CLR_BG, fg=_CLR_FG).pack()
            tk.Label(col, textvariable=var, font=_FONT_LABEL, bg=_CLR_BG, fg=_CLR_FG).pack()

    def update(self, chars: int, tokens: int, window: int, remaining: int) -> None:  # type: ignore[override]
        self._vars["Characters"].set(f"{chars:,}")
        self._vars["Tokens"].set(f"~{tokens:,}")
        self._vars["Window"].set(f"{window:,}")
        colour = _CLR_OK if remaining >= 0 else _CLR_ERR
        self._vars["Remaining"].set(f"{remaining:,}")


# ---------------------------------------------------------------------------
# Tab 1 — Text estimator
# ---------------------------------------------------------------------------

class _TextTab(tk.Frame):
    """Live token estimation for free-form text input."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_CLR_BG)
        self._build()
        self._after_id: str | None = None

    def _build(self) -> None:
        # ---- toolbar ----
        toolbar = tk.Frame(self, bg=_CLR_BG)
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(toolbar, text="Model:", bg=_CLR_BG, font=_FONT_LABEL).pack(side="left")
        self._model_var = tk.StringVar(value=_DEFAULT_MODEL)
        model_cb = ttk.Combobox(
            toolbar,
            textvariable=self._model_var,
            values=_MODELS,
            state="readonly",
            width=24,
        )
        model_cb.pack(side="left", padx=(4, 16))
        model_cb.bind("<<ComboboxSelected>>", lambda _: self._schedule_update())

        tk.Label(toolbar, text="Headroom (tokens):", bg=_CLR_BG, font=_FONT_LABEL).pack(side="left")
        self._headroom_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(
            toolbar,
            from_=0,
            to=1_000_000,
            increment=100,
            textvariable=self._headroom_var,
            width=10,
            command=self._schedule_update,
        )
        spin.pack(side="left", padx=(4, 16))
        spin.bind("<Return>", lambda _: self._schedule_update())

        ttk.Button(toolbar, text="Clear", command=self._clear).pack(side="right")

        # ---- text area ----
        text_frame = tk.Frame(self, bg=_CLR_BG)
        text_frame.pack(fill="both", expand=True, padx=8, pady=4)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self._text = tk.Text(
            text_frame,
            wrap="word",
            font=_FONT_MONO,
            yscrollcommand=scrollbar.set,
            relief="solid",
            borderwidth=1,
            undo=True,
        )
        self._text.pack(fill="both", expand=True)
        scrollbar.config(command=self._text.yview)
        self._text.bind("<<Modified>>", self._on_modified)

        # ---- stats ----
        self._stats = _StatsRow(self)
        self._stats.pack(fill="x", padx=8)

        self._bar = _UsageBar(self)
        self._bar.pack(fill="x", padx=8, pady=4)

        self._status = _StatusBar(self)
        self._status.pack(fill="x", padx=8, pady=(0, 8))

    # ---- helpers ----

    def _on_modified(self, _event: Any = None) -> None:
        if self._text.edit_modified():
            self._text.edit_modified(False)
            self._schedule_update()

    def _schedule_update(self, _event: Any = None) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(80, self._update)

    def _update(self) -> None:
        text = self._text.get("1.0", "end-1c")
        model = self._model_var.get()
        try:
            headroom = int(self._headroom_var.get())
        except (ValueError, tk.TclError):
            headroom = 0

        est: TokenEstimate = token_summary(text, model, headroom)
        self._stats.update(
            chars=len(text),
            tokens=est.tokens,
            window=est.window,
            remaining=est.remaining,
        )
        self._bar.update_pct(est.tokens + max(0, headroom), est.window)

        if est.fits:
            pct = (est.tokens + max(0, headroom)) / est.window * 100
            colour = _CLR_OK if pct < 80 else _CLR_WARN
            self._status.set(
                f"✓  Fits — {est.remaining:,} tokens remaining  ({pct:.1f} % used)",
                colour,
            )
        else:
            overflow = -est.remaining
            self._status.set(
                f"✗  Overflows by {overflow:,} tokens",
                _CLR_ERR,
            )

    def _clear(self) -> None:
        self._text.delete("1.0", "end")
        self._update()


# ---------------------------------------------------------------------------
# Tab 2 — Message builder
# ---------------------------------------------------------------------------

class _MessagesTab(tk.Frame):
    """Build a chat-style message list and estimate its total token count."""

    _ROLES = ["user", "assistant", "system"]

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_CLR_BG)
        self._messages: list[dict[str, str]] = []
        self._build()

    def _build(self) -> None:
        # ---- toolbar ----
        toolbar = tk.Frame(self, bg=_CLR_BG)
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(toolbar, text="Model:", bg=_CLR_BG, font=_FONT_LABEL).pack(side="left")
        self._model_var = tk.StringVar(value=_DEFAULT_MODEL)
        model_cb = ttk.Combobox(
            toolbar,
            textvariable=self._model_var,
            values=_MODELS,
            state="readonly",
            width=24,
        )
        model_cb.pack(side="left", padx=(4, 16))
        model_cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_total())

        ttk.Button(toolbar, text="Clear All", command=self._clear_all).pack(side="right")

        # ---- add-message form ----
        form = tk.LabelFrame(self, text=" New message ", bg=_CLR_BG, font=_FONT_LABEL)
        form.pack(fill="x", padx=8, pady=4)

        tk.Label(form, text="Role:", bg=_CLR_BG, font=_FONT_LABEL).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self._role_var = tk.StringVar(value="user")
        role_cb = ttk.Combobox(form, textvariable=self._role_var, values=self._ROLES, state="readonly", width=12)
        role_cb.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        tk.Label(form, text="Content:", bg=_CLR_BG, font=_FONT_LABEL).grid(row=1, column=0, sticky="nw", padx=4, pady=4)
        content_frame = tk.Frame(form, bg=_CLR_BG)
        content_frame.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        form.columnconfigure(1, weight=1)

        sb = ttk.Scrollbar(content_frame)
        sb.pack(side="right", fill="y")
        self._content_text = tk.Text(
            content_frame,
            height=4,
            wrap="word",
            font=_FONT_MONO,
            relief="solid",
            borderwidth=1,
            yscrollcommand=sb.set,
        )
        self._content_text.pack(fill="x", expand=True)
        sb.config(command=self._content_text.yview)

        btn_row = tk.Frame(form, bg=_CLR_BG)
        btn_row.grid(row=2, column=0, columnspan=3, sticky="e", padx=4, pady=4)
        ttk.Button(btn_row, text="Add Message", command=self._add_message).pack(side="right")

        # ---- message list ----
        list_frame = tk.LabelFrame(self, text=" Messages ", bg=_CLR_BG, font=_FONT_LABEL)
        list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        cols = ("role", "preview", "tokens")
        self._tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=8)
        self._tree.heading("role", text="Role")
        self._tree.heading("preview", text="Content")
        self._tree.heading("tokens", text="Tokens")
        self._tree.column("role", width=90, minwidth=70, anchor="center")
        self._tree.column("preview", width=400, minwidth=200)
        self._tree.column("tokens", width=70, minwidth=60, anchor="e")

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        ttk.Button(list_frame, text="Remove Selected", command=self._remove_selected).pack(anchor="e", padx=4, pady=4)

        # ---- totals ----
        self._stats = _StatsRow(self)
        self._stats.pack(fill="x", padx=8)

        self._bar = _UsageBar(self)
        self._bar.pack(fill="x", padx=8, pady=4)

        self._status = _StatusBar(self)
        self._status.pack(fill="x", padx=8, pady=(0, 8))

    # ---- helpers ----

    def _add_message(self) -> None:
        role = self._role_var.get()
        content = self._content_text.get("1.0", "end-1c").strip()
        if not content:
            return
        msg = {"role": role, "content": content}
        self._messages.append(msg)

        model = self._model_var.get()
        tok = estimate_tokens(role, model) + estimate_tokens(content, model)
        preview = content[:60] + ("…" if len(content) > 60 else "")
        self._tree.insert("", "end", values=(role, preview, tok))

        self._content_text.delete("1.0", "end")
        self._refresh_total()

    def _remove_selected(self) -> None:
        selected = self._tree.selection()
        if not selected:
            return
        for item in selected:
            idx = self._tree.index(item)
            self._tree.delete(item)
            if 0 <= idx < len(self._messages):
                self._messages.pop(idx)
        self._refresh_total()

    def _clear_all(self) -> None:
        self._messages.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._refresh_total()

    def _refresh_total(self) -> None:
        model = self._model_var.get()
        total = estimate_messages(self._messages, model)
        # Approximate window from first family lookup
        from tokenfit import _WINDOWS, _DEFAULT_FAMILY, _family  # noqa: PLC0415
        fam = _family(model)
        window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
        rem = window - total
        self._stats.update(chars=0, tokens=total, window=window, remaining=rem)
        self._bar.update_pct(total, window)
        if rem >= 0:
            pct = total / window * 100 if window > 0 else 0.0
            colour = _CLR_OK if pct < 80 else _CLR_WARN
            self._status.set(
                f"✓  Fits — {rem:,} tokens remaining  ({pct:.1f} % used)",
                colour,
            )
        else:
            self._status.set(f"✗  Overflows by {-rem:,} tokens", _CLR_ERR)


# ---------------------------------------------------------------------------
# Tab 3 — Model comparison
# ---------------------------------------------------------------------------

class _CompareTab(tk.Frame):
    """Show token estimates for the same text across all model families."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=_CLR_BG)
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text="Enter text below then click Compare:",
            bg=_CLR_BG,
            font=_FONT_LABEL,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        text_frame = tk.Frame(self, bg=_CLR_BG)
        text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        sb = ttk.Scrollbar(text_frame)
        sb.pack(side="right", fill="y")
        self._text = tk.Text(
            text_frame,
            height=8,
            wrap="word",
            font=_FONT_MONO,
            relief="solid",
            borderwidth=1,
            yscrollcommand=sb.set,
        )
        self._text.pack(fill="both", expand=True)
        sb.config(command=self._text.yview)
        self._text.bind("<<Modified>>", self._on_modified)

        ttk.Button(self, text="Compare All Models", command=self._compare).pack(anchor="e", padx=8, pady=4)

        # ---- result table ----
        cols = ("family", "tokens", "window", "pct", "fits")
        self._tree = ttk.Treeview(self, columns=cols, show="headings")
        self._tree.heading("family", text="Model Family")
        self._tree.heading("tokens", text="Tokens")
        self._tree.heading("window", text="Window")
        self._tree.heading("pct", text="% Used")
        self._tree.heading("fits", text="Fits?")
        self._tree.column("family", width=200, minwidth=140)
        self._tree.column("tokens", width=90, minwidth=70, anchor="e")
        self._tree.column("window", width=110, minwidth=80, anchor="e")
        self._tree.column("pct", width=80, minwidth=60, anchor="e")
        self._tree.column("fits", width=60, minwidth=50, anchor="center")

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 8))
        self._tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._tree.tag_configure("ok", foreground=_CLR_OK)
        self._tree.tag_configure("warn", foreground=_CLR_WARN)
        self._tree.tag_configure("err", foreground=_CLR_ERR)

        self._after_id: str | None = None

    def _on_modified(self, _event: Any = None) -> None:
        if self._text.edit_modified():
            self._text.edit_modified(False)

    def _compare(self) -> None:
        text = self._text.get("1.0", "end-1c")
        for item in self._tree.get_children():
            self._tree.delete(item)

        from tokenfit import _WINDOWS, _DEFAULT_FAMILY  # noqa: PLC0415

        for family in _MODELS:
            est = token_summary(text, family)
            pct = est.tokens / est.window * 100 if est.window > 0 else 0.0
            fits_str = "✓" if est.fits else "✗"
            tag = "ok" if est.fits and pct < 80 else ("warn" if est.fits else "err")
            self._tree.insert(
                "",
                "end",
                values=(
                    family,
                    f"{est.tokens:,}",
                    f"{est.window:,}",
                    f"{pct:.2f} %",
                    fits_str,
                ),
                tags=(tag,),
            )


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class TokenfitApp(tk.Tk):
    """Root window for the tokenfit graphical interface."""

    def __init__(self) -> None:
        super().__init__()
        self.title("tokenfit — Token Estimator")
        self.minsize(700, 520)
        self.configure(bg=_CLR_BG)
        self._build()

    def _build(self) -> None:
        # Header
        header = tk.Frame(self, bg=_CLR_FG, height=4)
        header.pack(fill="x")

        title_bar = tk.Frame(self, bg=_CLR_BG)
        title_bar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(
            title_bar,
            text="tokenfit",
            font=("TkDefaultFont", 16, "bold"),
            bg=_CLR_BG,
            fg=_CLR_FG,
        ).pack(side="left")

        from tokenfit import __version__  # noqa: PLC0415
        tk.Label(
            title_bar,
            text=f"v{__version__}  •  offline token estimator",
            font=("TkDefaultFont", 10),
            bg=_CLR_BG,
            fg="#7f8c8d",
        ).pack(side="left", padx=(8, 0))

        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", padx=8, pady=(6, 0))

        # Notebook
        style = ttk.Style(self)
        style.configure("TNotebook", background=_CLR_BG)
        style.configure("TNotebook.Tab", padding=(12, 4))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._text_tab = _TextTab(nb)
        self._msgs_tab = _MessagesTab(nb)
        self._cmp_tab = _CompareTab(nb)

        nb.add(self._text_tab, text="  Text  ")
        nb.add(self._msgs_tab, text="  Messages  ")
        nb.add(self._cmp_tab, text="  Compare  ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the tokenfit graphical interface."""
    app = TokenfitApp()
    app.mainloop()


if __name__ == "__main__":
    main()
