"""tokenfit GUI — interactive token-count estimator.

Launch with::

    python -m tokenfit.gui

or, after installation::

    tokenfit-gui

The window provides two tabs:

* **Text** — paste or type free-form text and see a live token estimate
  together with a context-window utilisation bar.
* **Messages** — build a chat-style message list and estimate the combined
  token count including per-message overhead.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    list_models,
)

# Colour palette (WCAG AA contrast on white / dark backgrounds)
_COLOUR = {
    "fit": "#2e7d32",       # dark green
    "warning": "#e65100",   # deep orange
    "overflow": "#c62828",  # dark red
    "bar_fill": "#1565c0",  # blue
    "bar_warn": "#f57f17",  # amber
    "bar_over": "#b71c1c",  # red
    "bg": "#f5f5f5",
    "frame_bg": "#ffffff",
    "accent": "#1565c0",
}

_PAD = 8


def _make_progress(parent: tk.Widget) -> tuple[tk.Canvas, tk.Rectangle]:
    """Return a canvas + the fill rectangle id for a custom progress bar."""
    canvas = tk.Canvas(parent, height=18, bg="#e0e0e0", highlightthickness=0)
    fill_id = canvas.create_rectangle(0, 0, 0, 18, fill=_COLOUR["bar_fill"], width=0)
    return canvas, fill_id


def _update_progress(
    canvas: tk.Canvas,
    fill_id: int,
    fraction: float,
) -> None:
    """Resize the fill rectangle to *fraction* (0–1) of the canvas width."""
    canvas.update_idletasks()
    width = canvas.winfo_width()
    filled = int(min(fraction, 1.0) * width)
    canvas.coords(fill_id, 0, 0, filled, 18)
    if fraction > 1.0:
        canvas.itemconfig(fill_id, fill=_COLOUR["bar_over"])
    elif fraction > 0.8:
        canvas.itemconfig(fill_id, fill=_COLOUR["bar_warn"])
    else:
        canvas.itemconfig(fill_id, fill=_COLOUR["bar_fill"])


# ---------------------------------------------------------------------------
# Text tab
# ---------------------------------------------------------------------------

class _TextTab(ttk.Frame):
    """Tab for estimating tokens from free-form text."""

    def __init__(self, parent: ttk.Notebook, model_var: tk.StringVar) -> None:
        super().__init__(parent, padding=_PAD)
        self._model_var = model_var
        self._headroom_var = tk.IntVar(value=0)
        self._build()

    def _build(self) -> None:
        # --- text input -------------------------------------------------------
        input_frame = ttk.LabelFrame(self, text="Input text", padding=_PAD)
        input_frame.pack(fill="both", expand=True, pady=(0, _PAD))

        self._text = tk.Text(
            input_frame,
            wrap="word",
            font=("Courier", 11),
            undo=True,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#bdbdbd",
        )
        scroll = ttk.Scrollbar(input_frame, command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._text.bind("<<Modified>>", self._on_text_change)
        self._text.bind("<KeyRelease>", self._on_text_change)
        self._text.bind("<ButtonRelease>", self._on_text_change)

        # --- headroom ---------------------------------------------------------
        hr_frame = ttk.Frame(self)
        hr_frame.pack(fill="x", pady=(0, _PAD))
        ttk.Label(hr_frame, text="Headroom (tokens):").pack(side="left")
        self._headroom_spin = ttk.Spinbox(
            hr_frame,
            from_=0,
            to=2_000_000,
            increment=500,
            textvariable=self._headroom_var,
            width=10,
            command=self._refresh,
        )
        self._headroom_spin.pack(side="left", padx=(_PAD, 0))
        self._headroom_spin.bind("<KeyRelease>", lambda _e: self._refresh())

        ttk.Button(hr_frame, text="Clear", command=self._clear).pack(
            side="right", padx=(_PAD, 0)
        )
        ttk.Button(hr_frame, text="Copy text", command=self._copy).pack(
            side="right", padx=(_PAD, 0)
        )

        # --- results ----------------------------------------------------------
        res_frame = ttk.LabelFrame(self, text="Estimate", padding=_PAD)
        res_frame.pack(fill="x")

        # token count row
        count_row = ttk.Frame(res_frame)
        count_row.pack(fill="x", pady=(0, 4))
        ttk.Label(count_row, text="Tokens:").pack(side="left")
        self._token_lbl = ttk.Label(count_row, text="0", font=("", 13, "bold"))
        self._token_lbl.pack(side="left", padx=_PAD)
        ttk.Label(count_row, text=" / ").pack(side="left")
        self._window_lbl = ttk.Label(count_row, text="")
        self._window_lbl.pack(side="left")

        # progress bar row
        self._canvas, self._fill = _make_progress(res_frame)
        self._canvas.pack(fill="x", pady=(0, 4))

        # status row
        self._status_lbl = ttk.Label(res_frame, text="", font=("", 11, "bold"))
        self._status_lbl.pack(anchor="w")

        # char count row
        self._char_lbl = ttk.Label(res_frame, text="Characters: 0", foreground="#757575")
        self._char_lbl.pack(anchor="w")

        self._refresh()

    # --- event handlers -------------------------------------------------------

    def _on_text_change(self, _event: object = None) -> None:
        if self._text.edit_modified():
            self._text.edit_modified(False)
        self._refresh()

    def _refresh(self) -> None:
        text = self._text.get("1.0", "end-1c")
        model = self._model_var.get()
        try:
            headroom = max(0, int(self._headroom_var.get()))
        except (ValueError, tk.TclError):
            headroom = 0

        tokens = estimate_tokens(text, model)
        window = context_window(model)
        used = tokens + headroom
        fraction = used / window if window else 0.0

        self._token_lbl.config(text=f"{tokens:,}")
        self._window_lbl.config(text=f"{window:,} (context window)")
        self._char_lbl.config(text=f"Characters: {len(text):,}")
        _update_progress(self._canvas, self._fill, fraction)

        pct = min(fraction * 100, 100)
        if used <= window:
            self._status_lbl.config(
                text=f"Fits in context  ({pct:.1f}% used)",
                foreground=_COLOUR["fit"] if fraction <= 0.8 else _COLOUR["warning"],
            )
        else:
            overflow = used - window
            self._status_lbl.config(
                text=f"Exceeds context by {overflow:,} tokens",
                foreground=_COLOUR["overflow"],
            )

    def _clear(self) -> None:
        self._text.delete("1.0", "end")
        self._refresh()

    def _copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._text.get("1.0", "end-1c"))

    def refresh_model(self) -> None:
        """Called when the model selector changes."""
        self._refresh()


# ---------------------------------------------------------------------------
# Messages tab
# ---------------------------------------------------------------------------

class _MessagesTab(ttk.Frame):
    """Tab for estimating tokens from a chat-style message list."""

    def __init__(self, parent: ttk.Notebook, model_var: tk.StringVar) -> None:
        super().__init__(parent, padding=_PAD)
        self._model_var = model_var
        self._messages: list[dict[str, str]] = []
        self._build()

    def _build(self) -> None:
        # --- message composer -------------------------------------------------
        compose_frame = ttk.LabelFrame(self, text="Add message", padding=_PAD)
        compose_frame.pack(fill="x", pady=(0, _PAD))

        role_row = ttk.Frame(compose_frame)
        role_row.pack(fill="x", pady=(0, 4))
        ttk.Label(role_row, text="Role:").pack(side="left")
        self._role_var = tk.StringVar(value="user")
        for role in ("user", "assistant", "system"):
            ttk.Radiobutton(
                role_row,
                text=role,
                variable=self._role_var,
                value=role,
            ).pack(side="left", padx=4)

        self._content_text = tk.Text(
            compose_frame,
            height=4,
            wrap="word",
            font=("Courier", 11),
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#bdbdbd",
        )
        self._content_text.pack(fill="x")

        btn_row = ttk.Frame(compose_frame)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="Add message", command=self._add_message).pack(side="left")
        ttk.Button(btn_row, text="Clear all", command=self._clear_all).pack(
            side="left", padx=_PAD
        )

        # --- message list -----------------------------------------------------
        list_frame = ttk.LabelFrame(self, text="Messages", padding=_PAD)
        list_frame.pack(fill="both", expand=True, pady=(0, _PAD))

        cols = ("role", "preview", "tokens")
        self._tree = ttk.Treeview(
            list_frame,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=8,
        )
        self._tree.heading("role", text="Role")
        self._tree.heading("preview", text="Content preview")
        self._tree.heading("tokens", text="Tokens")
        self._tree.column("role", width=90, stretch=False)
        self._tree.column("preview", width=340)
        self._tree.column("tokens", width=70, anchor="e", stretch=False)

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        ttk.Button(self, text="Remove selected", command=self._remove_selected).pack(
            anchor="w", pady=(0, _PAD)
        )

        # --- results ----------------------------------------------------------
        res_frame = ttk.LabelFrame(self, text="Estimate", padding=_PAD)
        res_frame.pack(fill="x")

        count_row = ttk.Frame(res_frame)
        count_row.pack(fill="x", pady=(0, 4))
        ttk.Label(count_row, text="Total tokens:").pack(side="left")
        self._total_lbl = ttk.Label(count_row, text="0", font=("", 13, "bold"))
        self._total_lbl.pack(side="left", padx=_PAD)
        ttk.Label(count_row, text=" / ").pack(side="left")
        self._window_lbl = ttk.Label(count_row, text="")
        self._window_lbl.pack(side="left")

        self._canvas, self._fill = _make_progress(res_frame)
        self._canvas.pack(fill="x", pady=(0, 4))

        self._status_lbl = ttk.Label(res_frame, text="", font=("", 11, "bold"))
        self._status_lbl.pack(anchor="w")

        self._msg_count_lbl = ttk.Label(
            res_frame, text="Messages: 0", foreground="#757575"
        )
        self._msg_count_lbl.pack(anchor="w")

        self._refresh()

    # --- helpers --------------------------------------------------------------

    def _add_message(self) -> None:
        content = self._content_text.get("1.0", "end-1c").strip()
        if not content:
            return
        role = self._role_var.get()
        self._messages.append({"role": role, "content": content})
        preview = content[:60] + ("…" if len(content) > 60 else "")
        tokens = estimate_tokens(role, self._model_var.get()) + estimate_tokens(
            content, self._model_var.get()
        )
        self._tree.insert("", "end", values=(role, preview, tokens))
        self._content_text.delete("1.0", "end")
        self._refresh()

    def _remove_selected(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        self._tree.delete(sel[0])
        if 0 <= idx < len(self._messages):
            self._messages.pop(idx)
        self._refresh()

    def _clear_all(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._messages.clear()
        self._refresh()

    def _refresh(self) -> None:
        model = self._model_var.get()
        total = estimate_messages(self._messages, model)
        window = context_window(model)
        fraction = total / window if window else 0.0

        self._total_lbl.config(text=f"{total:,}")
        self._window_lbl.config(text=f"{window:,} (context window)")
        self._msg_count_lbl.config(text=f"Messages: {len(self._messages)}")
        _update_progress(self._canvas, self._fill, fraction)

        pct = min(fraction * 100, 100)
        if total <= window:
            self._status_lbl.config(
                text=f"Fits in context  ({pct:.1f}% used)",
                foreground=_COLOUR["fit"] if fraction <= 0.8 else _COLOUR["warning"],
            )
        else:
            overflow = total - window
            self._status_lbl.config(
                text=f"Exceeds context by {overflow:,} tokens",
                foreground=_COLOUR["overflow"],
            )

    def refresh_model(self) -> None:
        """Called when the model selector changes."""
        self._refresh()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class TokenFitApp(tk.Tk):
    """Root window for the tokenfit GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"tokenfit {__version__} — token estimator")
        self.minsize(600, 520)
        self.configure(bg=_COLOUR["bg"])

        self._model_var = tk.StringVar()
        self._build_header()
        self._build_notebook()
        self._build_footer()

        # Set a sensible default model
        models = list_models()
        default = "gpt-4" if "gpt-4" in models else models[0]
        self._model_var.set(default)
        self._model_var.trace_add("write", self._on_model_change)

    # --- layout ---------------------------------------------------------------

    def _build_header(self) -> None:
        header = ttk.Frame(self, padding=(_PAD, _PAD, _PAD, 0))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="tokenfit",
            font=("", 16, "bold"),
            foreground=_COLOUR["accent"],
        ).pack(side="left")
        ttk.Label(
            header,
            text="  Estimate LLM token usage without API calls",
            foreground="#616161",
        ).pack(side="left", pady=4)

        # model selector on the right
        model_frame = ttk.Frame(header)
        model_frame.pack(side="right")
        ttk.Label(model_frame, text="Model:").pack(side="left")
        self._model_combo = ttk.Combobox(
            model_frame,
            textvariable=self._model_var,
            values=list_models(),
            width=24,
            state="readonly",
        )
        self._model_combo.pack(side="left", padx=(_PAD // 2, 0))

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self, padding=_PAD)
        self._nb.pack(fill="both", expand=True)

        self._text_tab = _TextTab(self._nb, self._model_var)
        self._nb.add(self._text_tab, text="  Text  ")

        self._msg_tab = _MessagesTab(self._nb, self._model_var)
        self._nb.add(self._msg_tab, text="  Messages  ")

    def _build_footer(self) -> None:
        footer = ttk.Frame(self, padding=(_PAD, 0, _PAD, _PAD // 2))
        footer.pack(fill="x")
        ttk.Label(
            footer,
            text=(
                "Estimates use character-to-token ratios — exact counts require "
                "the model's own tokeniser.  Counts always round up."
            ),
            foreground="#9e9e9e",
            font=("", 9),
        ).pack(side="left")

    # --- callbacks ------------------------------------------------------------

    def _on_model_change(self, *_args: object) -> None:
        self._text_tab.refresh_model()
        self._msg_tab.refresh_model()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the tokenfit GUI."""
    app = TokenFitApp()
    app.mainloop()


if __name__ == "__main__":
    main()
