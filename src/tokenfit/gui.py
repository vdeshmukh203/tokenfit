"""tokenfit.gui — interactive offline token-count estimator.

Launch via:
    tokenfit-gui          # after ``pip install tokenfit``
    python -m tokenfit    # from a source checkout
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, List

from tokenfit import (
    _DEFAULT_FAMILY,
    _WINDOWS,
    _family,
    estimate_messages,
    estimate_tokens,
    list_models,
)

# ---------------------------------------------------------------------------
# Colour scheme for the fit indicator
# ---------------------------------------------------------------------------
_GREEN = "#27ae60"   # fits with room to spare (<= 80 % used)
_AMBER = "#e67e22"   # fits but tight (> 80 % used)
_RED = "#c0392b"     # overflow


# ---------------------------------------------------------------------------
# Message-row widget
# ---------------------------------------------------------------------------

class _MessageRow(ttk.Frame):
    """One editable row in the chat-messages editor."""

    ROLES = ("user", "assistant", "system", "tool")

    def __init__(self, parent: tk.Widget, on_change: Any) -> None:
        super().__init__(parent, padding=(2, 2))
        self.columnconfigure(1, weight=1)

        self.role_var = tk.StringVar(value="user")
        role_cb = ttk.Combobox(
            self,
            textvariable=self.role_var,
            values=self.ROLES,
            width=12,
            state="readonly",
        )
        role_cb.grid(row=0, column=0, padx=(0, 6))
        role_cb.bind("<<ComboboxSelected>>", on_change)

        self.content_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.content_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 6)
        )
        self.content_var.trace_add("write", on_change)

        self._del_btn = ttk.Button(self, text="✕", width=3)
        self._del_btn.grid(row=0, column=2)

    def set_delete_command(self, cmd: Any) -> None:
        self._del_btn.configure(command=cmd)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role_var.get(), "content": self.content_var.get()}


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class TokenfitApp(tk.Tk):
    """Top-level GUI window for the tokenfit estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.title("tokenfit — Token Count Estimator")
        self.minsize(680, 540)
        self._rows: List[_MessageRow] = []
        self._build_ui()
        self._update()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_notebook()
        self._build_results_panel()

    def _build_top_bar(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=0)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Model:").grid(row=0, column=0, padx=(0, 4))
        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            top,
            textvariable=self._model_var,
            values=list_models(),
            width=26,
            state="readonly",
        )
        model_cb.grid(row=0, column=1, sticky="w", padx=(0, 20))
        self._model_var.trace_add("write", self._update)

        ttk.Label(top, text="Headroom (tokens):").grid(row=0, column=2, padx=(0, 4))
        self._headroom_var = tk.StringVar(value="0")
        ttk.Spinbox(
            top,
            from_=0,
            to=500_000,
            increment=500,
            textvariable=self._headroom_var,
            width=10,
        ).grid(row=0, column=3, sticky="w")
        self._headroom_var.trace_add("write", self._update)

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 4))
        self._nb.bind("<<NotebookTabChanged>>", self._update)

        self._build_text_tab()
        self._build_messages_tab()

    def _build_text_tab(self) -> None:
        frame = ttk.Frame(self._nb, padding=4)
        self._nb.add(frame, text="Plain Text")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._text = tk.Text(frame, wrap="word", undo=True, relief="flat", bd=1)
        self._text.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._text.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=vsb.set)
        self._text.bind("<<Modified>>", self._text_modified)

    def _build_messages_tab(self) -> None:
        outer = ttk.Frame(self._nb, padding=4)
        self._nb.add(outer, text="Chat Messages")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        # Scrollable canvas for the message rows
        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)

        self._rows_frame = ttk.Frame(canvas)
        self._rows_frame.columnconfigure(0, weight=1)
        cwin = canvas.create_window((0, 0), window=self._rows_frame, anchor="nw")
        self._rows_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(cwin, width=e.width),
        )

        ttk.Button(
            outer, text="+ Add Message", command=self._add_row
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self._add_row()  # seed with one empty row

    def _build_results_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="Estimate", padding=(10, 6))
        panel.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Tokens used:").grid(row=0, column=0, sticky="w")
        self._lbl_tokens = ttk.Label(panel, text="—")
        self._lbl_tokens.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(panel, text="Context window:").grid(row=1, column=0, sticky="w")
        self._lbl_window = ttk.Label(panel, text="—")
        self._lbl_window.grid(row=1, column=1, sticky="w", padx=8)

        self._bar = ttk.Progressbar(panel, maximum=100)
        self._bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))

        self._lbl_status = ttk.Label(panel, text="", font=("", 11, "bold"))
        self._lbl_status.grid(row=3, column=0, columnspan=3, sticky="w")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _text_modified(self, *_: Any) -> None:
        if self._text.edit_modified():
            self._update()
            self._text.edit_modified(False)

    def _add_row(self) -> None:
        row = _MessageRow(self._rows_frame, on_change=self._update)
        row.grid(row=len(self._rows), column=0, sticky="ew", pady=2)
        row.set_delete_command(lambda r=row: self._del_row(r))
        self._rows.append(row)
        self._update()

    def _del_row(self, row: _MessageRow) -> None:
        if len(self._rows) <= 1:
            return  # always keep at least one row
        self._rows.remove(row)
        row.destroy()
        for i, r in enumerate(self._rows):
            r.grid(row=i, column=0, sticky="ew", pady=2)
        self._update()

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def _update(self, *_: Any) -> None:
        model = self._model_var.get()
        try:
            headroom = max(0, int(self._headroom_var.get() or 0))
        except ValueError:
            headroom = 0

        tab_idx = self._nb.index("current")
        if tab_idx == 0:
            text = self._text.get("1.0", "end-1c")
            tokens = estimate_tokens(text, model)
        else:
            msgs = [r.to_dict() for r in self._rows]
            tokens = estimate_messages(msgs, model)

        fam = _family(model)
        window = _WINDOWS.get(fam, _WINDOWS[_DEFAULT_FAMILY])
        used = tokens + headroom
        pct = min(100.0, used / window * 100) if window else 0.0
        fits = used <= window

        # Update labels
        if headroom:
            token_text = f"{tokens:,}  + {headroom:,} headroom  =  {used:,} total"
        else:
            token_text = f"{tokens:,}"
        self._lbl_tokens.configure(text=token_text)
        self._lbl_window.configure(text=f"{window:,} tokens")
        self._bar["value"] = pct

        if fits:
            color = _GREEN if pct <= 80 else _AMBER
            status = f"✓  Fits  ({pct:.1f}% of window used)"
        else:
            overflow = used - window
            color = _RED
            status = f"✗  Overflow by {overflow:,} tokens  ({pct:.1f}%)"

        self._lbl_status.configure(text=status, foreground=color)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the tokenfit GUI."""
    app = TokenfitApp()
    app.mainloop()


if __name__ == "__main__":
    main()
