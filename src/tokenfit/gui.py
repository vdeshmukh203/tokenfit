"""tokenfit GUI — interactive token-count estimator built on Tkinter.

Launch with::

    python -m tokenfit
    # or, after pip install:
    tokenfit-gui
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List

from tokenfit import (
    __version__,
    context_window,
    estimate_messages,
    estimate_tokens,
    fits_in_context,
    supported_models,
)

_MODELS: List[str] = supported_models()
_ROLES = ["user", "assistant", "system"]

# Palette used for the "fits" indicator
_GREEN = "#1a7a1a"
_RED = "#b00020"


# ---------------------------------------------------------------------------
# Text-estimation tab
# ---------------------------------------------------------------------------

class _TextTab(ttk.Frame):
    """Estimate tokens for arbitrary plain text."""

    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=0)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build()

    def _build(self) -> None:
        # ---- Settings row --------------------------------------------------
        ctrl = ttk.LabelFrame(self, text=" Settings ", padding=(10, 6))
        ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ctrl.columnconfigure(5, weight=1)

        ttk.Label(ctrl, text="Model:").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            ctrl, textvariable=self._model_var,
            values=_MODELS, state="readonly", width=22,
        )
        model_cb.grid(row=0, column=1, padx=(4, 20), sticky="w")
        model_cb.bind("<<ComboboxSelected>>", self._refresh)

        ttk.Label(ctrl, text="Headroom (tokens):").grid(row=0, column=2, sticky="w")
        self._headroom_var = tk.StringVar(value="0")
        headroom_e = ttk.Entry(ctrl, textvariable=self._headroom_var, width=10)
        headroom_e.grid(row=0, column=3, padx=(4, 0), sticky="w")
        self._headroom_var.trace_add("write", lambda *_: self._refresh())

        # ---- Text area -----------------------------------------------------
        text_frame = ttk.LabelFrame(self, text=" Input text ", padding=(6, 6))
        text_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self._text = tk.Text(
            text_frame, wrap="word", undo=True,
            font=("TkDefaultFont", 10), relief="flat",
            borderwidth=1,
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=self._text.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=vsb.set)
        self._text.bind("<<Modified>>", self._on_text_modified)

        # ---- Results row ---------------------------------------------------
        res = ttk.LabelFrame(self, text=" Results ", padding=(10, 8))
        res.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))
        for col in range(4):
            res.columnconfigure(col, weight=1)

        self._token_var = tk.StringVar(value="—")
        self._window_var = tk.StringVar(value="—")
        self._pct_var = tk.StringVar(value="—")
        self._fits_var = tk.StringVar(value="—")

        metrics = [
            ("Estimated tokens", self._token_var),
            ("Context window", self._window_var),
            ("Window used", self._pct_var),
        ]
        for col, (label, var) in enumerate(metrics):
            ttk.Label(res, text=label, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, sticky="w", padx=6,
            )
            ttk.Label(res, textvariable=var, font=("TkDefaultFont", 15)).grid(
                row=1, column=col, sticky="w", padx=6, pady=(2, 0),
            )

        ttk.Label(res, text="Fits in context", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=3, sticky="w", padx=6,
        )
        self._fits_label = tk.Label(
            res, textvariable=self._fits_var,
            font=("TkDefaultFont", 15, "bold"), bg=res.cget("background"),
        )
        self._fits_label.grid(row=1, column=3, sticky="w", padx=6, pady=(2, 0))

    # ---- Callbacks ---------------------------------------------------------

    def _on_text_modified(self, _event: object) -> None:
        if self._text.edit_modified():
            self._text.edit_modified(False)
            self._refresh()

    def _refresh(self, *_: object) -> None:
        text = self._text.get("1.0", "end-1c")
        model = self._model_var.get()
        try:
            headroom = int(self._headroom_var.get())
        except ValueError:
            headroom = 0

        tokens = estimate_tokens(text, model)
        win = context_window(model)
        fits = fits_in_context(text, model, headroom=headroom)
        pct = (tokens / win * 100) if win > 0 else 0.0

        self._token_var.set(f"{tokens:,}")
        self._window_var.set(f"{win:,}")
        self._pct_var.set(f"{pct:.1f}%")
        self._fits_var.set("Yes" if fits else "No")
        self._fits_label.configure(fg=_GREEN if fits else _RED)


# ---------------------------------------------------------------------------
# Chat-messages estimation tab
# ---------------------------------------------------------------------------

class _MessagesTab(ttk.Frame):
    """Estimate tokens for a multi-turn chat messages list."""

    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=0)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        # iid -> {"role": str, "content": str}
        self._messages: Dict[str, Dict[str, str]] = {}
        self._build()

    def _build(self) -> None:
        # ---- Settings row --------------------------------------------------
        ctrl = ttk.LabelFrame(self, text=" Settings ", padding=(10, 6))
        ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        ttk.Label(ctrl, text="Model:").grid(row=0, column=0, sticky="w")
        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            ctrl, textvariable=self._model_var,
            values=_MODELS, state="readonly", width=22,
        )
        model_cb.grid(row=0, column=1, padx=(4, 0), sticky="w")
        model_cb.bind("<<ComboboxSelected>>", self._refresh)

        # ---- Message list --------------------------------------------------
        list_frame = ttk.LabelFrame(self, text=" Messages ", padding=(6, 6))
        list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        cols = ("role", "preview", "tokens")
        self._tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=10)
        self._tree.heading("role", text="Role")
        self._tree.heading("preview", text="Content preview")
        self._tree.heading("tokens", text="Tokens")
        self._tree.column("role", width=100, minwidth=80, stretch=False)
        self._tree.column("preview", width=480, minwidth=200)
        self._tree.column("tokens", width=80, minwidth=60, stretch=False, anchor="e")
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=vsb.set)

        # ---- Add-message form ----------------------------------------------
        add_frame = ttk.LabelFrame(self, text=" Add message ", padding=(10, 6))
        add_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        add_frame.columnconfigure(1, weight=1)

        ttk.Label(add_frame, text="Role:").grid(row=0, column=0, sticky="w")
        self._role_var = tk.StringVar(value="user")
        role_cb = ttk.Combobox(
            add_frame, textvariable=self._role_var,
            values=_ROLES, state="readonly", width=12,
        )
        role_cb.grid(row=0, column=1, padx=(4, 0), sticky="w")

        ttk.Label(add_frame, text="Content:").grid(row=1, column=0, sticky="nw", pady=(6, 0))
        self._content_text = tk.Text(
            add_frame, height=4, wrap="word", undo=True,
            font=("TkDefaultFont", 10), relief="flat", borderwidth=1,
        )
        self._content_text.grid(row=1, column=1, padx=(4, 0), pady=(6, 0), sticky="ew")
        # Ctrl+Return adds the message without needing the button
        self._content_text.bind("<Control-Return>", lambda _e: self._add_message())

        btn_row = ttk.Frame(add_frame)
        btn_row.grid(row=2, column=1, sticky="e", pady=(6, 0))
        ttk.Button(btn_row, text="Add  (Ctrl+↵)", command=self._add_message).pack(
            side="left", padx=(0, 6),
        )
        ttk.Button(btn_row, text="Remove selected", command=self._remove_selected).pack(
            side="left", padx=(0, 6),
        )
        ttk.Button(btn_row, text="Clear all", command=self._clear_all).pack(side="left")

        # ---- Results row ---------------------------------------------------
        res = ttk.LabelFrame(self, text=" Results ", padding=(10, 8))
        res.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 10))
        for col in range(3):
            res.columnconfigure(col, weight=1)

        self._count_var = tk.StringVar(value="0 messages")
        self._total_var = tk.StringVar(value="0")
        self._window_var = tk.StringVar(value="—")

        for col, (label, var) in enumerate([
            ("Messages", self._count_var),
            ("Total tokens", self._total_var),
            ("Context window", self._window_var),
        ]):
            ttk.Label(res, text=label, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, sticky="w", padx=6,
            )
            ttk.Label(res, textvariable=var, font=("TkDefaultFont", 15)).grid(
                row=1, column=col, sticky="w", padx=6, pady=(2, 0),
            )

    # ---- Callbacks ---------------------------------------------------------

    def _add_message(self) -> None:
        role = self._role_var.get()
        content = self._content_text.get("1.0", "end-1c").strip()
        if not content:
            messagebox.showwarning("Empty content", "Please enter message content before adding.")
            return
        model = self._model_var.get()
        tokens = estimate_tokens(role + content, model)
        preview = content[:90] + ("…" if len(content) > 90 else "")
        iid = self._tree.insert("", "end", values=(role, preview, tokens))
        self._messages[iid] = {"role": role, "content": content}
        self._content_text.delete("1.0", "end")
        self._refresh()

    def _remove_selected(self) -> None:
        selected = self._tree.selection()
        if not selected:
            return
        for iid in selected:
            self._tree.delete(iid)
            self._messages.pop(iid, None)
        self._refresh()

    def _clear_all(self) -> None:
        for iid in list(self._tree.get_children()):
            self._tree.delete(iid)
        self._messages.clear()
        self._refresh()

    def _refresh(self, *_: object) -> None:
        model = self._model_var.get()
        msgs = list(self._messages.values())
        total = estimate_messages(msgs, model)
        win = context_window(model)
        n = len(msgs)
        self._count_var.set(f"{n} message{'s' if n != 1 else ''}")
        self._total_var.set(f"{total:,}")
        self._window_var.set(f"{win:,}")


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

class _AboutTab(ttk.Frame):
    """Brief description and version info."""

    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=20)
        self.columnconfigure(0, weight=1)
        self._build()

    def _build(self) -> None:
        ttk.Label(
            self, text="tokenfit", font=("TkDefaultFont", 22, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self, text=f"v{__version__}  —  heuristic LLM token-count estimator",
            font=("TkDefaultFont", 11),
        ).grid(row=1, column=0, sticky="w", pady=(2, 16))

        body = (
            "tokenfit estimates how many tokens a text or chat conversation\n"
            "will consume for a given model — with no API calls and no external\n"
            "dependencies.\n\n"
            "Text tab   — paste or type any text and see token count, window\n"
            "             usage, and whether it fits with optional headroom.\n\n"
            "Messages tab — build a multi-turn conversation turn-by-turn and\n"
            "               see the cumulative token estimate.\n\n"
            "Supported model families:\n"
            "  GPT-3.5 · GPT-4 · GPT-4-Turbo · GPT-4o\n"
            "  Claude 3 / 3.5 Sonnet / Sonnet 4\n"
            "  Gemini Pro · Gemini 1.5 Pro · Gemini 2.0 Flash\n\n"
            "Estimates always round up so the count never under-reports.\n"
            "Non-English text and code may deviate from the ratio calibration."
        )
        ttk.Label(self, text=body, justify="left", font=("TkDefaultFont", 10)).grid(
            row=2, column=0, sticky="w",
        )
        ttk.Label(
            self, text="MIT License — https://github.com/vdeshmukh203/tokenfit",
            font=("TkDefaultFont", 9), foreground="gray",
        ).grid(row=3, column=0, sticky="w", pady=(20, 0))


# ---------------------------------------------------------------------------
# Application root
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """tokenfit main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"tokenfit {__version__} — LLM Token Estimator")
        self.minsize(700, 500)
        self.geometry("900x640")
        self._build()

    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.add(_TextTab(nb), text="  Text  ")
        nb.add(_MessagesTab(nb), text="  Chat messages  ")
        nb.add(_AboutTab(nb), text="  About  ")
        nb.pack(fill="both", expand=True, padx=8, pady=8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the tokenfit GUI application."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
