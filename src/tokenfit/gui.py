"""Graphical user interface for tokenfit."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

from . import (
    TokenEstimate,
    _WINDOWS,
    _family,
    estimate_messages_detailed,
    estimate_tokens_detailed,
)

_KNOWN_MODELS: List[str] = [
    # OpenAI
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    # Anthropic
    "claude-3.7-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-haiku-4",
    "claude-3-opus",
    "claude-3-haiku",
    # Google
    "gemini-2.5",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
    # Open-source
    "llama-3",
    "llama-2",
    "mistral-large",
    "mistral-7b",
    "deepseek-r1",
    "command-r",
]


def _fmt(n: int) -> str:
    return f"{n:,}"


class _ResultsPanel:
    """Manages the shared results section at the bottom of the window."""

    def __init__(self, parent: ttk.LabelFrame) -> None:
        parent.columnconfigure(1, weight=1)

        # Row 0 — token count + fits label
        ttk.Label(parent, text="Token estimate:").grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=4
        )
        self._count_var = tk.StringVar(value="—")
        ttk.Label(
            parent,
            textvariable=self._count_var,
            font=("TkDefaultFont", 22, "bold"),
        ).grid(row=0, column=1, sticky=tk.W, padx=8)

        self._status_var = tk.StringVar(value="")
        self._status_lbl = ttk.Label(
            parent,
            textvariable=self._status_var,
            font=("TkDefaultFont", 11, "bold"),
        )
        self._status_lbl.grid(row=0, column=2, sticky=tk.E, padx=12)

        # Row 1 — progress bar
        ttk.Label(parent, text="Context usage:").grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=(4, 2)
        )
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress = ttk.Progressbar(
            parent,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress.grid(
            row=1, column=1, columnspan=2, sticky=tk.EW, padx=8, pady=(4, 2)
        )

        # Row 2 — details
        ttk.Label(parent, text="Details:").grid(
            row=2, column=0, sticky=tk.NW, padx=8, pady=(2, 6)
        )
        self._details_var = tk.StringVar(value="Run estimation to see details.")
        ttk.Label(
            parent,
            textvariable=self._details_var,
            wraplength=460,
            justify=tk.LEFT,
        ).grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=8, pady=(2, 6))

    def update(self, est: TokenEstimate, headroom: int = 0) -> None:
        total = est.tokens + max(0, headroom)
        pct = min(100.0, total / est.window_size * 100) if est.window_size > 0 else 100.0
        remaining = max(0, est.window_size - total)
        fits = total <= est.window_size

        self._count_var.set(_fmt(est.tokens))
        self._progress_var.set(pct)

        if fits:
            self._status_var.set("✓  Fits in context")
            self._status_lbl.configure(foreground="#197a1f")
        else:
            self._status_var.set("✗  Exceeds context")
            self._status_lbl.configure(foreground="#c0392b")

        hr_note = f"  +{_fmt(headroom)} headroom" if headroom else ""
        self._details_var.set(
            f"Window: {_fmt(est.window_size)} tokens"
            f"  |  Used: {_fmt(total)}{hr_note}"
            f"  |  Remaining: {_fmt(remaining)}"
            f"  |  {pct:.1f}% full"
        )

    def clear(self) -> None:
        self._count_var.set("—")
        self._progress_var.set(0.0)
        self._status_var.set("")
        self._status_lbl.configure(foreground="black")
        self._details_var.set("Run estimation to see details.")


class TokenFitApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TokenFit — LLM Token Estimator")
        self.root.geometry("800x620")
        self.root.minsize(640, 520)

        self._messages: List[Dict[str, str]] = []
        self._after_id: Optional[str] = None

        self._setup_styles()
        self._build_ui()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass  # fall back to default theme on platforms where clam is absent

        style.configure("Header.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor="#e0e0e0",
            background="#27ae60",
        )
        style.configure(
            "red.Horizontal.TProgressbar",
            troughcolor="#e0e0e0",
            background="#e74c3c",
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(
            root_frame, text="TokenFit — LLM Token Estimator", style="Header.TLabel"
        ).pack(anchor=tk.W, pady=(0, 8))

        # Options row
        self._build_options(root_frame)

        # Notebook
        nb = ttk.Notebook(root_frame)
        nb.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        text_tab = ttk.Frame(nb, padding=6)
        nb.add(text_tab, text="  Plain Text  ")
        self._build_text_tab(text_tab)

        chat_tab = ttk.Frame(nb, padding=6)
        nb.add(chat_tab, text="  Chat Messages  ")
        self._build_chat_tab(chat_tab)

        # Results
        results_frame = ttk.LabelFrame(root_frame, text="Results", padding=6)
        results_frame.pack(fill=tk.X, pady=(8, 0))
        self._results = _ResultsPanel(results_frame)

    def _build_options(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Options", padding="6 4")
        frame.pack(fill=tk.X, pady=(0, 0))

        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(4, 2))

        self._model_var = tk.StringVar(value="gpt-4")
        model_cb = ttk.Combobox(
            frame,
            textvariable=self._model_var,
            values=_KNOWN_MODELS,
            width=22,
            state="normal",
        )
        model_cb.grid(row=0, column=1, sticky=tk.W, padx=(0, 16))
        model_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_model_change())

        ttk.Label(frame, text="Headroom (tokens):").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 2)
        )
        self._headroom_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self._headroom_var, width=10).grid(
            row=0, column=3, sticky=tk.W, padx=(0, 16)
        )

        self._window_var = tk.StringVar()
        ttk.Label(frame, textvariable=self._window_var, foreground="#555555").grid(
            row=0, column=4, sticky=tk.W
        )

        self._model_var.trace_add("write", lambda *_: self._on_model_change())
        self._on_model_change()

    def _build_text_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Enter text to estimate:").pack(anchor=tk.W)

        self._text_input = scrolledtext.ScrolledText(
            parent, height=11, wrap=tk.WORD, font=("TkFixedFont", 10)
        )
        self._text_input.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._text_input.bind("<KeyRelease>", self._schedule_text_estimate)

        info_row = ttk.Frame(parent)
        info_row.pack(fill=tk.X, pady=(4, 0))

        self._char_var = tk.StringVar(value="0 characters")
        ttk.Label(info_row, textvariable=self._char_var, foreground="#666666").pack(
            side=tk.LEFT
        )

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            btn_row, text="Estimate", command=self._estimate_text
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_row,
            text="Clear",
            command=lambda: (
                self._text_input.delete("1.0", tk.END),
                self._char_var.set("0 characters"),
                self._results.clear(),
            ),
        ).pack(side=tk.LEFT)

    def _build_chat_tab(self, parent: ttk.Frame) -> None:
        # Add-message form
        add_frame = ttk.LabelFrame(parent, text="Add message", padding="6 4")
        add_frame.pack(fill=tk.X, pady=(0, 6))
        add_frame.columnconfigure(3, weight=1)

        ttk.Label(add_frame, text="Role:").grid(
            row=0, column=0, sticky=tk.W, padx=(4, 2)
        )
        self._role_var = tk.StringVar(value="user")
        ttk.Combobox(
            add_frame,
            textvariable=self._role_var,
            values=["user", "assistant", "system"],
            width=10,
            state="readonly",
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

        ttk.Label(add_frame, text="Content:").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 2)
        )
        self._chat_content_var = tk.StringVar()
        content_entry = ttk.Entry(
            add_frame, textvariable=self._chat_content_var, width=42
        )
        content_entry.grid(row=0, column=3, sticky=tk.EW, padx=(0, 6))
        content_entry.bind("<Return>", lambda _e: self._add_message())

        ttk.Button(add_frame, text="Add", command=self._add_message).grid(
            row=0, column=4
        )

        # Message list
        list_frame = ttk.LabelFrame(parent, text="Messages", padding="4 4")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self._msg_listbox = tk.Listbox(
            list_frame, height=7, font=("TkFixedFont", 9), selectmode=tk.SINGLE
        )
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self._msg_listbox.yview
        )
        self._msg_listbox.configure(yscrollcommand=scrollbar.set)
        self._msg_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Chat buttons
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X)
        ttk.Button(
            btn_row, text="Remove selected", command=self._remove_message
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_row, text="Clear all", command=self._clear_messages
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Button(
            btn_row, text="Estimate messages", command=self._estimate_messages
        ).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_model_change(self) -> None:
        import warnings

        model = self._model_var.get()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fam = _family(model)
        window = _WINDOWS.get(fam, 8_192)
        self._window_var.set(f"Context window: {_fmt(window)} tokens")

    def _get_headroom(self) -> int:
        try:
            return max(0, int(self._headroom_var.get()))
        except ValueError:
            return 0

    def _schedule_text_estimate(self, _event: object = None) -> None:
        """Debounce live estimation to 300 ms after the last keystroke."""
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(300, self._estimate_text_live)

    def _estimate_text_live(self) -> None:
        self._after_id = None
        text = self._text_input.get("1.0", tk.END).rstrip("\n")
        self._char_var.set(f"{len(text):,} characters")
        if text.strip():
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = estimate_tokens_detailed(text, self._model_var.get())
            self._results.update(est, self._get_headroom())

    def _estimate_text(self) -> None:
        text = self._text_input.get("1.0", tk.END).rstrip("\n")
        self._char_var.set(f"{len(text):,} characters")
        if not text.strip():
            messagebox.showinfo("No input", "Please enter some text to estimate.")
            return
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = estimate_tokens_detailed(text, self._model_var.get())
        self._results.update(est, self._get_headroom())

    def _add_message(self) -> None:
        role = self._role_var.get()
        content = self._chat_content_var.get().strip()
        if not content:
            messagebox.showwarning("Empty content", "Please enter message content.")
            return
        self._messages.append({"role": role, "content": content})
        preview = content[:60] + "…" if len(content) > 60 else content
        self._msg_listbox.insert(tk.END, f"[{role}]  {preview}")
        self._chat_content_var.set("")

    def _remove_message(self) -> None:
        sel = self._msg_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._msg_listbox.delete(idx)
        self._messages.pop(idx)

    def _clear_messages(self) -> None:
        self._messages.clear()
        self._msg_listbox.delete(0, tk.END)
        self._results.clear()

    def _estimate_messages(self) -> None:
        if not self._messages:
            messagebox.showinfo("No messages", "Please add at least one message.")
            return
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = estimate_messages_detailed(self._messages, self._model_var.get())
        self._results.update(est, self._get_headroom())


def main() -> None:
    """Launch the TokenFit GUI."""
    root = tk.Tk()
    app = TokenFitApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
