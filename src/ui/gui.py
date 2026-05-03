"""
gui.py — Coevolutionary Recommendation System — Control Panel + Results Viewer
Place this file in your project's src/ directory (alongside main.py).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config.parameters import PARAMS
from dataloader.loader import load_dataset
from evolution.coevolution import run_coevolution

# ─────────────────────────────────────────────
# Attempt to import set_seed (optional utility)
# ─────────────────────────────────────────────
try:
    from utils.seeds import set_seed
except ImportError:
    def set_seed(seed):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (dark industrial / utilitarian — fits an EA "engine" theme)
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#1a1a2e"   # deep navy
PANEL     = "#16213e"   # slightly lighter panel
CARD      = "#0f3460"   # card / frame background
ACCENT    = "#e94560"   # hot red accent
ACCENT2   = "#53c0f0"   # cool blue accent
FG        = "#e0e0e0"   # primary text
FG2       = "#8899aa"   # secondary / label text
ENTRY_BG  = "#0d1b2a"
SUCCESS   = "#4ecb71"
FONT_H    = ("Courier New", 11, "bold")
FONT_BODY = ("Courier New", 10)
FONT_SM   = ("Courier New", 9)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def styled_label(parent, text, font=FONT_BODY, color=FG2, **kw):
    return tk.Label(parent, text=text, font=font, fg=color, bg=parent["bg"], **kw)


def styled_entry(parent, textvariable, width=10):
    return tk.Entry(
        parent, textvariable=textvariable, width=width,
        font=FONT_BODY, bg=ENTRY_BG, fg=FG,
        insertbackground=FG, relief="flat",
        highlightthickness=1, highlightcolor=ACCENT2,
        highlightbackground=FG2
    )


def styled_combo(parent, textvariable, values, width=18, **kw):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Dark.TCombobox",
        fieldbackground=ENTRY_BG, background=CARD,
        foreground=FG, arrowcolor=ACCENT2,
        selectbackground=CARD, selectforeground=FG,
        bordercolor=FG2, lightcolor=CARD, darkcolor=CARD
    )
    cb = ttk.Combobox(
        parent, textvariable=textvariable, values=values,
        width=width, state="readonly", style="Dark.TCombobox",
        font=FONT_BODY
    )
    return cb


def make_card(parent, title="", padx=12, pady=8):
    """A titled dark card frame."""
    outer = tk.Frame(parent, bg=CARD, bd=0)
    if title:
        tk.Label(
            outer, text=f"  {title}  ",
            font=FONT_H, fg=ACCENT2, bg=CARD,
            anchor="w"
        ).pack(fill="x", padx=padx, pady=(pady, 2))
        sep = tk.Frame(outer, bg=ACCENT2, height=1)
        sep.pack(fill="x", padx=padx)
    inner = tk.Frame(outer, bg=CARD)
    inner.pack(fill="both", expand=True, padx=padx, pady=pady)
    return outer, inner


def grid_row(frame, row, label_text, widget, col_label=0, col_widget=1,
             pady=5, padx=(0, 20)):
    styled_label(frame, label_text).grid(
        row=row, column=col_label, sticky="w", pady=pady, padx=(0, 10)
    )
    widget.grid(row=row, column=col_widget, sticky="w", pady=pady, padx=padx)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class CoevoApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Coevolutionary Recommendation Engine")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1050, 680)

        # Runtime state
        self._results   = None
        self._running   = False

        self._build_variables()
        self._build_ui()

        # Trace tournament-size enable/disable
        self.sel_var.trace_add("write", self._toggle_tournament)
        # Trace dataset choice
        self.dataset_var.trace_add("write", self._toggle_file_picker)

    # ──────────────────────────────────────────────────────────────────────────
    #  TKINTER VARIABLES
    # ──────────────────────────────────────────────────────────────────────────
    def _build_variables(self):
        p = PARAMS
        self.sel_var        = tk.StringVar(value=p.get("selection_method", "tournament").capitalize())
        self.cx_var         = tk.StringVar(value=p.get("crossover_method", "one_point").replace("_", "-").capitalize())
        self.mut_var        = tk.StringVar(value=p.get("mutation_method", "gaussian").replace("_", " ").capitalize())
        self.init_var       = tk.StringVar(value=p.get("init_method", "uniform").capitalize())
        self.dataset_var    = tk.StringVar(value="Synthetic")

        self.pop_size_var   = tk.StringVar(value=str(p.get("population_size", 50)))
        self.gen_var        = tk.StringVar(value=str(p.get("num_generations", 100)))
        self.mut_rate_var   = tk.StringVar(value=str(p.get("mutation_rate", 0.3)))
        self.cx_rate_var    = tk.StringVar(value=str(p.get("crossover_rate", 0.8)))
        self.tourn_var      = tk.StringVar(value=str(p.get("tournament_size", 3)))
        self.topn_var       = tk.StringVar(value="5")
        self.file_var       = tk.StringVar(value="")
        self.user_sel_var   = tk.StringVar(value="")

    # ──────────────────────────────────────────────────────────────────────────
    #  UI LAYOUT
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=PANEL, height=44)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(
            title_bar,
            text="⚙  COEVOLUTIONARY RECOMMENDATION ENGINE",
            font=("Courier New", 13, "bold"), fg=ACCENT, bg=PANEL
        ).pack(side="left", padx=20, pady=10)

        # Body: left controls  |  right results
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left  = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_controls(left)
        self._build_results(right)

    # ── LEFT: CONTROLS ────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        # ── Evolution Methods ──────────────────────────────────────────────────
        card, inner = make_card(parent, "EVOLUTION METHODS")
        card.pack(fill="x", pady=(0, 8))

        grid_row(inner, 0, "Selection",
                 styled_combo(inner, self.sel_var, ["Tournament", "Roulette"]))
        grid_row(inner, 1, "Crossover",
                 styled_combo(inner, self.cx_var, ["One-point", "Uniform"]))
        grid_row(inner, 2, "Mutation",
                 styled_combo(inner, self.mut_var, ["Gaussian", "Random reset"]))
        grid_row(inner, 3, "Initialization",
                 styled_combo(inner, self.init_var, ["Uniform", "Gaussian"]))

        # ── Hyperparameters ────────────────────────────────────────────────────
        card2, inner2 = make_card(parent, "HYPERPARAMETERS")
        card2.pack(fill="x", pady=(0, 8))

        entries = [
            ("Population size",  self.pop_size_var),
            ("Generations",      self.gen_var),
            ("Mutation rate",    self.mut_rate_var),
            ("Crossover rate",   self.cx_rate_var),
        ]
        for i, (lbl, var) in enumerate(entries):
            grid_row(inner2, i, lbl, styled_entry(inner2, var))

        # Tournament size (conditionally enabled)
        styled_label(inner2, "Tournament size").grid(
            row=len(entries), column=0, sticky="w", pady=5, padx=(0, 10))
        self._tourn_entry = styled_entry(inner2, self.tourn_var)
        self._tourn_entry.grid(row=len(entries), column=1, sticky="w", pady=5)
        self._toggle_tournament()

        # ── Dataset ────────────────────────────────────────────────────────────
        card3, inner3 = make_card(parent, "DATASET")
        card3.pack(fill="x", pady=(0, 8))

        grid_row(inner3, 0, "Source",
                 styled_combo(inner3, self.dataset_var, ["Synthetic", "Real (CSV)"]))

        self._file_frame = tk.Frame(inner3, bg=CARD)
        self._file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)

        styled_label(self._file_frame, "CSV path").pack(side="left", padx=(0, 8))
        self._file_entry = styled_entry(self._file_frame, self.file_var, width=18)
        self._file_entry.pack(side="left")
        tk.Button(
            self._file_frame, text="Browse",
            font=FONT_SM, fg=FG, bg=ACCENT2, activebackground=ACCENT,
            activeforeground=FG, relief="flat", bd=0, padx=6, pady=2,
            command=self._browse_file
        ).pack(side="left", padx=(6, 0))
        self._toggle_file_picker()

        # ── Results Options ─────────────────────────────────────────────────────
        card4, inner4 = make_card(parent, "RESULTS OPTIONS")
        card4.pack(fill="x", pady=(0, 8))
        grid_row(inner4, 0, "Top-N recommendations", styled_entry(inner4, self.topn_var, 5))

        # ── Run button ─────────────────────────────────────────────────────────
        self._run_btn = tk.Button(
            parent,
            text="▶  RUN EVOLUTION",
            font=("Courier New", 12, "bold"),
            fg=BG, bg=ACCENT,
            activebackground="#c73652", activeforeground=BG,
            relief="flat", bd=0, pady=10,
            command=self._on_run
        )
        self._run_btn.pack(fill="x", pady=(4, 0))

        self._status_lbl = tk.Label(
            parent, text="", font=FONT_SM, fg=ACCENT2, bg=BG, anchor="w"
        )
        self._status_lbl.pack(fill="x", pady=(4, 0))

    # ── RIGHT: RESULTS ─────────────────────────────────────────────────────────
    def _build_results(self, parent):
        parent.rowconfigure(0, weight=2)
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # ── Fitness Plot ───────────────────────────────────────────────────────
        plot_card, plot_inner = make_card(parent, "FITNESS OVER GENERATIONS")
        plot_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        plot_inner.columnconfigure(0, weight=1)
        plot_inner.rowconfigure(0, weight=1)

        self._fig = Figure(figsize=(6, 3.2), dpi=96, facecolor=CARD)
        self._ax  = self._fig.add_subplot(111)
        self._style_axes(self._ax)
        self._ax.set_title("Run the engine to see results", color=FG2, fontsize=9)

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_inner)
        self._canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

        # ── Recommendations ────────────────────────────────────────────────────
        rec_card, rec_inner = make_card(parent, "TOP-N RECOMMENDATIONS")
        rec_card.grid(row=1, column=0, sticky="nsew")
        rec_inner.columnconfigure(1, weight=1)

        styled_label(rec_inner, "Select user:").grid(
            row=0, column=0, sticky="w", pady=6, padx=(0, 10))
        self._user_combo = styled_combo(
            rec_inner, self.user_sel_var, [], width=12)
        self._user_combo.grid(row=0, column=1, sticky="w", pady=6)
        self._user_combo.bind("<<ComboboxSelected>>", self._on_user_selected)

        # Recommendations table
        self._rec_frame = tk.Frame(rec_inner, bg=CARD)
        self._rec_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)

        self._rec_header()

    def _rec_header(self):
        for w in self._rec_frame.winfo_children():
            w.destroy()
        cols = ["Rank", "Item ID", "Predicted Score", "Bar"]
        widths = [6, 12, 18, 30]
        for c, (col, w) in enumerate(zip(cols, widths)):
            tk.Label(
                self._rec_frame, text=col, width=w,
                font=("Courier New", 9, "bold"), fg=ACCENT2, bg=PANEL,
                anchor="w", padx=4, pady=3
            ).grid(row=0, column=c, sticky="ew", padx=1, pady=(0, 2))

    # ──────────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_tournament(self, *_):
        if self.sel_var.get().lower() == "tournament":
            self._tourn_entry.configure(state="normal", fg=FG)
        else:
            self._tourn_entry.configure(state="disabled", fg=FG2)

    def _toggle_file_picker(self, *_):
        if self.dataset_var.get().lower().startswith("real"):
            self._file_frame.grid()
        else:
            self._file_frame.grid_remove()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)

    def _on_run(self):
        if self._running:
            return
        self._running = True
        self._run_btn.configure(state="disabled", text="⏳ Running…")
        self._status_lbl.configure(text="Initialising…", fg=ACCENT2)
        self.update()
        threading.Thread(target=self._run_backend, daemon=True).start()

    def _run_backend(self):
        try:
            cfg = self._build_config()
            set_seed(cfg.get("random_seed", 42))

            self._set_status("Loading dataset…")
            dataset = self._load_data()

            self._set_status(f"Evolving for {cfg['num_generations']} generations…")
            results = run_coevolution(dataset=dataset, config=cfg)

            self._results   = results
            self._dataset   = dataset
            self._cfg       = cfg

            self.after(0, self._display_results)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self.after(0, self._reset_run_btn)

    def _build_config(self):
        def _f(v): return float(v.get())
        def _i(v): return int(v.get())

        sel = self.sel_var.get().lower()
        cx  = self.cx_var.get().lower().replace("-", "_").replace(" ", "_")
        mut = self.mut_var.get().lower().replace(" ", "_")
        ini = self.init_var.get().lower()

        return {
            **PARAMS,
            "population_size":   _i(self.pop_size_var),
            "num_generations":   _i(self.gen_var),
            "mutation_rate":     _f(self.mut_rate_var),
            "crossover_rate":    _f(self.cx_rate_var),
            "tournament_size":   _i(self.tourn_var),
            "selection_method":  sel,
            "crossover_method":  cx,
            "mutation_method":   mut,
            "init_method":       ini,
        }

    def _load_data(self):
        mode = self.dataset_var.get().lower()
        if mode.startswith("real"):
            path = self.file_var.get().strip()
            if not path:
                raise ValueError("Please select a CSV file for the real dataset.")
            return load_dataset(mode="real", path=path)
        return load_dataset(mode="synthetic")

    def _display_results(self):
        r = self._results
        self._plot_fitness(r["user_history"], r["item_history"])

        # Populate user dropdown
        user_ids = sorted(
            [u.user_id for u in r["users"]],
            key=lambda x: (str(type(x)), x)
        )
        self._user_combo.configure(values=[str(uid) for uid in user_ids])
        if user_ids:
            self.user_sel_var.set(str(user_ids[0]))
            self._on_user_selected()

        best_u = max(r["users"], key=lambda u: u.fitness or -1e9)
        best_i = max(r["items"], key=lambda i: i.fitness or -1e9)
        self._set_status(
            f"Done — best user fitness: {best_u.fitness:.4f}  |  "
            f"best item fitness: {best_i.fitness:.4f}",
            color=SUCCESS
        )
        self._reset_run_btn()

    def _on_user_selected(self, *_):
        if not self._results:
            return
        uid_str = self.user_sel_var.get()
        if not uid_str:
            return
        try:
            uid = int(uid_str)
        except ValueError:
            uid = uid_str

        users = self._results["users"]
        items = self._results["items"]

        user = next((u for u in users if u.user_id == uid), None)
        if user is None:
            return

        try:
            topn = max(1, int(self.topn_var.get()))
        except ValueError:
            topn = 5

        scores = {}
        for item in items:
            s = float(np.dot(user.vector, item.vector))
            if item.item_id not in scores or s > scores[item.item_id]:
                scores[item.item_id] = s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
        self._show_recommendations(ranked)

    def _show_recommendations(self, ranked):
        self._rec_header()

        if not ranked:
            styled_label(self._rec_frame, "No recommendations available.").grid(
                row=1, column=0, columnspan=4, pady=8)
            return

        max_score = max(abs(s) for _, s in ranked) or 1.0

        for rank, (iid, score) in enumerate(ranked, 1):
            bg_row = PANEL if rank % 2 == 0 else CARD
            bar_width = int(28 * abs(score) / max_score)
            bar_str = "█" * bar_width + "░" * (28 - bar_width)
            bar_col = ACCENT if score >= 0 else "#e06c75"

            cells = [str(rank), str(iid), f"{score:+.4f}", bar_str]
            colors = [FG2, FG, SUCCESS if score >= 0 else "#e06c75", bar_col]
            widths = [6, 12, 18, 30]

            for c, (val, col, w) in enumerate(zip(cells, colors, widths)):
                tk.Label(
                    self._rec_frame, text=val, width=w,
                    font=FONT_SM, fg=col, bg=bg_row,
                    anchor="w", padx=4, pady=4
                ).grid(row=rank, column=c, sticky="ew", padx=1, pady=1)

    # ──────────────────────────────────────────────────────────────────────────
    #  PLOTTING
    # ──────────────────────────────────────────────────────────────────────────
    def _style_axes(self, ax):
        ax.set_facecolor(ENTRY_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(FG2)
            spine.set_linewidth(0.6)
        ax.tick_params(colors=FG2, labelsize=8)
        ax.xaxis.label.set_color(FG2)
        ax.yaxis.label.set_color(FG2)
        ax.title.set_color(FG2)
        self._fig.patch.set_facecolor(CARD)
        ax.set_facecolor(ENTRY_BG)

    def _plot_fitness(self, user_hist, item_hist):
        self._ax.clear()
        self._style_axes(self._ax)

        gens = range(1, len(user_hist) + 1)
        self._ax.plot(gens, user_hist, color=ACCENT2,  linewidth=1.8,
                      label="User best fitness")
        self._ax.plot(gens, item_hist, color=ACCENT,   linewidth=1.8,
                      label="Item best fitness",  linestyle="--")

        self._ax.set_xlabel("Generation", fontsize=8)
        self._ax.set_ylabel("Fitness (−MSE)", fontsize=8)
        self._ax.set_title("Best Fitness per Generation", color=FG, fontsize=9)
        self._ax.legend(fontsize=8, facecolor=PANEL, edgecolor=FG2,
                        labelcolor=FG, framealpha=0.8)
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    # ──────────────────────────────────────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=ACCENT2):
        self.after(0, lambda: self._status_lbl.configure(text=msg, fg=color))

    def _reset_run_btn(self):
        self._running = False
        self._run_btn.configure(state="normal", text="▶  RUN EVOLUTION")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = CoevoApp()
    app.mainloop()
