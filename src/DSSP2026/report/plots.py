from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared base: show() / save() interface
# ---------------------------------------------------------------------------

class _PlotBase:
    """Minimal base for simple plot objects.

    Subclasses implement ``_figure()`` and inherit:
        show()       — display inline (notebook) or open a window
        save(path)   — write PNG to disk
        figure()     — return the raw matplotlib Figure
        _repr_png_() — notebook auto-display hook
    """

    def _figure(self):
        raise NotImplementedError

    def figure(self):
        """Return the matplotlib Figure."""
        return self._figure()

    def show(self):
        """Display the plot. In a notebook renders inline; otherwise opens a
        matplotlib window."""
        import matplotlib.pyplot as plt
        fig = self._figure()
        try:
            # IPython / Jupyter: display inline.
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
        finally:
            plt.close(fig)

    def save(self, path, *, dpi: int = 220):
        """Save the plot to *path* (PNG inferred from extension; pass .png).
        Returns the path string."""
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    # Back-compat alias used by older code.
    def to_png(self, path, *, dpi: int = 220):
        return self.save(path, dpi=dpi)

    def _repr_png_(self):
        import io
        import matplotlib.pyplot as plt
        fig = self._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


class ConfusionPlot(_PlotBase):
    """A confusion-matrix figure that renders inline and exports to PNG.

    Wraps the integer count matrix + display options; rendering goes through the
    package's shared tile-grid primitive so it matches the AT&T heatmap styling.
    Rows are the true class, columns the predicted class; the diagonal is
    outlined. ``normalize`` colors by row-normalized rate (diagonal = recall)
    while still printing raw counts.
    """

    def __init__(self, counts, class_labels, *, normalize=False,
                 title="Confusion matrix", true_axis="True",
                 pred_axis="Predicted", figsize=None, row_labels=None,
                 value_matrix=None, value_text="both"):
        self.counts = np.asarray(counts, dtype=float)
        self.class_labels = list(class_labels)            # column labels
        self.row_labels = list(row_labels) if row_labels is not None \
            else list(class_labels)                        # defaults to columns
        self.normalize = normalize
        self.title = title
        self.true_axis = true_axis
        self.pred_axis = pred_axis
        self.figsize = figsize
        # Monetary mode: cells colored by total realized value (count x payoff)
        # on a diverging firebrick->gold->forest-green scale, symmetric about 0.
        self.value_matrix = (np.asarray(value_matrix, dtype=float)
                             if value_matrix is not None else None)
        self.value_text = value_text   # "both" | "count" | "value"

    # @staticmethod
    # def _money_cmap():
    #     from matplotlib.colors import LinearSegmentedColormap
    #     # firebrick (loss) -> gold (zero) -> forest green (gain)
    #     return LinearSegmentedColormap.from_list(
    #         "cost_diverging", ["#B22222", "#FFFFFF", "#228B22"])

    @staticmethod
    def _fmt_money(v):
        v = float(v)
        sign = "-" if v < 0 else ""
        a = abs(v)
        for thresh, div, suffix in (
            (1e12, 1e12, "T"),
            (1e9, 1e9, "B"),
            (1e6, 1e6, "M"),
            (1e3, 1e3, "k"),
        ):
            if a >= thresh:
                return f"{sign}${a / div:.1f}{suffix}"
        return f"{sign}${a:.0f}"

    def _figure(self):
        from DSSP2026.core.heatmap import plot_tile_grid
        n_rows, n_cols = self.counts.shape
        diagonal = [(i, j) for i in range(n_rows) for j in range(n_cols)
                    if i < len(self.row_labels) and j < len(self.class_labels)
                    and self.row_labels[i] == self.class_labels[j]]

        if self.value_matrix is not None:
            # Monetary coloring: total realized value per cell.
            from matplotlib.colors import Normalize
            vals = self.value_matrix
            finite = vals[np.isfinite(vals)]
            max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
            if max_abs == 0:
                max_abs = 1.0
            norm = Normalize(vmin=-max_abs, vmax=max_abs)   # symmetric about 0
            if self.value_text == "count":
                cell_text = [[f"{int(self.counts[i, j]):,}" for j in range(n_cols)]
                             for i in range(n_rows)]
            elif self.value_text == "value":
                cell_text = [[self._fmt_money(vals[i, j]) for j in range(n_cols)]
                             for i in range(n_rows)]
            else:  # both
                cell_text = [[f"{int(self.counts[i, j]):,}\n{self._fmt_money(vals[i, j])}"
                              for j in range(n_cols)] for i in range(n_rows)]
            return plot_tile_grid(
                vals, row_labels=self.row_labels, col_labels=self.class_labels,
                row_axis_label=self.true_axis, col_axis_label=self.pred_axis,
                title=self.title, cell_text=cell_text,
                colorbar_label="Realized value ($)",
                cmap="att_diverging", norm=norm,
                highlight=diagonal, invert_yaxis=True, figsize=self.figsize)

        if self.normalize:
            row_sums = self.counts.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                color_values = np.where(row_sums > 0, self.counts / row_sums, 0.0)
            cell_text = [[f"{int(self.counts[i, j]):,}\n{color_values[i, j]:.0%}"
                          for j in range(n_cols)] for i in range(n_rows)]
            colorbar_label = "Row-normalized rate"
        else:
            color_values = self.counts
            cell_text = [[f"{int(self.counts[i, j]):,}" for j in range(n_cols)]
                         for i in range(n_rows)]
            colorbar_label = "Count"
        return plot_tile_grid(
            color_values,
            row_labels=self.row_labels, col_labels=self.class_labels,
            row_axis_label=self.true_axis, col_axis_label=self.pred_axis,
            title=self.title, cell_text=cell_text, colorbar_label=colorbar_label,
            highlight=diagonal, invert_yaxis=True, figsize=self.figsize)

    def figure(self):
        """Return the matplotlib Figure (for further tweaking or display)."""
        return self._figure()

    def to_png(self, path, *, dpi: int = 220):
        """Render and save the confusion matrix to a PNG. Returns the path."""
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    def _repr_png_(self):
        # Inline display in notebooks.
        import io
        fig = self._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

class ROCPlot(_PlotBase):
    """An ROC-comparison figure (one curve per model) that exports to PNG.

    ``curves`` is a list of {label, fpr, tpr, auc}, sorted strongest-first.
    Rendered in AT&T palette with the chance diagonal and AUC in the legend.
    """

    def __init__(self, curves, *, average="macro", title="ROC comparison"):
        self.curves = curves
        self.average = average
        self.title = title

    def _figure(self):
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_PALETTE, ATT_COLORS

        fig, ax = plt.subplots(figsize=(7.5, 6))
        for i, c in enumerate(self.curves):
            ax.plot(c["fpr"], c["tpr"], linewidth=2.0,
                    color=ATT_PALETTE[i % len(ATT_PALETTE)],
                    label=f"{c['label']} — AUC {c['auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
                color=ATT_COLORS.get("gray_500", "#888888"))
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False positive rate", fontsize=13)
        ax.set_ylabel("True positive rate", fontsize=13)
        ax.set_title(self.title, fontsize=13, color=ATT_COLORS.get("navy", "#003057"),
                     weight="bold")
        ax.legend(loc="lower right", fontsize=10, frameon=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
        ax.set_axisbelow(True)
        return fig

    def figure(self):
        return self._figure()

    def to_png(self, path, *, dpi: int = 220):
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    def _repr_png_(self):
        import io
        import matplotlib.pyplot as plt
        fig = self._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

class ThresholdSweepPlot(_PlotBase):
    """A threshold-tuning figure: selected metrics vs decision threshold, with a
    vertical line at the threshold that optimizes each chosen metric (one or
    many)."""

    _DISPLAY = {"precision": "Precision", "recall": "Recall", "f1": "F1",
                "youden": "Youden's J", "accuracy": "Accuracy",
                "specificity": "Specificity",
                "false_negative_rate": "FNR", "false_positive_rate": "FPR"}

    def __init__(self, sweep_df, *, plot_metrics, marks, display_aliases=None,
                 target_class, model, title):
        self.sweep_df = sweep_df
        self.plot_metrics = plot_metrics
        # marks: list of {metric, display, threshold, value} — one optimum line each.
        self.marks = list(marks)
        self.display_aliases = dict(display_aliases or {})
        self.target_class = target_class
        self.model = model
        self.title = title

    def _label(self, key):
        """Display label for a metric column, honoring a requested alias."""
        return self.display_aliases.get(key, self._DISPLAY.get(key, key))

    def _figure(self):
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_PALETTE, ATT_COLORS

        df = self.sweep_df.sort_values("threshold")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, m in enumerate(self.plot_metrics):
            ax.plot(df["threshold"], df[m], linewidth=2.0,
                    color=ATT_PALETTE[i % len(ATT_PALETTE)],
                    label=self._label(m))

        # One dashed vertical line per requested optimum, distinctly colored.
        line_palette = [ATT_COLORS.get("magenta", "#c4007a"),
                        ATT_COLORS.get("navy", "#003057"),
                        ATT_COLORS.get("green", "#2a7d4f"),
                        ATT_COLORS.get("orange", "#e07b00"),
                        "#7a3ea0", "#b00020"]
        # Stagger annotation heights so multiple labels don't overlap.
        for j, mk in enumerate(self.marks):
            c = line_palette[j % len(line_palette)]
            disp = mk["display"] if mk["display"] != mk["metric"] else self._label(mk["metric"])
            ax.axvline(mk["threshold"], linestyle="--", linewidth=1.5, color=c,
                       label=f"{disp} optimum (thr={mk['threshold']:.3f})")
            ax.annotate(
                f"{disp}\nthr={mk['threshold']:.3f}, {mk['value']:.3f}",
                xy=(mk["threshold"], 0.02 + 0.12 * (j % 3)),
                xycoords=("data", "axes fraction"),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color=c)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Decision threshold", fontsize=13)
        ax.set_ylabel("Metric value", fontsize=13)
        ax.set_title(self.title, fontsize=13,
                     color=ATT_COLORS.get("navy", "#003057"), weight="bold")
        ax.legend(loc="best", fontsize=9, frameon=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
        ax.set_axisbelow(True)
        return fig

    def figure(self):
        return self._figure()

    def to_png(self, path, *, dpi: int = 220):
        from DSSP2026.core.figure import save_figure
        return save_figure(self._figure(), path, dpi=dpi)

    def _repr_png_(self):
        import io
        import matplotlib.pyplot as plt
        fig = self._figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# FeatureImportancePlot
# ---------------------------------------------------------------------------

class FeatureImportancePlot(_PlotBase):
    """Horizontal bar chart of feature importances for one model.

    Parameters
    ----------
    importance_df : DataFrame
        Columns ``Feature`` and ``Importance``, already filtered to the
        requested ``importance_type`` and sorted descending.
    model : str
        Model display name (used in the title).
    importance_type : str
        The importance type being shown (e.g. "gain", "abs_coef").
    top_n : int, optional
        Truncate to the top-N features. None shows all.
    title : str, optional
        Override the default title.
    figsize : tuple, optional
    """

    def __init__(self, importance_df, *, model, importance_type,
                 top_n=None, title=None, figsize=None):
        self.importance_df = importance_df
        self.model = model
        self.importance_type = importance_type
        self.top_n = top_n
        self.title = title
        self.figsize = figsize

    def _figure(self):
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_COLORS, ATT_SEQUENTIAL

        df = self.importance_df.copy()
        if self.top_n is not None:
            df = df.head(self.top_n)
        df = df.iloc[::-1]   # largest at the top after reversal for hbar

        n = len(df)
        # palette[0] = navy (darkest). Bar indices run bottom→top after
        # iloc[::-1], so bar i=0 is the LEAST important. We want the MOST
        # important bar (i = n-1) to be navy, so assign colors in reverse:
        # colors[n-1] = palette[0] = navy, colors[0] = palette[n-1 % len].
        palette = list(reversed(ATT_SEQUENTIAL))   # navy first
        colors = [palette[(n - 1 - i) % len(palette)] for i in range(n)]

        fig, ax = plt.subplots(
            figsize=self.figsize or (9, max(3, 0.5 * n + 1.5)))
        bars = ax.barh(df["Feature"], df["Importance"],
                       color=colors, zorder=3)
        for y, val in enumerate(df["Importance"]):
            ax.text(val, y, f"  {val:.4f}", va="center", ha="left",
                    fontsize=12, color=ATT_COLORS["gray_700"])
        ax.set_xlabel("Importance")
        ax.set_xlim(0, df["Importance"].max() * 1.15)
        title = self.title or (
            f"Feature importance — {self.model} ({self.importance_type})")
        ax.set_title(title, fontsize=13,
                     color=ATT_COLORS["navy"], weight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="x",
                color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# ClassDistributionPlot
# ---------------------------------------------------------------------------

class ClassDistributionPlot(_PlotBase):
    """Horizontal bar chart of class label counts in a dataset split.

    Parameters
    ----------
    counts : dict or Series
        ``{label: count}`` mapping, ordered by descending count.
    split : str
        One of ``"all"``, ``"train"``, ``"test"`` — used in the title.
    title : str, optional
        Override the default title.
    figsize : tuple, optional

    Colors cycle through ``ATT_SEQUENTIAL`` dark-to-light (same convention as
    ``FeatureImportancePlot``): the most frequent class gets navy, then repeats
    once the palette is exhausted.
    """

    def __init__(self, counts, *, split="all", title=None, figsize=None):
        import pandas as pd
        if isinstance(counts, dict):
            self.counts = pd.Series(counts)
        else:
            self.counts = counts.copy()
        self.counts = self.counts.sort_values(ascending=False)
        self.split = split
        self.title = title
        self.figsize = figsize

    def _figure(self):
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_COLORS, ATT_SEQUENTIAL

        counts = self.counts.iloc[::-1]   # smallest at bottom, largest at top
        n = len(counts)
        # Same convention as FeatureImportancePlot: most frequent (top bar,
        # index n-1 after reversal) gets navy = palette[0].
        palette = list(reversed(ATT_SEQUENTIAL))
        colors = [palette[(n - 1 - i) % len(palette)] for i in range(n)]

        fig, ax = plt.subplots(
            figsize=self.figsize or (9, max(3, 0.5 * n + 1.5)))
        ax.barh(counts.index.astype(str), counts.values,
                color=colors, zorder=3)

        total = counts.sum()
        for y, (label, val) in enumerate(counts.items()):
            pct = 100 * val / total if total > 0 else 0
            ax.text(val, y, f"  {val:,}  ({pct:.1f}%)",
                    va="center", ha="left",
                    fontsize=12, color=ATT_COLORS["gray_700"])

        ax.set_xlabel("Count")
        ax.set_xlim(0, counts.max() * 1.25)
        split_label = {"all": "full dataset",
                       "train": "training set",
                       "test": "test / eval set"}.get(self.split, self.split)
        title = self.title or f"Class distribution — {split_label}"
        ax.set_title(title, fontsize=13,
                     color=ATT_COLORS["navy"], weight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="x",
                color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return fig




class TrainingCurvePlot(_PlotBase):
    """Aggregated training / eval loss curves for the top-N Optuna trials.

    Reads pre-stored curves from ``report.db`` (written by
    ``sidecar.write_trial_curves`` and ETL'd by ``report_builder``).
    Renders mean ± 1 std bands for train and eval loss across the stored
    trials, ranked by CV objective.

    Parameters
    ----------
    train_curves : list of list of float
        Per-trial training loss curves (ragged lengths allowed).
    eval_curves : list of list of float
        Per-trial eval loss curves (same length as train_curves).
    ranks : list of int
        Rank of each trial (1 = best CV value).
    model : str
        Model display name (used in the title).
    title : str, optional
    figsize : tuple, optional
    """

    def __init__(self, train_curves, eval_curves, ranks, *,
                 model, title=None, figsize=None):
        self.train_curves = train_curves
        self.eval_curves = eval_curves
        self.ranks = ranks
        self.model = model
        self.title = title
        self.figsize = figsize

    @staticmethod
    def _stack(curves):
        """Stack ragged curves into a (n, max_len) array, NaN-padded."""
        import numpy as np
        L = max(len(c) for c in curves)
        M = np.full((len(curves), L), np.nan)
        for i, c in enumerate(curves):
            M[i, :len(c)] = c
        return M

    def _figure(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from DSSP2026.core.style import ATT_COLORS

        train_M = self._stack(self.train_curves)
        eval_M = self._stack(self.eval_curves)
        n = train_M.shape[0]

        epochs = np.arange(train_M.shape[1])
        tr_mean = np.nanmean(train_M, axis=0)
        tr_std  = np.nanstd(train_M, axis=0)
        ev_epochs = np.arange(eval_M.shape[1])
        ev_mean = np.nanmean(eval_M, axis=0)
        ev_std  = np.nanstd(eval_M, axis=0)

        fig, ax = plt.subplots(figsize=self.figsize or (9, 5.5))

        ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                        color=ATT_COLORS["deep_blue"], alpha=0.15, linewidth=0)
        ax.plot(epochs, tr_mean, color=ATT_COLORS["deep_blue"],
                linewidth=2.4, label="Train loss")

        ax.fill_between(ev_epochs, ev_mean - ev_std, ev_mean + ev_std,
                        color=ATT_COLORS["orange"], alpha=0.15, linewidth=0)
        ax.plot(ev_epochs, ev_mean, color=ATT_COLORS["orange"],
                linewidth=2.4, label="Eval loss")

        best_ep = int(np.nanargmin(ev_mean))
        ax.axvline(best_ep, color=ATT_COLORS.get("gray_500", "#888"),
                   linestyle="--", linewidth=1.3, alpha=0.8,
                   label=f"Min eval loss (epoch {best_ep})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log loss")
        title = self.title or (
            f"Training curves — {self.model}, top {n} trials (mean ± 1 std)")
        ax.set_title(title, fontsize=13,
                     color=ATT_COLORS.get("navy", "#003057"), weight="bold")
        ax.legend(loc="best", fontsize=10, frameon=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return fig

# ---------------------------------------------------------------------------
# CostThresholdPlot
# ---------------------------------------------------------------------------

class CostThresholdPlot(_PlotBase):
    """Expected net value vs. decision threshold for one or more classes.

    Built by :meth:`CostDecision.threshold_plot`. The curve is colored green
    where expected value > 0 and red where < 0. A dashed vertical line marks
    the threshold that maximises expected net value, annotated with its
    threshold and value.

    Parameters
    ----------
    curves : dict
        ``{class_label: (thresholds_array, ev_array)}`` as produced by
        ``CostDecision._compute_sweep``.
    class_order : list
        Full class list from the experiment (used for subplot ordering).
    model : str
        Model display name (used in titles).
    title : str, optional
        Suptitle for multi-panel figures. Single-panel uses it as the axes
        title.
    figsize : tuple, optional
    """

    def __init__(self, curves, *, class_order, model, title=None, figsize=None):
        self.curves = curves          # {label: (thresholds, ev)}
        self.class_order = class_order
        self.model = model
        self.title = title
        self.figsize = figsize

    def _figure(self):
        import matplotlib.pyplot as plt
        import matplotlib.collections as mc
        import numpy as np
        from DSSP2026.core.style import ATT_COLORS

        GREEN = ATT_COLORS.get("green",  "#1A7A4A")
        RED   = ATT_COLORS.get("magenta", "#C8003A")
        NAVY  = ATT_COLORS["navy"]
        GRAY  = ATT_COLORS.get("gray_300", "#CCCCCC")

        # Preserve class_order ordering for subplot layout.
        ordered_labels = [c for c in self.class_order if c in self.curves]
        n_panels = len(ordered_labels)

        cols = min(n_panels, 2)
        rows = int(np.ceil(n_panels / cols))
        default_figsize = (cols * 7, rows * 4.5)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=self.figsize or default_figsize,
            squeeze=False,
        )

        for idx, label in enumerate(ordered_labels):
            ax = axes[idx // cols][idx % cols]
            thresholds, ev = self.curves[label]

            # --- signed-color line segments ---
            # Build a LineCollection: each segment is two adjacent points,
            # colored by the sign of the midpoint value.
            points = np.array([thresholds, ev]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            mid_ev = (ev[:-1] + ev[1:]) / 2
            seg_colors = [GREEN if v >= 0 else RED for v in mid_ev]
            lc = mc.LineCollection(segments, colors=seg_colors, linewidth=2.5,
                                   zorder=3)
            ax.add_collection(lc)
            ax.set_xlim(thresholds[0], thresholds[-1])
            margin = max(abs(ev.max()), abs(ev.min()), 1e-6) * 0.15
            ax.set_ylim(ev.min() - margin, ev.max() + margin)

            # --- optimum marker ---
            opt_idx = int(np.argmax(ev))
            opt_t   = thresholds[opt_idx]
            opt_ev  = ev[opt_idx]
            ax.axvline(opt_t, color=GRAY, linestyle="--",
                       linewidth=1.3, zorder=2)
            ax.scatter([opt_t], [opt_ev],
                       color=NAVY, s=60, zorder=4, clip_on=False)
            # Annotation: nudge left if near right edge
            ha = "left" if opt_t < 0.75 else "right"
            x_off = 0.012 if ha == "left" else -0.012
            sign = "+" if opt_ev >= 0 else ""
            ax.annotate(
                f"t={opt_t:.2f}\n{sign}{opt_ev:.3f}",
                xy=(opt_t, opt_ev),
                xytext=(opt_t + x_off, opt_ev),
                fontsize=10,
                ha=ha, va="center",
                color=NAVY,
            )

            # --- zero baseline (subtle) ---
            if ev.min() < 0 < ev.max():
                ax.axhline(0, color=GRAY, linewidth=0.8, zorder=1)

            ax.set_xlabel("Threshold", fontsize=11)
            ax.set_ylabel("Expected net value / sample", fontsize=11)
            panel_title = (f"Cost threshold sweep — {label} vs. rest\n"
                           f"{self.model}")
            ax.set_title(panel_title, fontsize=12,
                         color=NAVY, weight="bold")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(True, color=ATT_COLORS.get("gray_100", "#eeeeee"),
                    linewidth=0.7)
            ax.set_axisbelow(True)

        # Hide any unused subplots
        for idx in range(n_panels, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        # Suptitle for multi-panel; single panel uses axes title already set.
        sup = self.title
        if sup is None and n_panels > 1:
            sup = f"Cost threshold sweep — {self.model}"
        if sup and n_panels > 1:
            fig.suptitle(sup, fontsize=13, color=NAVY, weight="bold", y=1.01)

        fig.tight_layout()
        return fig
