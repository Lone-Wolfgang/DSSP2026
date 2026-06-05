from __future__ import annotations

import numpy as np
import pandas as pd

class ConfusionPlot:
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

class ROCPlot:
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

class ThresholdSweepPlot:
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
