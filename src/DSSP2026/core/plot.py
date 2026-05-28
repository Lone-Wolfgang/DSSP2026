"""
core/plot.py — model-agnostic plotting helpers.

Holds figure utilities shared across every model family (saving figures,
the cross-family MSPE/R² comparison chart) plus small helpers for applying
AT&T styling to matplotlib and plotly figures.
"""

from typing import Optional, Sequence, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from DSSP2026.core.style import ATT_COLORS, ATT_PALETTE, ATT_SEQUENTIAL


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_R2_VARIANTS = ("R²", "R2")


def r2_col(df: pd.DataFrame) -> str:
    """Return whichever R² column name is present in *df*.

    Accepts the superscript unicode variant (``R²``) and the plain-ASCII
    fallback (``R2``). Raises KeyError with a helpful message if neither is
    found.
    """
    for name in _R2_VARIANTS:
        if name in df.columns:
            return name
    raise KeyError(
        f"DataFrame has no R² column. Expected one of {_R2_VARIANTS}; "
        f"got columns: {list(df.columns)}"
    )


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    *,
    dpi: int = 220,
    facecolor: str = ATT_COLORS["white"],
    **kwargs,
) -> str:
    """Save *fig* to *path* and close it. Returns the path as a string."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor, **kwargs)
    plt.close(fig)
    return str(path)


def apply_plotly_att_style(
    fig,
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 1000,
    height: int = 600,
    equal_aspect: bool = False,
):
    """Apply the shared AT&T plotly layout (fonts, colors, axes, hoverlabel).

    Centralizes the layout boilerplate that the interactive prediction plots
    would otherwise each repeat.
    """
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=22, color=ATT_COLORS["black"])),
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor=ATT_COLORS["white"],
        paper_bgcolor=ATT_COLORS["white"],
        font=dict(family="Helvetica, Arial, sans-serif",
                  size=14, color=ATT_COLORS["gray_900"]),
        hoverlabel=dict(
            bgcolor=ATT_COLORS["white"],
            bordercolor=ATT_COLORS["navy"],
            font=dict(family="Helvetica, Arial, sans-serif",
                      size=13, color=ATT_COLORS["gray_900"]),
            align="left",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=ATT_COLORS["gray_300"],
            borderwidth=1,
        ),
        width=width, height=height,
        margin=dict(l=70, r=40, t=80, b=70),
    )
    fig.update_xaxes(showgrid=True, gridcolor=ATT_COLORS["gray_300"],
                     zeroline=False, linecolor=ATT_COLORS["gray_900"],
                     title_font=dict(size=16))
    y_kwargs = dict(showgrid=True, gridcolor=ATT_COLORS["gray_300"],
                    zeroline=False, linecolor=ATT_COLORS["gray_900"],
                    title_font=dict(size=16))
    if equal_aspect:
        y_kwargs.update(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(**y_kwargs)
    return fig


def plot_class_distribution(
    labels,
    *,
    label_order: Optional[Sequence] = None,
    title: str = "Class distribution",
    x_label: str = "Count",
    y_label: str = "Class",
    palette=ATT_SEQUENTIAL,
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Horizontal bar plot for class-label distributions.

    Parameters
    ----------
    labels : array-like, Series, or dict
        Class labels or pre-aggregated {label: count} mapping.
    label_order : sequence, optional
        Explicit class ordering. Defaults to highest-frequency → lowest.
    palette : sequence of colors
        Defaults to ATT_SEQUENTIAL. Can also pass ATT_CATEGORICAL.
    """
    if isinstance(labels, dict):
        counts = pd.Series(labels)
    else:
        counts = pd.Series(labels).value_counts(dropna=False)

    # -----------------------------------------------------------------------
    # Ordering
    # -----------------------------------------------------------------------
    if label_order is not None:
        missing = [x for x in label_order if x not in counts.index]
        if missing:
            raise ValueError(f"Labels not found in data: {missing}")

        counts = counts.reindex(label_order)
    else:
        counts = counts.sort_values(ascending=False)

    class_names = [str(x) for x in counts.index.tolist()]
    class_counts = counts.values.astype(float)

    if ax is None:
        fig_height = max(4, len(class_names) * 0.65)
        fig, ax = plt.subplots(figsize=figsize or (8, fig_height))
    else:
        fig = ax.figure

    # -----------------------------------------------------------------------
    # Dynamic color mapping
    # -----------------------------------------------------------------------
    if palette == ATT_SEQUENTIAL:
        norm = Normalize(
            vmin=float(class_counts.min()),
            vmax=float(class_counts.max()),
        )

        cmap = LinearSegmentedColormap.from_list(
            "att_class_distribution",
            ATT_SEQUENTIAL,
        )

        bar_colors = [cmap(norm(v)) for v in class_counts]

    else:
        palette_list = list(palette)

        bar_colors = (
            palette_list * ((len(class_names) // len(palette_list)) + 1)
        )[: len(class_names)]

    bars = ax.barh(
        class_names,
        class_counts,
        color=bar_colors,
        edgecolor=ATT_COLORS["white"],
        linewidth=1.2,
        height=0.72,
        zorder=3,
    )

    # Highest count at top
    ax.invert_yaxis()

    # -----------------------------------------------------------------------
    # Value labels
    # -----------------------------------------------------------------------
    x_offset = max(class_counts) * 0.012 if len(class_counts) else 0.5

    for bar, value in zip(bars, class_counts):
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=ATT_COLORS["gray_900"],
        )

    # -----------------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------------
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)

    ax.grid(
        axis="x",
        color=ATT_COLORS["gray_300"],
        linewidth=0.8,
        alpha=0.8,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(ATT_COLORS["gray_500"])
    ax.spines["bottom"].set_color(ATT_COLORS["gray_500"])

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xlim(0, max(class_counts) * 1.12 if len(class_counts) else 1)

    fig.tight_layout()
    return fig

def save_class_distribution_png(
    labels,
    path,
    *,
    label_order: Optional[Sequence] = None,
    title: str = "Class distribution",
    x_label: str = "Count",
    y_label: str = "Class",
    palette=ATT_SEQUENTIAL,
    dpi: int = 220,
    **kwargs,
):
    """Render and save a class-distribution bar chart."""
    fig = plot_class_distribution(
        labels,
        label_order=label_order,
        title=title,
        x_label=x_label,
        y_label=y_label,
        palette=palette,
        **kwargs,
    )

    return save_figure(fig, path, dpi=dpi)

# ---------------------------------------------------------------------------
# Tile / heatmap primitive (shared by grid-search heatmap, confusion matrix)
# ---------------------------------------------------------------------------

ATT_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("att_sequential", ATT_SEQUENTIAL)


def _text_color_for(rgba) -> str:
    """Pick black or white text for legibility on a given cell color.

    Uses relative luminance (WCAG-style) so light cells get dark text and
    dark cells get light text — independent of which cell is 'best'.
    """
    r, g, b = rgba[0], rgba[1], rgba[2]

    def _lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return ATT_COLORS["gray_900"] if lum > 0.5 else ATT_COLORS["white"]


def plot_tile_grid(
    values,
    *,
    row_labels: Sequence,
    col_labels: Sequence,
    row_axis_label: str = "",
    col_axis_label: str = "",
    title: str = "",
    cell_text=None,
    cmap=ATT_SEQUENTIAL_CMAP,
    colorbar_label: str = "",
    highlight=None,
    text_fontsize: int = 14,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
    invert_yaxis: bool = True,
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Render a labelled grid of colored tiles (heatmap).

    Shared primitive for the grid-search heatmap and the classification
    confusion matrix. Text color is chosen per-cell by luminance, so values
    stay readable on both light and dark tiles.

    Parameters
    ----------
    values : 2-D array-like, shape (n_rows, n_cols)
        The numeric values that drive cell color.
    row_labels, col_labels : sequences
        Tick labels for each axis.
    cell_text : 2-D array-like of str, optional
        Text drawn in each cell. Defaults to the formatted `values`.
    highlight : (row, col) or list of (row, col), optional
        Cells to outline (e.g. the best grid point, or the diagonal).
    invert_yaxis : bool
        True puts the first row at the top (natural for matrices).
    """
    grid = np.asarray(values, dtype=float)
    n_rows, n_cols = grid.shape

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (1.5 * n_cols + 3, 1.15 * n_rows + 2.5))
    else:
        fig = ax.figure

    finite = grid[np.isfinite(grid)]
    norm = Normalize(vmin=finite.min(), vmax=finite.max()) if finite.size else Normalize()
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    im = ax.imshow(grid, cmap=cmap_obj, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=tick_fontsize)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
    ax.set_xlabel(col_axis_label, fontsize=label_fontsize)
    ax.set_ylabel(row_axis_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title)

    highlight_set = set()
    if highlight is not None:
        if isinstance(highlight, tuple):
            highlight = [highlight]
        highlight_set = {tuple(h) for h in highlight}

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(grid[i, j]):
                continue
            if cell_text is not None:
                label = str(cell_text[i][j])
            else:
                label = f"{grid[i, j]:,.0f}"
            rgba = cmap_obj(norm(grid[i, j]))
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=text_fontsize, color=_text_color_for(rgba),
                    fontweight="bold" if (i, j) in highlight_set else "normal")
            if (i, j) in highlight_set:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=ATT_COLORS["orange"], linewidth=3))

    if invert_yaxis:
        ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=label_fontsize)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-family model comparison chart (MSPE / R²)
# ---------------------------------------------------------------------------

def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Two-panel bar chart comparing models by MSPE and R².

    Expects columns: 'Model', 'MSPE', and an R² column ('R²' or 'R2').
    """
    model_labels = comparison_df["Model"].tolist()
    mspe_vals = comparison_df["MSPE"].tolist()
    r2_vals = comparison_df[r2_col(comparison_df)].tolist()

    bar_palette = [ATT_COLORS["deep_blue"], ATT_COLORS["teal"], ATT_COLORS["orange"]]
    bar_palette = (bar_palette * len(model_labels))[: len(model_labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bars1 = axes[0].bar(model_labels, mspe_vals, color=bar_palette, width=0.5, zorder=3)
    for bar, val in zip(bars1, mspe_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(mspe_vals) * 0.01,
                     f"{val:.2f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
    axes[0].set_title("MSPE (lower is better)")
    axes[0].set_ylabel("Mean Squared Prediction Error")
    axes[0].set_ylim(0, max(mspe_vals) * 1.25)

    bars2 = axes[1].bar(model_labels, r2_vals, color=bar_palette, width=0.5, zorder=3)
    for bar, val in zip(bars2, r2_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
    axes[1].set_title("R² (higher is better)")
    axes[1].set_ylabel("R² Score")
    axes[1].set_ylim(0, 1.1)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=11)

    fig.suptitle("60/40 External CV — Model Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def save_model_comparison_png(
    comparison_df: pd.DataFrame,
    path: Union[str, Path],
    *,
    dpi: int = 220,
) -> str:
    """Render and save the MSPE/R² comparison chart."""
    fig = plot_model_comparison(comparison_df)
    return save_figure(fig, path, dpi=dpi)


# ---------------------------------------------------------------------------
# Classification evaluation plots (shared by every classifier family)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm,
    *,
    class_labels: Optional[Sequence] = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Confusion matrix heatmap built on the shared tile primitive.

    Parameters
    ----------
    cm : 2-D array-like or DataFrame
        Confusion matrix (rows = true, cols = predicted). If a DataFrame, its
        index/columns are used as class labels unless `class_labels` is given.
    normalize : bool
        If True, color by row-normalized rates while still printing the raw
        counts in each cell — so the diagonal reads as per-class recall and
        imbalanced classes don't wash out. Color and text decouple here.
    """
    if hasattr(cm, "values"):  # DataFrame
        counts = np.asarray(cm.values, dtype=float)
        if class_labels is None:
            class_labels = [str(c) for c in cm.columns]
    else:
        counts = np.asarray(cm, dtype=float)
        if class_labels is None:
            class_labels = [str(i) for i in range(counts.shape[0])]

    n = counts.shape[0]
    diagonal = [(i, i) for i in range(n)]

    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            color_values = np.where(row_sums > 0, counts / row_sums, 0.0)
        cell_text = [[f"{int(counts[i, j]):,}\n{color_values[i, j]:.0%}"
                      for j in range(n)] for i in range(n)]
        colorbar_label = "Row-normalized rate"
    else:
        color_values = counts
        cell_text = [[f"{int(counts[i, j]):,}" for j in range(n)] for i in range(n)]
        colorbar_label = "Count"

    return plot_tile_grid(
        color_values,
        row_labels=class_labels,
        col_labels=class_labels,
        row_axis_label="True",
        col_axis_label="Predicted",
        title=title,
        cell_text=cell_text,
        colorbar_label=colorbar_label,
        highlight=diagonal,
        invert_yaxis=True,
        figsize=figsize,
        ax=ax,
    )


def plot_roc_curve(
    curves,
    *,
    title: str = "ROC curve",
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Plot one or more ROC curves with the chance diagonal.

    Parameters
    ----------
    curves : tuple or list of tuples
        Either a single (fpr, tpr, auc) tuple, or a list of
        (label, fpr, tpr, auc) tuples to overlay several models.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 7))
    else:
        fig = ax.figure

    # Normalize input into a list of (label, fpr, tpr, auc)
    if isinstance(curves, tuple):
        fpr, tpr, auc = curves
        series = [(None, fpr, tpr, auc)]
    else:
        series = list(curves)

    for idx, (label, fpr, tpr, auc) in enumerate(series):
        color = ATT_PALETTE[idx % len(ATT_PALETTE)]
        auc_str = "" if (auc is None or np.isnan(auc)) else f" (AUC = {auc:.3f})"
        name = (label or "ROC") + auc_str
        ax.plot(fpr, tpr, color=color, linewidth=2.6, zorder=3, label=name)

    ax.plot([0, 1], [0, 1], color=ATT_COLORS["gray_500"], linestyle="--",
            linewidth=1.5, zorder=2, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def save_confusion_matrix_png(cm, path, *, class_labels=None, normalize=False,
                              title="Confusion matrix", dpi=220):
    """Render and save a confusion-matrix heatmap."""
    fig = plot_confusion_matrix(cm, class_labels=class_labels, normalize=normalize,
                                title=title)
    return save_figure(fig, path, dpi=dpi)


def save_roc_curve_png(curves, path, *, title="ROC curve", dpi=220):
    """Render and save one or more ROC curves."""
    fig = plot_roc_curve(curves, title=title)
    return save_figure(fig, path, dpi=dpi)


def plot_roc_threshold_tuning(
    tuning_result,
    *,
    title: str = "ROC threshold tuning",
    metric_cols: Sequence[str] = ("f1", "recall", "precision", "false_negative_rate"),
    figsize: Optional[tuple] = None,
):
    roc_df = tuning_result.roc_df
    sweep_df = tuning_result.sweep_df
    best_threshold = tuning_result.best_threshold
    best_row = tuning_result.best_row

    missing_roc = [
        col for col in ("false_positive_rate", "true_positive_rate", "roc_auc")
        if col not in roc_df.columns
    ]
    if missing_roc:
        raise ValueError(f"tuning_result.roc_df is missing columns: {missing_roc}")

    missing_sweep = [
        col for col in ("threshold", *metric_cols)
        if col not in sweep_df.columns
    ]
    if missing_sweep:
        raise ValueError(f"tuning_result.sweep_df is missing columns: {missing_sweep}")

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize or (13, 5.5),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    roc_ax, metric_ax = axes

    auc = roc_df["roc_auc"].dropna()
    auc_value = float(auc.iloc[0]) if not auc.empty else float("nan")
    auc_str = "" if np.isnan(auc_value) else f" (AUC = {auc_value:.3f})"

    roc_ax.plot(
        roc_df["false_positive_rate"],
        roc_df["true_positive_rate"],
        color=ATT_PALETTE[0],
        linewidth=2.6,
        zorder=3,
        label=f"ROC{auc_str}",
    )
    roc_ax.plot(
        [0, 1],
        [0, 1],
        color=ATT_COLORS["gray_500"],
        linestyle="--",
        linewidth=1.5,
        zorder=2,
        label="Chance",
    )
    roc_ax.scatter(
        [best_row["false_positive_rate"]],
        [best_row["recall"]],
        s=95,
        color=ATT_COLORS["orange"],
        edgecolor=ATT_COLORS["white"],
        linewidth=1.2,
        zorder=4,
        label=f"Selected threshold = {best_threshold:.3f}",
    )
    roc_ax.set_xlim(-0.02, 1.02)
    roc_ax.set_ylim(-0.02, 1.02)
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC curve")
    roc_ax.set_aspect("equal", adjustable="box")
    roc_ax.legend(loc="lower right")

    ordered = sweep_df.sort_values("threshold")
    for idx, col in enumerate(metric_cols):
        metric_ax.plot(
            ordered["threshold"],
            ordered[col],
            color=ATT_PALETTE[idx % len(ATT_PALETTE)],
            linewidth=2.2,
            label=col.replace("_", " ").title(),
        )

    metric_ax.axvline(
        best_threshold,
        color=ATT_COLORS["orange"],
        linestyle="--",
        linewidth=1.8,
        label=f"Selected = {best_threshold:.3f}",
    )
    metric_ax.set_xlabel("Decision threshold")
    metric_ax.set_ylabel("Metric value")
    metric_ax.set_title("Threshold diagnostics")
    metric_ax.set_xlim(
        max(0, float(ordered["threshold"].min()) - 0.02),
        min(1, float(ordered["threshold"].max()) + 0.02),
    )
    metric_ax.set_ylim(-0.02, 1.02)
    metric_ax.legend(loc="best")
    metric_ax.grid(
        True,
        color=ATT_COLORS["gray_300"],
        linewidth=0.8,
        alpha=0.8,
    )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_roc_threshold_tuning_png(
    tuning_result,
    path,
    *,
    title: str = "ROC threshold tuning",
    dpi: int = 220,
    **kwargs,
):
    fig = plot_roc_threshold_tuning(tuning_result, title=title, **kwargs)
    return save_figure(fig, path, dpi=dpi)