"""
core/heatmap.py — the matplotlib tile/heatmap primitive and its callers.

This is one of the two matplotlib figures that deliberately did **not** move to
the grammar-of-graphics engine (``core.gg``). The reason: ``plot_tile_grid``
does per-cell luminance-based text contrast and decouples cell *color* from cell
*text* (e.g. color by row-normalized recall while printing raw counts), which
``geom_tile`` + ``geom_text`` can't express without dropping back to matplotlib
anyway. So the primitive stays here, hand-rolled, and earns its keep.

Contents
--------
- ``plot_tile_grid``        : labelled heatmap primitive (color + per-cell text,
                              luminance-chosen contrast, optional cell highlight)
- ``plot_confusion_matrix`` : classification confusion matrix built on it
- ``save_*_png``            : render-and-save companions

The grid-search heatmap (``tree.plots``) is another caller — it builds its grid
and calls ``plot_tile_grid`` from here.
"""

from typing import Optional, Sequence, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from DSSP2026.core.style import ATT_COLORS, ATT_SEQUENTIAL
from DSSP2026.core.figure import save_figure


ATT_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "att_sequential", ATT_SEQUENTIAL)


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
    norm=None,
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
    if norm is None:
        norm = (Normalize(vmin=finite.min(), vmax=finite.max())
                if finite.size else Normalize())
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


def save_tile_grid_png(values, path, *, dpi: int = 220, **kwargs) -> str:
    """Render and save a tile-grid heatmap."""
    return save_figure(plot_tile_grid(values, **kwargs), path, dpi=dpi)


def save_confusion_matrix_png(cm, path, *, class_labels=None, normalize=False,
                              title="Confusion matrix", dpi=220):
    """Render and save a confusion-matrix heatmap."""
    fig = plot_confusion_matrix(cm, class_labels=class_labels, normalize=normalize,
                                title=title)
    return save_figure(fig, path, dpi=dpi)


# ---------------------------------------------------------------------------
# Diverging / dual-ramp colormaps for the configurations below
# ---------------------------------------------------------------------------

from DSSP2026.core.style import ATT_DIVERGING

# Full signed diverging ramp (red <- neutral -> blue), for signed z-scores.
ATT_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "att_diverging", ATT_DIVERGING)
# Neutral -> blue half, used as a sequential ramp for |z| (magnitude only):
# light = at the row mean, dark = stands out (either direction).
ATT_DIVERGING_HALF_CMAP = LinearSegmentedColormap.from_list(
    "att_diverging_half", ATT_DIVERGING[2:])
# White -> blue (true positives) and white -> red (false positives) for the
# tp/fp divergence matrix.
TP_CMAP = LinearSegmentedColormap.from_list("tp_white_blue",
                                            ["#FFFFFF", ATT_COLORS["deep_blue"]])
FP_CMAP = LinearSegmentedColormap.from_list("fp_white_red",
                                            ["#FFFFFF", ATT_COLORS["magenta"]])


# ---------------------------------------------------------------------------
# Configuration 1 — z-score color scaling
# ---------------------------------------------------------------------------

def plot_zscore_grid(
    values,
    *,
    row_labels: Sequence,
    col_labels: Sequence,
    row_axis_label: str = "",
    col_axis_label: str = "",
    title: str = "",
    cell_text=None,
    signed: bool = False,
    cmap=None,
    z_thresh: Optional[float] = 1.0,
    already_zscored: bool = False,
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Heatmap coloured by per-row z-score.

    Each row is z-scored across its columns (centre on the row mean, scale by
    the row std), so rows on very different scales become comparable. Two modes:

    - ``signed=False`` (default): colour by the **magnitude** ``|z|`` on the
      neutral->blue half-ramp — light = at the row mean (typical), dark = stands
      out in *either* direction.
    - ``signed=True``: colour by the **signed** z on the full red<->blue
      diverging ramp, symmetric about 0 — red = below the row mean, blue = above.

    The cell *text* defaults to the raw input ``values`` (so colour shows
    deviation while text shows the real numbers); pass ``cell_text`` to override.
    Cells with ``|z| >= z_thresh`` are outlined (set ``z_thresh=None`` to skip).

    Parameters
    ----------
    values : 2-D array-like
        Raw values (rows = series, cols = groups). Z-scoring is done here unless
        ``already_zscored=True``, in which case ``values`` is treated as z.
    signed : bool
        Magnitude (False, default) vs. signed (True) colouring.
    cmap : str or Colormap, optional
        Override the colormap. By default the ramp is chosen from ``signed``
        (magnitude -> neutral->blue half ramp; signed -> full red<->blue
        diverging). Pass any matplotlib colormap name or object to use your own;
        the ``signed`` semantics (which values map / symmetric clim) are kept,
        only the colours change. For ``signed=True`` give a *diverging* ramp
        with a neutral middle; for magnitude give a *sequential* ramp.
    already_zscored : bool
        If True, skip the internal z-scoring and treat ``values`` as z-scores.
    """
    raw = np.asarray(values, dtype=float)

    if already_zscored:
        z = raw.copy()
    else:
        mu = np.nanmean(raw, axis=1, keepdims=True)
        sd = np.nanstd(raw, axis=1, ddof=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 0, (raw - mu) / sd, 0.0)
    z = np.nan_to_num(z, nan=0.0)

    # Cell text defaults to the raw values (1-decimal), not the z-scores.
    if cell_text is None:
        cell_text = [[f"{raw[i, j]:,.1f}" for j in range(raw.shape[1])]
                     for i in range(raw.shape[0])]

    # Outline the cells that stand out (|z| above threshold).
    highlight = None
    if z_thresh is not None:
        highlight = [(i, j) for i in range(z.shape[0]) for j in range(z.shape[1])
                     if abs(z[i, j]) >= z_thresh]

    if signed:
        colour_values = z
        default_cmap = ATT_DIVERGING_CMAP
        colorbar_label = "Row z-score (− below mean, + above)"
    else:
        colour_values = np.abs(z)
        default_cmap = ATT_DIVERGING_HALF_CMAP
        colorbar_label = "|row z-score|  (0 = at row mean, dark = stands out)"
    # A caller-supplied cmap overrides the signed-chosen default; everything
    # else (which values are coloured, the symmetric clim for signed) stays.
    cmap = default_cmap if cmap is None else cmap

    fig = plot_tile_grid(
        colour_values,
        row_labels=row_labels, col_labels=col_labels,
        row_axis_label=row_axis_label, col_axis_label=col_axis_label,
        title=title, cell_text=cell_text, cmap=cmap,
        colorbar_label=colorbar_label, highlight=highlight,
        figsize=figsize, ax=ax,
    )

    # For the signed view, pin the colour scale symmetric about 0 so the neutral
    # midpoint lands exactly on z = 0 (plot_tile_grid auto-normalises to min/max
    # otherwise, which would shift white off-centre).
    if signed:
        vmax = float(np.nanmax(np.abs(z))) or 1.0
        fig.axes[0].images[0].set_clim(-vmax, vmax)

    return fig


def save_zscore_grid_png(values, path, *, dpi: int = 220, **kwargs) -> str:
    """Render and save a z-score-scaled heatmap."""
    return save_figure(plot_zscore_grid(values, **kwargs), path, dpi=dpi)


# ---------------------------------------------------------------------------
# Configuration 2 — true-positive / false-positive divergence
# ---------------------------------------------------------------------------

def plot_tp_fp_matrix(
    values,
    *,
    row_labels: Sequence,
    col_labels: Sequence,
    row_axis_label: str = "Flag",
    col_axis_label: str = "True cause",
    title: str = "True positive (blue) vs. false positive (red)",
    vmin: float = 0.0,
    vmax: float = 1.0,
    text_fontsize: int = 12,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Square fire-rate matrix with split true/false-positive colouring.

    ``values[i, j]`` is the rate (mean of a boolean, in ``[0, 1]``) at which
    detector *i* fires on class *j*. The **diagonal** (detector meets its target
    class) is read as a true-positive rate and coloured white->blue; every
    **off-diagonal** cell is a false-positive rate and coloured white->red. Both
    ramps share the ``[vmin, vmax]`` domain (default 0..1) so a 0.5 blue and a
    0.5 red are equally saturated. Cell text is the rate as a percentage, with
    per-cell black/white contrast.

    Requires a square matrix (n detectors == n classes); the diagonal is the
    set of correct detector->class pairs.
    """
    grid = np.asarray(values, dtype=float)
    n_rows, n_cols = grid.shape
    if n_rows != n_cols:
        raise ValueError(
            f"plot_tp_fp_matrix needs a square matrix (diagonal = true "
            f"positives); got shape {grid.shape}.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (1.5 * n_cols + 3, 1.15 * n_rows + 2.5))
    else:
        fig = ax.figure

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Per-cell RGBA: diagonal on the blue (TP) ramp, off-diagonal on red (FP).
    # A single imshow cmap can't express two ramps, so build the image directly.
    rgba = np.zeros((n_rows, n_cols, 4))
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(grid[i, j]):
                rgba[i, j] = (1, 1, 1, 1)
                continue
            cmap = TP_CMAP if i == j else FP_CMAP
            rgba[i, j] = cmap(norm(grid[i, j]))
    ax.imshow(rgba, aspect="auto")

    ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=tick_fontsize)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
    ax.set_xlabel(col_axis_label, fontsize=label_fontsize)
    ax.set_ylabel(row_axis_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title)

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(grid[i, j]):
                continue
            ax.text(j, i, f"{grid[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=text_fontsize, color=_text_color_for(rgba[i, j]))

    # Two colourbars: TP (blue) and FP (red), sharing the [vmin, vmax] domain.
    from matplotlib.cm import ScalarMappable
    cb_tp = fig.colorbar(ScalarMappable(norm=norm, cmap=TP_CMAP), ax=ax,
                         fraction=0.046, pad=0.03, shrink=0.85)
    cb_tp.set_label("True-positive rate  [diagonal]", fontsize=label_fontsize)
    cb_fp = fig.colorbar(ScalarMappable(norm=norm, cmap=FP_CMAP), ax=ax,
                         fraction=0.046, pad=0.14, shrink=0.85)
    cb_fp.set_label("False-positive rate  [off-diagonal]", fontsize=label_fontsize)

    fig.tight_layout()
    return fig


def save_tp_fp_matrix_png(values, path, *, dpi: int = 220, **kwargs) -> str:
    """Render and save a true/false-positive divergence matrix."""
    return save_figure(plot_tp_fp_matrix(values, **kwargs), path, dpi=dpi)