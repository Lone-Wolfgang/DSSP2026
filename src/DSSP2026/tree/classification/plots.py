"""
tree/classification/plots.py — plots for classification trees and forests.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from DSSP2026.core.figure import save_figure
from DSSP2026.core.heatmap import plot_tile_grid
from DSSP2026.core.style import ATT_COLORS


def plot_cv_depth_curve(result, *, figsize: Optional[tuple] = None) -> plt.Figure:
    """Plot mean CV score vs max_depth with a ±1 SE band and the selected depths.

    Parameters
    ----------
    result : DepthTuneResult
        From ``tree.classify_tune.tune_dt_depth_cv``.

    Marks
    -----
    - the mean CV-score curve (deep blue) with a ±1 SE band;
    - the one-SE threshold line (best mean − 1 SE) the selection rule uses;
    - the best-by-mean depth (gray dashed);
    - the one-SE-selected depth (orange ring) — the depth actually chosen.
    """
    df = result.results_df
    depths = df["max_depth"].to_numpy()
    mean = df["mean_score"].to_numpy()
    se = df["se_score"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))

    # ±1 SE band + mean curve.
    ax.fill_between(depths, mean - se, mean + se,
                    color=ATT_COLORS["deep_blue"], alpha=0.15, linewidth=0,
                    zorder=1, label="±1 SE")
    ax.plot(depths, mean, marker="o", color=ATT_COLORS["deep_blue"],
            linewidth=2.4, zorder=3, label=f"Mean CV {result.scoring}")

    # One-SE threshold (best mean − 1 SE): depths whose mean sits at/above this
    # qualify; the rule takes the shallowest such depth.
    ax.axhline(result.se_threshold, color=ATT_COLORS["gray_500"], linestyle=":",
               linewidth=1.4, zorder=2,
               label=f"best − 1 SE = {result.se_threshold:.3f}")

    # Best-by-mean depth.
    ax.axvline(result.best_depth, color=ATT_COLORS["gray_700"], linestyle="--",
               linewidth=1.4, alpha=0.8, zorder=2,
               label=f"Best mean (depth = {result.best_depth})")

    # One-SE selected depth — the chosen, parsimonious tree.
    sel = df.loc[df["max_depth"] == result.one_se_depth]
    if len(sel):
        ax.scatter(sel["max_depth"], sel["mean_score"], s=170,
                   facecolors="none", edgecolors=ATT_COLORS["orange"],
                   linewidth=2.6, zorder=4,
                   label=f"1-SE choice (depth = {result.one_se_depth})")

    ax.set_xlabel("Max depth")
    ax.set_ylabel(f"Cross-validated {result.scoring}")
    ax.set_title(f"Decision tree depth selection — {result.n_splits}-fold CV "
                 f"(1-SE rule)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def save_cv_depth_curve_png(result, path, *, dpi: int = 220) -> str:
    """Render and save the CV depth-selection curve."""
    return save_figure(plot_cv_depth_curve(result), path, dpi=dpi)


def plot_rf_grid_heatmap(
    cv_results,
    *,
    scoring: str = "f1_macro",
    best_n=None,
    best_mf=None,
    figsize: Optional[tuple] = None,
):
    """Heatmap of mean CV score over an (n_estimators × max_features) grid.

    Expects sklearn's ``cv_results_`` from a grid over exactly those two params.
    The score is the CV ``scoring`` metric (macro-F1 by default), plotted as-is
    because higher is better — the best cell is the maximum and is ringed.

    Parameters
    ----------
    cv_results : dict
        ``GridSearchCV.cv_results_`` (from ``classify_rf_tune.tune_rf_grid_classify``).
    scoring : str
        Name of the CV metric, used only for the colorbar/title label.
    best_n, best_mf : optional
        The chosen ``n_estimators`` / ``max_features`` to ring as the best cell.
    """
    params = cv_results["params"]
    n_vals = sorted({p["n_estimators"] for p in params})
    mf_vals_raw = list({p["max_features"] for p in params})
    # Stable, readable order: ints ascending, then strings.
    mf_vals = (sorted(v for v in mf_vals_raw if not isinstance(v, str)) +
               sorted(v for v in mf_vals_raw if isinstance(v, str)))

    mean_score = cv_results["mean_test_score"]
    grid = np.full((len(mf_vals), len(n_vals)), np.nan)
    for p, s in zip(params, mean_score):
        i = mf_vals.index(p["max_features"])
        j = n_vals.index(p["n_estimators"])
        grid[i, j] = s          # macro-F1: positive, higher is better (no negation)

    highlight = None
    if best_n is not None and best_mf is not None:
        highlight = (mf_vals.index(best_mf), n_vals.index(best_n))

    # 3-decimal cell text (F1 sits in [0, 1]; the regressor's "{:,.0f}" default
    # would round every cell to 0 or 1).
    cell_text = [[("" if not np.isfinite(grid[i, j]) else f"{grid[i, j]:.3f}")
                  for j in range(len(n_vals))] for i in range(len(mf_vals))]

    return plot_tile_grid(
        grid,
        row_labels=[str(v) for v in mf_vals],
        col_labels=[str(v) for v in n_vals],
        row_axis_label="max_features",
        col_axis_label="n_estimators",
        title=f"Grid search — mean CV {scoring} (higher is better)",
        cell_text=cell_text,
        colorbar_label=f"Mean CV {scoring}",
        highlight=highlight,
        invert_yaxis=False,
        figsize=figsize,
    )


def save_rf_grid_heatmap(cv_results, path, *, scoring="f1_macro",
                         best_n=None, best_mf=None, dpi=220):
    """Render and save the classification grid-search heatmap."""
    fig = plot_rf_grid_heatmap(cv_results, scoring=scoring,
                               best_n=best_n, best_mf=best_mf)
    return save_figure(fig, path, dpi=dpi)
