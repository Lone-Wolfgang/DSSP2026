"""
tree/plots.py — presentation-quality plots for tree models.

Starts with the depth-vs-MSPE elbow curve produced by tree/tune.py. Add tree
diagrams, feature-importance bars, etc. here as needed.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_SEQUENTIAL, ATT_COLORS


def plot_depth_elbow(results_df: pd.DataFrame, elbow: int, *,
                     figsize: Optional[tuple] = None) -> plt.Figure:
    """Plot test MSPE against max_depth and mark the selected elbow."""
    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))

    ax.plot(results_df["max_depth"], results_df["MSPE"], marker="o",
            color=ATT_COLORS["deep_blue"], linewidth=2.4, zorder=3,
            label="Test MSPE")

    elbow_row = results_df.loc[results_df["max_depth"] == elbow]
    if len(elbow_row):
        ax.scatter(elbow_row["max_depth"], elbow_row["MSPE"], s=160,
                   facecolors="none", edgecolors=ATT_COLORS["orange"],
                   linewidth=2.6, zorder=4, label=f"Elbow (depth = {elbow})")
        ax.axvline(elbow, color=ATT_COLORS["orange"], linestyle="--",
                   linewidth=1.4, alpha=0.7, zorder=2)

    ax.set_xlabel("Max depth")
    ax.set_ylabel("Mean Squared Prediction Error")
    ax.set_title("Decision tree depth selection")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def save_dt_depth_elbow_png(results_df, elbow, path, *, dpi=220):
    """Render and save the depth-elbow curve."""
    from modules.core.plot import save_figure
    fig = plot_depth_elbow(results_df, elbow)
    return save_figure(fig, path, dpi=dpi)


def save_dt_depth_table_png(results_df, elbow, path, *, context="report", dpi=220):
    """Save the depth/metrics table as a PNG, highlighting the elbow row."""
    from modules.core.tables import save_generic_table_png
    highlight = None
    matches = results_df.index[results_df["max_depth"] == elbow].tolist()
    if matches:
        highlight = int(results_df.index.get_loc(matches[0]))
    fmt = {"MSPE": "{:,.3f}".format, "R²": "{:,.3f}".format, "R2": "{:,.3f}".format}
    return save_generic_table_png(
        results_df, path, title=f"Decision tree depth sweep — elbow at depth {elbow}",
        col_fmts={k: v for k, v in fmt.items() if k in results_df.columns},
        highlight_row=highlight, context=context, dpi=dpi)


def plot_feature_importance(importance_df, *, top_n=None, figsize=None):
    """Horizontal bar chart of feature importances (descending)."""
    df = importance_df.copy()
    if top_n is not None:
        df = df.head(top_n)
    df = df.iloc[::-1]  # so the largest sits at the top of the chart

    fig, ax = plt.subplots(figsize=figsize or (9, max(3, 0.6 * len(df) + 1.5)))
    ax.barh(df["Feature"], df["Importance"], color=ATT_SEQUENTIAL, zorder=3)
    for y, val in zip(range(len(df)), df["Importance"]):
        ax.text(val, y, f"  {val:.3f}", va="center", ha="left",
                fontsize=14, fontweight="bold", color=ATT_COLORS["gray_700"])
    ax.set_xlabel("Importance")
    title = "Random forest feature importance"
    if top_n is not None:
        title += f" (top {top_n})"
    ax.set_title(title)
    ax.set_xlim(0, df["Importance"].max() * 1.15)
    fig.tight_layout()
    return fig


def save_rf_feature_importance_png(importance_df, path, *, top_n=None, dpi=220):
    from modules.core.plot import save_figure
    fig = plot_feature_importance(importance_df, top_n=top_n)
    return save_figure(fig, path, dpi=dpi)


def plot_grid_search_heatmap(cv_results, *, best_n=None, best_mf=None, figsize=None):
    """Heatmap of mean CV score over an (n_estimators × max_features) grid.

    Expects sklearn's cv_results_ dict from a grid over those two params.
    Scores are negated MSE; we plot positive MSE (lower is better) via the
    shared tile primitive in core.plot.
    """
    from modules.core.plot import plot_tile_grid

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
        grid[i, j] = -s  # negate: neg_MSE -> MSE

    highlight = None
    if best_n is not None and best_mf is not None:
        highlight = (mf_vals.index(best_mf), n_vals.index(best_n))

    return plot_tile_grid(
        grid,
        row_labels=[str(v) for v in mf_vals],
        col_labels=[str(v) for v in n_vals],
        row_axis_label="max_features",
        col_axis_label="n_estimators",
        title="Grid search — mean CV MSE (lower is better)",
        colorbar_label="Mean CV MSE",
        highlight=highlight,
        invert_yaxis=False,
        figsize=figsize,
    )


def save_grid_search_heatmap(cv_results, path, *, best_n=None, best_mf=None, dpi=220):
    from modules.core.plot import save_figure
    fig = plot_grid_search_heatmap(cv_results, best_n=best_n, best_mf=best_mf)
    return save_figure(fig, path, dpi=dpi)