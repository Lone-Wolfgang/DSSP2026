"""
tree/_shared.py — task-agnostic helpers for tree models.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from DSSP2026.core.figure import save_figure
from DSSP2026.core.style import ATT_COLORS, ATT_SEQUENTIAL


def feature_importance_df(model, features: Sequence[str]) -> pd.DataFrame:
    """Tidy, descending-sorted feature-importance DataFrame."""
    return pd.DataFrame({
        "Feature": list(features),
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


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
    fig = plot_feature_importance(importance_df, top_n=top_n)
    return save_figure(fig, path, dpi=dpi)
