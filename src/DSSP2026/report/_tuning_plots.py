"""
report/_tuning_plots.py — low-level renderers for the report's tuning views.

These take already-prepared frames (the TuningMixin does the report.db loading
and the model-selection math) and render AT&T-styled figures. Kept separate from
``report/plots.py`` so the evaluation plots and the tuning plots don't entangle.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS


def plot_elbow_curve(curve_df: pd.DataFrame, *, elbow, param_name: str,
                     model: str, scoring: str = "CV score",
                     figsize: Optional[tuple] = None) -> plt.Figure:
    """CV-vs-single-parameter elbow curve with the parsimony pick marked.

    ``curve_df`` has columns ``param`` (the swept value) and ``cv_value`` (best
    CV at that value), one row per distinct value, ascending. ``elbow`` is the
    selected value (smallest near-optimal). Higher CV is better, so — unlike the
    minimisation MSPE curve in ``tree/regression/plots.py`` — the y-axis reads as
    a score and the elbow sits where the curve has effectively plateaued.
    """
    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))

    ax.plot(curve_df["param"], curve_df["cv_value"], marker="o",
            color=ATT_COLORS["deep_blue"], linewidth=2.4, zorder=3,
            label=scoring)

    row = curve_df.loc[curve_df["param"] == elbow]
    if len(row):
        ax.scatter(row["param"], row["cv_value"], s=170, facecolors="none",
                   edgecolors=ATT_COLORS["orange"], linewidth=2.6, zorder=4,
                   label=f"Elbow ({param_name} = {elbow})")
        ax.axvline(elbow, color=ATT_COLORS["orange"], linestyle="--",
                   linewidth=1.4, alpha=0.7, zorder=2)

    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(f"{model} — {param_name} selection")
    ax.grid(True, color=ATT_COLORS.get("gray_100", "#eeeeee"), linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
