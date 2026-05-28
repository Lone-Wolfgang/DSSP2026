"""
logistic/plots.py — presentation-quality plots specific to logistic regression.

The odds-ratio forest plot is the logistic-specific view: each predictor's
odds ratio with its confidence interval, against the OR = 1 no-effect line.

Classification evaluation plots (confusion matrix, ROC) are family-agnostic
and live in core/plot.py — call plot_confusion_matrix / plot_roc_curve with
the predictions from logistic/fit.py.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS


def plot_odds_ratios(
    coef_df: pd.DataFrame,
    *,
    drop_intercept: bool = True,
    log_scale: bool = True,
    title: str = "Odds ratios with 95% CI",
    figsize: Optional[tuple] = None,
    ax=None,
):
    """Forest plot of odds ratios and their confidence intervals.

    Expects the columns produced by logistic.fit.make_logit_coef_df:
    'Term', 'Odds Ratio', and the two 'OR CI ...' bounds. Predictors whose
    CI excludes 1 (significant) are drawn in the brand accent color; the rest
    are muted. A reference line marks OR = 1 (no effect).

    Parameters
    ----------
    log_scale : bool
        Plot the OR axis on a log scale (default). Odds ratios are
        multiplicative, so a log axis renders 0.5 and 2.0 symmetrically about 1.
    """
    or_col = "Odds Ratio"
    ci_cols = [c for c in coef_df.columns if c.startswith("OR CI ")]
    if or_col not in coef_df.columns or len(ci_cols) != 2:
        raise ValueError(
            "coef_df must have an 'Odds Ratio' column and two 'OR CI ...' "
            "bound columns (use logistic.fit.make_logit_coef_df)."
        )
    low_col, high_col = ci_cols[0], ci_cols[1]

    df = coef_df.copy()
    if drop_intercept:
        df = df[df["Term"] != "Intercept"]
    df = df.iloc[::-1].reset_index(drop=True)  # first term at the top

    terms = df["Term"].tolist()
    or_vals = df[or_col].to_numpy()
    lo = df[low_col].to_numpy()
    hi = df[high_col].to_numpy()
    y = np.arange(len(terms))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (9, max(3, 0.6 * len(terms) + 1.8)))
    else:
        fig = ax.figure

    significant = (lo > 1) | (hi < 1)  # CI excludes 1
    for yi, orv, l, h, sig in zip(y, or_vals, lo, hi, significant):
        color = ATT_COLORS["deep_blue"] if sig else ATT_COLORS["gray_500"]
        ax.plot([l, h], [yi, yi], color=color, linewidth=2.4, zorder=3,
                solid_capstyle="round")
        ax.scatter([orv], [yi], s=90, color=color, edgecolor="white",
                   linewidth=1.0, zorder=4)

    ax.axvline(1.0, color=ATT_COLORS["orange"], linestyle="--", linewidth=1.6,
               zorder=2, label="No effect (OR = 1)")

    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.set_xlabel("Odds ratio")
    ax.set_title(title)
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (log scale)")
    ax.margins(y=0.12)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def save_odds_ratios_png(coef_df, path, *, drop_intercept=True, log_scale=True,
                         dpi=220, **kwargs):
    from DSSP2026.core.plot import save_figure
    fig = plot_odds_ratios(coef_df, drop_intercept=drop_intercept,
                           log_scale=log_scale, **kwargs)
    return save_figure(fig, path, dpi=dpi)