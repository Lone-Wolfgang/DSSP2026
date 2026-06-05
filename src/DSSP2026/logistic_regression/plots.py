"""
logistic/plots.py — presentation-quality plots specific to logistic regression.

The odds-ratio forest plot is the logistic-specific view: each predictor's
odds ratio with its confidence interval, against the OR = 1 no-effect line.

Built on the shared seaborn-Objects engine (``core.viz``) so it inherits the
AT&T theme and palette: the point estimates and CI bars come from
``so.Dot`` + ``so.Range`` with significance mapped through ``att_nominal``.
The OR = 1 reference line and the log scale stay as direct matplotlib on the
same axes (via ``.on(ax)``), since neither is a data layer.

Classification evaluation plots (confusion matrix, ROC) are family-agnostic
and live in core — call those with the predictions from logistic/fit.py.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure
from DSSP2026.core.color_scales import so, att_nominal


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

    # Tidy frame for the Objects API. Significance becomes a real column so the
    # color split goes through the brand scale instead of a hand-set loop.
    plot_df = pd.DataFrame({
        "Term": df["Term"].astype(str),
        "or": df[or_col].to_numpy(),
        "lo": df[low_col].to_numpy(),
        "hi": df[high_col].to_numpy(),
    })
    plot_df["Significance"] = np.where(
        (plot_df["lo"] > 1) | (plot_df["hi"] < 1), "Significant", "Not significant")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (9, max(3, 0.6 * len(plot_df) + 1.8)))
    else:
        fig = ax.figure

    # Order so the first term sits at the top; categorical y preserves order.
    order = plot_df["Term"].tolist()
    # Significant -> deep_blue (first palette slot), Not -> gray.
    sig_scale = att_nominal(order=["Significant", "Not significant"])

    p = (
        so.Plot(plot_df, y="Term", color="Significance")
          .add(so.Range(linewidth=2.4), xmin="lo", xmax="hi")
          .add(so.Dot(pointsize=9, edgecolor="white", edgewidth=1.0), x="or")
          .scale(y=so.Nominal(order=order),
                 color=so.Nominal(values=[ATT_COLORS["deep_blue"],
                                          ATT_COLORS["gray_500"]],
                                  order=["Significant", "Not significant"]))
          .label(x="Odds ratio", y="", title=title)
    )
    p.on(ax).plot()

    # Reference line + log scale are not data layers — add them directly.
    ax.axvline(1.0, color=ATT_COLORS["orange"], linestyle="--", linewidth=1.6,
               zorder=2, label="No effect (OR = 1)")
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (log scale)")
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig


def save_odds_ratios(coef_df, path, *, drop_intercept=True, log_scale=True,
                     dpi=220, **kwargs):
    """Render the forest plot and save it (matplotlib formats: png/pdf/svg…)."""
    fig = plot_odds_ratios(coef_df, drop_intercept=drop_intercept,
                           log_scale=log_scale, **kwargs)
    return save_figure(fig, path, dpi=dpi)


# Backwards-compatible alias for the old name.
save_odds_ratios_png = save_odds_ratios