"""
core/residuals.py — the residual-diagnostics quad (matplotlib survivor #2).

The second matplotlib figure that stays out of the grammar-of-graphics engine.
A 2x2 diagnostic panel — residuals-vs-fitted, normal Q-Q, scale-location, and
Cook's distance — relies on ``scipy.stats.probplot`` (Q-Q), lowess smoothers,
and a stem plot, none of which the grammar expresses cleanly. So it stays
hand-rolled here.

Family-agnostic by construction: this renders from a *tidy diagnostics
DataFrame* (columns ``fitted``, ``residuals``, ``stud_resid``, ``cooks``,
``obs``), not from a statsmodels model. The regression family computes that
frame (``regression.fit.make_ols_diagnostics_df``) and passes it in, so ``core``
never imports a family — the dependency direction stays one-way.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure


def plot_residual_diagnostics(
    diag_df: pd.DataFrame,
    *,
    n_obs: Optional[int] = None,
    title: str = "",
    figsize: Optional[tuple] = None,
):
    """Four-panel residual diagnostics from a tidy diagnostics DataFrame.

    Parameters
    ----------
    diag_df : DataFrame
        Must have columns ``fitted``, ``residuals``, ``stud_resid``, ``cooks``,
        ``obs`` — as produced by ``regression.fit.make_ols_diagnostics_df``.
    n_obs : int, optional
        Number of observations, used for the Cook's-distance 4/n threshold line.
        Defaults to ``len(diag_df)``.
    title : str
        Optional figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    required = {"fitted", "residuals", "stud_resid", "cooks", "obs"}
    missing = required - set(diag_df.columns)
    if missing:
        raise ValueError(
            f"diag_df is missing columns {sorted(missing)}; expected {sorted(required)} "
            f"(use regression.fit.make_ols_diagnostics_df).")

    n = int(n_obs) if n_obs is not None else len(diag_df)

    fig, axes = plt.subplots(2, 2, figsize=figsize or (13, 10))
    ax_resid, ax_qq, ax_scale, ax_cook = axes.ravel()

    # Residuals vs fitted
    ax_resid.scatter(diag_df["fitted"], diag_df["residuals"], s=34, alpha=0.65,
                     color=ATT_COLORS["att_blue"], edgecolor="white", linewidth=0.4)
    ax_resid.axhline(0, color=ATT_COLORS["gray_700"], linewidth=1.0, linestyle="--")
    if len(diag_df) >= 5:
        smooth = lowess(diag_df["residuals"], diag_df["fitted"], frac=0.45,
                        return_sorted=True)
        ax_resid.plot(smooth[:, 0], smooth[:, 1], color=ATT_COLORS["orange"], linewidth=2.0)
    ax_resid.set_title("Residuals vs fitted")
    ax_resid.set_xlabel("Fitted values")
    ax_resid.set_ylabel("Residuals")

    # Normal Q-Q
    stats.probplot(diag_df["stud_resid"], dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor(ATT_COLORS["att_blue"])
    ax_qq.get_lines()[0].set_markeredgecolor("white")
    ax_qq.get_lines()[0].set_alpha(0.65)
    ax_qq.get_lines()[1].set_color(ATT_COLORS["orange"])
    ax_qq.get_lines()[1].set_linewidth(2.0)
    ax_qq.set_title("Normal Q-Q")

    # Scale-location
    ax_scale.scatter(diag_df["fitted"], np.sqrt(np.abs(diag_df["stud_resid"])),
                     s=34, alpha=0.65, color=ATT_COLORS["att_blue"],
                     edgecolor="white", linewidth=0.4)
    if len(diag_df) >= 5:
        smooth = lowess(np.sqrt(np.abs(diag_df["stud_resid"])), diag_df["fitted"],
                        frac=0.45, return_sorted=True)
        ax_scale.plot(smooth[:, 0], smooth[:, 1], color=ATT_COLORS["orange"], linewidth=2.0)
    ax_scale.set_title("Scale-location")
    ax_scale.set_xlabel("Fitted values")
    ax_scale.set_ylabel("sqrt(|studentized residual|)")

    # Cook's distance
    ax_cook.stem(diag_df["obs"], diag_df["cooks"],
                 linefmt=ATT_COLORS["att_blue"], markerfmt=" ", basefmt=" ")
    ax_cook.axhline(4 / max(n, 1), color=ATT_COLORS["orange"],
                    linewidth=1.6, linestyle="--")
    ax_cook.set_title("Cook's distance")
    ax_cook.set_xlabel("Observation")
    ax_cook.set_ylabel("Cook's D")

    if title:
        fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def save_residual_diagnostics_png(diag_df, path, *, n_obs=None, title="", dpi=220):
    """Render and save the residual-diagnostics quad."""
    fig = plot_residual_diagnostics(diag_df, n_obs=n_obs, title=title)
    return save_figure(fig, path, dpi=dpi)
