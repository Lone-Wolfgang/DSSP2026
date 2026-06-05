from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure
from DSSP2026.core.color_scales import so, att_nominal


def _normalize_roc_input(roc):
    if isinstance(roc, tuple) and len(roc) == 3:
        fpr, tpr, auc = roc
        return [(None, np.asarray(fpr), np.asarray(tpr), auc)]
    curves = []
    for item in roc:
        label, fpr, tpr, auc = item
        curves.append((label, np.asarray(fpr), np.asarray(tpr), auc))
    return curves


def plot_roc_curve(
    roc,
    *,
    title: str = "ROC curve",
    figsize: Optional[tuple] = None,
    ax=None,
):
    curves = _normalize_roc_input(roc)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 7))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], color=ATT_COLORS["gray_500"], linestyle="--",
            linewidth=1.4, zorder=1, label="Chance")

    frames = []
    legend_names = []
    for i, (label, fpr, tpr, auc) in enumerate(curves):
        name = label if label is not None else "ROC"
        legend = f"{name} (AUC = {auc:.3f})" if auc == auc else name
        legend_names.append(legend)
        frames.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "curve": legend}))
    roc_df = pd.concat(frames, ignore_index=True)

    (
        so.Plot(roc_df, x="fpr", y="tpr", color="curve")
          .add(so.Line(linewidth=2.8))
          .scale(color=att_nominal(order=legend_names))
          .label(x="False positive rate", y="True positive rate")
          .on(ax).plot()
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def save_roc_curve_png(roc, path, *, title="ROC curve", dpi: int = 220,
                       figsize: Optional[tuple] = None) -> str:
    return save_figure(plot_roc_curve(roc, title=title, figsize=figsize), path, dpi=dpi)


def plot_roc_threshold_tuning(
    tuning,
    *,
    title: str = "ROC threshold tuning",
    figsize: Optional[tuple] = None,
    ax=None,
):
    roc_df = tuning.roc_df
    auc = float(roc_df["roc_auc"].iloc[0]) if "roc_auc" in roc_df else float("nan")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 7))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], color=ATT_COLORS["gray_500"], linestyle="--",
            linewidth=1.4, zorder=1, label="Chance")

    (
        so.Plot(roc_df, x="false_positive_rate", y="true_positive_rate")
          .add(so.Line(linewidth=2.8, color=ATT_COLORS["deep_blue"]))
          .label(x="False positive rate", y="True positive rate")
          .on(ax).plot()
    )

    best = tuning.best_row
    op_fpr = float(best.get("false_positive_rate", np.nan))
    op_tpr = float(best.get("recall", best.get("true_positive_rate", np.nan)))
    if np.isfinite(op_fpr) and np.isfinite(op_tpr):
        ax.scatter([op_fpr], [op_tpr], s=130, color=ATT_COLORS["orange"],
                   edgecolor="white", linewidth=1.5, zorder=4,
                   label=(f"Selected: thr = {tuning.best_threshold:.3f}, "
                          f"{tuning.best_metric} = {tuning.best_value:.3f}"))

    if np.isfinite(auc):
        ax.plot([], [], " ", label=f"AUC = {auc:.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def save_roc_threshold_tuning_png(tuning, path, *, title="ROC threshold tuning",
                                  dpi: int = 220, figsize: Optional[tuple] = None) -> str:
    return save_figure(
        plot_roc_threshold_tuning(tuning, title=title, figsize=figsize), path, dpi=dpi)
