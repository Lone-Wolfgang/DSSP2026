from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from DSSP2026.core.color_scales import att_nominal, so
from DSSP2026.core.figure import save_figure
from DSSP2026.core.style import ATT_COLORS, ATT_PALETTE, att_table_styles
from DSSP2026.core.tables import _CAPTION_STYLE, save_generic_table_png, save_table_by_extension
from DSSP2026.core.threshold import CVThresholdResult


_DEFAULT_PLOT_METRICS = ("f1", "precision", "recall", "accuracy")


def plot_cv_threshold_metrics(
    result: CVThresholdResult,
    *,
    metrics: Sequence[str] = _DEFAULT_PLOT_METRICS,
    show_band: bool = True,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    ax=None,
):
    metrics = [m for m in metrics if f"{m}_mean" in result.summary_df.columns]
    if not metrics:
        raise ValueError("None of the requested metrics are in the summary.")

    long = result.summary_df.melt(
        id_vars=["threshold"],
        value_vars=[f"{m}_mean" for m in metrics],
        var_name="metric", value_name="mean")
    long["metric"] = long["metric"].str.removesuffix("_mean")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure

    if show_band:
        palette = att_palette_for(metrics)
        for m, color in zip(metrics, palette):
            mean = result.summary_df[f"{m}_mean"].to_numpy()
            std = result.summary_df[f"{m}_std"].to_numpy()
            thr = result.summary_df["threshold"].to_numpy()
            ax.fill_between(thr, mean - std, mean + std, color=color,
                            alpha=0.15, linewidth=0, zorder=1)

    p = (
        so.Plot(long, x="threshold", y="mean", color="metric")
          .add(so.Line(linewidth=2.6))
          .scale(color=att_nominal(order=metrics))
          .label(x="Decision threshold", y="Metric (mean across folds)",
                 color="Metric")
    )
    p.on(ax).plot()

    ax.axvline(result.mean_best_threshold, color=ATT_COLORS["gray_700"],
               linestyle="--", linewidth=1.6, zorder=2,
               label=f"mean best @ {result.mean_best_threshold:.2f} ({result.metric})")
    ax.set_ylim(-0.02, 1.02)
    if title is None:
        title = (f"Classification metrics vs threshold — {result.n_splits}-fold CV "
                 f"(tuned on {result.metric})")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cv_roc(
    result: CVThresholdResult,
    *,
    show_folds: bool = True,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 7))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], color=ATT_COLORS["gray_500"], linestyle="--",
            linewidth=1.4, zorder=1, label="Chance")

    if show_folds:
        (so.Plot(result.roc_per_fold, x="fpr", y="tpr", group="fold")
            .add(so.Line(linewidth=1.0, alpha=0.35, color=ATT_COLORS["gray_500"]))
            .on(ax).plot())

    g = result.roc_mean
    ax.fill_between(g["fpr_grid"], g["tpr_mean"] - g["tpr_std"],
                    g["tpr_mean"] + g["tpr_std"], color=ATT_COLORS["deep_blue"],
                    alpha=0.15, linewidth=0, zorder=2)
    ax.plot(g["fpr_grid"], g["tpr_mean"], color=ATT_COLORS["deep_blue"],
            linewidth=2.8, zorder=3,
            label=f"Mean ROC (AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    if title is None:
        title = f"ROC across folds — {result.n_splits}-fold CV"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_cv_threshold_metrics(result, path, *, dpi=220, **kwargs):
    return save_figure(plot_cv_threshold_metrics(result, **kwargs), path, dpi=dpi)


def save_cv_roc(result, path, *, dpi=220, **kwargs):
    return save_figure(plot_cv_roc(result, **kwargs), path, dpi=dpi)


def _summary_caption(result: CVThresholdResult) -> str:
    return (f"{result.n_splits}-fold internal CV — mean ± std across folds. "
            f"Tuned on {result.metric}: mean best threshold = "
            f"{result.mean_best_threshold:.3f}, mean {result.metric} = "
            f"{result.mean_best_value:.3f}, mean ROC-AUC = "
            f"{result.mean_auc:.3f} ± {result.std_auc:.3f}.")


def make_cv_summary_df(result: CVThresholdResult,
                       metrics: Sequence[str] = _DEFAULT_PLOT_METRICS) -> pd.DataFrame:
    rows = result.fold_best.copy()
    rows = rows.rename(columns={
        "fold": "Fold", "best_threshold": "Best threshold",
        "best_value": f"Best {result.metric}", "roc_auc": "ROC-AUC"})
    rows["Fold"] = rows["Fold"].astype(int).astype(str)

    mean_row = {
        "Fold": "mean ± std",
        "Best threshold": f"{result.mean_best_threshold:.3f} ± "
                          f"{result.fold_best['best_threshold'].std(ddof=0):.3f}",
        f"Best {result.metric}": f"{result.mean_best_value:.3f} ± "
                          f"{result.fold_best['best_value'].std(ddof=0):.3f}",
        "ROC-AUC": f"{result.mean_auc:.3f} ± {result.std_auc:.3f}",
    }
    disp = rows.copy()
    disp["Best threshold"] = disp["Best threshold"].map(lambda v: f"{v:.3f}")
    disp[f"Best {result.metric}"] = disp[f"Best {result.metric}"].map(lambda v: f"{v:.3f}")
    disp["ROC-AUC"] = disp["ROC-AUC"].map(lambda v: f"{v:.3f}")
    return pd.concat([disp, pd.DataFrame([mean_row])], ignore_index=True)


def style_cv_summary(result: CVThresholdResult,
                     metrics: Sequence[str] = _DEFAULT_PLOT_METRICS,
                     *, context: str = "report"):
    summary = make_cv_summary_df(result, metrics)

    def _emphasize_mean(row):
        if str(row.get("Fold", "")).startswith("mean"):
            return [f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
                    for _ in row]
        return ["" for _ in row]

    return (
        summary.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .apply(_emphasize_mean, axis=1)
            .set_caption(_summary_caption(result))
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_cv_summary(result: CVThresholdResult, path, *,
                    metrics: Sequence[str] = _DEFAULT_PLOT_METRICS,
                    context: str = "report", dpi: int = 220) -> str:
    summary = make_cv_summary_df(result, metrics)
    return save_table_by_extension(
        summary, path,
        styler=style_cv_summary(result, metrics, context=context),
        png_renderer=lambda df, p, **kw: _save_cv_summary_png(
            df, p, result, context=context, dpi=dpi),
    )


def _save_cv_summary_png(summary_df, path, result, *, context="report", dpi=220):
    return save_generic_table_png(
        summary_df, path, title=_summary_caption(result),
        context=context, dpi=dpi)


def att_palette_for(keys):
    pal = list(ATT_PALETTE)
    return [pal[i % len(pal)] for i in range(len(keys))]
