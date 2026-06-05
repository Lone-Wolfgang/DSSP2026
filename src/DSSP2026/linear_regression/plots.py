"""
regression/plot.py — presentation-quality plots for OLS regression.

Fit lines, actual-vs-fitted, partial-regression grids, residual diagnostics,
model-comparison curves, and (static + interactive) prediction plots.

Plotting only. Statistics come from regression.fit; tables from
regression.tables. Functions take models/DataFrames and return figures.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.nonparametric.smoothers_lowess import lowess

from DSSP2026.core.style import ATT_COLORS, ATT_PALETTE, ATT_SEQUENTIAL
from DSSP2026.linear_regression.fit import (
    numeric_predictors,
    categorical_predictors,
    make_ols_diagnostics_df,
    _build_comparison_prediction_grid,
    _detect_extrapolation,
    _resolve_comparison_training_data,
    _resolve_comparison_x,
    _top_ols_models_by_metric,
)
from DSSP2026.linear_regression.interactive import _predict_plot_interactive


# ---------------------------------------------------------------------------
# Fit line
# ---------------------------------------------------------------------------

def plot_ols_fit(model, df, formula, *, alpha=0.05, figsize=None, ax=None,
                 facet=False, show_points=False):
    """Plot the fitted regression line over a single numeric predictor.

    If a categorical predictor is present it becomes the grouping variable;
    facet=True lays groups out in small multiples.
    """
    numeric_preds = numeric_predictors(model, df)
    if len(numeric_preds) < 1:
        raise ValueError(
            f"plot_ols_fit requires at least one numeric predictor; "
            f"found {len(numeric_preds)}: {numeric_preds}. "
            f"Try plot_actual_vs_fitted or plot_ols_partial."
        )
    x_name = numeric_preds[0]
    y_name = model.model.endog_names
    categorical_preds = categorical_predictors(model, df)
    group_name = categorical_preds[0] if categorical_preds else None

    groups = [None]
    if group_name is not None:
        groups = list(pd.Series(df[group_name]).dropna().unique())

    if facet and group_name is not None:
        n_groups = len(groups)
        ncols = min(n_groups, 3)
        nrows = int(np.ceil(n_groups / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=figsize or (6 * ncols, 4.5 * nrows),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for extra_ax in axes[n_groups:]:
            extra_ax.set_visible(False)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 6))
        else:
            fig = ax.figure
        axes = [ax]

    for idx, group_value in enumerate(groups):
        color = ATT_PALETTE[idx % len(ATT_PALETTE)]
        label = "Observations" if group_value is None else str(group_value)
        plot_ax = axes[idx] if facet and group_name is not None else axes[0]
        plot_df = df if group_value is None else df[df[group_name] == group_value]

        if show_points:
            plot_ax.scatter(plot_df[x_name], plot_df[y_name], s=45, alpha=0.3,
                            color=color, edgecolor="white", linewidth=0.5,
                            label=label, zorder=2)

        plot_x = plot_df[x_name].to_numpy()
        plot_x = plot_x[~pd.isna(plot_x)]
        if len(plot_x) == 0:
            continue
        x_grid = np.linspace(np.nanmin(plot_x), np.nanmax(plot_x), 200)
        pred_df = _build_fit_prediction_grid(model, df, x_name, x_grid)
        if group_value is not None:
            pred_df[group_name] = group_value

        pred = model.get_prediction(pred_df).summary_frame(alpha=alpha)
        plot_ax.plot(x_grid, pred["mean"], color=color, linewidth=4.0, zorder=4,
                     path_effects=[pe.Stroke(linewidth=6.5, foreground="white"), pe.Normal()])

        if facet and group_name is not None:
            plot_ax.set_title(str(group_value))

    for plot_ax in axes:
        if not plot_ax.get_visible():
            continue
        plot_ax.set_xlabel(x_name)
        plot_ax.set_ylabel(y_name)

    if facet and group_name is not None:
        fig.suptitle(f"OLS fit: {formula}", fontweight="bold")
        _annotate_fit_stats(axes[0], model)
        fig.tight_layout()
    else:
        axes[0].set_title(f"OLS fit: {formula}")
        _annotate_fit_stats(axes[0], model)
        axes[0].legend(loc="best", title=group_name) if group_name else axes[0].legend(loc="best")
    return fig


def _build_fit_prediction_grid(model, df, x_name, x_grid):
    pred_df = pd.DataFrame({x_name: x_grid})
    y_name = model.model.endog_names
    for col in df.columns:
        if col in (x_name, y_name) or col in pred_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            pred_df[col] = df[col].mean()
        else:
            mode = df[col].mode(dropna=True)
            pred_df[col] = mode.iloc[0] if len(mode) else df[col].iloc[0]
    return pred_df


def _annotate_fit_stats(ax, model):
    """Drop a small box of fit statistics in the upper-left of the axes."""
    text = (
        f"N = {int(model.nobs)}\n"
        f"R² = {model.rsquared:.3f}\n"
        f"Adj. R² = {model.rsquared_adj:.3f}\n"
        f"F = {model.fvalue:.2f}  (p = {model.f_pvalue:.3g})"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, family="monospace",
            bbox=dict(facecolor="white", edgecolor=ATT_COLORS["gray_300"],
                      boxstyle="round,pad=0.5", alpha=0.92))


# ---------------------------------------------------------------------------
# Actual vs fitted
# ---------------------------------------------------------------------------

def plot_actual_vs_fitted(model, *, figsize=None, ax=None, equal_aspect=True,
                          show_points=False):
    """Scatter of actual vs. fitted values with a y = x reference line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 8))
    else:
        fig = ax.figure

    fitted = model.fittedvalues
    actual = model.model.endog

    if show_points:
        ax.scatter(fitted, actual, s=45, alpha=0.5, color=ATT_COLORS["att_blue"],
                   edgecolor="white", linewidth=0.5)
    lo = float(min(np.min(fitted), np.min(actual)))
    hi = float(max(np.max(fitted), np.max(actual)))
    ax.plot([lo, hi], [lo, hi], color=ATT_COLORS["gray_700"], linestyle="--",
            linewidth=1.8, label="y = x")

    ax.set_xlabel("Fitted")
    ax.set_ylabel(f"Actual ({model.model.endog_names})")
    ax.set_title("Actual vs. fitted")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    else:
        margin = (hi - lo) * 0.05
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
    _annotate_fit_stats(ax, model)
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# Partial regression
# ---------------------------------------------------------------------------

def plot_ols_partial(model, *, figsize=None):
    """Small-multiples of added-variable (partial regression) plots."""
    from statsmodels.graphics.regressionplots import plot_partregress_grid
    preds = [p for p in model.model.exog_names if p != "Intercept"]
    if len(preds) == 0:
        raise ValueError("No predictors to plot.")
    ncols = min(len(preds), 3)
    nrows = int(np.ceil(len(preds) / ncols))
    fig = plt.figure(figsize=figsize or (5 * ncols, 4.5 * nrows))
    plot_partregress_grid(model, fig=fig)
    for ax in fig.axes:
        for child in ax.get_children():
            if hasattr(child, "set_color") and child.__class__.__name__ == "PathCollection":
                child.set_color(ATT_COLORS["att_blue"])
                child.set_alpha(0.55)
    fig.suptitle("Partial regression plots", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite diagnostics figure (fit + actual-vs-fitted + partial grid)
# ---------------------------------------------------------------------------

def plot_ols_diagnostics(model, df, formula, *, alpha=0.05, figsize=None,
                         show_points=False):
    """Composite: fit line (if applicable) + actual-vs-fitted + partial grid.

    Partial plots are computed directly from the design matrix (Frisch-Waugh),
    so they work for numeric, categorical-dummy, and interaction terms.
    """
    numeric_preds = numeric_predictors(model, df)
    other_preds = [p for p in model.model.exog_names if p != "Intercept"]
    n_partial = len(other_preds)
    has_fit = len(numeric_preds) == 1

    top_cols = 2 if has_fit else 1
    partial_cols = min(n_partial, 3) if n_partial else 1
    partial_rows = int(np.ceil(n_partial / partial_cols)) if n_partial else 0
    total_cols = max(top_cols, partial_cols)

    fig = plt.figure(figsize=figsize or (6 * total_cols, 5 * (1 + partial_rows)))
    gs = fig.add_gridspec(1 + partial_rows, total_cols, hspace=0.45, wspace=0.35)

    if has_fit:
        ax_fit = fig.add_subplot(gs[0, 0])
        plot_ols_fit(model, df, formula, alpha=alpha, ax=ax_fit, show_points=show_points)
        ax_avf = fig.add_subplot(gs[0, 1])
    else:
        ax_avf = fig.add_subplot(gs[0, :])
    plot_actual_vs_fitted(model, ax=ax_avf, equal_aspect=False, show_points=show_points)

    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    y = pd.Series(model.model.endog, name=model.model.endog_names)

    for i, term in enumerate(other_preds):
        r = 1 + i // partial_cols
        c = i % partial_cols
        ax_p = fig.add_subplot(gs[r, c])

        others = [col for col in X.columns if col != term]
        X_others = X[others].to_numpy()
        x_j = X[term].to_numpy()

        beta_y, *_ = np.linalg.lstsq(X_others, y.to_numpy(), rcond=None)
        beta_x, *_ = np.linalg.lstsq(X_others, x_j, rcond=None)
        y_resid = y.to_numpy() - X_others @ beta_y
        x_resid = x_j - X_others @ beta_x

        ax_p.scatter(x_resid, y_resid, s=35, alpha=0.5, color=ATT_COLORS["att_blue"],
                     edgecolor="white", linewidth=0.5)
        slope = model.params[term]
        xr = np.array([x_resid.min(), x_resid.max()])
        ax_p.plot(xr, slope * xr, color=ATT_COLORS["navy"], linewidth=2.2)

        ax_p.set_xlabel(f"e({term} | others)")
        ax_p.set_ylabel(f"e({y.name} | others)")
        ax_p.set_title(f"Partial: {term}", fontsize=12)
        ax_p.axhline(0, color=ATT_COLORS["gray_500"], linewidth=0.8, linestyle=":")
        ax_p.axvline(0, color=ATT_COLORS["gray_500"], linewidth=0.8, linestyle=":")

    fig.suptitle(f"OLS diagnostics: {formula}", fontweight="bold", y=0.995)
    return fig


# ---------------------------------------------------------------------------
# Residual diagnostics quad (residuals/QQ/scale-location/Cook's)
# ---------------------------------------------------------------------------

def plot_residual_diagnostics(model, title="", *, figsize=None, bins=40,
                              studentized="internal", sample=None, random_state=42):
    """Four-panel residual diagnostics. Returns (figure, diagnostics_df)."""
    diag_df = make_ols_diagnostics_df(model, studentized=studentized,
                                      sample=sample, random_state=random_state)

    fig, axes = plt.subplots(2, 2, figsize=figsize or (13, 10))
    ax_resid, ax_qq, ax_scale, ax_cook = axes.ravel()

    ax_resid.scatter(diag_df["fitted"], diag_df["residuals"], s=34, alpha=0.65,
                     color=ATT_COLORS["att_blue"], edgecolor="white", linewidth=0.4)
    ax_resid.axhline(0, color=ATT_COLORS["gray_700"], linewidth=1.0, linestyle="--")
    if len(diag_df) >= 5:
        smooth = lowess(diag_df["residuals"], diag_df["fitted"], frac=0.45, return_sorted=True)
        ax_resid.plot(smooth[:, 0], smooth[:, 1], color=ATT_COLORS["orange"], linewidth=2.0)
    ax_resid.set_title("Residuals vs fitted")
    ax_resid.set_xlabel("Fitted values")
    ax_resid.set_ylabel("Residuals")

    stats.probplot(diag_df["stud_resid"], dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor(ATT_COLORS["att_blue"])
    ax_qq.get_lines()[0].set_markeredgecolor("white")
    ax_qq.get_lines()[0].set_alpha(0.65)
    ax_qq.get_lines()[1].set_color(ATT_COLORS["orange"])
    ax_qq.get_lines()[1].set_linewidth(2.0)
    ax_qq.set_title("Normal Q-Q")

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

    ax_cook.stem(diag_df["obs"], diag_df["cooks"],
                 linefmt=ATT_COLORS["att_blue"], markerfmt=" ", basefmt=" ")
    ax_cook.axhline(4 / max(int(model.nobs), 1), color=ATT_COLORS["orange"],
                    linewidth=1.6, linestyle="--")
    ax_cook.set_title("Cook's distance")
    ax_cook.set_xlabel("Observation")
    ax_cook.set_ylabel("Cook's D")

    if title:
        fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, diag_df


# ---------------------------------------------------------------------------
# Model comparison curves
# ---------------------------------------------------------------------------

def _comparison_line_colors(top_rows):
    return [ATT_SEQUENTIAL[::-1][idx % len(ATT_SEQUENTIAL)] for idx in range(len(top_rows))]


def plot_ols_model_comparison(comparison_df, models, *, best_by, top_n=5,
                              x=None, figsize=None):
    """Overlay the top-N models' fitted curves over a shared predictor."""
    top_rows = _top_ols_models_by_metric(comparison_df, best_by=best_by, top_n=top_n)
    selected_models = [models[int(row["_fit_index"])] for _, row in top_rows.iterrows()]
    x_name = _resolve_comparison_x(selected_models, x)
    training_data = _resolve_comparison_training_data(selected_models, x_name)
    y_name = selected_models[0].model.endog_names

    fig, ax = plt.subplots(figsize=figsize or (10, 6))
    ax.scatter(training_data[x_name], training_data[y_name], s=40, alpha=0.45,
               color=ATT_COLORS["gray_300"], edgecolor="white", linewidth=0.5,
               label="Training data", zorder=1)

    x_grid = np.linspace(training_data[x_name].min(), training_data[x_name].max(), 200)
    line_colors = _comparison_line_colors(top_rows)

    for idx, ((_, row), model, color) in enumerate(
            zip(top_rows.iterrows(), selected_models, line_colors)):
        pred_df = _build_comparison_prediction_grid(model, training_data, x_name, x_grid)
        y_pred = model.predict(pred_df)
        label = f"{idx + 1} | {row['formula']}"
        linewidth = 3.2 if idx == 0 else 2.2
        zorder = 4 if idx == 0 else 3
        ax.plot(x_grid, y_pred, color=color, linewidth=linewidth, label=label, zorder=zorder)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"OLS model comparison: top {len(top_rows)} models")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              title=f"Ranked by {best_by}", loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction plots (static)
# ---------------------------------------------------------------------------

def plot_predictions(predictions, *, model, new_data, new_filled,
                     training_data=None, formula=None, alpha=0.05,
                     interval="prediction", x=None, show_training=True,
                     figsize=None, ax=None, interactive=False):
    """Plot predictions on new data, vs a predictor or vs actual values."""
    if interval not in ("none", "prediction", "confidence", "all"):
        raise ValueError("interval must be 'none', 'prediction', 'confidence', or 'all'")

    y_name = model.model.endog_names
    conf_level = int(round((1 - alpha) * 100))

    if x is None:
        numeric_preds = [
            name for name in model.model.exog_names
            if name != "Intercept" and name in new_filled.columns
            and pd.api.types.is_numeric_dtype(new_filled[name])
        ]
        x = numeric_preds[0] if numeric_preds else "actual"

    if interactive:
        return _predict_plot_interactive(
            new_data=new_data, new_filled=new_filled, predictions=predictions,
            training_data=training_data, x=x, y_name=y_name, interval=interval,
            conf_level=conf_level, show_training=show_training, formula=formula)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure

    if x == "actual":
        if y_name not in new_data.columns:
            raise ValueError(
                f"x='actual' requires the target column '{y_name}' in new_data.")
        _plot_predicted_vs_actual(ax, new_data[y_name].to_numpy(), predictions,
                                  training_data, model, y_name, interval=interval,
                                  conf_level=conf_level, show_training=show_training)
    else:
        if x not in new_filled.columns:
            raise ValueError(f"Column '{x}' not found in new_data.")
        _plot_predicted_vs_x(ax, new_filled[x].to_numpy(), predictions, training_data,
                             model, x, y_name, interval=interval, conf_level=conf_level,
                             show_training=show_training)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.subplots_adjust(bottom=0.25)
    return fig


def _plot_predicted_vs_x(ax, x_vals, predictions, training_data, model, x_name, y_name,
                         *, interval, conf_level, show_training):
    show_ci = interval in ("confidence", "all")
    show_pi = interval in ("prediction", "all")
    ci_color = ATT_COLORS["orange"]
    pi_color = ATT_COLORS["navy"]

    if show_training and training_data is not None and x_name in training_data.columns:
        ax.scatter(training_data[x_name], training_data[y_name], s=25, alpha=0.25,
                   color=ATT_COLORS["gray_500"], edgecolor="none",
                   label="Training data", zorder=1)

    x_arr = np.asarray(x_vals)
    if training_data is not None:
        row_is_extrap, _ = _detect_extrapolation(pd.DataFrame({x_name: x_arr}), training_data)
    else:
        row_is_extrap = [False] * len(x_arr)

    marker_size = 90
    offset = (float(np.ptp(x_arr)) * 0.014) if (show_ci and show_pi and len(x_arr) > 1) else 0.0

    for i in range(len(x_arr)):
        extrap = row_is_extrap[i] if row_is_extrap else False
        mk_ci = ATT_COLORS["magenta"] if extrap else ci_color
        mk_pi = ATT_COLORS["magenta"] if extrap else pi_color
        if show_ci:
            ax.scatter(x_arr[i], predictions["ci_upper"].iloc[i], marker="^", s=marker_size,
                       facecolors="none", edgecolors=mk_ci, linewidth=2.0, zorder=4,
                       label=f"{conf_level}% CI" if i == 0 else "_nolegend_")
            ax.scatter(x_arr[i], predictions["ci_lower"].iloc[i], marker="v", s=marker_size,
                       facecolors="none", edgecolors=mk_ci, linewidth=2.0, zorder=4)
        if show_pi:
            ax.scatter(x_arr[i] + offset, predictions["pi_upper"].iloc[i], marker="^",
                       s=marker_size, color=mk_pi, edgecolor=mk_pi, linewidth=1.2, zorder=4,
                       label=f"{conf_level}% PI" if i == 0 else "_nolegend_")
            ax.scatter(x_arr[i] + offset, predictions["pi_lower"].iloc[i], marker="v",
                       s=marker_size, color=mk_pi, edgecolor=mk_pi, linewidth=1.2, zorder=4)

    for i in range(len(x_arr)):
        extrap = row_is_extrap[i] if row_is_extrap else False
        pt_color = ATT_COLORS["magenta"] if extrap else ATT_COLORS["att_blue"]
        border = ATT_COLORS["magenta"] if extrap else ATT_COLORS["navy"]
        first_extrap = i == list(row_is_extrap).index(True) if (extrap and True in list(row_is_extrap)) else False
        first_normal = i == next((j for j, e in enumerate(row_is_extrap) if not e), 0)
        label = "Extrapolation" if (extrap and first_extrap) else \
                ("New observations" if (not extrap and first_normal) else "_nolegend_")
        ax.scatter(x_arr[i], predictions["mean"].iloc[i], s=80, color=pt_color,
                   edgecolor=border, linewidth=1.2, zorder=5, label=label)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)


def _plot_predicted_vs_actual(ax, actual, predictions, training_data, model, y_name,
                              *, interval, conf_level, show_training):
    mean = predictions["mean"].to_numpy()
    if interval == "confidence":
        lower, upper = predictions["ci_lower"].to_numpy(), predictions["ci_upper"].to_numpy()
        band_label = f"{conf_level}% CI"
    else:
        lower, upper = predictions["pi_lower"].to_numpy(), predictions["pi_upper"].to_numpy()
        band_label = f"{conf_level}% PI"

    if show_training and training_data is not None:
        try:
            ax.scatter(model.model.endog, model.fittedvalues, s=25, alpha=0.25,
                       color=ATT_COLORS["gray_500"], edgecolor="none",
                       label="Training data", zorder=1)
        except Exception:
            pass

    lo = float(min(np.min(actual), np.min(lower)))
    hi = float(max(np.max(actual), np.max(upper)))
    ax.plot([lo, hi], [lo, hi], color=ATT_COLORS["gray_700"], linestyle="--",
            linewidth=1.5, label="y = x", zorder=2)
    ax.errorbar(actual, mean, yerr=[mean - lower, upper - mean], fmt="o", markersize=8,
                color=ATT_COLORS["att_blue"], markeredgecolor=ATT_COLORS["navy"],
                markeredgewidth=1.2, ecolor=ATT_COLORS["att_blue"], elinewidth=1.5,
                capsize=4, alpha=0.85, label=f"Predicted ± {band_label}", zorder=4)
    ax.set_xlabel(f"Actual ({y_name})")
    ax.set_ylabel(f"Predicted ({y_name})")

