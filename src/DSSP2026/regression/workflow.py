"""
regression/workflow.py — high-level convenience wrappers.

These tie together fit + plots + tables for common one-call workflows.
Everything here is composition; the real logic lives in the three modules
it imports. Reach for these in notebooks; reach for the underlying modules
when you need control.
"""

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from DSSP2026.regression.fit import fit_ols, predict_ols, _fill_missing_predictors
from DSSP2026.regression import plots as rplots
from DSSP2026.regression import tables as rtables


def fit_and_plot_ols(df, formula, *, alpha=0.05, plot_kind="auto",
                     table_context="report", figsize=None, ax=None, facet=False,
                     show_points=False):
    """Fit an OLS model, then produce its plot and styled coefficient table.

    Returns
    -------
    result : OLSResult        (model, coef_df, training_data, formula)
    figure : matplotlib Figure
    table  : pandas Styler
    """
    from modules.regression.fit import resolve_plot_kind

    result = fit_ols(df, formula)
    model = result.model
    table = rtables.style_ols_coefficients(result.coef_df, model, context=table_context)
    kind = resolve_plot_kind(plot_kind, model, df)

    if kind == "fit":
        fig = rplots.plot_ols_fit(model, df, formula, alpha=alpha, figsize=figsize,
                                  ax=ax, facet=facet, show_points=show_points)
    elif kind == "actual_vs_fitted":
        fig = rplots.plot_actual_vs_fitted(model, figsize=figsize, ax=ax,
                                           show_points=show_points)
    elif kind == "partial":
        fig = rplots.plot_ols_partial(model, figsize=figsize)
    elif kind == "all":
        fig = rplots.plot_ols_diagnostics(model, df, formula, alpha=alpha,
                                          figsize=figsize, show_points=show_points)
    else:
        raise ValueError(f"Unknown plot_kind: {plot_kind}")

    return result, fig, table


def predict_and_plot(model, new_data, *, training_data=None, formula=None,
                     alpha=0.05, interval="prediction", x=None, show_training=True,
                     figsize=None, ax=None, interactive=False):
    """Predict on new data, then plot and build a styled summary table.

    Returns
    -------
    prediction : PredictionResult  (predictions, table_df)
    figure     : matplotlib or plotly Figure
    table      : pandas Styler
    """
    prediction = predict_ols(model, new_data, training_data=training_data, alpha=alpha)
    predictions = prediction.predictions
    new_filled = _fill_missing_predictors(new_data, model, training_data)

    fig = rplots.plot_predictions(
        predictions, model=model, new_data=new_data, new_filled=new_filled,
        training_data=training_data, formula=formula, alpha=alpha, interval=interval,
        x=x, show_training=show_training, figsize=figsize, ax=ax, interactive=interactive)

    conf_level = int(round((1 - alpha) * 100))
    table, summary_df = rtables.style_prediction_summary(
        predictions, conf_level=conf_level, new_data=new_data,
        training_data=training_data, model=model, y_name=model.model.endog_names)
    prediction.table_df = summary_df

    return prediction, fig, table