"""
regression/tune.py — model/feature selection for regression.

Provides model ranking by information criteria for fitted OLS models.
"""

from typing import Sequence

import pandas as pd

from DSSP2026.linear_regression.fit import OLSResult


def rank_ols_models(
    fits: Sequence,
    *,
    best_by: str = "BIC",
) -> tuple[pd.DataFrame, list]:
    """Rank a list of OLSResult or RegressionResultsWrapper by AIC, BIC, or Adjusted R².

    Returns
    -------
    comparison_df : pd.DataFrame
        Sorted with the best model first. Columns: formula, n_parameters,
        Adjusted R2, AIC, BIC.
    models : list[RegressionResultsWrapper]
        In the same order as comparison_df.
    """
    return _rank_ols_models(fits, best_by=best_by)


def _extract_ols_model_info(fits):
    rows, models = [], []
    for i, fit in enumerate(fits, start=1):
        if isinstance(fit, OLSResult):
            model, formula = fit.model, fit.formula
        else:
            model = fit
            formula = getattr(getattr(model, "model", None), "formula", None)
        if formula is None:
            formula = f"Model {i}"
        models.append(model)
        rows.append({
            "_fit_index":   len(models) - 1,
            "formula":      formula,
            "n_parameters": int(len(model.params)),
            "Adjusted R2":  float(model.rsquared_adj),
            "AIC":          float(model.aic),
            "BIC":          float(model.bic),
        })
    return pd.DataFrame(rows), models


def _resolve_ols_comparison_metric(best_by):
    metric_lookup = {
        "adjusted r2":        "Adjusted R2",
        "adj r2":             "Adjusted R2",
        "adjusted r-squared": "Adjusted R2",
        "aic":                "AIC",
        "bic":                "BIC",
    }
    key = best_by.lower()
    if key not in metric_lookup:
        raise ValueError(
            f"best_by must be one of ['Adjusted R2', 'AIC', 'BIC']; got {best_by!r}."
        )
    return metric_lookup[key]


def _rank_ols_models(fits, *, best_by):
    comparison_df, models = _extract_ols_model_info(fits)
    best_col = _resolve_ols_comparison_metric(best_by)
    ascending = best_col.lower() != "adjusted r2"
    best_idx = (
        comparison_df[best_col].idxmin() if ascending
        else comparison_df[best_col].idxmax()
    )
    best_row = comparison_df.loc[[best_idx]]
    remaining = comparison_df.drop(index=best_idx).sort_values(
        ["n_parameters", "formula"], ascending=[True, True]
    )
    comparison_df = pd.concat([best_row, remaining]).reset_index(drop=True)
    return comparison_df, models
