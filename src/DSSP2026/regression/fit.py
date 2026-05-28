"""
regression.fit.py — OLS fitting, prediction, and diagnostics.

Responsibilities
----------------
- Fit OLS models (with formula-transform helpers)
- Extract tidy DataFrames from fitted models (coefficients, predictions, diagnostics)
- Rank / compare models by information criteria

This module is presentation-free. It never imports matplotlib, seaborn, or
anything from modules.presentation. Callers who want figures should pass the
DataFrames and models returned here into the presentation layer.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import EvalEnvironment
from statsmodels.regression.linear_model import RegressionResultsWrapper


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class OLSResult:
    """Returned by fit_ols. Pure data — no figures, no Stylers."""
    model: RegressionResultsWrapper
    coef_df: pd.DataFrame
    training_data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None


@dataclass
class OLSDiagnosticsResult:
    """Returned by make_ols_diagnostics_df."""
    data: pd.DataFrame


@dataclass
class PredictionResult:
    """Returned by predict_ols.

    predictions columns: mean, ci_lower, ci_upper, pi_lower, pi_upper
    table_df is an optional pre-formatted summary DataFrame (wider form,
    suitable for display); callers produce the Styler themselves.
    """
    predictions: pd.DataFrame
    table_df: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_ols(
    df: pd.DataFrame,
    formula: str,
) -> OLSResult:
    """Fit an OLS model and return an OLSResult.

    The formula environment is augmented with common numpy transforms so
    expressions like ``log(x)``, ``sqrt(x)``, ``exp(x)`` work without
    prefixing ``np.``.
    """
    formula_env = EvalEnvironment.capture(1).with_outer_namespace({
        "np": np,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "exp": np.exp,
    })
    model = smf.ols(formula, data=df, eval_env=formula_env).fit()
    coef_df = make_ols_coef_df(model)
    return OLSResult(
        model=model,
        coef_df=coef_df,
        training_data=df,
        formula=formula,
    )


# ---------------------------------------------------------------------------
# DataFrame extractors
# ---------------------------------------------------------------------------

def make_ols_coef_df(
    model: RegressionResultsWrapper,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Tidy DataFrame of coefficients, standard errors, t-stats, p-values, CIs."""
    conf_level = int(round((1 - alpha) * 100))
    ci = model.conf_int(alpha=alpha)
    ci.columns = [f"CI {conf_level}% low", f"CI {conf_level}% high"]
    coef_df = pd.concat(
        [
            model.params.rename("Coefficient"),
            model.bse.rename("Std Error"),
            model.tvalues.rename("t"),
            model.pvalues.rename("p-value"),
            ci,
        ],
        axis=1,
    )
    coef_df.index.name = "Term"
    return coef_df.reset_index()


def predict_ols(
    model: RegressionResultsWrapper,
    new_data: pd.DataFrame,
    *,
    training_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> PredictionResult:
    """Return a PredictionResult with mean, CI, and PI columns.

    Missing predictors are filled from ``training_data`` means/modes when
    provided; raises ValueError if they are needed but unavailable.
    """
    new_filled = _fill_missing_predictors(new_data, model, training_data)
    pred_summary = model.get_prediction(new_filled).summary_frame(alpha=alpha)
    predictions = pd.DataFrame({
        "mean":     pred_summary["mean"].to_numpy(),
        "ci_lower": pred_summary["mean_ci_lower"].to_numpy(),
        "ci_upper": pred_summary["mean_ci_upper"].to_numpy(),
        "pi_lower": pred_summary["obs_ci_lower"].to_numpy(),
        "pi_upper": pred_summary["obs_ci_upper"].to_numpy(),
    }, index=new_data.index)
    return PredictionResult(predictions=predictions)


def make_ols_diagnostics_df(
    model: RegressionResultsWrapper,
    *,
    studentized: str = "internal",
    sample: Optional[int] = None,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """DataFrame of residuals, leverage, Cook's distance, and danger-zone flags.

    Columns
    -------
    fitted, residuals, stud_resid, cooks, leverage, obs,
    in_danger_zone (bool), bubble_size (Cook's D when flagged, else min)
    """
    influence = model.get_influence()
    if studentized == "internal":
        stud_resid = influence.resid_studentized_internal
    elif studentized == "external":
        stud_resid = influence.resid_studentized_external
    else:
        raise ValueError("studentized must be 'internal' or 'external'.")

    df = pd.DataFrame({
        "fitted":    model.fittedvalues,
        "residuals": model.resid,
        "stud_resid": stud_resid,
        "cooks":     influence.cooks_distance[0],
        "leverage":  influence.hat_matrix_diag,
        "obs":       np.arange(1, int(model.nobs) + 1),
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["stud_resid"])

    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=random_state).sort_values("obs")

    p = len(model.params)
    n = int(model.nobs)
    lev_threshold = 2 * p / n
    res_threshold = 2
    min_cooks = df["cooks"].min() if len(df) else 0
    df["in_danger_zone"] = (
        (df["leverage"] > lev_threshold) &
        (df["stud_resid"].abs() > res_threshold)
    )
    df["bubble_size"] = np.where(df["in_danger_zone"], df["cooks"], min_cooks)
    return df


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Predictor introspection (used by the presentation layer)
# ---------------------------------------------------------------------------

def numeric_predictors(model: RegressionResultsWrapper, df: pd.DataFrame) -> list[str]:
    """Names of predictors in the model that map to numeric columns in df."""
    return [
        name for name in model.params.index
        if name != "Intercept"
        and name in df.columns
        and pd.api.types.is_numeric_dtype(df[name])
    ]


def categorical_predictors(model: RegressionResultsWrapper, df: pd.DataFrame) -> list[str]:
    """Names of predictors in the model that map to categorical columns in df."""
    y_name = model.model.endog_names
    design_info = getattr(model.model.data, "design_info", None)
    design_factors = set()
    if design_info is not None:
        for factor_info in design_info.factor_infos.values():
            factor_name = factor_info.factor.name()
            if factor_name.startswith("C(") and factor_name.endswith(")"):
                factor_name = factor_name[2:-1]
            design_factors.add(factor_name)

    return [
        col for col in df.columns
        if col != y_name
        and not pd.api.types.is_numeric_dtype(df[col])
        and col in design_factors
    ]


def resolve_plot_kind(plot_kind: str, model, df: pd.DataFrame) -> str:
    """Resolve 'auto' to either 'fit' or 'actual_vs_fitted'."""
    if plot_kind != "auto":
        return plot_kind
    num = numeric_predictors(model, df)
    cat = categorical_predictors(model, df)
    if cat and len(num) >= 1:
        return "fit"
    return "fit" if len(num) == 1 else "actual_vs_fitted"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fill_missing_predictors(new_data, model, training_data):
    """Fill columns the model needs that are absent from new_data.

    Uses training-data means for numeric columns and modes for categorical.
    Raises ValueError if training_data is needed but not supplied.
    """
    needed = set()
    for name in model.model.exog_names:
        if name == "Intercept":
            continue
        if "(" in name and ")" in name:
            inner = name[name.index("(") + 1:name.index(")")]
            needed.add(inner)
        else:
            base = name.split("[")[0].split(":")[0]
            needed.add(base)

    missing = needed - set(new_data.columns)
    if not missing:
        return new_data.copy()
    if training_data is None:
        raise ValueError(
            f"new_data is missing columns {missing} and no training_data "
            f"was supplied. Either include those columns or pass training_data."
        )
    filled = new_data.copy()
    for col in missing:
        if col in training_data.columns:
            if pd.api.types.is_numeric_dtype(training_data[col]):
                filled[col] = training_data[col].mean()
            else:
                filled[col] = training_data[col].mode().iloc[0]
    return filled


def _detect_extrapolation(new_data, training_data):
    """Return per-row extrapolation flags.

    Returns
    -------
    row_is_extrap : list[bool]
    col_flags : list[dict]  — {col_name: True} for each out-of-range column
    """
    n = len(new_data)
    row_is_extrap = [False] * n
    col_flags = [{} for _ in range(n)]

    if training_data is None:
        return row_is_extrap, col_flags

    for col in new_data.columns:
        if col not in training_data.columns:
            continue
        if not pd.api.types.is_numeric_dtype(training_data[col]):
            continue
        train_min = training_data[col].min()
        train_max = training_data[col].max()
        for idx in range(n):
            val = new_data[col].iloc[idx]
            try:
                if val < train_min or val > train_max:
                    col_flags[idx][col] = True
                    row_is_extrap[idx] = True
            except TypeError:
                pass

    return row_is_extrap, col_flags


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


def _top_ols_models_by_metric(comparison_df, *, best_by, top_n):
    ascending = best_by.lower() != "adjusted r2"
    top_rows = comparison_df.sort_values(
        [best_by, "n_parameters", "formula"],
        ascending=[ascending, True, True],
    ).head(max(1, min(int(top_n), len(comparison_df))))
    best_fit_index = comparison_df.loc[0, "_fit_index"]
    top_rows = top_rows[top_rows["_fit_index"] != best_fit_index]
    top_rows = pd.concat([comparison_df.iloc[[0]], top_rows]).head(top_n)
    return top_rows.reset_index(drop=True)


def _resolve_comparison_x(models, x):
    if x is not None:
        return x
    predictor_sets, ordered_candidates = [], []
    for model in models:
        frame = getattr(getattr(model.model, "data", None), "frame", None)
        if frame is None:
            predictor_sets.append(set())
            continue
        predictors = {
            name for name in model.model.exog_names
            if name != "Intercept"
            and name in frame.columns
            and pd.api.types.is_numeric_dtype(frame[name])
        }
        predictor_sets.append(predictors)
        if not ordered_candidates:
            ordered_candidates = [
                name for name in model.model.exog_names
                if name != "Intercept"
                and name in frame.columns
                and pd.api.types.is_numeric_dtype(frame[name])
            ]
    shared = set.intersection(*predictor_sets) if predictor_sets else set()
    if shared:
        for name in ordered_candidates:
            if name in shared:
                return name
    raise ValueError(
        "Could not infer an x-axis variable for the comparison plot. "
        "Pass x='column_name'."
    )


def _resolve_comparison_training_data(models, x_name):
    for model in models:
        frame = getattr(getattr(model.model, "data", None), "frame", None)
        y_name = model.model.endog_names
        if frame is not None and x_name in frame.columns and y_name in frame.columns:
            return frame
    raise ValueError(
        f"Could not find training data containing both x='{x_name}' and the model target."
    )


def _build_comparison_prediction_grid(model, training_data, x_name, x_grid):
    pred_df = pd.DataFrame({x_name: x_grid})
    frame = getattr(getattr(model.model, "data", None), "frame", training_data)
    for col in frame.columns:
        if col in (x_name, model.model.endog_names) or col in pred_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            pred_df[col] = frame[col].mean()
        else:
            mode = frame[col].mode(dropna=True)
            pred_df[col] = mode.iloc[0] if len(mode) else frame[col].iloc[0]
    return pred_df