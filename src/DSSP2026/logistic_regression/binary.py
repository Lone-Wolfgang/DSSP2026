"""
logistic/fit.py — logistic-regression fitting and prediction (statsmodels).

Responsibilities
----------------
- Fit binary logistic models via the formula interface (with transform helpers)
- Extract a tidy coefficient table with odds ratios, CIs, and p-values
- Predict class probabilities and (thresholded) class labels

Presentation-free. Mirrors regression/fit.py: returns models and DataFrames;
figures and styled tables are built by the logistic/core presentation layers.
Classification *evaluation* (confusion matrix, ROC, metrics) is family-agnostic
and lives in core — call those with the predictions produced here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import EvalEnvironment
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from DSSP2026.eda.encoders import FORMULA_NAMESPACE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LogitResult:
    """Returned by fit_logit. Pure data — no figures, no Stylers."""
    model: BinaryResultsWrapper
    coef_df: pd.DataFrame
    training_data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_logit(
    df: pd.DataFrame,
    formula: str,
    *,
    maxiter: int = 100,
    disp: bool = False,
) -> LogitResult:
    """Fit a binary logistic regression and return a LogitResult.

    The LHS of the formula must be binary (0/1, bool, or two-level categorical;
    statsmodels treats the higher/with-`C()` level as the positive outcome).
    The formula environment is augmented with common numpy transforms so
    ``log(x)``, ``sqrt(x)``, etc. work without the ``np.`` prefix.

    Parameters
    ----------
    maxiter : int
        Max Newton iterations.
    disp : bool
        If True, statsmodels prints convergence output.
    """
    formula_env = EvalEnvironment.capture(1).with_outer_namespace(FORMULA_NAMESPACE)
    _check_binary_target(df, formula)
    model = smf.logit(formula, data=df, eval_env=formula_env).fit(
        maxiter=maxiter, disp=disp)
    coef_df = make_logit_coef_df(model)
    return LogitResult(model=model, coef_df=coef_df, training_data=df, formula=formula)


def _check_binary_target(df: pd.DataFrame, formula: str) -> None:
    """Validate the LHS is a numeric 0/1 column before handing it to statsmodels.

    statsmodels' logit requires a numeric endog; a bool or string target
    raises a cryptic 'multiple columns' error. We catch the common case (a
    bare column name on the LHS) early with an actionable message. Transformed
    or C()-wrapped LHS expressions are left to statsmodels.
    """
    lhs = formula.split("~")[0].strip()
    if lhs not in df.columns:
        return  # transformed/categorical LHS — let statsmodels handle it
    col = df[lhs]
    if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
        raise ValueError(
            f"Logit target '{lhs}' must be numeric 0/1, but its dtype is "
            f"{col.dtype}. Convert it first, e.g. "
            f"df['{lhs}'] = df['{lhs}'].astype(int) for booleans, or map a "
            f"two-level category to 0/1 (and note which level is the positive "
            f"class)."
        )
    uniques = set(pd.unique(col.dropna()))
    if not uniques <= {0, 1}:
        raise ValueError(
            f"Logit target '{lhs}' must contain only 0/1, but found values "
            f"{sorted(uniques)[:6]}{'...' if len(uniques) > 6 else ''}."
        )


# ---------------------------------------------------------------------------
# Coefficient table (odds ratios)
# ---------------------------------------------------------------------------

def make_logit_coef_df(
    model: BinaryResultsWrapper,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Tidy coefficient table on both log-odds and odds-ratio scales.

    Columns: Term, Coefficient (log-odds), Std Error, z, p-value,
    Odds Ratio, OR CI low, OR CI high. The CI is on the odds-ratio scale
    (exp of the log-odds CI), which is what readers usually want to report.
    """
    conf_level = int(round((1 - alpha) * 100))
    ci = model.conf_int(alpha=alpha)
    ci.columns = ["ci_low", "ci_high"]

    coef_df = pd.concat(
        [
            model.params.rename("Coefficient"),
            model.bse.rename("Std Error"),
            model.tvalues.rename("z"),
            model.pvalues.rename("p-value"),
            np.exp(model.params).rename("Odds Ratio"),
            np.exp(ci["ci_low"]).rename(f"OR CI {conf_level}% low"),
            np.exp(ci["ci_high"]).rename(f"OR CI {conf_level}% high"),
        ],
        axis=1,
    )
    coef_df.index.name = "Term"
    return coef_df.reset_index()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_proba(
    model: BinaryResultsWrapper,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """Predicted probability of the positive class for each row of new_data."""
    return np.asarray(model.predict(new_data))


def predict_class(
    model: BinaryResultsWrapper,
    new_data: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predicted 0/1 labels at a probability threshold (default 0.5)."""
    proba = predict_proba(model, new_data)
    return (proba >= threshold).astype(int)


@dataclass
class LogitPrediction:
    """Bundle of probabilities and thresholded labels for new data."""
    proba: np.ndarray
    labels: np.ndarray
    threshold: float


def predict_logit(
    model: BinaryResultsWrapper,
    new_data: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> LogitPrediction:
    """Predict probabilities and labels in one call.

    Feed `prediction.labels` / `prediction.proba` straight into the core
    classification evaluators (make_confusion_matrix, classification_metrics,
    roc_curve_points).
    """
    proba = predict_proba(model, new_data)
    labels = (proba >= threshold).astype(int)
    return LogitPrediction(proba=proba, labels=labels, threshold=threshold)


# ---------------------------------------------------------------------------
# Target extraction (handy when scoring against the held-out truth)
# ---------------------------------------------------------------------------

def get_endog(model: BinaryResultsWrapper, data: pd.DataFrame) -> np.ndarray:
    """Return the 0/1 outcome vector for `data` using the model's formula.

    Useful for pulling true labels out of a test frame in exactly the encoding
    statsmodels used to fit. Re-evaluates the formula's left-hand side against
    `data` via the fitted model's design info, so bool / categorical / numeric
    targets all come back as 0/1.
    """
    from patsy import dmatrices

    formula = model.model.formula
    formula_env = EvalEnvironment.capture(1).with_outer_namespace(FORMULA_NAMESPACE)
    y, _ = dmatrices(formula, data, return_type="dataframe", eval_env=formula_env)
    return np.asarray(y.iloc[:, 0]).astype(int)