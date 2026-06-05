"""
logistic/multiclass.py — multinomial logistic regression (statsmodels mnlogit).

The multiclass sibling of ``logistic/fit.py``. Where ``fit.py`` fits a binary
logit (one coefficient vector, a scalar positive-class probability, a tunable
decision threshold), this fits a multinomial logit over K > 2 classes:

    - coefficients are a *matrix* — one column per non-baseline class, each read
      as "log-odds of this class vs. the baseline class";
    - prediction is ``argmax`` over the K class probabilities, so there is **no
      decision threshold** (and hence no ``tune`` / ``workflow`` analogue — those
      stay binary-only by design);
    - evaluation (confusion matrix, classification report, per-class metrics) is
      family-agnostic and already multiclass-capable, so it is reused verbatim
      from ``core`` — call those with the predictions produced here.

Design parallels ``fit.py`` deliberately: a ``*_Result`` dataclass of pure data,
a ``fit_*`` entry point, a tidy coefficient table with odds ratios, and
prediction helpers whose output feeds straight into the ``core`` evaluators.

Label handling
--------------
statsmodels' ``mnlogit`` needs a numeric endog, and a bare string/categorical
target raises a cryptic "multiple columns" error. So we encode the target to
integer codes internally, fit on the codes, and map everything back to the
original labels on the way out — the user-facing API speaks in real class
labels (e.g. "C1".."C8"), never codes. The baseline class (code 0, the first in
sorted order) is the reference level every odds ratio is relative to.

Presentation-free except for the two logistic-specific views (the odds-ratio
coefficient table and the per-class forest plot), which follow the same
``core.tables`` dispatch / ``core.figure.save_figure`` contracts as the binary
module's tables and plots.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import EvalEnvironment
from statsmodels.discrete.discrete_model import MultinomialResultsWrapper

from DSSP2026.eda.encoders import FORMULA_NAMESPACE
from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure
from DSSP2026.core.color_scales import so, att_nominal
from DSSP2026.core.tables import (
    att_table_styles, _CAPTION_STYLE, format_pvalue, save_table_by_extension,
    save_generic_table_png,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MNLogitResult:
    """Returned by fit_mnlogit. Pure data — no figures, no Stylers.

    Attributes
    ----------
    model : MultinomialResultsWrapper
        The fitted statsmodels result (in terms of integer class codes).
    coef_df : pd.DataFrame
        Long-format coefficient table; one block of rows per non-baseline
        class (see make_mnlogit_coef_df).
    classes_ : list
        Original class labels in code order; classes_[k] is the label for
        integer code k. classes_[0] is the baseline/reference class.
    baseline_ : object
        The reference class label (== classes_[0]); every odds ratio is
        "class vs. baseline_".
    target : str
        The original LHS target column name.
    """
    model: MultinomialResultsWrapper
    coef_df: pd.DataFrame
    classes_: list
    baseline_: object
    training_data: Optional[pd.DataFrame] = None
    formula: Optional[str] = None
    target: Optional[str] = None
    # Internal code column injected into the design frame (e.g. "_OnTime_code").
    _code_col: Optional[str] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_mnlogit(
    df: pd.DataFrame,
    formula: str,
    *,
    method: str = "lbfgs",
    maxiter: int = 200,
    disp: bool = False,
) -> MNLogitResult:
    """Fit a multinomial logistic regression and return an MNLogitResult.

    The LHS of *formula* must be a bare column name with K > 2 distinct
    classes (numeric, string, bool, or categorical — all accepted). It is
    encoded to integer codes internally; the first class in sorted order is the
    baseline that the other classes' odds ratios are measured against.

    The RHS keeps the full statsmodels/patsy formula vocabulary, and the
    evaluation namespace is augmented with the same transforms as the binary
    fitter (``log``, ``sqrt``, ``cyc_sin``, …) so formulas are interchangeable.

    Parameters
    ----------
    method : str
        Optimizer for the MLE. Defaults to ``"lbfgs"`` rather than statsmodels'
        ``"newton"``: Newton's method diverges to NaN coefficients on
        multinomial problems with many classes or near-separated features
        (exactly the RCA case), and it fails *silently* — returning NaN params
        and predicting one class for everything. lbfgs/bfgs are robust here.
    maxiter : int
        Max optimizer iterations.
    disp : bool
        If True, statsmodels prints convergence output.

    Raises
    ------
    ValueError
        If the LHS is not a bare column, has fewer than 3 classes (use
        ``logistic.fit.fit_logit`` for the binary case), or if the optimizer
        fails to converge (NaN coefficients — try scaling the features, a
        different ``method``, or more ``maxiter``).
    """
    lhs = formula.split("~")[0].strip()
    if lhs not in df.columns:
        raise ValueError(
            f"fit_mnlogit needs a bare target column on the LHS to encode its "
            f"classes; '{lhs}' is not a column in df. Transformed/categorical "
            f"LHS expressions aren't supported here."
        )

    classes = sorted(pd.unique(df[lhs].dropna()), key=_sort_key)
    if len(classes) < 3:
        raise ValueError(
            f"Multinomial target '{lhs}' has {len(classes)} class(es) "
            f"{classes}; mnlogit needs 3+. For a two-class target use "
            f"logistic.fit.fit_logit instead."
        )

    # Encode to integer codes; code 0 (first sorted class) is the baseline.
    code_of = {c: i for i, c in enumerate(classes)}
    code_col = f"_{lhs}_code"
    work = df.copy()
    work[code_col] = work[lhs].map(code_of)

    # Refit the formula against the code column, keeping the RHS untouched.
    rhs = formula.split("~", 1)[1]
    code_formula = f"{code_col} ~{rhs}"

    formula_env = EvalEnvironment.capture(1).with_outer_namespace(FORMULA_NAMESPACE)
    model = smf.mnlogit(code_formula, data=work, eval_env=formula_env).fit(
        method=method, maxiter=maxiter, disp=disp)

    # Newton (and occasionally others) can diverge to NaN without raising;
    # catch it here so the caller gets an actionable error rather than a model
    # that silently predicts one class for everything.
    if np.isnan(np.asarray(model.params, dtype=float)).any():
        raise ValueError(
            f"mnlogit failed to converge with method={method!r} (NaN "
            f"coefficients). Scale/standardise the predictors, increase "
            f"maxiter, or try method='bfgs'. Standardising continuous features "
            f"is the usual fix for near-separated multinomial problems."
        )

    coef_df = make_mnlogit_coef_df(model, classes=classes)
    return MNLogitResult(
        model=model, coef_df=coef_df, classes_=list(classes),
        baseline_=classes[0], training_data=df, formula=formula,
        target=lhs, _code_col=code_col,
    )


def _sort_key(v):
    """Stable sort key that orders mixed label types deterministically.

    Numerics sort numerically; everything else by string. Keeps the baseline
    (code 0) reproducible across runs regardless of label dtype.
    """
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return (0, float(v), "")
    return (1, 0.0, str(v))


# ---------------------------------------------------------------------------
# Coefficient table (odds ratios), long format with a Class column
# ---------------------------------------------------------------------------

def make_mnlogit_coef_df(
    model: MultinomialResultsWrapper,
    *,
    classes: Sequence,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Tidy multinomial coefficient table on log-odds and odds-ratio scales.

    One block of rows per non-baseline class (each "class vs. baseline"),
    stacked long. Columns mirror the binary ``make_logit_coef_df`` plus a
    leading ``Class`` column:

        Class, Term, Coefficient (log-odds), Std Error, z, p-value,
        Odds Ratio, OR CI {conf}% low, OR CI {conf}% high

    For a non-baseline class, an Odds Ratio > 1 on a term means that increasing
    that term raises the odds of *this* class relative to the baseline class.

    Parameters
    ----------
    classes : sequence
        Original class labels in code order (classes[0] is the baseline). Used
        to label each block with the real class name instead of an integer code.
    """
    conf_level = int(round((1 - alpha) * 100))

    params = model.params           # rows: terms, cols: non-baseline class codes

    # Standard errors, z, p-values and CIs all require the parameter covariance.
    # Under perfect/quasi separation (common with boolean flag predictors) the
    # covariance is singular and statsmodels raises when these are requested —
    # but the point estimates and predictions remain valid. Degrade gracefully:
    # report the coefficients and odds ratios, fill the inference columns with
    # NaN, and flag it, rather than crashing the whole run.
    nan_like = params * np.nan
    try:
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        ci = model.conf_int(alpha=alpha)
        ci.columns = ["ci_low", "ci_high"]
        ci_level0 = ci.index.get_level_values(0).astype(str)
        have_inference = True
    except ValueError:
        import warnings
        warnings.warn(
            "Parameter covariance is unavailable (likely perfect separation in "
            "the predictors); standard errors, z, p-values and CIs are reported "
            "as NaN. Coefficients and odds ratios are still valid, as are "
            "predictions.", RuntimeWarning)
        bse = tvalues = pvalues = nan_like
        ci = None
        ci_level0 = None
        have_inference = False

    low_name = f"OR CI {conf_level}% low"
    high_name = f"OR CI {conf_level}% high"

    blocks = []
    # statsmodels mnlogit drops the baseline (code 0) and numbers the remaining
    # coefficient columns 0..K-2 by POSITION — column j is class code j+1, not
    # code j. The conf_int MultiIndex confirms this: its level-0 keys are the
    # actual non-baseline codes as STRINGS ('1','2',...). So map column position
    # -> code via +1, and slice conf_int by the stringified code.
    for pos, col in enumerate(params.columns):
        code = pos + 1                          # class code (baseline 0 excluded)
        class_label = classes[code] if 0 <= code < len(classes) else code
        terms = params.index

        if have_inference:
            mask = ci_level0 == str(code)
            ci_block = ci[mask].copy()
            ci_block.index = ci_block.index.get_level_values(-1)   # to term level
            or_lo = np.exp(ci_block.loc[terms, "ci_low"].to_numpy())
            or_hi = np.exp(ci_block.loc[terms, "ci_high"].to_numpy())
        else:
            or_lo = np.full(len(terms), np.nan)
            or_hi = np.full(len(terms), np.nan)

        block = pd.DataFrame({
            "Class": str(class_label),
            "Term": terms.astype(str),
            "Coefficient": params[col].to_numpy(),
            "Std Error": bse[col].to_numpy(),
            "z": tvalues[col].to_numpy(),
            "p-value": pvalues[col].to_numpy(),
            "Odds Ratio": np.exp(params[col].to_numpy()),
            low_name: or_lo,
            high_name: or_hi,
        })
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_proba(
    result: MNLogitResult,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """Class-probability matrix for new_data, columns labelled by class.

    Returns a DataFrame with one column per class (original labels, in code
    order) and one row per observation. Feed the argmax (see predict_class) or
    a single class column into the core evaluators.
    """
    proba = np.asarray(result.model.predict(new_data))
    return pd.DataFrame(proba, columns=list(result.classes_),
                        index=new_data.index)


def predict_class(
    result: MNLogitResult,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """Predicted class label (argmax over class probabilities) for each row."""
    proba = np.asarray(result.model.predict(new_data))
    codes = proba.argmax(axis=1)
    classes = np.asarray(result.classes_, dtype=object)
    return classes[codes]


@dataclass
class MNLogitPrediction:
    """Bundle of the probability matrix and argmax labels for new data."""
    proba: pd.DataFrame          # (n, K), columns = class labels
    labels: np.ndarray           # (n,), predicted class labels
    classes_: list


def predict_mnlogit(
    result: MNLogitResult,
    new_data: pd.DataFrame,
) -> MNLogitPrediction:
    """Predict probabilities and argmax labels in one call.

    ``prediction.labels`` and the true labels from ``get_endog`` go straight
    into the core multiclass evaluators::

        pred = predict_mnlogit(res, test)
        y_true = get_endog(res, test)
        cm = core.metrics.make_confusion_matrix(y_true, pred.labels, labels=res.classes_)
        m  = core.metrics.classification_metrics(y_true, pred.labels,
                                                 y_score=pred.proba.to_numpy(),
                                                 average="macro")
    """
    proba = predict_proba(result, new_data)
    classes = np.asarray(result.classes_, dtype=object)
    labels = classes[proba.to_numpy().argmax(axis=1)]
    return MNLogitPrediction(proba=proba, labels=labels,
                             classes_=list(result.classes_))


def get_endog(result: MNLogitResult, data: pd.DataFrame) -> np.ndarray:
    """Return the true class labels for `data` in the original label space.

    Reads the target column named by the fitted result and returns it as an
    array of original labels (not integer codes), so it lines up with
    ``predict_class`` / ``predict_mnlogit`` output for the core evaluators.
    """
    if result.target is None or result.target not in data.columns:
        raise ValueError(
            f"Target column '{result.target}' not found in data; cannot extract "
            f"true labels.")
    return np.asarray(data[result.target].to_numpy(), dtype=object)


# ---------------------------------------------------------------------------
# Coefficient table — notebook view + extension-dispatching saver
# ---------------------------------------------------------------------------

def _mn_coef_formats(coef_df: pd.DataFrame, *, as_callables: bool):
    """Per-column formatters for the multinomial coefficient table.

    Mirrors the binary module's ``_coef_formats``: spec strings for
    ``Styler.format`` (as_callables=False) or bound callables for the PNG
    renderer's ``col_fmts`` (as_callables=True). One source of truth so the
    notebook view and saved file can't drift.
    """
    base = {
        "Coefficient": "{:,.4f}",
        "Std Error":   "{:,.4f}",
        "z":           "{:,.2f}",
        "Odds Ratio":  "{:,.3f}",
    }
    for col in coef_df.columns:
        if col.startswith("OR CI "):
            base[col] = "{:,.3f}"
    fmt = {k: (v.format if as_callables else v) for k, v in base.items()}
    fmt["p-value"] = format_pvalue  # callable in both modes
    return {k: v for k, v in fmt.items() if k in coef_df.columns}


def _mn_caption(result: MNLogitResult, *, short: bool = False) -> str:
    """Caption describing the fit and the reference class."""
    model = result.model
    pseudo_r2 = getattr(model, "prsquared", float("nan"))
    ref = result.baseline_
    if short:
        return (f"Multinomial logistic coefficients — N = {int(model.nobs)}, "
                f"pseudo-R² = {pseudo_r2:.3f}. Baseline class = {ref} "
                f"(OR > 1 raises odds of the row's class vs. baseline).")
    return (f"Multinomial logistic coefficients — N = {int(model.nobs)}, "
            f"pseudo-R² = {pseudo_r2:.3f}, LLR p = {model.llr_pvalue:.3g}. "
            f"Each block is one class vs. the baseline ({ref}); Odds Ratio > 1 "
            f"raises the odds of that class relative to the baseline.")


def style_mnlogit_coefficients(result: MNLogitResult, *, context: str = "report"):
    """Styler for the multinomial coefficient table (odds-ratio scale).

    Significant rows (p < 0.05) are highlighted and the odds-ratio column is
    emphasized, matching the binary table. The ``Class`` column tells the reader
    which class-vs-baseline contrast each block belongs to.
    """
    coef_df = result.coef_df
    or_col = "Odds Ratio"

    def _highlight_sig(val):
        try:
            if val < 0.05:
                return f"background-color: {ATT_COLORS['pale_sky']}; font-weight: bold;"
        except TypeError:
            pass
        return ""

    return (
        coef_df.style
            .set_table_styles(att_table_styles(context=context))
            .hide(axis="index")
            .format(_mn_coef_formats(coef_df, as_callables=False))
            .map(_highlight_sig, subset=["p-value"])
            .set_properties(subset=[or_col],
                            **{"font-weight": "bold", "color": ATT_COLORS["navy"]})
            .set_caption(_mn_caption(result))
            .set_table_styles(_CAPTION_STYLE, overwrite=False)
    )


def save_mnlogit_coefficients(result: MNLogitResult, path: Union[str, Path], *,
                              context: str = "report", dpi: int = 220) -> str:
    """Save the multinomial coefficient table, dispatching on file extension.

    .html -> styled Styler (highlighted significant rows, caption).
    .csv / .xlsx -> raw long-format table.
    .png -> matplotlib table image matching the notebook view.
    """
    return save_table_by_extension(
        result.coef_df, path,
        styler=style_mnlogit_coefficients(result, context=context),
        png_renderer=lambda df, p, **kw: save_mnlogit_coefficients_png(
            result, p, context=context, dpi=dpi),
    )


def save_mnlogit_coefficients_png(result: MNLogitResult, path: Union[str, Path], *,
                                  context: str = "report", dpi: int = 220) -> str:
    """Render the multinomial odds-ratio coefficient table as a PNG."""
    return save_generic_table_png(
        result.coef_df, path,
        title=_mn_caption(result, short=True),
        col_fmts=_mn_coef_formats(result.coef_df, as_callables=True),
        context=context, dpi=dpi)


# ---------------------------------------------------------------------------
# Odds-ratio forest plot, one panel per non-baseline class
# ---------------------------------------------------------------------------

def plot_mnlogit_odds_ratios(
    result: MNLogitResult,
    *,
    drop_intercept: bool = True,
    log_scale: bool = True,
    title: Optional[str] = None,
    sharex: bool = True,
    figsize: Optional[tuple] = None,
):
    """Faceted forest plot: one column of odds ratios per non-baseline class.

    Each subplot is the binary forest plot for one "class vs. baseline"
    contrast — point estimate plus CI, brand-accent if the CI excludes 1, with
    an OR = 1 reference line. Subplots share the OR axis by default so classes
    are visually comparable.

    Built per-axes with the same seaborn-Objects layers as the binary
    ``plot_odds_ratios`` (so.Range + so.Dot through ``.on(ax)``), the OR = 1
    line and log scale added as direct matplotlib.
    """
    coef_df = result.coef_df.copy()
    if drop_intercept:
        coef_df = coef_df[coef_df["Term"] != "Intercept"]

    ci_cols = [c for c in coef_df.columns if c.startswith("OR CI ")]
    if "Odds Ratio" not in coef_df.columns or len(ci_cols) != 2:
        raise ValueError(
            "result.coef_df must have 'Odds Ratio' and two 'OR CI ...' columns "
            "(use make_mnlogit_coef_df).")
    low_col, high_col = ci_cols[0], ci_cols[1]

    # One panel per non-baseline class, in code order.
    classes = [c for c in result.classes_ if str(c) in set(coef_df["Class"])]
    n = len(classes)
    if figsize is None:
        n_terms = coef_df["Term"].nunique()
        figsize = (4.2 * n, max(3.0, 0.55 * n_terms + 1.8))

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=sharex, squeeze=False)
    axes = axes[0]

    for ax, cls in zip(axes, classes):
        sub = coef_df[coef_df["Class"] == str(cls)].iloc[::-1].reset_index(drop=True)
        # Near-separated multinomial fits can produce ORs of 0 / inf and CI
        # bounds that overflow; a log axis can't render those. Clip to a wide
        # but finite positive window purely for display (the table keeps the
        # raw values). The clip window spans 20 orders of magnitude, well
        # beyond any interpretable effect size.
        lo_raw = sub[low_col].to_numpy(dtype=float)
        hi_raw = sub[high_col].to_numpy(dtype=float)
        or_raw = sub["Odds Ratio"].to_numpy(dtype=float)
        clip_lo, clip_hi = 1e-9, 1e9
        plot_df = pd.DataFrame({
            "Term": sub["Term"].astype(str),
            "or": np.clip(np.nan_to_num(or_raw, nan=1.0,
                                        posinf=clip_hi, neginf=clip_lo),
                          clip_lo, clip_hi),
            "lo": np.clip(np.nan_to_num(lo_raw, nan=clip_lo,
                                        posinf=clip_hi, neginf=clip_lo),
                          clip_lo, clip_hi),
            "hi": np.clip(np.nan_to_num(hi_raw, nan=clip_hi,
                                        posinf=clip_hi, neginf=clip_lo),
                          clip_lo, clip_hi),
        })
        # Significance from the RAW bounds (not the clipped display values).
        plot_df["Significance"] = np.where(
            (lo_raw > 1) | (hi_raw < 1), "Significant", "Not significant")
        order = plot_df["Term"].tolist()

        (
            so.Plot(plot_df, y="Term", color="Significance")
              .add(so.Range(linewidth=2.4), xmin="lo", xmax="hi")
              .add(so.Dot(pointsize=9, edgecolor="white", edgewidth=1.0), x="or")
              .scale(y=so.Nominal(order=order),
                     color=so.Nominal(values=[ATT_COLORS["deep_blue"],
                                              ATT_COLORS["gray_500"]],
                                      order=["Significant", "Not significant"]))
              .label(x="Odds ratio", y="", title=f"{cls} vs. {result.baseline_}")
              .on(ax).plot()
        )
        ax.axvline(1.0, color=ATT_COLORS["orange"], linestyle="--", linewidth=1.6,
                   zorder=2)
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlabel("Odds ratio (log scale)")
        ax.margins(y=0.12)

    if title is None:
        title = f"Odds ratios by class (vs. baseline {result.baseline_}), 95% CI"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_mnlogit_odds_ratios(result: MNLogitResult, path, *, drop_intercept=True,
                             log_scale=True, dpi=220, **kwargs) -> str:
    """Render the per-class forest plot and save it (png/pdf/svg…)."""
    fig = plot_mnlogit_odds_ratios(
        result, drop_intercept=drop_intercept, log_scale=log_scale, **kwargs)
    return save_figure(fig, path, dpi=dpi)