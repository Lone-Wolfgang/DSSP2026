"""
experiment/cv.py — the single, authoritative home for all cross-validation.

This module centralizes every fold-based computation in DSSP2026. Before this
existed, CV logic was duplicated across tree/, mlp/, xgboost/, logistic/, and
core/ — each re-constructing a ``StratifiedKFold`` and re-implementing fold
loops. Those are now retired; everything routes through here.

Three concerns live here:

1. **Splitter construction** — ``make_splitter`` is the one place a
   ``StratifiedKFold`` / ``KFold`` is built (shuffle + seed policy in one spot).

2. **Hyperparameter-scoring CV** — the objective for any model (whether its
   study uses Optuna's TPE or Grid sampler) calls one of:
     - ``cv_score`` for plain sklearn estimators (decision tree, random forest,
       regression forest): a fold loop equivalent to ``cross_val_score`` that
       returns mean / std / standard-error across folds.
     - ``cv_fold_scores`` for custom per-fold fitting that can't use a plain
       ``estimator.fit`` (the MLP's epoch loop with loss curves; XGBoost): a
       primitive that yields fold indices to a callback and aggregates whatever
       the callback returns.

3. **Decision-threshold CV** — ``cross_validate_threshold_generic`` and the
   per-family wrappers (``cross_validate_threshold`` for logistic,
   ``cross_validate_tree_threshold``, ``cross_validate_rf_threshold``). These
   sweep the positive-class probability cutoff across folds. They reuse the
   model-agnostic threshold *math* (``tune_threshold``, ``roc_curve_points``)
   which remains in ``core`` as a non-fold primitive.

Note on grids: there is deliberately no grid-search function here. RF / regression
grids are expressed as Optuna ``GridSampler`` studies in the experiment layer,
each trial calling ``cv_score`` — so grid search and continuous search share one
substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# Threshold *math* (non-fold) and the result containers stay in core; the fold
# logic that drives them lives here.
from DSSP2026.core.metrics import roc_curve_points
from DSSP2026.core.threshold import (
    _ALLOWED_METRICS,
    CVThresholdResult,
    tune_threshold,
)

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_SCORING = "f1_macro"


# ---------------------------------------------------------------------------
# 1. Splitter construction (single source of truth)
# ---------------------------------------------------------------------------

def make_splitter(*, stratified: bool = True, n_splits: int = DEFAULT_N_SPLITS,
                  random_state: int = DEFAULT_RANDOM_STATE):
    """Build the cross-validation splitter.

    The one place fold splitters are constructed. Classification uses
    ``StratifiedKFold``; regression (or any unstratified need) uses ``KFold``.
    Both shuffle with a fixed seed so folds are reproducible across the codebase.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    cls = StratifiedKFold if stratified else KFold
    return cls(n_splits=n_splits, shuffle=True, random_state=random_state)


# ---------------------------------------------------------------------------
# 2a. Estimator CV — mean / std / SE across folds
# ---------------------------------------------------------------------------

@dataclass
class CVScore:
    """Aggregate of a fold-scored estimator (one hyperparameter candidate)."""
    mean: float
    std: float                 # population std across folds (ddof=0)
    se: float                  # standard error of the mean = std / sqrt(k)
    fold_scores: list          # per-fold scores, in fold order
    n_splits: int
    scoring: str


def cv_score(
    estimator_factory: Callable[[], object],
    X,
    y,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    scoring: str = DEFAULT_SCORING,
    stratified: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CVScore:
    """Cross-validate a plain sklearn estimator and return aggregate scores.

    Equivalent to ``cross_val_score(estimator, X, y, cv=splitter, scoring=...)``
    but built on the shared splitter and returning mean/std/SE in one object so
    the one-standard-error rule (and Optuna objectives) have what they need.

    ``estimator_factory`` is a zero-arg callable returning a fresh, unfitted
    estimator each fold (so no state leaks between folds). ``X`` may be a
    DataFrame or array; ``y`` an array-like of labels.

    The estimator is fit on each fold's training rows and scored on the held-out
    rows; no preprocessing is fit here, so if the estimator needs preprocessing
    it must be a Pipeline (or use ``cv_fold_scores`` for manual control).
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    splitter = make_splitter(stratified=stratified, n_splits=n_splits,
                             random_state=random_state)

    X_df = X if hasattr(X, "iloc") else np.asarray(X)
    y_arr = np.asarray(y)

    def _take(data, idx):
        return data.iloc[idx] if hasattr(data, "iloc") else data[idx]

    fold_scores = []
    for train_i, val_i in splitter.split(X_df, y_arr):
        est = estimator_factory()
        est.fit(_take(X_df, train_i), y_arr[train_i])
        fold_scores.append(float(scorer(est, _take(X_df, val_i), y_arr[val_i])))

    arr = np.asarray(fold_scores, dtype=float)
    std = float(arr.std(ddof=0))
    return CVScore(
        mean=float(arr.mean()),
        std=std,
        se=std / np.sqrt(n_splits),
        fold_scores=fold_scores,
        n_splits=n_splits,
        scoring=scoring,
    )


# ---------------------------------------------------------------------------
# 2b. Manual fold loop — for custom per-fold fitting (MLP curves, XGBoost)
# ---------------------------------------------------------------------------

def cv_fold_scores(
    fold_fn: Callable,
    X,
    y,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    stratified: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Run a custom per-fold computation and collect its outputs.

    The primitive behind any CV whose per-fold step is *not* a plain
    ``estimator.fit`` + scorer — e.g. the MLP's ``partial_fit`` epoch loop that
    records train/eval loss curves, or a bespoke XGBoost fit. It owns the fold
    split (via the shared splitter) and nothing else: for each fold it calls

        fold_fn(X_train, y_train, X_val, y_val) -> result

    and returns the list of per-fold ``result`` objects in fold order. The
    caller decides what a fold result is (a scalar score, or a tuple of
    score + curves) and how to aggregate it. ``X`` may be a DataFrame or array.
    """
    splitter = make_splitter(stratified=stratified, n_splits=n_splits,
                             random_state=random_state)
    X_df = X if hasattr(X, "iloc") else np.asarray(X)
    y_arr = np.asarray(y)

    def _take(data, idx):
        return data.iloc[idx] if hasattr(data, "iloc") else data[idx]

    results = []
    for train_i, val_i in splitter.split(X_df, y_arr):
        results.append(fold_fn(
            _take(X_df, train_i), y_arr[train_i],
            _take(X_df, val_i), y_arr[val_i]))
    return results


def average_curves(curves):
    """Average ragged per-fold curves onto a common length (NaN-padded mean).

    Moved here from ``tuning.search``: fold loss curves can differ in length
    (early stopping), so pad to the max length with NaN and take the column-wise
    nanmean. Returned as a plain list.
    """
    length = max(len(c) for c in curves)
    matrix = np.full((len(curves), length), np.nan)
    for i, curve in enumerate(curves):
        matrix[i, :len(curve)] = curve
    return np.nanmean(matrix, axis=0).tolist()


# ---------------------------------------------------------------------------
# 3. Decision-threshold CV (generic + per-family wrappers)
# ---------------------------------------------------------------------------

def cross_validate_threshold_generic(
    df,
    *,
    fold_proba_fn,
    strat_labels,
    formula,
    metric="f1",
    n_splits=DEFAULT_N_SPLITS,
    thresholds=None,
    pos_label=1,
    min_recall=None,
    max_false_negative_rate=None,
    random_state=0,
) -> CVThresholdResult:
    """Stratified k-fold CV that tunes a binary decision threshold.

    Model-agnostic: each fold's positive-class probabilities are produced by the
    ``fold_proba_fn(train, val) -> (y_val, proba)`` callback, then the shared
    threshold grid is swept via ``core.threshold.tune_threshold``. Fold metric
    curves are averaged by threshold and per-fold ROC curves interpolated onto a
    common FPR grid, returning a ``CVThresholdResult`` for the threshold plots.

    (Moved verbatim from ``core/threshold.py`` so all fold logic lives in cv.py;
    the threshold *math* it calls — ``tune_threshold`` / ``roc_curve_points`` —
    remains in ``core`` as a non-fold primitive.)
    """
    if metric not in _ALLOWED_METRICS:
        raise ValueError(
            f"metric must be one of {list(_ALLOWED_METRICS)}; got {metric!r}.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    thresholds = np.asarray(thresholds, dtype=float)

    splitter = make_splitter(stratified=True, n_splits=n_splits,
                             random_state=random_state)

    sweep_frames = []
    roc_rows = []
    fold_best_rows = []
    fpr_grid = np.linspace(0.0, 1.0, 101)
    interp_tprs = []

    for fold_idx, (train_i, val_i) in enumerate(splitter.split(df, strat_labels)):
        train = df.iloc[train_i]
        val = df.iloc[val_i]
        y_val, proba = fold_proba_fn(train, val)

        try:
            sweep = tune_threshold(
                y_val, proba, metric=metric, thresholds=thresholds,
                pos_label=pos_label, min_recall=min_recall,
                max_false_negative_rate=max_false_negative_rate)
        except ValueError as e:
            raise ValueError(
                f"Fold {fold_idx}: {e} (constraint min_recall={min_recall}, "
                f"max_false_negative_rate={max_false_negative_rate} could not be "
                f"met on the threshold grid).") from e
        sf = sweep.sweep_df.copy()
        sf.insert(0, "fold", fold_idx)
        sweep_frames.append(sf)

        fold_best_rows.append({
            "fold": fold_idx,
            "best_threshold": sweep.best_threshold,
            "best_value": sweep.best_value,
            "roc_auc": float(sweep.sweep_df["roc_auc"].iloc[0]),
        })

        fpr, tpr, _, _ = roc_curve_points(y_val, proba, pos_label=pos_label)
        roc_rows.append(pd.DataFrame({"fold": fold_idx, "fpr": fpr, "tpr": tpr}))
        interp_tprs.append(np.interp(fpr_grid, fpr, tpr))

    per_fold_wide = pd.concat(sweep_frames, ignore_index=True)

    metric_cols = [c for c in _ALLOWED_METRICS if c in per_fold_wide.columns]
    grouped = per_fold_wide.groupby("threshold")[metric_cols]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=0).add_suffix("_std")
    summary_df = pd.concat([means, stds], axis=1).reset_index()
    summary_df = summary_df.sort_values("threshold").reset_index(drop=True)

    per_fold_df = per_fold_wide.melt(
        id_vars=["fold", "threshold"], value_vars=metric_cols,
        var_name="metric", value_name="value")

    roc_per_fold = pd.concat(roc_rows, ignore_index=True)
    interp_tprs = np.vstack(interp_tprs)
    interp_tprs[:, 0] = 0.0
    roc_mean = pd.DataFrame({
        "fpr_grid": fpr_grid,
        "tpr_mean": interp_tprs.mean(axis=0),
        "tpr_std": interp_tprs.std(axis=0, ddof=0),
    })

    fold_best = pd.DataFrame(fold_best_rows)

    return CVThresholdResult(
        summary_df=summary_df,
        per_fold_df=per_fold_df,
        roc_per_fold=roc_per_fold,
        roc_mean=roc_mean,
        fold_best=fold_best,
        metric=metric,
        formula=formula,
        n_splits=n_splits,
        mean_best_threshold=float(fold_best["best_threshold"].mean()),
        mean_best_value=float(fold_best["best_value"].mean()),
        mean_auc=float(fold_best["roc_auc"].mean()),
        std_auc=float(fold_best["roc_auc"].std(ddof=0)),
    )


def cross_validate_threshold(
    df: pd.DataFrame,
    formula: str,
    *,
    metric: str = "f1",
    n_splits: int = DEFAULT_N_SPLITS,
    thresholds: Optional[np.ndarray] = None,
    pos_label: int = 1,
    min_recall: Optional[float] = None,
    max_false_negative_rate: Optional[float] = None,
    random_state: Optional[int] = 0,
    maxiter: int = 100,
) -> CVThresholdResult:
    """Logistic-regression decision-threshold CV (statsmodels per fold).

    Fits ``fit_logit`` on each fold and sweeps the threshold grid. The formula's
    LHS must be a bare 0/1 target column (used for stratification + per-fold
    truth). Moved here from ``logistic_regression/workflow.py``.
    """
    # Imported lazily to avoid a hard cv -> logistic dependency at import time.
    from DSSP2026.logistic_regression.binary import fit_logit, predict_proba
    from DSSP2026.logistic_regression.multiclass import get_endog

    lhs = formula.split("~")[0].strip()
    if lhs not in df.columns:
        raise ValueError(
            f"workflow needs a bare 0/1 target column on the LHS for "
            f"stratification; '{lhs}' is not a column in df.")

    def fold_proba_fn(train, val):
        res = fit_logit(train, formula, maxiter=maxiter)
        proba = predict_proba(res.model, val)
        y_val = get_endog(res.model, val)
        return y_val, proba

    return cross_validate_threshold_generic(
        df,
        fold_proba_fn=fold_proba_fn,
        strat_labels=df[lhs].to_numpy(),
        formula=formula,
        metric=metric,
        n_splits=n_splits,
        thresholds=thresholds,
        pos_label=pos_label,
        min_recall=min_recall,
        max_false_negative_rate=max_false_negative_rate,
        random_state=random_state,
    )


def cross_validate_tree_threshold(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    metric: str = "f1",
    max_depth: Optional[int] = None,
    class_weight="balanced",
    n_splits: int = DEFAULT_N_SPLITS,
    thresholds: Optional[np.ndarray] = None,
    pos_label: int = 1,
    min_recall: Optional[float] = None,
    max_false_negative_rate: Optional[float] = None,
    random_state: Optional[int] = 0,
    estimator_factory=None,
) -> CVThresholdResult:
    """Decision-tree decision-threshold CV. Moved from tree/classification/tune.py.

    Fits a ``DecisionTreeClassifier`` (``class_weight='balanced'`` by default for
    imbalanced data) per fold and sweeps the positive-class threshold. ``df``'s
    ``target`` must be a 0/1 column. ``estimator_factory`` overrides the default
    tree (used by the RF wrapper).
    """
    from sklearn.tree import DecisionTreeClassifier

    if target not in df.columns:
        raise ValueError(f"target {target!r} is not a column in df.")
    if estimator_factory is None:
        def estimator_factory():
            return DecisionTreeClassifier(
                max_depth=max_depth, class_weight=class_weight,
                random_state=random_state)

    features = list(features)

    def fold_proba_fn(train, val):
        clf = estimator_factory()
        clf.fit(train[features], train[target])
        classes = list(clf.classes_)
        if pos_label not in classes:
            raise ValueError(
                f"pos_label={pos_label!r} not among fitted classes {classes}.")
        pos_col = classes.index(pos_label)
        proba = clf.predict_proba(val[features])[:, pos_col]
        y_val = val[target].to_numpy()
        return y_val, proba

    label = f"{target} ~ " + " + ".join(features)

    return cross_validate_threshold_generic(
        df,
        fold_proba_fn=fold_proba_fn,
        strat_labels=df[target].to_numpy(),
        formula=label,
        metric=metric,
        n_splits=n_splits,
        thresholds=thresholds,
        pos_label=pos_label,
        min_recall=min_recall,
        max_false_negative_rate=max_false_negative_rate,
        random_state=random_state,
    )


def cross_validate_rf_threshold(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    metric: str = "f1",
    n_estimators: int = 300,
    max_features="sqrt",
    max_depth: Optional[int] = None,
    class_weight="balanced",
    n_splits: int = DEFAULT_N_SPLITS,
    thresholds: Optional[np.ndarray] = None,
    pos_label: int = 1,
    min_recall: Optional[float] = None,
    max_false_negative_rate: Optional[float] = None,
    random_state: Optional[int] = 0,
) -> CVThresholdResult:
    """Random-forest decision-threshold CV. Moved from tree/classification/tune.py.

    Forest counterpart of :func:`cross_validate_tree_threshold` — a forest's
    finely-graded ``predict_proba`` gives the threshold sweep real resolution.
    """
    from sklearn.ensemble import RandomForestClassifier

    def _factory():
        return RandomForestClassifier(
            n_estimators=n_estimators, max_features=max_features,
            max_depth=max_depth, class_weight=class_weight,
            random_state=random_state, n_jobs=4)

    return cross_validate_tree_threshold(
        df, features, target, metric=metric, n_splits=n_splits,
        thresholds=thresholds, pos_label=pos_label, min_recall=min_recall,
        max_false_negative_rate=max_false_negative_rate,
        random_state=random_state, estimator_factory=_factory)