"""
experiment/logistic_adapter.py — binary/multiclass routing for logistic studies.

The experiment layer's eval contract is uniform on the *multiclass* shape: every
study produces a ``ClassificationResult``-like object whose ``y_proba`` is an
``(n, K)`` matrix with columns following ``classes_``. ``sidecar.write_eval``
reads ``proba.shape[1]`` and the report's ROC / cost layers index ``y_proba[:, k]``
per class — so binary logit, whose native ``predict_proba`` is a 1-D positive-class
vector, must be lifted to ``(n, 2)`` before it can flow downstream unchanged.

This module is the *single* place the class-count check lives. It exposes:

  - ``is_binary(train, target)`` — the routing predicate.
  - ``fold_macro_f1(...)``       — per-fold CV score (used by the objective).
  - ``refit_eval(...)``          — refit on TRAIN + predict on the held-out set,
                                   returned as a uniform ``LogisticEval`` that
                                   satisfies the multiclass contract for both
                                   cardinalities.

Nothing else in the experiment layer should import from
``logistic_regression.binary`` / ``.multiclass`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Routing predicate (the one class-count check)
# ---------------------------------------------------------------------------

def is_binary(train: pd.DataFrame, target: str) -> bool:
    """True iff the target has exactly two classes."""
    return int(train[target].nunique()) == 2


# ---------------------------------------------------------------------------
# Uniform refit/eval result — the multiclass contract for both cardinalities
# ---------------------------------------------------------------------------

@dataclass
class LogisticEval:
    """ClassificationResult-like bundle that ``_build_eval_record`` consumes.

    ``y_proba`` is always an ``(n, K)`` DataFrame with columns == ``classes_``,
    so the binary case (K=2) is structurally identical to multiclass.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: pd.DataFrame
    classes_: List


# ---------------------------------------------------------------------------
# Binary -> (n, 2) shape lift
# ---------------------------------------------------------------------------

def _binary_proba_matrix(p_pos, class_order, index) -> pd.DataFrame:
    """Lift a 1-D positive-class probability vector to a 2-column matrix.

    ``class_order`` is ``[neg_label, pos_label]``; the returned columns are
    ``[1 - p, p]`` in that order, matching how the multiclass path lays out
    ``classes_`` so ``label_binarize(y_true, classes=class_order)`` and
    ``y_proba[:, k]`` line up downstream.
    """
    p_pos = np.asarray(p_pos, dtype=float).reshape(-1)
    mat = np.column_stack([1.0 - p_pos, p_pos])
    return pd.DataFrame(mat, columns=list(class_order), index=index)


# ---------------------------------------------------------------------------
# CV scoring (used by objectives.logistic_objective)
# ---------------------------------------------------------------------------

def fold_macro_f1(train_rows, val_rows, *, formula, target, binary, maxiter=100):
    """Fit on a fold's train rows, return macro-F1 on its val rows.

    Routes binary vs multiclass internally; both return labels in the original
    label space so ``f1_score(..., average='macro')`` is computed identically.
    """
    from sklearn.metrics import f1_score

    if binary:
        from DSSP2026.logistic_regression.binary import (
            fit_logit, predict_logit, get_endog)
        res = fit_logit(train_rows, formula, maxiter=maxiter)
        pred = predict_logit(res.model, val_rows)
        y_true = get_endog(res.model, val_rows)
        y_pred = pred.labels
    else:
        from DSSP2026.logistic_regression.multiclass import (
            fit_mnlogit, predict_mnlogit, get_endog)
        res = fit_mnlogit(train_rows, formula, maxiter=maxiter)
        pred = predict_mnlogit(res, val_rows)
        y_true = get_endog(res, val_rows)
        y_pred = pred.labels

    return f1_score(y_true, y_pred, average="macro")


# ---------------------------------------------------------------------------
# Refit + evaluate (used by study._refit_winner)
# ---------------------------------------------------------------------------

def refit_eval(train, evaluation, *, formula, target, binary,
               maxiter=100) -> LogisticEval:
    """Refit the winner on TRAIN, score on the held-out set, normalize shape.

    Returns a ``LogisticEval`` whose ``y_proba`` is always ``(n, K)`` so the
    binary and multiclass branches are indistinguishable to the eval/report
    layers downstream.
    """
    if binary:
        from DSSP2026.logistic_regression.binary import (
            fit_logit, predict_logit, get_endog)

        res = fit_logit(train, formula, maxiter=maxiter)
        pred = predict_logit(res.model, evaluation)        # .proba is 1-D (n,)
        y_true_codes = get_endog(res.model, evaluation)     # 0/1 ints

        # statsmodels' binary endog is the higher level == positive class == 1.
        # class_order = [neg, pos] = [0, 1] in the model's 0/1 encoding.
        class_order = [0, 1]
        y_proba = _binary_proba_matrix(pred.proba, class_order, evaluation.index)
        return LogisticEval(
            y_true=np.asarray(y_true_codes),
            y_pred=np.asarray(pred.labels),
            y_proba=y_proba,
            classes_=class_order,
        )

    from DSSP2026.logistic_regression.multiclass import (
        fit_mnlogit, predict_mnlogit, get_endog)

    res = fit_mnlogit(train, formula, maxiter=maxiter)
    pred = predict_mnlogit(res, evaluation)                 # .proba is (n, K) df
    y_true = get_endog(res, evaluation)
    return LogisticEval(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(pred.labels),
        y_proba=pred.proba,
        classes_=list(pred.proba.columns),
    )
