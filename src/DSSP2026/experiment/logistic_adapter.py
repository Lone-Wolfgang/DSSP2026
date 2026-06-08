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

Target encoding
---------------
statsmodels' ``logit`` requires a numeric 0/1 endog column. This module handles
that encoding transparently so callers never need to pre-encode:

- **Numeric 0/1**: passed through unchanged.
- **Boolean**: cast to int.
- **String / categorical (two levels)**: sorted lexicographically; the higher
  sort value becomes the positive class (1). The mapping is recorded and all
  outputs (``y_true``, ``y_pred``, ``classes_``) are decoded back to the
  original label strings before returning, so ``report.db`` always stores
  the real label names.

The positive-class convention (higher sort value = 1) is consistent and
predictable. If callers need a specific positive class they should encode
manually before passing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Routing predicate (the one class-count check)
# ---------------------------------------------------------------------------

def is_binary(train: pd.DataFrame, target: str) -> bool:
    """True iff the target has exactly two classes."""
    return int(train[target].nunique()) == 2


# ---------------------------------------------------------------------------
# Target encoding helpers
# ---------------------------------------------------------------------------

def _encode_target(series: pd.Series) -> Tuple[pd.Series, Optional[dict]]:
    """Return (encoded_series, label_map) where encoded_series is numeric 0/1.

    If the series is already numeric 0/1 (or boolean), returns it cast to int
    with ``label_map=None``.

    For string/categorical targets, sorts the two unique values lexicographically
    and maps the lower to 0 and the higher to 1. Returns the encoded series and
    a ``{0: neg_label, 1: pos_label}`` dict for decoding.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int), None

    if pd.api.types.is_numeric_dtype(series):
        uniques = set(pd.unique(series.dropna()))
        if uniques <= {0, 1}:
            return series.astype(int), None
        # Numeric but not 0/1 — fall through to sort-based encoding below
        # (handles e.g. {1, 2} which statsmodels also rejects).

    levels = sorted(series.dropna().unique(), key=str)
    if len(levels) != 2:
        raise ValueError(
            f"Binary logistic target must have exactly 2 unique values; "
            f"found {len(levels)}: {levels}."
        )
    label_map = {0: levels[0], 1: levels[1]}
    encoded = series.map({levels[0]: 0, levels[1]: 1})
    return encoded.astype(int), label_map


def _encode_df(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Return a copy of df with the target encoded 0/1, plus the label map."""
    encoded_col, label_map = _encode_target(df[target])
    if label_map is None:
        return df, None
    df2 = df.copy()
    df2[target] = encoded_col
    return df2, label_map


def _decode(arr: np.ndarray, label_map: Optional[dict]) -> np.ndarray:
    """Decode a 0/1 integer array back to original labels using label_map.

    Returns arr unchanged if label_map is None.
    """
    if label_map is None:
        return arr
    return np.vectorize(label_map.__getitem__)(np.asarray(arr, dtype=int))


# ---------------------------------------------------------------------------
# Uniform refit/eval result — the multiclass contract for both cardinalities
# ---------------------------------------------------------------------------

@dataclass
class LogisticEval:
    """ClassificationResult-like bundle that ``_build_eval_record`` consumes.

    ``y_proba`` is always an ``(n, K)`` DataFrame with columns == ``classes_``,
    so the binary case (K=2) is structurally identical to multiclass.

    ``y_true``, ``y_pred``, and ``classes_`` are always in the *original* label
    space (e.g. ``"FRAUD"`` / ``"NOTFRAUD"``), never the encoded 0/1 integers.
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

    For binary, the target is encoded to 0/1 transparently if needed; the
    resulting F1 is computed in the encoded space (0/1 or original strings),
    which is identical since macro-F1 is label-agnostic.
    """
    from sklearn.metrics import f1_score

    if binary:
        from DSSP2026.logistic_regression.binary import (
            fit_logit, predict_logit, get_endog)

        enc_train, label_map = _encode_df(train_rows, target)
        enc_val, _           = _encode_df(val_rows, target)
        res   = fit_logit(enc_train, formula, maxiter=maxiter)
        pred  = predict_logit(res.model, enc_val)
        y_true_enc = get_endog(res.model, enc_val)
        # Macro-F1 is the same regardless of whether we decode; skip decode.
        return f1_score(y_true_enc, pred.labels, average="macro")
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

    Returns a ``LogisticEval`` whose ``y_proba`` is always ``(n, K)`` and whose
    ``y_true``, ``y_pred``, ``classes_`` are always in the *original* label
    space — so ``report.db`` stores ``"FRAUD"``/``"NOTFRAUD"`` rather than
    ``0``/``1`` regardless of how the target was originally encoded.
    """
    if binary:
        from DSSP2026.logistic_regression.binary import (
            fit_logit, predict_logit, get_endog)

        enc_train, label_map = _encode_df(train, target)
        enc_eval, _          = _encode_df(evaluation, target)

        res            = fit_logit(enc_train, formula, maxiter=maxiter)
        pred           = predict_logit(res.model, enc_eval)
        y_true_enc     = get_endog(res.model, enc_eval)   # 0/1 ints

        # class_order in original label space: [neg_label, pos_label].
        # If no encoding was needed (target was already 0/1) label_map is None
        # and class_order stays [0, 1] — unchanged from the prior behaviour.
        if label_map is not None:
            class_order = [label_map[0], label_map[1]]
            y_true      = _decode(y_true_enc, label_map)
            y_pred      = _decode(pred.labels, label_map)
        else:
            class_order = [0, 1]
            y_true      = np.asarray(y_true_enc)
            y_pred      = np.asarray(pred.labels)

        y_proba = _binary_proba_matrix(pred.proba, class_order, evaluation.index)
        return LogisticEval(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            classes_=class_order,
        )

    from DSSP2026.logistic_regression.multiclass import (
        fit_mnlogit, predict_mnlogit, get_endog)

    res  = fit_mnlogit(train, formula, maxiter=maxiter)
    pred = predict_mnlogit(res, evaluation)
    y_true = get_endog(res, evaluation)
    return LogisticEval(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(pred.labels),
        y_proba=pred.proba,
        classes_=list(pred.proba.columns),
    )
