"""
core/threshold.py — shared decision-threshold tuning machinery.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from DSSP2026.core.metrics import roc_curve_points


# Metrics that are valid optimization targets. Excludes the count columns
# (tp/fp/tn/fn) and threshold/roc_auc that also live in the sweep frame but
# are not things you'd pick a threshold to maximize.
_ALLOWED_METRICS = (
    "f1",
    "youden",
    "accuracy",
    "precision",
    "recall",
    "false_negative_rate",
    "specificity",
    "false_positive_rate",
)
_MINIMIZE_METRICS = {"false_negative_rate", "false_positive_rate"}


@dataclass
class ThresholdSweepResult:
    """Returned by tune_threshold."""
    sweep_df: pd.DataFrame      # threshold + metric columns at each cutoff
    roc_df: pd.DataFrame
    best_threshold: float
    best_metric: str
    best_value: float
    best_row: pd.Series


@dataclass
class CVThresholdResult:
    """Returned by cross_validate_threshold. Pure data — no figures, no Stylers."""
    summary_df: pd.DataFrame        # threshold + <metric>_mean / <metric>_std cols
    per_fold_df: pd.DataFrame       # long: fold, threshold, metric, value
    roc_per_fold: pd.DataFrame      # fold, fpr, tpr (per-fold ROC points)
    roc_mean: pd.DataFrame          # fpr_grid, tpr_mean, tpr_std (interpolated)
    fold_best: pd.DataFrame         # fold, best_threshold, best_value, roc_auc
    metric: str
    formula: str
    n_splits: int
    # Headline means across folds:
    mean_best_threshold: float
    mean_best_value: float
    mean_auc: float
    std_auc: float


def tune_threshold(
    y_true,
    y_proba,
    *,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
    pos_label: int = 1,
    min_recall: Optional[float] = None,
    max_false_negative_rate: Optional[float] = None,
) -> ThresholdSweepResult:
    """Sweep the decision threshold and pick the best by a chosen criterion.

    Parameters
    ----------
    y_true : array-like
        True 0/1 labels.
    y_proba : array-like
        Predicted probability of the positive class.
    metric : {"f1", "youden", "accuracy", "precision", "recall", "false_negative_rate", "specificity", "false_positive_rate"}
        Criterion to maximize. "youden" maximizes Youden's J
        (sensitivity + specificity − 1), a balanced operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if not 0 < np.sum(y_true == pos_label) < y_true.shape[0]:
        raise ValueError("y_true must contain both positive and negative classes.")
    if metric not in _ALLOWED_METRICS:
        raise ValueError(
            f"metric must be one of {list(_ALLOWED_METRICS)}; got {metric!r}.")

    fpr, tpr, roc_thresholds, auc = roc_curve_points(
        y_true, y_proba, pos_label=pos_label)
    roc_df = pd.DataFrame({
        "threshold": roc_thresholds,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "recall": tpr,
        "false_negative_rate": 1 - tpr,
        "specificity": 1 - fpr,
        "roc_auc": auc,
    })

    if thresholds is None:
        thresholds = roc_thresholds[np.isfinite(roc_thresholds)]
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = np.unique(thresholds[np.isfinite(thresholds)])
    if thresholds.size == 0:
        raise ValueError("No finite thresholds are available to evaluate.")

    pos = y_true == pos_label
    neg = ~pos
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    order = np.argsort(y_proba)
    sorted_proba = y_proba[order]
    sorted_pos = pos[order].astype(int)
    sorted_neg = neg[order].astype(int)
    pos_below = np.searchsorted(sorted_proba, thresholds, side="left")
    tp = n_pos - np.r_[0, np.cumsum(sorted_pos)][pos_below]
    fp = n_neg - np.r_[0, np.cumsum(sorted_neg)][pos_below]
    fn = n_pos - tp
    tn = n_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = (tp + tn) / y_true.shape[0]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                              where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float),
                           where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision, dtype=float),
                       where=(precision + recall) != 0)
        false_negative_rate = np.divide(fn, tp + fn,
                                        out=np.zeros_like(tp, dtype=float),
                                        where=(tp + fn) != 0)
        false_positive_rate = np.divide(fp, tn + fp,
                                        out=np.zeros_like(fp, dtype=float),
                                        where=(tn + fp) != 0)

    specificity = 1 - false_positive_rate
    sweep_df = pd.DataFrame({
        "threshold": thresholds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "youden": recall + specificity - 1,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "tn": tn.astype(int),
        "fn": fn.astype(int),
        "roc_auc": auc,
    })

    candidates = sweep_df
    if min_recall is not None:
        candidates = candidates[candidates["recall"] >= min_recall]
    if max_false_negative_rate is not None:
        candidates = candidates[
            candidates["false_negative_rate"] <= max_false_negative_rate]
    if candidates.empty:
        raise ValueError("No thresholds satisfy the requested recall/FNR constraints.")

    minimize_metrics = _MINIMIZE_METRICS
    if metric in minimize_metrics:
        best_idx = candidates[metric].idxmin()
    else:
        best_idx = candidates[metric].idxmax()
    best_row = sweep_df.loc[best_idx]
    return ThresholdSweepResult(
        sweep_df=sweep_df,
        roc_df=roc_df,
        best_threshold=float(best_row["threshold"]),
        best_metric=metric,
        best_value=float(best_row[metric]),
        best_row=best_row,
    )


def tune_roc_threshold(
    y_true,
    y_proba,
    *,
    metric: str = "f1",
    pos_label: int = 1,
    min_recall: Optional[float] = None,
    max_false_negative_rate: Optional[float] = None,
) -> ThresholdSweepResult:
    return tune_threshold(
        y_true,
        y_proba,
        metric=metric,
        thresholds=None,
        pos_label=pos_label,
        min_recall=min_recall,
        max_false_negative_rate=max_false_negative_rate,
    )

# ---------------------------------------------------------------------------
# Multiclass per-class threshold tuning
# ---------------------------------------------------------------------------

def per_class_thresholds(
    y_true,
    y_proba,
    class_order,
    *,
    metric: str = "f1",
) -> dict:
    """Tune a one-vs-all decision threshold independently for each class.

    For every class ``k`` in ``class_order``, treats ``k`` as the positive
    class and everything else as negative, sweeps the threshold, and records
    the value that maximises ``metric`` (e.g. F1 or Youden's J). Returns a
    ``{class_label: threshold}`` mapping.

    A class that is degenerate in ``y_true`` (all-positive or all-negative,
    so a sweep is undefined) falls back to a 0.5 threshold.

    Parameters
    ----------
    y_true : array-like
        True labels (original label space, any dtype).
    y_proba : ndarray (n_samples, n_classes)
        Class probabilities, columns aligned to ``class_order``.
    class_order : sequence
        Class labels, one per probability column.
    metric : str
        Optimization target passed to :func:`tune_threshold`
        (``"f1"``, ``"youden"``, etc.).
    """
    y_true = np.asarray(y_true, dtype=object)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = [str(c) for c in class_order]
    thresholds = {}
    for k, label in enumerate(classes):
        y_bin = (y_true.astype(str) == label).astype(int)
        # tune_threshold needs both classes present; otherwise default to 0.5.
        if not 0 < int(y_bin.sum()) < y_bin.shape[0]:
            thresholds[label] = 0.5
            continue
        try:
            res = tune_threshold(y_bin, y_proba[:, k], metric=metric, pos_label=1)
            thresholds[label] = float(res.best_threshold)
        except Exception:
            thresholds[label] = 0.5
    return thresholds


def decisions_from_thresholds(
    y_proba,
    class_order,
    thresholds: dict,
) -> np.ndarray:
    """Assign each sample the class maximising ``p(k|x) - t_k``.

    This is the multi-threshold decision rule: each class is offset by its own
    tuned threshold, and the highest threshold-adjusted score wins. It reduces
    to plain argmax when all thresholds are equal, rewards a class for clearing
    its threshold by a wide margin, and always yields exactly one label per
    sample (no empty or ambiguous predictions).

    Parameters
    ----------
    y_proba : ndarray (n_samples, n_classes)
        Class probabilities, columns aligned to ``class_order``.
    class_order : sequence
        Class labels, one per column.
    thresholds : dict
        ``{class_label: threshold}`` (e.g. from :func:`per_class_thresholds`).
        Missing classes default to 0.5.

    Returns
    -------
    ndarray of object
        Decided class label per sample.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    classes = [str(c) for c in class_order]
    t = np.array([float(thresholds.get(c, 0.5)) for c in classes], dtype=float)
    adjusted = y_proba - t[None, :]          # (n, K)
    decided_idx = adjusted.argmax(axis=1)
    return np.asarray(classes, dtype=object)[decided_idx]


def per_class_thresholds_cv(oof_y_true, oof_y_proba, class_order, *,
                            metric: str = "f1") -> dict:
    """Per-class one-vs-all thresholds from pooled out-of-fold predictions.

    Identical tuning logic to :func:`per_class_thresholds`, but intended to be
    called on **cross-validated out-of-fold** probabilities (every train row
    scored by a model that did not see it during fitting) rather than on
    resubstitution or held-out predictions. The caller is responsible for
    producing the OOF arrays — collecting each fold's validation-slice
    probabilities and concatenating them in the original row order is enough,
    since per-class tuning only needs (label, prob-of-class) pairs and is
    order-independent.

    Parameters
    ----------
    oof_y_true : array-like
        True labels for every pooled OOF row (original label space).
    oof_y_proba : ndarray (n_oof, n_classes)
        OOF class probabilities, columns aligned to ``class_order``.
    class_order : sequence
        Class labels, one per probability column.
    metric : str
        Optimization target ("f1", "youden", ...), passed to
        :func:`tune_threshold` via the shared one-vs-all sweep.

    Returns
    -------
    dict
        ``{class_label: threshold}`` — feed straight into
        :func:`decisions_from_thresholds`.
    """
    # The math is the same one-vs-all sweep as per_class_thresholds; reusing it
    # keeps a single definition of "tune a class's threshold".
    return per_class_thresholds(
        oof_y_true, oof_y_proba, class_order, metric=metric)
