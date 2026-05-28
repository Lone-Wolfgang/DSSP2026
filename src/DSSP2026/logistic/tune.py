"""
logistic/tune.py — tuning for logistic regression.

Currently: decision-threshold selection. Given a fitted model and a holdout
set, sweep the probability cutoff and pick the threshold that optimizes a
chosen metric (F1 by default, or Youden's J for a balanced operating point).

Fitting lives in logistic/fit.py; metric computation is delegated to
core/metrics.py so the criteria stay consistent across the codebase.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from modules.core.metrics import roc_curve_points


@dataclass
class ThresholdSweepResult:
    """Returned by tune_threshold."""
    sweep_df: pd.DataFrame      # threshold + metric columns at each cutoff
    roc_df: pd.DataFrame
    best_threshold: float
    best_metric: str
    best_value: float
    best_row: pd.Series


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
    allowed_metrics = [
        "f1",
        "youden",
        "accuracy",
        "precision",
        "recall",
        "false_negative_rate",
        "specificity",
        "false_positive_rate",
    ]
    if metric not in sweep_df.columns:
        raise ValueError(
            f"metric must be one of "
            f"{allowed_metrics}; got {metric!r}.")

    candidates = sweep_df
    if min_recall is not None:
        candidates = candidates[candidates["recall"] >= min_recall]
    if max_false_negative_rate is not None:
        candidates = candidates[
            candidates["false_negative_rate"] <= max_false_negative_rate]
    if candidates.empty:
        raise ValueError("No thresholds satisfy the requested recall/FNR constraints.")

    minimize_metrics = {"false_negative_rate", "false_positive_rate"}
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


def _confusion_counts(y_true, y_pred, *, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == pos_label
    neg = ~pos
    tp = int(np.sum(y_pred[pos] == pos_label))
    fn = int(np.sum(pos) - tp)
    tn = int(np.sum(y_pred[neg] != pos_label))
    fp = int(np.sum(neg) - tn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _youden_j(y_true, y_pred, *, pos_label=1):
    """Youden's J = sensitivity + specificity − 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == pos_label
    neg = ~pos
    tp = np.sum(y_pred[pos] == pos_label)
    fn = np.sum(pos) - tp
    tn = np.sum(y_pred[neg] != pos_label)
    fp = np.sum(neg) - tn
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity + specificity - 1.0