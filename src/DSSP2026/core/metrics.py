"""
core/metrics.py — model-agnostic scoring utilities.

Shared by every model family's fit/tune layer. Presentation-free.
Regression metrics (MSPE, R²) and classification metrics (accuracy,
precision, recall, F1, ROC-AUC) live here so any family can reuse them.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
    roc_curve as sk_roc_curve,
)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    """Return {'MSPE': ..., 'R²': ...} for a set of predictions."""
    return {
        "MSPE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def make_regression_metrics_df(rows) -> pd.DataFrame:
    """Build a metrics DataFrame from an iterable of metric dicts/rows."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true,
    y_pred,
    *,
    y_score=None,
    average: str = "binary",
    pos_label=1,
) -> dict:
    """Accuracy / precision / recall / F1 (+ ROC-AUC when scores are given).

    Parameters
    ----------
    y_true, y_pred : array-like
        True labels and predicted labels.
    y_score : array-like, optional
        Predicted scores/probabilities for the positive class (binary) or a
        (n_samples, n_classes) matrix (multiclass). Enables ROC-AUC.
    average : {"binary", "macro", "micro", "weighted"}
        Averaging for multi-class precision/recall/F1. Use "binary" for the
        two-class case.
    pos_label : label
        Positive class for the binary case.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average,
                                     pos_label=pos_label, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average,
                               pos_label=pos_label, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average,
                       pos_label=pos_label, zero_division=0),
    }
    if y_score is not None:
        metrics["ROC-AUC"] = _safe_roc_auc(y_true, y_score, average=average,
                                           pos_label=pos_label)
    return metrics


def _safe_roc_auc(y_true, y_score, *, average, pos_label):
    """ROC-AUC that tolerates the degenerate single-class case."""
    try:
        y_score = np.asarray(y_score)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        if y_score.ndim == 1:
            return roc_auc_score(y_true, y_score)
        multi_avg = "macro" if average == "binary" else average
        return roc_auc_score(y_true, y_score, average=multi_avg, multi_class="ovr")
    except (ValueError, IndexError):
        return float("nan")


def make_confusion_matrix(y_true, y_pred, *, labels=None, normalize=None):
    """Confusion matrix as a DataFrame indexed/columned by class label.

    Rows are true classes, columns predicted. `normalize` follows sklearn:
    None (counts), 'true' (row-normalized rates), 'pred', or 'all'.
    """
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=pd.Index(labels, name="True"),
                        columns=pd.Index(labels, name="Predicted"))


def roc_curve_points(y_true, y_score, *, pos_label=1):
    """Return (fpr, tpr, thresholds, auc) for a binary ROC curve."""
    y_score = np.asarray(y_score)
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]
    fpr, tpr, thresholds = sk_roc_curve(y_true, y_score, pos_label=pos_label)
    auc = _safe_roc_auc(y_true, y_score, average="binary", pos_label=pos_label)
    return fpr, tpr, thresholds, auc


def make_classification_report_df(
    y_true,
    y_pred,
    *,
    labels=None,
    target_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-class precision/recall/F1/support as a tidy DataFrame.

    Includes accuracy, macro-avg, and weighted-avg rows at the bottom.
    """
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0)

    rows = []
    for key, vals in report.items():
        if key == "accuracy":
            rows.append({"Class": "accuracy", "Precision": np.nan, "Recall": np.nan,
                         "F1": vals, "Support": report["macro avg"]["support"]})
        else:
            rows.append({"Class": key, "Precision": vals["precision"],
                         "Recall": vals["recall"], "F1": vals["f1-score"],
                         "Support": vals["support"]})
    return pd.DataFrame(rows)