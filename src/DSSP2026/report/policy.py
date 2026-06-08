"""
report/policy.py — metric-driven decision policies (shared, schedule-free).

The cost layer (``cost_api``/``cost_fit``) optimizes a *dollar* objective: it
sweeps (model × policy) by net benefit under a payoff schedule. This module is
its metric-only sibling — the same policy vocabulary applied to a pure
classification objective, with no schedule involved.

A *policy* is just a decision rule mapping stored class probabilities to class
labels:

- **ArgMax**     — highest probability wins (the stored predictions).
- **F1**         — per-class one-vs-all thresholds tuned to maximise F1, then
  the threshold-adjusted argmax decision rule.
- **Youden's J** — same, tuned to maximise Youden's J.

Both ``compare_models`` (for scoring rows) and ``Report.fit`` (for choosing and
deploying a winner) route through here, so the two never drift on what a policy
*means*. Everything here is pure (no DB, no Streamlit): it consumes
``(class_order, y_true, y_proba)`` triples — exactly what ``_read_predictions``
returns — and a metric name.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Display name -> the metric key the threshold sweep understands. ArgMax has no
# tuned metric (it's plain argmax), so it is absent from this map.
POLICIES = ("ArgMax", "F1", "Youden's J")
POLICY_METRIC = {"F1": "f1", "Youden's J": "youden"}


def validate_policy(policy: str) -> str:
    """Return ``policy`` if recognised, else raise with the valid set."""
    if policy not in POLICIES:
        raise ValueError(
            f"unknown policy {policy!r}; valid policies: {list(POLICIES)}.")
    return policy


def policy_thresholds(policy: str, class_order, y_true, y_proba) -> Optional[dict]:
    """Per-class thresholds for ``policy``, or None for ArgMax.

    For F1 / Youden's J this runs the shared one-vs-all sweep
    (``core.threshold.per_class_thresholds``) and returns a ``{class: cutoff}``
    mapping; for ArgMax there is no tuning, so it returns None.
    """
    validate_policy(policy)
    if policy == "ArgMax":
        return None
    from DSSP2026.core.threshold import per_class_thresholds
    metric = POLICY_METRIC[policy]
    return per_class_thresholds(y_true, y_proba, class_order, metric=metric)


def decisions_under_policy(policy: str, class_order, y_proba,
                           thresholds: Optional[dict] = None) -> np.ndarray:
    """Class decisions for one policy from a probability matrix.

    ArgMax → plain argmax over ``y_proba``. F1 / Youden's J → the
    threshold-adjusted argmax (``decisions_from_thresholds``) using
    ``thresholds`` (recomputed here when not supplied — though callers that
    will reuse the cutoffs, e.g. ``fit``, should pass them in).
    """
    validate_policy(policy)
    co = [str(c) for c in class_order]
    y_proba = np.asarray(y_proba, dtype=float)
    if policy == "ArgMax":
        return np.asarray([co[i] for i in y_proba.argmax(axis=1)], dtype=object)
    from DSSP2026.core.threshold import decisions_from_thresholds
    if thresholds is None:
        raise ValueError(
            "thresholds are required for a tuned policy; call "
            "policy_thresholds(...) first or pass them in.")
    return decisions_from_thresholds(y_proba, co, thresholds)


def metrics_from_decisions(y_true, y_pred, class_order) -> dict:
    """Held-out classification metrics from explicit decisions.

    Returns the same metric keys ``compare_models`` reads from report.db
    (``accuracy``, ``precision``, ``recall``, ``f1``), computed with the same
    binary-vs-macro averaging convention used elsewhere in the report layer.
    ROC-AUC is intentionally omitted: it is a property of the *probabilities*,
    not the thresholded decisions, so a policy can't change it — callers carry
    the stored ROC-AUC through unchanged.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    co = [str(c) for c in class_order]
    binary = len(co) == 2
    pos = co[-1] if binary else None
    avg = "binary" if binary else "macro"

    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)

    def _safe(fn, **kw):
        try:
            return float(fn(y_true, y_pred, **kw))
        except Exception:
            return float("nan")

    kw = (dict(average=avg, pos_label=pos, zero_division=0)
          if binary else dict(average=avg, zero_division=0))
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": _safe(precision_score, **kw),
        "recall":    _safe(recall_score, **kw),
        "f1":        _safe(f1_score, **kw),
    }


def score_under_policy(policy: str, class_order, y_true, y_proba) -> dict:
    """Convenience: decide under ``policy`` and return the metric dict.

    Recomputes thresholds internally; for ArgMax this is plain argmax. Used by
    ``compare_models`` to rebuild a per-model row for a chosen policy.
    """
    thr = policy_thresholds(policy, class_order, y_true, y_proba)
    y_pred = decisions_under_policy(policy, class_order, y_proba, thr)
    return metrics_from_decisions(y_true, y_pred, class_order)
