"""
report/cost_math.py — shared cost / net-benefit math.

The single source of truth for the economic scoring used by both the dashboard
and ``Report.cost_optimize``. Everything here is pure (no Streamlit, no DB) and
operates on a *schedule*: a per-class table with three signed columns.

Schedule columns
----------------
- ``TP/FP`` : value applied every time the model *fires* for a class (right or
  wrong). Typically a negative intervention cost.
- ``TP``    : extra benefit collected when a fire is *correct* (a true positive).
- ``FN``    : value applied every time the model *misses* a true case of the
  class. Typically negative.

A fourth, derived quantity — ``TP Net Benefit`` = ``TP/FP + TP - FN`` — is the
per-event swing of a correct catch and is read back into the net-benefit sum.

Net benefit
-----------
Per class ``k``::

    net_k =  TP_k · (TP/FP + TP - FN)_k     # correct catch: full swing
           + FP_k · (TP/FP)_k               # false alarm: fire cost only
           + FN_k · (FN)_k                  # miss: the miss value

The model total is the sum over classes. The "No Action" reference (never fire)
incurs only the miss cost on every class's full support; net benefit is reported
relative to it, so doing nothing scores zero.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# Schedule column names (the three signed inputs + one derived).
TPFP_COL = "TP/FP"
FN_COL = "FN"
TP_COL = "TP"
TPNET_COL = "TP Net Benefit"
SCHEDULE_COLS = (TPFP_COL, TP_COL, FN_COL)


def fmt_currency(value, *, abbreviate: bool = True, decimals: int = 1) -> str:
    """Currency string, sign-aware, no cents.

    ``abbreviate=True`` → compact ``$7.2k`` / ``-$3.4M``; values under 1,000 show
    in full. ``abbreviate=False`` → full dollars with separators. Pure.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    sign = "-" if v < 0 else ""
    n = abs(v)
    if not abbreviate:
        return f"{sign}${n:,.0f}"
    if n < 1000:
        return f"{sign}${n:,.0f}"
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{sign}${n / scale:,.{decimals}f}{suffix}"
    return f"{sign}${n:,.0f}"


def fmt_count(value) -> str:
    """Abbreviate an integer count to <=3 significant digits.

    <1000 shows in full (e.g. 942); 1,000 -> 1.0k; 1,110,000 -> 1.1M;
    billions -> B, trillions -> T. Sign-aware. Used for confusion-matrix cells.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    sign = "-" if v < 0 else ""
    n = abs(v)
    if n < 1000:
        return f"{sign}{int(round(n))}"
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{sign}{n / scale:.1f}{suffix}"
    return f"{sign}{int(round(n))}"


def per_class_counts(cm: pd.DataFrame) -> pd.DataFrame:
    """Per-class TP / FP / FN / fires from a wide (true x pred) confusion matrix."""
    labels = [str(x) for x in cm.index]
    arr = np.asarray(cm.values, dtype=float)
    rows = {}
    for k, label in enumerate(labels):
        tp = arr[k, k]
        fp = arr[:, k].sum() - tp
        fn = arr[k, :].sum() - tp
        rows[label] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "fires": int(tp + fp),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def default_schedule(class_labels: Sequence[str]) -> pd.DataFrame:
    """All-zero editable schedule: one row per class, columns TP/FP, TP, FN."""
    return pd.DataFrame(
        {TPFP_COL: 0.0, TP_COL: 0.0, FN_COL: 0.0},
        index=[str(c) for c in class_labels],
    )


def derive_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Append the derived ``TP Net Benefit`` column = TP/FP + TP - FN (pure)."""
    out = schedule.copy()
    out[TPNET_COL] = out[TPFP_COL] + out[TP_COL] - out[FN_COL]
    return out


def net_benefit_matrix(class_labels: Sequence[str],
                       schedule: pd.DataFrame) -> pd.DataFrame:
    """Long-form (Truth, Action, Net) net-benefit matrix from the schedule.

    Diagonal (correct action for truth i): ``(TP/FP)_i + (TP)_i - (FN)_i``.
    Off-diagonal (action j on truth i): ``(TP/FP)_j + (FN)_i`` — the fire cost of
    the action column plus the miss cost of the truth row. Tidy frame for a
    rect/heatmap. Pure.
    """
    labels = [str(c) for c in class_labels]
    tpfp = schedule[TPFP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    tp = schedule[TP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    fn = schedule[FN_COL].reindex(labels).fillna(0.0).to_numpy(float)

    diag = tpfp + tp - fn
    recs = []
    for i, truth in enumerate(labels):
        for j, action in enumerate(labels):
            net = diag[i] if i == j else (tpfp[j] + fn[i])
            recs.append({"Truth": truth, "Action": action, "Net": float(net)})
    return pd.DataFrame(recs)


def class_contributions(counts: pd.DataFrame,
                        schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-class economic contribution for one model (itemised TP/FP/FN values)."""
    idx = counts.index
    tpfp = schedule[TPFP_COL].reindex(idx).fillna(0.0)
    tp = schedule[TP_COL].reindex(idx).fillna(0.0)
    fn = schedule[FN_COL].reindex(idx).fillna(0.0)
    tp_net = tpfp - fn + tp

    tp_value = counts["tp"] * tp_net
    fp_value = counts["fp"] * tpfp
    fn_value = counts["fn"] * fn
    return pd.DataFrame({
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "investment": (counts["tp"] + counts["fp"]) * tpfp,
        "waste": counts["fp"] * tpfp,
        "loss": counts["fn"] * fn,
        "gains": counts["tp"] * tp,
        "tp_value": tp_value,
        "fp_value": fp_value,
        "fn_value": fn_value,
        "net_benefit": tp_value + fp_value + fn_value,
    })


def net_benefit(counts: pd.DataFrame, schedule: pd.DataFrame) -> float:
    """Total net benefit for one model: sum of per-class contributions."""
    return float(class_contributions(counts, schedule)["net_benefit"].sum())


def expected_value_decisions(class_order: Sequence[str], y_proba: np.ndarray,
                             schedule: pd.DataFrame) -> np.ndarray:
    """Bayes-optimal class decision per sample under the schedule.

    ``EV(j) = (TP/FP)_j + Σ_k p_k·(FN)_k + p_j·((TP)_j - 2·(FN)_j)``,
    decision is ``argmax_j EV(j)``. Vectorised; returns decided class labels.
    """
    classes = [str(c) for c in class_order]
    tpfp = schedule[TPFP_COL].reindex(classes).fillna(0.0).to_numpy(float)
    fn = schedule[FN_COL].reindex(classes).fillna(0.0).to_numpy(float)
    tp = schedule[TP_COL].reindex(classes).fillna(0.0).to_numpy(float)
    P = np.asarray(y_proba, dtype=float)

    miss_all = P @ fn
    ev = (
        tpfp[None, :]
        + miss_all[:, None]
        + P * (tp[None, :] - 2.0 * fn[None, :])
    )
    decided_idx = ev.argmax(axis=1)
    return np.asarray(classes, dtype=object)[decided_idx]


def confusion_from_decisions(y_true: Sequence[str], y_pred: Sequence[str],
                             class_order: Sequence[str]) -> pd.DataFrame:
    """Wide (true x pred) integer confusion matrix from explicit decisions."""
    classes = [str(c) for c in class_order]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti, pi = idx.get(str(t)), idx.get(str(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    out = pd.DataFrame(cm, index=classes, columns=classes)
    out.index.name = "True"
    out.columns.name = "Predicted"
    return out


def optimized_confusion(pred, schedule: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Confusion matrix after Bayes (expected-value) re-decision. None if no preds."""
    if pred is None:
        return None
    class_order, y_true, y_proba = pred
    decisions = expected_value_decisions(class_order, y_proba, schedule)
    return confusion_from_decisions(y_true, decisions, class_order)


def threshold_tuned_confusion(pred, metric: str,
                              score_pred=None) -> Optional[pd.DataFrame]:
    """Confusion after per-class threshold-tuned decisions. None if no preds.

    Thresholds are tuned on ``pred``; when ``score_pred`` is given, those
    thresholds are *applied* to ``score_pred`` (leak-free: tune on OOF, score on
    test). When ``score_pred`` is None the thresholds are applied to ``pred``
    itself (legacy single-set behaviour).
    """
    if pred is None:
        return None
    from DSSP2026.core.threshold import (
        per_class_thresholds, decisions_from_thresholds)
    class_order, y_true, y_proba = pred
    thr = per_class_thresholds(y_true, y_proba, class_order, metric=metric)
    if score_pred is not None:
        class_order, y_true, y_proba = score_pred
    decisions = decisions_from_thresholds(y_proba, class_order, thr)
    return confusion_from_decisions(y_true, decisions, class_order)


def no_action_baseline(cm: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """The 'No Action' reference policy: never fire on anything.

    Zero investment/waste/gains; the only cost is the FN value applied to each
    class's full support. Returns a row dict shaped like ``cost_optimize``'s
    policy-table rows. Pure.
    """
    labels = [str(x) for x in cm.index]
    support = np.asarray(cm.values, dtype=float).sum(axis=1)
    fn = schedule[FN_COL].reindex(labels).fillna(0.0).to_numpy(float)
    loss = float((support * fn).sum())
    return {
        "Model": "No Action",
        "Feature set": "—",
        "Investment": 0.0,
        "Waste": 0.0,
        "Loss": loss,
        "Gains": 0.0,
        "Gross": loss,
        "Net benefit": 0.0,
    }


# ---------------------------------------------------------------------------
# Shared table styling (symmetric zero-centred RdYlGn — matches the dashboard)
# ---------------------------------------------------------------------------

def style_cost_table(frame: pd.DataFrame, *, currency_cols=None,
                     abbreviate: bool = True):
    """Return a pandas Styler with the dashboard's red/yellow/green shading.

    Currency columns are shaded on a symmetric, zero-centred RdYlGn gradient so
    0 is neutral yellow and equal-magnitude gains/losses read equally; values
    are formatted with :func:`fmt_currency`. Non-currency columns (labels,
    integer counts) are left unstyled. If ``currency_cols`` is None, every
    float-dtype column is treated as currency.
    """
    if currency_cols is None:
        currency_cols = [c for c in frame.columns
                         if pd.api.types.is_float_dtype(frame[c])]
    currency_cols = [c for c in currency_cols if c in frame.columns]

    fmt = {c: (lambda v: fmt_currency(v, abbreviate=abbreviate))
           for c in currency_cols}
    sty = (frame.style
           .format(fmt)
           .set_properties(**{"text-align": "right"})
           .hide(axis="index"))

    if currency_cols:
        vals = frame[currency_cols].to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
        if not np.isfinite(max_abs) or max_abs == 0.0:
            max_abs = 1.0
        sty = sty.background_gradient(
            subset=currency_cols, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs)
    return sty


def cost_unit_matrix(class_labels, schedule, *, viewpoint="net_benefit"):
    """Per-cell *unit* value V[i, j] for a cost-aware confusion matrix.

    Rows i = true class, columns j = predicted class.

    - off-diagonal (i != j): ``(TP/FP)_j + (FN)_i`` — fire cost of the predicted
      class plus the miss cost of the true class.
    - diagonal (i == j):
        * ``viewpoint="net_benefit"`` → ``(TP/FP)_i + (TP)_i - (FN)_i``
          (credits the avoided miss — the positive-spin view).
        * ``viewpoint="gross"``       → ``(TP/FP)_i + (TP)_i``
          (honest: you still spent the intervention cost on a correct catch).

    Returns a (K, K) float array aligned to ``class_labels``.
    """
    if viewpoint not in ("net_benefit", "gross"):
        raise ValueError(
            f"viewpoint must be 'net_benefit' or 'gross'; got {viewpoint!r}.")
    labels = [str(c) for c in class_labels]
    tpfp = schedule[TPFP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    tp = schedule[TP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    fn = schedule[FN_COL].reindex(labels).fillna(0.0).to_numpy(float)
    K = len(labels)
    V = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                V[i, j] = tpfp[i] + tp[i] - (fn[i] if viewpoint == "net_benefit" else 0.0)
            else:
                V[i, j] = tpfp[j] + fn[i]
    return V
