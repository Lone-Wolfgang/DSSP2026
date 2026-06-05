"""
workflows/dashboard.py — interactive cost / net-benefit comparison dashboard.

A Streamlit app that reads the append-only study database (the same artifact the
CLI writes after every run) and turns each model's stored confusion matrix into a
*net benefit* under a user-defined payoff structure. The point of the app is that
one mechanism covers every framing:

    net_benefit = TP*v_tp + FP*v_fp + FN*v_fn + TN*v_tn

- **Cost-reduction framing.** A missed event is very expensive (``v_fn`` strongly
  negative), each intervention costs a little (``v_tp``, ``v_fp`` mildly negative),
  doing nothing on a true negative is free (``v_tn`` = 0). Every model loses money;
  the best one loses the least by catching events without over-intervening.
- **Revenue framing.** A detected event pays out (``v_tp`` positive), a false
  positive returns nothing or costs the stake (``v_fp`` negative), the rest are 0.
  The best model earns more on true positives than it bleeds on false positives.

The payoffs are sliders, so the ranking re-sorts live as the user changes the
framing — the histogram (top) is invariant to *how* you frame it because it always
plots the same scalar; only the numbers move.

The eight-class TeleLogs problem is collapsed to binary TP/FP/FN/TN by letting the
user choose which class label(s) count as the *positive* (intervene-worthy) set.
Everything not in that set is negative. This keeps the binary payoff vocabulary the
user is reasoning in while honouring the real multiclass model.

Data logic (DB reads, the collapse, the net-benefit sum) lives in plain functions
at the top so it is unit-testable without Streamlit; the ``render`` section below
is presentation only. Launch via::

    python -m DSSP2026.workflows.cli dashboard
    python -m DSSP2026.workflows.cli dashboard --study-db path/to/study.db
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# AT&T brand palette — single source of truth lives in core.style.
try:
    from DSSP2026.core.style import ATT_COLORS
except Exception:  # pragma: no cover - defensive; style import should succeed
    ATT_COLORS = {
        "deep_blue": "#0057B8", "navy": "#002A5C", "att_blue": "#00A8E0",
        "orange": "#E55A0B", "teal": "#00857C", "gold": "#C99000",
        "magenta": "#C8102E", "purple": "#6E3FA3", "green": "#2E7031",
        "gray_900": "#1A1A1A", "gray_700": "#333333", "gray_500": "#666666",
        "gray_300": "#BBBBBB", "gray_100": "#F2F2F2", "white": "#FFFFFF",
    }


# ===========================================================================
# Data layer  (no Streamlit imports here — pure, testable)
# ===========================================================================

# The user-editable schedule: one row per class, two editable values each.
#   tpfp : applied every time the model *fires* for that class (TP + FP)
#   fn   : applied every time the model *fails to fire* (FN)
# A third, derived value is surfaced for the user but never an input:
#   tp_net = tpfp - fn   (the per-event swing of firing vs. missing)
TPFP_COL = "TP/FP"
FN_COL = "FN"
TP_COL = "TP"
TPNET_COL = "TP Net Benefit"
SCHEDULE_COLS = (TPFP_COL, TP_COL, FN_COL)


def fmt_currency(value, *, abbreviate: bool = True, decimals: int = 1) -> str:
    """Currency string in one of two styles, sign-aware, no cents.

    - ``abbreviate=True``  (default): compact ``$7.2k`` / ``-$3.4M`` / ``$2.5T``;
      values under 1,000 show in full (``$950``). ``decimals`` sets the
      abbreviated precision (k/M/B/T); small values are always whole dollars.
    - ``abbreviate=False``: full dollars with thousands separators and no cents
      (``$7,200`` / ``-$3,400,000``).

    Used everywhere a dollar value is shown (tables, metrics, tooltips) so the
    dashboard reads consistently. Pure / testable.
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
        # Whole dollars for small magnitudes (no ".0" noise).
        return f"{sign}${n:,.0f}"
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{sign}${n / scale:,.{decimals}f}{suffix}"
    return f"{sign}${n:,.0f}"


def per_class_counts(cm: pd.DataFrame) -> pd.DataFrame:
    """Per-class fires / misses from a wide (true x pred) confusion matrix.

    For each class label ``k`` (taken from the matrix index/columns, which share
    the C1..C8 order):

    - ``tp``    : correct hits          = cm[k, k]
    - ``fp``    : wrong fires for k      = column_k sum - tp   (truly other, predicted k)
    - ``fn``    : misses of k            = row_k sum - tp      (truly k, predicted other)
    - ``fires`` : everything predicted k = tp + fp             (the events that "fire")

    The economic schedule prices ``fires`` (with the TP/FP rate) and ``fn`` (with
    the FN rate); ``tp``/``fp`` are kept for the itemised policy table.
    Returns a DataFrame indexed by class label.
    """
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
    """An all-zero editable schedule: one row per class, columns TP/FP, TP, FN."""
    return pd.DataFrame(
        {TPFP_COL: 0.0, TP_COL: 0.0, FN_COL: 0.0},
        index=[str(c) for c in class_labels],
    )


def derive_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Append the derived ``TP Net Benefit`` column = TP/FP - FN.

    Pure: does not mutate the input. This is the per-event value of a *correct*
    catch: firing on a true case earns the fire value (TP/FP) and also avoids the
    miss cost (FN), so the swing is ``(TP/FP) - (FN)``. It is read back into the
    net-benefit sum (see :func:`class_contributions`), not merely displayed.
    """
    out = schedule.copy()
    out[TPNET_COL] = out[TPFP_COL] + out[TP_COL] - out[FN_COL]
    return out


def net_benefit_matrix(class_labels: Sequence[str],
                       schedule: pd.DataFrame) -> pd.DataFrame:
    """Long-form (Truth, Action, Net) net-benefit matrix from the schedule.

    The matrix prices each cell by *both* axes. For true class ``i`` and action
    ``j``:

    - **diagonal** (``j == i``) — the single correct action, a true positive,
      worth ``(TP/FP)_i + (TP)_i - (FN)_i``;
    - **off-diagonal** (``j != i``) — taking action ``j`` when the truth is
      ``i``: you pay the fire cost of acting as ``j`` (``(TP/FP)_j``, a property
      of the action *column*) and the miss cost of the true class ``i``
      (``(FN)_i``, a property of the truth *row*), so the cell is
      ``(TP/FP)_j + (FN)_i``.

    ``schedule`` is the signed table (costs already negative, see the sign-fixing
    in :func:`render`), so diagonals are typically positive and off-diagonals
    negative — which is what makes a zero-centred currency cmap read correctly.
    Returns a tidy frame with columns ``Truth``, ``Action``, ``Net`` (one row per
    cell), ready for an Altair ``mark_rect`` heatmap. Pure / testable.
    """
    labels = [str(c) for c in class_labels]
    tpfp = schedule[TPFP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    tp = schedule[TP_COL].reindex(labels).fillna(0.0).to_numpy(float)
    fn = schedule[FN_COL].reindex(labels).fillna(0.0).to_numpy(float)

    diag = tpfp + tp - fn        # correct action per truth: TP/FP + TP - FN
    recs = []
    for i, truth in enumerate(labels):
        for j, action in enumerate(labels):
            # Diagonal: the correct catch for truth i. Off-diagonal: fire cost of
            # the ACTION taken (column j) plus the miss cost of the TRUTH (row i).
            net = diag[i] if i == j else (tpfp[j] + fn[i])
            recs.append({
                "Truth": truth,
                "Action": action,
                "Net": float(net),
            })
    return pd.DataFrame(recs)


def class_contributions(counts: pd.DataFrame,
                        schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-class economic contribution for one model.

    Each confusion outcome is priced by what the model actually did:

        net_benefit_k = TP_k * (TP Net Benefit)_k    # correct catch: full swing
                      + FP_k * (TP/FP)_k             # false alarm: fire cost only
                      + FN_k * (FN)_k                # miss: the miss value

    where ``(TP Net Benefit)_k = (TP/FP)_k - (FN)_k``. A correct catch earns the
    full swing because firing on a true case both collects the fire value and
    avoids the miss cost; a false alarm collects only the fire value; a miss
    incurs the miss value. ``counts`` is the output of :func:`per_class_counts`;
    ``schedule`` is the user's per-class table. Missing classes are treated as
    zero. Returns a DataFrame indexed by class with the itemised TP / FP / FN
    contributions and their sum.
    """
    idx = counts.index
    tpfp = schedule[TPFP_COL].reindex(idx).fillna(0.0)
    tp = schedule[TP_COL].reindex(idx).fillna(0.0)
    fn = schedule[FN_COL].reindex(idx).fillna(0.0)
    tp_net = tpfp - fn + tp                    # the derived TP Net Benefit, per class

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
    """Cost-optimal class decision per sample under the schedule.

    For a sample with class-probability vector ``p`` (one entry per class in
    ``class_order``), deciding class ``j`` yields, per the scoring in
    :func:`class_contributions`:

    - if the true class is ``j`` (prob ``p_j``): a correct catch worth the full
      swing ``(TP/FP)_j - (FN)_j``;
    - if the true class is some ``k != j``: a false alarm for ``j`` *and* a miss
      of ``k``, worth ``(TP/FP)_j + (FN)_k``.

    Taking the expectation over ``p`` and simplifying::

        EV(j) = (TP/FP)_j  +  Σ_k p_k · (FN)_k  -  2 · p_j · (FN)_j

    (The ``p_j·(FN)_j`` term is subtracted twice: once to exclude j from the
    miss sum, once for the FN-avoidance credit a correct catch earns.) The chosen
    class is ``argmax_j EV(j)``. This maximises *expected* net benefit under the
    model's own probabilities, so optimized beats baseline in expectation and
    almost always on the realised labels (a poorly calibrated model can
    occasionally dip slightly below).

    Vectorised over all samples. Returns an array of decided class labels.
    When the schedule is all zeros, every EV(j) is equal, and numpy's argmax
    picks the first column — so callers wanting argmax-equivalence at zero cost
    should use the stored predictions directly rather than this path.
    """
    classes = [str(c) for c in class_order]
    tpfp = schedule[TPFP_COL].reindex(classes).fillna(0.0).to_numpy(float)  # (C,)
    fn = schedule[FN_COL].reindex(classes).fillna(0.0).to_numpy(float)      # (C,)
    P = np.asarray(y_proba, dtype=float)                                    # (N, C)

    # Σ_k p_k·FN_k is common to every candidate j; the j-specific term
    # p_j·FN_j is removed twice (mask out j from the miss sum, then credit the
    # correct-catch FN avoidance). Broadcast tpfp across rows.
    miss_all = P @ fn                       # (N,)  Σ_k p_k·FN_k
    tp = schedule[TP_COL].reindex(classes).fillna(0.0).to_numpy(float)

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
    """Wide (true x pred) integer confusion matrix from explicit decisions.

    Mirrors the shape of ``study_db.confusion_matrix`` so the same
    ``per_class_counts`` / ``class_contributions`` scoring applies to optimized
    decisions exactly as to the stored argmax predictions.
    """
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
    """Confusion matrix after re-deciding every sample to maximise net benefit.

    ``pred`` is the ``(class_order, y_true, y_proba)`` tuple from
    ``study_db.read_predictions`` (or None). Returns None when no predictions
    are stored, so the caller can fall back to the argmax confusion matrix.
    """
    if pred is None:
        return None
    class_order, y_true, y_proba = pred
    decisions = expected_value_decisions(class_order, y_proba, schedule)
    return confusion_from_decisions(y_true, decisions, class_order)


# Decision modes the UI toggles between.
MODE_BASELINE = "Baseline (argmax)"
MODE_OPTIMIZED = "Cost-optimized decisions"
DECISION_MODES = (MODE_BASELINE, MODE_OPTIMIZED)


def confusion_for_mode(result_id: int, mode: str, cm_lookup, pred_lookup,
                       schedule: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Resolve the confusion matrix for one result under the chosen mode.

    Baseline -> the stored argmax confusion matrix (``cm_lookup``).
    Optimized -> rebuilt from cost-optimal decisions (``pred_lookup`` + schedule),
    falling back to the baseline matrix if this result has no stored predictions.
    """
    if mode == MODE_OPTIMIZED:
        opt = optimized_confusion(pred_lookup(result_id), schedule)
        if opt is not None:
            return opt
    return cm_lookup(result_id)


def no_action_baseline(cm: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """The 'No Action' reference policy: never fire on anything.

    If the model never fires, no sample is a TP or FP, so there is **zero
    investment and zero waste** (no firing cost is ever incurred). Every true
    case of every class is therefore missed, so the only cost is the miss (FN)
    cost applied to the *full support* of each class:

        Net = Loss = Σ_class  support_class × (FN)_class

    where ``support_class`` is the number of true samples of that class (the
    confusion-matrix row sum) and ``(FN)_class`` is the signed miss cost from the
    schedule. ``cm`` is any model's (true × pred) confusion matrix for this run —
    class supports are a property of the eval set, identical across models — so
    the baseline is model-independent. Returns a row dict shaped like
    :func:`build_comparison`'s rows. Pure / testable.
    """
    labels = [str(x) for x in cm.index]
    support = np.asarray(cm.values, dtype=float).sum(axis=1)        # row sums
    fn = schedule[FN_COL].reindex(labels).fillna(0.0).to_numpy(float)
    loss = float((support * fn).sum())
    return {
        "Model": "No Action",
        "Feature set": "—",
        "Investment": 0.0,
        "Waste": 0.0,
        "Loss": loss,
        "Gains": 0.0,
        "Gross": loss,              # Investment(0) + Loss + Gains(0)
        "Net benefit": 0.0,         # the reference point: relative to itself
    }


def build_comparison(rows: Iterable[dict], cm_lookup, schedule: pd.DataFrame,
                     *, mode: str = MODE_BASELINE, pred_lookup=None) -> pd.DataFrame:
    """One net-benefit row per model under the schedule and decision mode.

    ``rows`` is an iterable of mappings with at least ``result_id``, ``model``,
    ``feature_set`` (rows from ``study_db.read_results``). ``cm_lookup`` maps a
    ``result_id`` to its stored (argmax) confusion DataFrame. For
    ``mode == MODE_OPTIMIZED`` a ``pred_lookup`` (``result_id -> predictions
    tuple``) must be supplied so decisions can be recomputed. Sorted best-first.
    """
    out = []
    for r in rows:
        rid = int(r["result_id"])
        if mode == MODE_OPTIMIZED and pred_lookup is not None:
            cm = confusion_for_mode(rid, mode, cm_lookup, pred_lookup, schedule)
        else:
            cm = cm_lookup(rid)
        if cm is None or cm.empty:
            continue
        counts = per_class_counts(cm)
        contrib = class_contributions(counts, schedule)
        investment = float(contrib["investment"].sum())
        loss = float(contrib["loss"].sum())
        gains = float(contrib["gains"].sum())
        out.append({
            "Model": str(r["model"]),
            "Feature set": str(r["feature_set"]),
            "Investment": investment,
            "Waste": float(contrib["waste"].sum()),
            "Loss": loss,
            "Gains": gains,
            # Gross = the policy's absolute outcome: firing cost + miss cost + TP
            # benefit. (Waste is a breakout of Investment, so it is NOT added in.)
            "Gross": investment + loss + gains,
            # Net benefit is filled in by the caller as Gross relative to the
            # No-Action baseline's Gross (so No Action itself scores 0).
            "Net benefit": np.nan,
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("Gross", ascending=False).reset_index(drop=True)
    return df



# ===========================================================================
# Presentation layer  (Streamlit)
# ===========================================================================

def render() -> None:  # pragma: no cover - UI, exercised by `streamlit run`
    import os

    import altair as alt
    import streamlit as st

    from DSSP2026.workflows import config as C
    from DSSP2026.workflows import study_adapter as A
    from DSSP2026.workflows.dashboard_style import inject_dashboard_css

    st.set_page_config(page_title="Cost Comparison Dashboard",
                       layout="wide", page_icon="📊")

    # --- brand styling: full AT&T-aligned theme from dashboard_style, whose
    # tokens are sourced from core.style so the chrome matches the figures. ---
    inject_dashboard_css(st)
    # Chart colors still come straight from the palette (used below).
    pos_c, neg_c = ATT_COLORS["green"], ATT_COLORS["magenta"]
    st.title("Interactive Cost Comparison")
    st.markdown('<hr class="att-rule"/>', unsafe_allow_html=True)

    # Shade currency columns on a symmetric, zero-centered RdYlGn scale so 0 is
    # always neutral yellow and equal-magnitude gains/losses read equally.
    def shade_currency(styler, columns, frame):
        cols = [c for c in columns if c in frame.columns]
        if not cols:
            return styler
        max_abs = float(np.nanmax(np.abs(frame[cols].to_numpy(dtype=float))))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            max_abs = 1.0  # avoid a degenerate vmin == vmax range
        return styler.background_gradient(
            subset=cols, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs)

        # Currency columns are the float-valued ones (counts are ints, labels are
    # strings). Detect them per-table so new currency columns are formatted and
    # shaded automatically, with no hand-maintained lists to fall out of sync.
    def currency_cols(frame):
        return [c for c in frame.columns
                if pd.api.types.is_float_dtype(frame[c])]

    def style_currency(frame, fmt=None):
        cols = currency_cols(frame)
        # Default: the shared abbreviated currency formatter ($1.2k / $3.4M).
        # A custom ``fmt`` (str template or callable) still overrides per call.
        if fmt is None:
            formatter = {c: fmt_currency for c in cols}
        elif callable(fmt):
            formatter = {c: fmt for c in cols}
        else:
            formatter = {c: fmt for c in cols}
        sty = (frame.style
                .format(formatter)
                .set_properties(**{"text-align": "right"}))
        return shade_currency(sty, cols, frame)

    def render_cost_matrix(class_labels, schedule, id2label=None):
        """Color-coded net-benefit matrix: truth (rows) x action (columns).

        Each square is the net value of taking the column's action when the
        truth is the row's class (see :func:`net_benefit_matrix`), shaded on the
        same symmetric, zero-centred RdYlGn currency scale the tables use so
        green = good action / red = bad action / yellow = neutral. Hovering a
        square shows the truth, the action, and the net value.

        The matrix is built on the raw (stored) class labels so the schedule
        lookup is unchanged; ``id2label`` (optional ``{stored -> friendly}``) is
        applied only to the rendered axis/tooltip labels, never the economics.
        """
        mat = net_benefit_matrix(class_labels, schedule)
        max_abs = float(np.nanmax(np.abs(mat["Net"].to_numpy(float))))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            max_abs = 1.0  # avoid a degenerate vmin == vmax range
        # Display-only relabel: remap Truth/Action text and the sort order to the
        # friendly names (identity when no map). Net values are untouched.
        disp = {str(k): str(v) for k, v in (id2label or {}).items()}
        if disp:
            mat = mat.assign(
                Truth=mat["Truth"].map(lambda s: disp.get(str(s), s)),
                Action=mat["Action"].map(lambda s: disp.get(str(s), s)))
        order = [disp.get(str(c), str(c)) for c in class_labels]
        # Abbreviated currency for the tooltip (Altair tooltips take d3 format
        # strings, not Python callables, so precompute the display string).
        mat = mat.assign(NetLabel=mat["Net"].map(money))
        heat = (
            alt.Chart(mat)
            .mark_rect(stroke=ATT_COLORS["white"], strokeWidth=1)
            .encode(
                x=alt.X("Action:N", sort=order, title="Action (predicted)",
                        axis=alt.Axis(orient="top", labelAngle=0)),
                y=alt.Y("Truth:N", sort=order, title="Truth (actual)"),
                color=alt.Color(
                    "Net:Q",
                    scale=alt.Scale(scheme="redyellowgreen",
                                    domain=[-max_abs, max_abs]),
                    legend=alt.Legend(title="Net")),
                tooltip=[
                    alt.Tooltip("Truth:N", title="TRUTH"),
                    alt.Tooltip("Action:N", title="ACTION"),
                    alt.Tooltip("NetLabel:N", title="NET"),
                ],
            )
            .properties(height=max(160, 34 * len(order)))
        )
        return heat

    # --- locate the study DB ----------------------------------------------
    default_db = os.getenv("DSSP_DASHBOARD_DB") or str(C.STUDY_DB)
    db_path = st.sidebar.text_input("Study database", value=str(default_db))
    if not Path(db_path).exists():
        st.warning(f"No study database found at `{db_path}`.\n\n"
                   "Run the workflow first (`python -m DSSP2026.workflows.cli "
                   "run`) or point the sidebar at an existing study database.")
        st.stop()

    # The adapter normalises both the study (runs/results) and experiment
    # (experiments/models) schemas to a single run_id / result_id vocabulary.
    try:
        schema = A.detect_schema(db_path)
    except ValueError as e:
        st.error(f"Unrecognised database schema: {e}")
        st.stop()

    # --- pick a run --------------------------------------------------------
    runs = A.list_runs(db_path)
    if runs.empty:
        st.warning("The study database has no recorded runs yet.")
        st.stop()

    run_labels = {
        int(r.run_id): f"Run #{int(r.run_id)} — {r.timestamp} "
                       f"({r.eval_kind}, {int(r.n_results)} results)"
        for r in runs.itertuples()
    }
    run_id = st.sidebar.selectbox(
        "Run", list(run_labels), format_func=lambda x: run_labels[x])
    best_only = st.sidebar.checkbox(
        "Best feature set per model only", value=True,
        help="Compare one row per model family. (No effect on databases that "
             "already store one row per model.)")

    result_rows = A.list_results(db_path, run_id, best_only=best_only)
    if result_rows.empty:
        st.warning("No results for this run with the current filter.")
        st.stop()

    # --- discover the class label space from a real confusion matrix ------
    first_cm = A.confusion_matrix(db_path, int(result_rows.iloc[0].result_id))
    class_labels = [str(x) for x in first_cm.index] if not first_cm.empty \
        else list(C.CLASS_LABELS)
    # Display-only friendly names registered on the experiment (empty -> identity).
    # class_labels stays raw: all schedule/decision math keys on the stored labels.
    label_display = A.id2label(db_path, run_id)

    # --- currency display style (shorthand $7.2k vs full $7,200), no cents ---
    _money_style = st.sidebar.radio(
        "Currency display", ["Shorthand ($7.2k)", "Full ($7,200)"],
        index=0, horizontal=True,
        help="How dollar values are shown across every table, metric, and "
             "tooltip. Shorthand abbreviates large numbers (k / M / B); Full "
             "shows the whole figure with separators. No cents either way.")
    _abbrev = _money_style.startswith("Shorthand")

    def money(value):
        """The active currency formatter, bound to the sidebar toggle."""
        return fmt_currency(value, abbreviate=_abbrev)

    # --- per-class economic schedule (editable table) --------------------
    st.sidebar.markdown("### Cost schedule")
    # st.sidebar.caption(
    #     "One row per class. **TP/FP**: value each time the model fires for that "
    #     "class (right or wrong). **FN**: value each time it misses that class. "
    #     "Enter positive magnitudes only. Costs are signed automatically.**TP Net Benefit** = TP/FP − FN is "
    #     "derived. Defaults are 0 (no economics → flat comparison).")

    # default = default_schedule(class_labels)
    # # Editor holds ONLY the two real inputs — no derived column inside it.
    # st.sidebar.markdown("### Cost schedule")
    # st.sidebar.caption(
    #     "Enter positive magnitudes only. Intervention costs (TP/FP) and missed-event "
    #     "costs (FN) are automatically treated as negative values. True-positive "
    #     "benefits (TP) remain positive.")

    default = default_schedule(class_labels)

    # Display-only: show friendly class names as the editor's row index. The
    # economics still key on the raw labels, so we restore the raw index on the
    # edited frame below (row order is preserved by data_editor, so a positional
    # restore is exact). Identity when no map is registered.
    default_display = default[[TPFP_COL, TP_COL, FN_COL]].copy()
    if label_display:
        default_display.index = [label_display.get(str(c), c)
                                 for c in default_display.index]

    edited = st.sidebar.data_editor(
        default_display,
        key="schedule_editor",
        use_container_width=True,
        column_config={
        TPFP_COL: st.column_config.NumberColumn(
            TPFP_COL,
            step=1.0,
            min_value=0.0,
            help="""
**The COST OF DOING BUSINESS**

- Represents the cost of taking action
- Applied every time the model fires for this class
- Includes both true and false positives
- Applied automatically as a negative value
"""
            ),
            TP_COL: st.column_config.NumberColumn(
                TP_COL,
                step=1.0,
                min_value=0.0,
                help="""
**TRUE POSITIVE BENEFIT**

- Applied when the model predicts this class correctly
- Earned only for true positives
- Applied as a positive value
    """,
            ),
            FN_COL: st.column_config.NumberColumn(
                FN_COL,
                step=1.0,
                min_value=0.0,
                help="""
**COST OF A MISSED EVENT**

- Applied when the model misses this class
- Incurred only for false negatives
- Applied automatically as a negative value
    """,
            ),
        },
    )

    # Convert user-entered magnitudes into the signed schedule used internally.
    schedule = edited[[TPFP_COL, TP_COL, FN_COL]].astype(float)
    # Restore the raw class-label index (the editor showed friendly names; row
    # order is preserved, so reassign positionally) so every downstream
    # reindex(class_labels) lookup keys on the stored labels.
    schedule.index = list(class_labels)

    schedule[TPFP_COL] = -schedule[TPFP_COL].abs()  # always a cost
    schedule[TP_COL] = schedule[TP_COL].abs()       # always a benefit
    schedule[FN_COL] = -schedule[FN_COL].abs()      # always a cost

    st.sidebar.markdown("### Net benefit matrix")
    # st.sidebar.caption(
    #     "Net value of each **action** (column) given the true class (row): the "
    #     "diagonal is a correct catch (TP/FP + TP); off-diagonal cells are false "
    #     "positives (TP/FP + FN). Hover a square for the exact value.")
    st.sidebar.altair_chart(
        render_cost_matrix(class_labels, schedule, id2label=label_display),
        use_container_width=True,
    )



    # --- compute (cached confusion lookup) --------------------------------
    @st.cache_data(show_spinner=False)
    def _cm(_db, rid):
        return A.confusion_matrix(_db, rid)

    @st.cache_data(show_spinner=False)
    def _pred(_db, rid):
        return A.read_predictions(_db, rid)

    cm_lookup = lambda rid: _cm(db_path, rid)
    pred_lookup = lambda rid: _pred(db_path, rid)

    # --- decision mode toggle (baseline argmax vs cost-optimized) ---------
    have_preds = A.has_predictions(db_path)
    if have_preds:
        mode = st.radio(
            "Decision rule", DECISION_MODES, horizontal=True,
            help="Baseline uses each model's stored argmax predictions. "
                 "Cost-optimized re-decides every sample to maximise expected "
                 "net benefit under the schedule: it picks the class j with the "
                 "highest EV(j) = TP/FP_j + Σ p_k·FN_k·[k≠j].")
    else:
        mode = MODE_BASELINE
        st.info("This study has no stored probability matrices, so only the "
                "baseline (argmax) decisions are available. Re-run the workflow "
                "with prediction logging to enable cost-optimized decisions.")

    table = build_comparison(
        result_rows.to_dict("records"), cm_lookup, schedule,
        mode=mode, pred_lookup=pred_lookup)
    if table.empty:
        st.warning("No confusion matrices stored for these results.")
        st.stop()

    # 'No Action' reference policy (never fire): zero investment/waste/gains, so
    # its Gross is the full miss cost. Net benefit is defined RELATIVE to this
    # baseline's Gross — Net = Gross − Gross(No Action) — so No Action scores 0
    # and each model's Net is how much better (or worse) than doing nothing it
    # does. Class supports come from any model's confusion matrix (identical
    # across models for this eval set), so the baseline is model-independent.
    _ref_cm = confusion_for_mode(
        int(result_rows.iloc[0]["result_id"]), mode, cm_lookup, pred_lookup,
        schedule)
    no_action = (no_action_baseline(_ref_cm, schedule)
                 if _ref_cm is not None and not _ref_cm.empty else None)
    gross_no_action = float(no_action["Gross"]) if no_action is not None else 0.0

    # Fill each model's Net benefit relative to the No-Action gross, then re-sort
    # best-first by it (equivalently by Gross, since they differ by a constant).
    table["Net benefit"] = table["Gross"] - gross_no_action
    table = table.sort_values("Net benefit", ascending=False).reset_index(drop=True)

    # Display table: No Action pinned to the TOP row as the reference, models below.
    table_display = (
        pd.concat([pd.DataFrame([no_action]), table], ignore_index=True)
        if no_action is not None else table)

    # ===================================================================
    # TOP PANEL — net benefit by model
    # ===================================================================
    st.subheader(f"Net benefit by model — {mode}")
    all_zero = bool(
        (schedule[[TPFP_COL, TP_COL, FN_COL]].to_numpy() == 0).all()
    )
    if all_zero:
        st.info("The cost schedule is all zeros, so every model scores 0. "
                "Edit the table in the sidebar to assign per-class costs and "
                "benefits — the ranking updates live.")

    plot_df = table_display.assign(
        Label=table_display["Model"] + "  (" + table_display["Feature set"] + ")",
        Sign=np.where(table_display["Model"] == "No Action", "No Action",
                      np.where(table_display["Net benefit"] >= 0, "Gain", "Loss")))
    # Abbreviated currency strings for the tooltip (numeric columns still drive
    # the bar lengths / sort; these are display-only companions).
    for col in ("Net benefit", "Investment", "Waste", "Loss"):
        if col in plot_df.columns:
            plot_df[f"{col} $"] = plot_df[col].map(money)
    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Net benefit:Q", title="Net benefit (model currency)"),
            y=alt.Y("Label:N", sort="-x", title=None),
            color=alt.Color(
                "Sign:N",
                scale=alt.Scale(
                    domain=["Gain", "Loss", "No Action"],
                    range=[pos_c, neg_c, ATT_COLORS["gray_500"]]),
                legend=alt.Legend(title=None)),
            tooltip=["Model", "Feature set",
                     alt.Tooltip("Net benefit $:N", title="Net benefit"),
                     alt.Tooltip("Investment $:N", title="Investment"),
                     alt.Tooltip("Waste $:N", title="Waste"),
                     alt.Tooltip("Loss $:N", title="Loss")],
        )
        .properties(height=max(120, 46 * len(plot_df)))
    )
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(
        color=ATT_COLORS["gray_500"], strokeDash=[4, 3]).encode(x="z:Q")
    st.altair_chart(bars + zero, use_container_width=True)

    best = table.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", f"{best['Model']}",
              help=f"feature set: {best['Feature set']}")
    c2.metric("Best net benefit", money(best['Net benefit']))
    spread = table["Net benefit"].max() - table["Net benefit"].min()
    c3.metric("Spread (best − worst)", money(spread))

    # ===================================================================
    # BOTTOM PANEL — how net benefit is derived
    # ===================================================================
    st.subheader("Policy table")
    st.caption(
        "Per model: net benefit = Σ over classes of "
        "TP·(TP Net Benefit) + FP·(TP/FP) + FN·(FN), using the schedule "
        "on the left.")

    st.dataframe(style_currency(table_display, fmt=money),
                 use_container_width=True, hide_index=True)

    # Per-class breakdown for a chosen model — shows where the benefit comes from.
    st.subheader("Per-class breakdown")
    st.caption(
        "For the selected model: per class, the confusion COUNTS (TP / FP / FN), "
        "each priced into a currency VALUE, and their sum as Net. The Total row "
        "ties back to the model's net benefit above. Hover a column header for "
        "its definition.")
    pick = st.selectbox(
        "Model", table["Model"].tolist(),
        help="See how each class contributes to the chosen model's net benefit.")
    rid = next(int(r["result_id"]) for r in result_rows.to_dict("records")
               if str(r["model"]) == pick)
    counts = per_class_counts(
        confusion_for_mode(rid, mode, cm_lookup, pred_lookup, schedule))
    contrib = class_contributions(counts, schedule)

    # Seven-column breakdown: the three confusion COUNTS, their three CURRENCY
    # values, and the per-class Net (= the three values summed). Net ties back to
    # this model's total in the top panel when summed over classes.
    detail = pd.DataFrame({
        "Class":     [label_display.get(str(c), c) for c in counts.index],
        "TP":        contrib["tp"].astype(int).values,
        "FP":        contrib["fp"].astype(int).values,
        "FN":        contrib["fn"].astype(int).values,
        "TP Value":  contrib["tp_value"].values,
        "FP Value":  contrib["fp_value"].values,
        "FN Value":  contrib["fn_value"].values,
        "Net":       contrib["net_benefit"].values,
    })
    # Append a Total row (counts sum; values sum; Net sums) so the table closes
    # to the model's overall net benefit.
    totals = {
        "Class": "Total",
        "TP": int(detail["TP"].sum()),
        "FP": int(detail["FP"].sum()),
        "FN": int(detail["FN"].sum()),
        "TP Value": float(detail["TP Value"].sum()),
        "FP Value": float(detail["FP Value"].sum()),
        "FN Value": float(detail["FN Value"].sum()),
        "Net": float(detail["Net"].sum()),
    }
    detail = pd.concat([detail, pd.DataFrame([totals])], ignore_index=True)

    # The value columns are formatted to STRINGS in the dataframe itself (not via
    # a Styler number-format), because st.dataframe's column_config takes
    # precedence over a Styler's pandas format and would otherwise re-render the
    # raw float. With the cells already strings and typed as TextColumn, the
    # chosen currency style (shorthand/full) always shows. Coloring is computed
    # from the NUMERIC values on a shared symmetric RdYlGn scale and applied to
    # the string cells; the Total row is excluded from the scale so one big
    # number can't wash out the per-class hues.
    import matplotlib as _mpl

    value_cols = ["TP Value", "FP Value", "FN Value", "Net"]
    count_cols = ["TP", "FP", "FN"]
    num = detail                              # numeric source (incl. Total row)
    per_class_rows = num.iloc[:-1]
    max_abs = float(np.nanmax(np.abs(
        per_class_rows[value_cols].to_numpy(dtype=float))))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0

    # Display frame: currency strings for values, grouped integers for counts.
    disp = num.copy()
    for c in value_cols:
        disp[c] = num[c].map(money)
    for c in count_cols:
        disp[c] = num[c].map(lambda x: f"{int(x):,}")

    _cmap = _mpl.colormaps["RdYlGn"]
    _norm = _mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)

    def _value_colors(_frame):
        css = pd.DataFrame("", index=num.index, columns=num.columns)
        for c in value_cols:
            for i in num.index:
                r, g, b, _ = _cmap(_norm(float(num.at[i, c])))
                css.at[i, c] = (f"background-color: "
                                f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)")
        return css

    detail_sty = (
        disp.style
        .apply(_value_colors, axis=None)
        .set_properties(**{"text-align": "right"})
        .set_properties(subset=["Class"], **{"text-align": "left"})
        # Bold the Total row (last row).
        .apply(lambda row: ["font-weight: 700"] * len(row)
               if row["Class"] == "Total" else [""] * len(row), axis=1)
    )

    st.dataframe(
        detail_sty, use_container_width=True, hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn(
                "Class",
                help="The true class. One row per class, plus a Total row that "
                     "sums to this model's overall net benefit."),
            "TP": st.column_config.TextColumn(
                "TP",
                help="True positives — samples of this class the model correctly "
                     "fired on (the confusion-matrix diagonal for this class)."),
            "FP": st.column_config.TextColumn(
                "FP",
                help="False positives — samples of OTHER classes the model "
                     "wrongly assigned to this class (false alarms)."),
            "FN": st.column_config.TextColumn(
                "FN",
                help="False negatives — samples of this class the model missed "
                     "(assigned to some other class)."),
            "TP Value": st.column_config.TextColumn(
                "TP Value",
                help="Currency value of the true positives: TP × (TP/FP + TP − "
                     "FN) — a correct catch collects the fire value and the "
                     "benefit, and avoids the miss cost."),
            "FP Value": st.column_config.TextColumn(
                "FP Value",
                help="Currency value of the false positives: FP × (TP/FP) — each "
                     "false alarm incurs only the fire cost."),
            "FN Value": st.column_config.TextColumn(
                "FN Value",
                help="Currency value of the false negatives: FN × (FN) — each "
                     "miss incurs the miss cost."),
            "Net": st.column_config.TextColumn(
                "Net",
                help="Net benefit for the class: TP Value + FP Value + FN Value. "
                     "Positive = the class earns more than it costs."),
        },
    )

    with st.expander("How the schedule and decision rule work"):
        st.markdown(

            "Your schedule sets three values per class: **TP/FP** (cost of intervening), "
            "**TP** (benefit of a correct intervention), and **FN** (cost of missing a true "
            "case). The sidebar accepts positive magnitudes only; TP/FP and FN are "
            "automatically applied as negative values internally."
            "Each model's confusion matrix gives, per class **k**, its true "
            "positives (**TP**, correct catches), false positives (**FP**, false "
            "alarms) and false negatives (**FN**, misses). Your schedule sets two "
            "values per class: **TP/FP**, earned whenever the model fires for k, "
            "and **FN**, earned whenever it misses k. The derived "
            "**TP Net Benefit** = TP/FP − FN is the per-event value of a *correct* "
            "catch: firing on a true case both collects the fire value and avoids "
            "the miss cost.\n\n"
            "Net benefit sums, across classes, "
            "**TP·(TP Net Benefit) + FP·(TP/FP) + FN·(FN)** — a correct catch "
            "earns the full swing, a false alarm earns only the fire value, and a "
            "miss incurs the miss value. It is naturally positive for a model that "
            "catches more than it misses, and goes negative when errors dominate.\n\n"
            "**Baseline** scores each model's stored argmax predictions as-is. "
            "**Cost-optimized** re-decides every sample to maximise its expected "
            "net benefit: for class probabilities **p**, deciding *j* is worth "
            "EV(*j*) = (TP/FP)ⱼ + Σₖ pₖ·(FN)ₖ − 2·pⱼ·(FN)ⱼ, and the decision is "
            "argmaxⱼ EV(*j*). This maximises *expected* net benefit under the "
            "model's own probabilities, so optimized beats baseline in "
            "expectation (and almost always on the realised labels); a "
            "well-calibrated model with an uneven schedule shows the largest "
            "gains. Where a model is poorly calibrated the realised optimized "
            "score can occasionally dip slightly below baseline.")


if __name__ == "__main__":  # pragma: no cover
    render()