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
    python -m DSSP2026.workflows.cli dashboard --report-db path/to/report.db
"""

from __future__ import annotations

import json
import sqlite3
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
# Data layer  (shared cost math now lives in report.cost.math)
# ===========================================================================

from DSSP2026.report.cost.math import (
    TPFP_COL, FN_COL, TP_COL, TPNET_COL, SCHEDULE_COLS,
    fmt_currency, per_class_counts, default_schedule, derive_schedule,
    net_benefit_matrix, class_contributions, net_benefit,
    expected_value_decisions, confusion_from_decisions,
    optimized_confusion, threshold_tuned_confusion, no_action_baseline,
    cost_unit_matrix, fmt_count,
)


# Decision modes the UI toggles between.
MODE_BASELINE = "Baseline (argmax)"
MODE_OPTIMIZED = "Cost-optimized decisions"
MODE_F1 = "F1-tuned thresholds"
MODE_YOUDEN = "Youden-tuned thresholds"
DECISION_MODES = (MODE_BASELINE, MODE_OPTIMIZED)

# Map a tuned mode to the metric name passed to the threshold tuner.
_MODE_METRIC = {MODE_F1: "f1", MODE_YOUDEN: "youden"}


def confusion_for_mode(result_id: int, mode: str, cm_lookup, pred_lookup,
                       schedule: pd.DataFrame,
                       tuning_lookup=None) -> Optional[pd.DataFrame]:
    """Resolve the confusion matrix for one result under the chosen mode.

    Baseline   -> the stored argmax confusion matrix (``cm_lookup``).
    Optimized  -> rebuilt from cost-optimal decisions (``pred_lookup`` + schedule).
    F1/Youden  -> per-class thresholds tuned on ``tuning_lookup`` (train OOF when
    available) and applied to ``pred_lookup`` (test); falls back to single-set
    tuning when ``tuning_lookup`` is None. Any tuned/optimized mode falls back to
    the baseline matrix when this result has no stored predictions.
    """
    if mode == MODE_OPTIMIZED:
        opt = optimized_confusion(pred_lookup(result_id), schedule)
        if opt is not None:
            return opt
    elif mode in _MODE_METRIC:
        tune_src = (tuning_lookup(result_id) if tuning_lookup is not None
                    else pred_lookup(result_id))
        score_src = pred_lookup(result_id)
        tuned = threshold_tuned_confusion(
            tune_src, _MODE_METRIC[mode],
            score_pred=(score_src if tuning_lookup is not None else None))
        if tuned is not None:
            return tuned
    return cm_lookup(result_id)


def build_comparison(rows: Iterable[dict], cm_lookup, schedule: pd.DataFrame,
                     *, mode: str = MODE_BASELINE, pred_lookup=None,
                     tuning_lookup=None) -> pd.DataFrame:
    """One net-benefit row per model under the schedule and decision mode.

    ``rows`` is an iterable of mappings with at least ``result_id``, ``model``,
    ``feature_set``. ``cm_lookup`` maps a
    ``result_id`` to its stored (argmax) confusion DataFrame. For
    ``mode == MODE_OPTIMIZED`` a ``pred_lookup`` (``result_id -> predictions
    tuple``) must be supplied so decisions can be recomputed. ``tuning_lookup``
    (train-OOF predictions) tunes thresholds leak-free when supplied. Sorted
    best-first.
    """
    out = []
    for r in rows:
        rid = int(r["result_id"])
        if mode != MODE_BASELINE and pred_lookup is not None:
            cm = confusion_for_mode(rid, mode, cm_lookup, pred_lookup, schedule,
                                    tuning_lookup=tuning_lookup)
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


def _report_runs(db_path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(
            "SELECT e.experiment_id AS run_id, e.timestamp AS timestamp, "
            "e.eval_kind AS eval_kind, COUNT(*) AS n_results "
            "FROM models m JOIN experiments e "
            "ON e.experiment_id = m.experiment_id "
            "GROUP BY e.experiment_id ORDER BY e.timestamp DESC, "
            "e.experiment_id DESC", conn)
    finally:
        conn.close()


def _report_results(db_path, experiment_id) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT m.model_id AS result_id, m.model AS model, "
            "m.feature_set AS feature_set, m.f1 AS f1 "
            "FROM models m WHERE m.experiment_id = ? ORDER BY m.f1 DESC",
            conn, params=[experiment_id])
    finally:
        conn.close()
    df["is_best_for_model"] = 1
    return df


def _report_predictions(db_path, result_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT class_order, y_true, y_proba FROM predictions "
            "WHERE model_id = ?", (int(result_id),)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return (json.loads(row[0]),
            np.asarray(json.loads(row[1]), dtype=object),
            np.asarray(json.loads(row[2]), dtype=float))


def _report_confusion(db_path, result_id) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT true_label, pred_label, count FROM confusion "
            "WHERE model_id = ?", conn, params=(int(result_id),))
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="true_label", columns="pred_label", values="count")
    labels = sorted(set(wide.index) | set(wide.columns), key=str)
    wide = wide.reindex(index=labels, columns=labels)
    wide.index.name = "True"
    wide.columns.name = "Predicted"
    return wide.fillna(0).astype(int)


def _report_id2label(db_path, experiment_id) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id2label FROM experiments WHERE experiment_id = ?",
            (experiment_id,)).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return {}
    return {str(k): str(v) for k, v in json.loads(row[0]).items()}



# ===========================================================================
# Presentation layer  (Streamlit)
# ===========================================================================

def render() -> None:  # pragma: no cover - UI, exercised by `streamlit run`
    import os

    import altair as alt
    import streamlit as st

    from DSSP2026.dashboards.style import inject_dashboard_css

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

    # --- locate the report DB ---------------------------------------------
    default_db = os.getenv("DSSP_DASHBOARD_DB") or "report.db"
    db_path = st.sidebar.text_input("Report database", value=str(default_db))
    if not Path(db_path).exists():
        st.warning(f"No report database found at `{db_path}`.\n\n"
                   "Run an experiment first (`Experiment.run(build_report=True)`) "
                   "or point the sidebar at an existing report.db.")
        st.stop()

    try:
        runs = _report_runs(db_path)
    except Exception as e:
        st.error(f"Unrecognised database schema: {e}")
        st.stop()

    if runs.empty:
        st.warning("The report database has no recorded experiments yet.")
        st.stop()

    run_labels = {
        str(r.run_id): f"{r.run_id} — {r.timestamp} "
                       f"({r.eval_kind}, {int(r.n_results)} models)"
        for r in runs.itertuples()
    }
    run_id = st.sidebar.selectbox(
        "Run", list(run_labels), format_func=lambda x: run_labels[x])

    result_rows = _report_results(db_path, run_id)
    if result_rows.empty:
        st.warning("No results for this run with the current filter.")
        st.stop()

    first_cm = _report_confusion(db_path, int(result_rows.iloc[0].result_id))
    class_labels = [str(x) for x in first_cm.index] if not first_cm.empty \
        else list(C.CLASS_LABELS)
    label_display = _report_id2label(db_path, run_id)

    # --- decision policy (controls the whole dashboard) ------------------
    _METRIC_OPTIONS = [
        "ArgMax",
        "F1",
        "Youden's J",
        "Bayes",
    ]
    st.sidebar.markdown("### Decision policy")
    try:
        _sel_metric = st.sidebar.segmented_control(
            label="Decision policy",
            options=_METRIC_OPTIONS,
            default="ArgMax",
            key="ts_metric",
            label_visibility="collapsed",
        )
        if _sel_metric is None:
            _sel_metric = "ArgMax"
    except AttributeError:
        _sel_metric = st.sidebar.radio(
            "Decision policy",
            _METRIC_OPTIONS,
            key="ts_metric",
            label_visibility="collapsed",
        )

    # Currency is always abbreviated ($7.2k); no per-session toggle.
    _abbrev = True

    def money(value):
        """The active currency formatter."""
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
    # Leak-free sourcing: scoring/display uses persisted TEST predictions;
    # threshold tuning (F1/Youden) uses persisted train OOF predictions. Falls
    # back to stored eval predictions only when those artifacts are absent
    # (older report.db), preserving behaviour for legacy runs.
    @st.cache_data(show_spinner=False)
    def _cm(_db, rid):
        return _report_confusion(_db, rid)

    @st.cache_data(show_spinner=False)
    def _pred(_db, rid):
        return _report_predictions(_db, rid)

    # result_id -> model name, so model-keyed test/OOF tables can be reached
    # from the result-keyed dashboard rows.
    _rid2model = {int(r["result_id"]): str(r["model"])
                  for r in result_rows.to_dict("records")}

    @st.cache_resource(show_spinner=False)
    def _report_for_lookups(path, experiment_id):
        from DSSP2026.report import Report as _R
        return _R(str(path), experiment_id=experiment_id)

    def _test_pred(rid):
        """Test predictions tuple for a result's model, or None."""
        model = _rid2model.get(int(rid))
        if model is None:
            return None
        try:
            rep = _report_for_lookups(str(db_path), str(run_id))
            return rep._read_test_predictions(model)
        except Exception:
            return None

    def _oof_pred(rid):
        """Train-OOF predictions tuple for a result's model, or None."""
        model = _rid2model.get(int(rid))
        if model is None:
            return None
        try:
            rep = _report_for_lookups(str(db_path), str(run_id))
            return rep._read_oof_cached([model])
        except Exception:
            return None

    cm_lookup = lambda rid: _cm_test_or_eval(rid)
    def _cm_test_or_eval(rid):
        """Baseline (argmax) confusion matrix from TEST predictions when
        available; falls back to the stored eval confusion table otherwise.

        This is what fixes the per-class breakdown and the confusion matrix
        showing the validation count: previously baseline mode read the stored
        eval ``confusion`` table directly. Now the argmax confusion is rebuilt
        from the persisted test probabilities so every mode — baseline included
        — reflects the test partition.
        """
        tp = _test_pred(rid)
        if tp is not None:
            co, y_true, y_proba = tp
            y_pred = [co[i] for i in y_proba.argmax(axis=1)]
            return confusion_from_decisions(y_true, y_pred, co)
        return _cm(db_path, rid)
    # pred_lookup feeds decision recomputation; prefer TEST preds, else eval.
    def pred_lookup(rid):
        tp = _test_pred(rid)
        return tp if tp is not None else _pred(db_path, rid)
    # tuning_lookup feeds threshold selection; prefer OOF, else fall back to the
    # same preds (legacy behaviour) so nothing breaks on old DBs.
    def tuning_lookup(rid):
        op = _oof_pred(rid)
        return op if op is not None else pred_lookup(rid)

    # Decision mode is determined by the policy selector in the main panel.
    # It is set there and referenced by the per-class breakdown below.
    # Initialise to baseline so downstream code always has a valid value
    # before the selector runs.
    mode = MODE_BASELINE

    # ===================================================================
    # MAIN PANEL
    # ===================================================================

    report_db_path = Path(db_path)
    _has_report_db = report_db_path.exists()

    all_zero = bool(
        (schedule[[TPFP_COL, TP_COL, FN_COL]].to_numpy() == 0).all()
    )
    if all_zero:
        st.info("The cost schedule is all zeros — edit the sidebar table to "
                "assign costs and benefits. The ranking updates live.")

    # Pre-load Report and build model list sorted by net benefit
    _report        = None
    _models_sorted = []
    _model_nb      = {}
    if _has_report_db:
        try:
            from DSSP2026.report import Report as _Report
            from DSSP2026.report.base import ENSEMBLE_NAME as _ENS

            @st.cache_resource(show_spinner=False)
            def _load_report(path):
                return _Report(str(path))

            _report = _load_report(str(report_db_path))

            def _nb_from_report(model_name):
                try:
                    got = _report._read_test_predictions(model_name)
                    if got is None:
                        got = _report._read_predictions(model_name)
                    co, y_true, y_proba = got
                    y_pred = [co[i] for i in y_proba.argmax(axis=1)]
                    cm_df  = confusion_from_decisions(y_true, y_pred, co)
                    return net_benefit(per_class_counts(cm_df), schedule)
                except Exception:
                    return float("-inf")

            _all_models    = _report.models(include_ensemble=True)
            _model_nb      = {m: _nb_from_report(m) for m in _all_models}
            _models_sorted = sorted(
                _all_models,
                key=lambda m: _model_nb.get(m, float("-inf")),
                reverse=True)
        except Exception:
            pass

    # Map the sidebar's decision-policy selection to the dashboard decision mode.
    # ArgMax → baseline argmax; F1/Youden → per-class threshold-tuned decisions;
    # Bayes → expected-value (Bayes-optimal) decisions under the live schedule.
    if _sel_metric == "Bayes":
        mode = MODE_OPTIMIZED
    elif _sel_metric == "F1":
        mode = MODE_F1
    elif _sel_metric == "Youden's J":
        mode = MODE_YOUDEN
    else:
        mode = MODE_BASELINE

    # Recompute table under the chosen mode now that mode is known.
    table = build_comparison(
        result_rows.to_dict("records"), cm_lookup, schedule,
        mode=mode, pred_lookup=pred_lookup, tuning_lookup=tuning_lookup)
    if table.empty:
        st.warning("No confusion matrices stored for these results.")
        st.stop()

    _ref_cm = confusion_for_mode(
        int(result_rows.iloc[0]["result_id"]), mode, cm_lookup, pred_lookup,
        schedule, tuning_lookup=tuning_lookup)
    no_action = (no_action_baseline(_ref_cm, schedule)
                 if _ref_cm is not None and not _ref_cm.empty else None)
    gross_no_action = float(no_action["Gross"]) if no_action is not None else 0.0
    table["Net benefit"] = table["Gross"] - gross_no_action
    table = table.sort_values("Net benefit", ascending=False).reset_index(drop=True)

    # Ensemble row
    _ens_report_db = Path(db_path)
    if _ens_report_db.exists():
        try:
            from DSSP2026.report import Report as _RptEns
            from DSSP2026.report.base import ENSEMBLE_NAME as _ENS_NAME
            _rpt_ens = _RptEns(str(_ens_report_db))
            # Leak-free: score the ensemble on persisted TEST predictions; tune
            # F1/Youden thresholds on persisted train OOF. Fall back to eval
            # mean-proba only if test predictions aren't persisted.
            _ens_test = _rpt_ens._read_test_predictions(_ENS_NAME)
            if _ens_test is not None:
                _co_ens, _yt_ens, _yp_ens = _ens_test
            else:
                _co_ens, _yt_ens, _yp_ens = _rpt_ens._ensemble_proba()
            if mode == MODE_OPTIMIZED:
                _y_pred_ens = expected_value_decisions(
                    _co_ens, _yp_ens, schedule)
            elif mode in _MODE_METRIC:
                from DSSP2026.core.threshold import (
                    per_class_thresholds, decisions_from_thresholds)
                _ens_oof = _rpt_ens._read_oof_cached(
                    _rpt_ens.models(include_ensemble=False))
                _tune_co, _tune_yt, _tune_yp = (
                    _ens_oof if _ens_oof is not None else (_co_ens, _yt_ens, _yp_ens))
                _thr_ens = per_class_thresholds(
                    _tune_yt, _tune_yp, _tune_co, metric=_MODE_METRIC[mode])
                _y_pred_ens = decisions_from_thresholds(
                    _yp_ens, _co_ens, _thr_ens)
            else:
                _y_pred_ens = [_co_ens[i] for i in _yp_ens.argmax(axis=1)]
            _cm_ens     = confusion_from_decisions(_yt_ens, _y_pred_ens, _co_ens)
            _counts_ens = per_class_counts(_cm_ens)
            _contrib_ens = class_contributions(_counts_ens, schedule)
            _inv_ens    = float(_contrib_ens["investment"].sum())
            _loss_ens   = float(_contrib_ens["loss"].sum())
            _gains_ens  = float(_contrib_ens["gains"].sum())
            _gross_ens  = _inv_ens + _loss_ens + _gains_ens
            _nb_ens     = _gross_ens - gross_no_action
            _ens_row    = pd.DataFrame([{
                "Model": _ENS_NAME, "Feature set": "ensemble",
                "Investment": _inv_ens,
                "Waste": float(_contrib_ens["waste"].sum()),
                "Loss": _loss_ens, "Gains": _gains_ens,
                "Gross": _gross_ens, "Net benefit": _nb_ens,
            }])
            table = pd.concat([table, _ens_row], ignore_index=True)
            table = table.sort_values(
                "Net benefit", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    # Display columns: the three additive components (Investment + Loss + Gains)
    # sum to Gross. Net benefit is Gross relative to the No-Action baseline.
    _display_cols = ["Model", "Feature set", "Investment", "Loss", "Gains",
                     "Gross", "Net benefit"]
    table_display = (
        pd.concat([pd.DataFrame([no_action]), table], ignore_index=True)
        if no_action is not None else table)
    table_display = table_display[
        [c for c in _display_cols if c in table_display.columns]]

    # ── Threshold tuning controls + plot ─────────────────────────────────
    st.subheader("Threshold tuning")
    if not _has_report_db:
        st.info("No `report.db` found — run `Experiment.build_report_db()` "
                "to enable threshold tuning.")
    elif not _models_sorted:
        st.warning("Could not load models from report.db.")
    else:
        ctrl_model, ctrl_class = st.columns([3, 1])
        with ctrl_model:
            _sel_model = st.selectbox(
                "Model",
                _models_sorted,
                index=0,
                format_func=lambda m: (
                    f"{m}  (NB: "
                    f"{fmt_currency(_model_nb.get(m, 0), abbreviate=_abbrev)})"
                ),
                key="ts_model",
            )
        with ctrl_class:
            _sel_class = st.selectbox(
                "Class",
                class_labels,
                format_func=lambda c: label_display.get(str(c), c),
                key="ts_class",
            )

        import matplotlib.pyplot as _plt
        try:
            if _sel_metric == "ArgMax":
                _ts = _report.threshold_sweep(
                    model=_sel_model, target_class=_sel_class,
                    optimize="f1", metrics=["precision", "recall", "f1"])
                fig = _ts.figure()
                ax  = fig.axes[0]
                for _ln in ax.get_lines():
                    if _ln.get_linestyle() in ("--", "dashed") and \
                            _ln.get_color() not in (
                                ATT_COLORS["deep_blue"],
                                ATT_COLORS.get("teal",   "#00857C"),
                                ATT_COLORS.get("orange", "#E55A0B")):
                        _ln.remove()
                ax.axvline(0.5, color=ATT_COLORS["navy"], linewidth=2.0,
                           linestyle="--", zorder=5,
                           label="ArgMax threshold (0.5)")
                ax.legend(loc="best", fontsize=9, frameon=True)
                ax.set_title(
                    f"ArgMax — {_sel_model}  "
                    f"({label_display.get(_sel_class, _sel_class)} vs. rest)",
                    fontsize=11, color=ATT_COLORS["navy"], weight="bold")
                fig.set_size_inches(10, 3.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                _plt.close(fig)

            elif _sel_metric == "F1":
                _ts = _report.threshold_sweep(
                    model=_sel_model, target_class=_sel_class,
                    optimize="f1", metrics=["precision", "recall", "f1"])
                fig = _ts.figure()
                fig.axes[0].set_title(
                    f"F1 — {_sel_model}  "
                    f"({label_display.get(_sel_class, _sel_class)} vs. rest)",
                    fontsize=11, color=ATT_COLORS["navy"], weight="bold")
                fig.set_size_inches(10, 3.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                _plt.close(fig)

            elif _sel_metric == "Youden's J":
                _ts = _report.threshold_sweep(
                    model=_sel_model, target_class=_sel_class,
                    optimize="youden",
                    metrics=["precision", "recall", "f1", "specificity"])
                fig = _ts.figure()
                fig.axes[0].set_title(
                    f"Youden's J — {_sel_model}  "
                    f"({label_display.get(_sel_class, _sel_class)} vs. rest)",
                    fontsize=11, color=ATT_COLORS["navy"], weight="bold")
                fig.set_size_inches(10, 3.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                _plt.close(fig)

            elif _sel_metric == "Bayes":
                _co   = [str(c) for c in class_labels]
                _payoff_df = schedule.reindex(_co)
                fig = _report.cost_threshold_plot(
                    payoff=_payoff_df, model=_sel_model,
                    target_class=_sel_class).figure()
                fig.axes[0].set_title(
                    f"Bayes — {_sel_model}  "
                    f"({label_display.get(_sel_class, _sel_class)} vs. rest)",
                    fontsize=11, color=ATT_COLORS["navy"], weight="bold")
                fig.set_size_inches(10, 3.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                _plt.close(fig)

        except Exception as _e:
            st.warning(f"Could not render threshold plot: {_e}")

    # ── Three summary stats ───────────────────────────────────────────────
    best   = table.iloc[0]
    spread = table["Net benefit"].max() - table["Net benefit"].min()
    s1, s2, s3 = st.columns(3)
    s1.metric("Best model", best["Model"],
              help=f"feature set: {best['Feature set']}")
    s2.metric("Best NB",    money(best["Net benefit"]))
    s3.metric("Spread",     money(spread),
              help="Best − worst net benefit")

    # ── Model ranking hbar ────────────────────────────────────────────────
    plot_df = table_display.assign(
        Label=table_display["Model"] + "  ("
              + table_display["Feature set"] + ")",
        Sign=np.where(
            table_display["Model"] == "No Action", "No Action",
            np.where(table_display["Net benefit"] >= 0, "Gain", "Loss")))
    for col in ("Net benefit", "Investment", "Waste", "Loss"):
        if col in plot_df.columns:
            plot_df[f"{col} $"] = plot_df[col].map(money)
    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Net benefit:Q", title="Net benefit"),
            y=alt.Y("Label:N", sort="-x", title=None),
            color=alt.Color(
                "Sign:N",
                scale=alt.Scale(
                    domain=["Gain", "Loss", "No Action"],
                    range=[pos_c, neg_c, ATT_COLORS["gray_500"]]),
                legend=alt.Legend(title=None)),
            tooltip=["Model", "Feature set",
                     alt.Tooltip("Net benefit $:N", title="Net benefit"),
                     alt.Tooltip("Investment $:N",  title="Investment"),
                     alt.Tooltip("Waste $:N",       title="Waste"),
                     alt.Tooltip("Loss $:N",        title="Loss")],
        )
        .properties(height=max(80, 34 * len(plot_df)))
    )
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(
        color=ATT_COLORS["gray_500"], strokeDash=[4, 3]).encode(x="z:Q")
    st.altair_chart(bars + zero, use_container_width=True)

    st.markdown("---")

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

    # Ensemble has no result_id in result_rows — derive its confusion matrix
    # from report.db mean-proba predictions instead.
    _ens_pick_db = Path(db_path)
    from DSSP2026.report.base import ENSEMBLE_NAME as _ENS_PICK_NAME
    if pick == _ENS_PICK_NAME and _ens_pick_db.exists():
        try:
            from DSSP2026.report import Report as _RptPick
            from DSSP2026.report.base import ENSEMBLE_NAME as _ENS_PICK
            _rpt_pick = _RptPick(str(_ens_pick_db))
            _pick_test = _rpt_pick._read_test_predictions(_ENS_PICK)
            if _pick_test is not None:
                _co_pick, _yt_pick, _yp_pick = _pick_test
            else:
                _co_pick, _yt_pick, _yp_pick = _rpt_pick._ensemble_proba()
            if mode == MODE_OPTIMIZED:
                _y_pred_pick = expected_value_decisions(
                    _co_pick, _yp_pick, schedule)
            elif mode in _MODE_METRIC:
                from DSSP2026.core.threshold import (
                    per_class_thresholds, decisions_from_thresholds)
                _pick_oof = _rpt_pick._read_oof_cached(
                    _rpt_pick.models(include_ensemble=False))
                _tp_co, _tp_yt, _tp_yp = (
                    _pick_oof if _pick_oof is not None else (_co_pick, _yt_pick, _yp_pick))
                _thr_pick = per_class_thresholds(
                    _tp_yt, _tp_yp, _tp_co, metric=_MODE_METRIC[mode])
                _y_pred_pick = decisions_from_thresholds(
                    _yp_pick, _co_pick, _thr_pick)
            else:
                _y_pred_pick = [_co_pick[i] for i in _yp_pick.argmax(axis=1)]
            _pick_cm = confusion_from_decisions(_yt_pick, _y_pred_pick, _co_pick)
            counts = per_class_counts(_pick_cm)
            _pick_class_order = list(_co_pick)
            _pick_confusion = _pick_cm
        except Exception as _pick_e:
            st.warning(f"Could not compute Ensemble breakdown: {_pick_e}")
            counts = None
            _pick_confusion = None
            _pick_class_order = None
    else:
        rid = next(
            (int(r["result_id"]) for r in result_rows.to_dict("records")
             if str(r["model"]) == pick),
            None)
        if rid is not None:
            _pick_confusion = confusion_for_mode(
                rid, mode, cm_lookup, pred_lookup, schedule,
                tuning_lookup=tuning_lookup)
            counts = per_class_counts(_pick_confusion)
            _pick_class_order = [str(c) for c in _pick_confusion.index]
        else:
            counts = None
            _pick_confusion = None
            _pick_class_order = None

    if counts is None:
        st.info("Per-class breakdown unavailable for this model.")
    else:
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

        # -- Cost-aware confusion matrix -----------------------------------
        # Same decisions as the breakdown above, laid out as a true × predicted
        # grid where each cell shows the count and its currency value, shaded by
        # the per-cell cost on a symmetric RdYlGn scale.
        if _pick_confusion is not None:
            st.subheader("Cost-aware confusion matrix")
            st.caption(
                "Rows are the true class, columns the predicted class. Cells are "
                "shaded by per-cell cost (green = value gained, red = value lost); "
                "toggle whether each cell shows the count or its currency value.")
            _cc_view = st.radio(
                "Viewpoint", ["Net Benefit", "Gross"], horizontal=True,
                key="cc_viewpoint",
                help="Net Benefit credits the avoided miss on correct catches "
                     "(positive spin). Gross shows the honest intervention cost, "
                     "so a correct catch can still read negative.")
            _vp = "net_benefit" if _cc_view == "Net Benefit" else "gross"

            _cc_show = st.radio(
                "Cell display", ["Counts", "Currency"], horizontal=True,
                key="cc_celltext",
                help="Counts shows the number of samples in each cell; Currency "
                     "shows the dollar value (count × per-cell cost).")

            _cc_co = _pick_class_order
            _cc_counts = _pick_confusion.reindex(
                index=_cc_co, columns=_cc_co).fillna(0).astype(int)
            _cc_unit = cost_unit_matrix(_cc_co, schedule, viewpoint=_vp)
            _cc_value = _cc_counts.to_numpy(float) * _cc_unit

            _cc_labels = [label_display.get(str(c), c) for c in _cc_co]
            # Prefix axis ticks so each axis is self-describing without relying on
            # the caption: rows = true class, columns = predicted class.
            _cc_row_labels = [f"True: {l}" for l in _cc_labels]
            _cc_col_labels = [f"Pred: {l}" for l in _cc_labels]
            if _cc_show == "Counts":
                _cc_cells = [[fmt_count(int(_cc_counts.iat[i, j]))
                              for j in range(len(_cc_co))]
                             for i in range(len(_cc_co))]
            else:
                _cc_cells = [[money(_cc_value[i, j])
                              for j in range(len(_cc_co))]
                             for i in range(len(_cc_co))]
            _cc_text = pd.DataFrame(
                _cc_cells, index=_cc_row_labels, columns=_cc_col_labels)
            # Name the axes themselves (renders as corner headers in Streamlit).
            _cc_text.index.name = "True \\ Predicted"

            _cc_max = float(np.nanmax(np.abs(_cc_value))) if _cc_value.size else 0.0
            if not np.isfinite(_cc_max) or _cc_max == 0.0:
                _cc_max = 1.0
            _cc_cmap = _mpl.colormaps["RdYlGn"]
            _cc_norm = _mpl.colors.Normalize(vmin=-_cc_max, vmax=_cc_max)

            def _cc_colors(_frame):
                css = pd.DataFrame("", index=_cc_text.index, columns=_cc_text.columns)
                for i in range(len(_cc_co)):
                    for j in range(len(_cc_co)):
                        r, g, b, _ = _cc_cmap(_cc_norm(_cc_value[i, j]))
                        css.iat[i, j] = (
                            f"background-color: rgba({int(r*255)},{int(g*255)},"
                            f"{int(b*255)},0.85); color: #1A1A1A; "
                            f"font-weight: 600; text-align: center;")
                return css

            _cc_sty = _cc_text.style.apply(_cc_colors, axis=None)
            st.dataframe(_cc_sty, use_container_width=True)

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