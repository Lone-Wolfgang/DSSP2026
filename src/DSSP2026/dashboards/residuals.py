"""
workflows/residuals_dashboard.py — standalone interactive OLS residual diagnostics.

A self-contained Streamlit app, deliberately independent of the cost/net-benefit
``dashboard.py``. It reads a *raw dataset*, fits an OLS model from a formula you
type, and renders the classic four-panel quad as Altair charts:

    - Residuals vs fitted        (linearity; lowess overlay)
    - Normal Q-Q                 (residual normality)
    - Scale-location             (homoscedasticity)
    - Cook's distance            (influence; 4/n threshold rule)

Live model
----------
The dashboard computes diagnostics in-process: point it at a CSV/Parquet, type an
OLS formula, and click **Refit**. Changing the formula refits (cheap: ~0.1–1.8s
at 50k rows; the cost scales with the number of expanded columns, not row count);
the fit is cached on ``(data, formula)`` so re-selecting a tried formula is
instant. Changing the *color* encoding never refits — it re-renders the cached
diagnostics, coloring the scatter panels by any column (categorical palette for
nominal columns, sequential scale for numeric).

Canvas rendering keeps ~15k plotted points fluid; larger frames are downsampled
for display (all rows above the Cook's threshold are always kept) while trend
lines and the 4/n line use the full data.

Note: the batch ``persist_ols_diagnostics`` → report.db path still exists as the
regression→report bridge for logged runs, but this dashboard is the live,
formula-driven explorer and does not read those artifacts.

Launch via::

    python -m DSSP2026.workflows.cli residuals --path project/data/diamonds.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from DSSP2026.core.style import ATT_COLORS
except Exception:  # pragma: no cover - defensive
    ATT_COLORS = {
        "att_blue": "#00A8E0", "orange": "#E55A0B", "teal": "#00857C",
        "gray_900": "#1A1A1A", "gray_700": "#333333", "gray_500": "#666666",
        "gray_300": "#BBBBBB", "gray_100": "#F2F2F2", "white": "#FFFFFF",
    }

# Columns the renderer needs; provenance (formula/n_obs) is optional on top.
_REQUIRED = ("fitted", "residuals", "stud_resid", "cooks", "obs")


# ===========================================================================
# Data layer — pure, testable, Streamlit-free.
# ===========================================================================

def read_tabular(path) -> pd.DataFrame:
    """Read a CSV or Parquet by sniffing content, not the file extension.

    The ``Could not read data: 'utf-8' codec can't decode byte ...`` error comes
    from feeding a binary Parquet file to ``pd.read_csv`` because its extension
    didn't match. Parquet files begin (and end) with the magic bytes ``PAR1``, so
    we detect that directly; everything else is treated as delimited text, with a
    latin-1 fallback for CSVs that aren't valid UTF-8.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError as exc:
        raise ValueError(f"cannot open {path}: {exc}") from exc

    if head == b"PAR1":
        return pd.read_parquet(path)

    # Delimited text. Try UTF-8, then latin-1 (covers most stray byte issues),
    # rather than letting a single bad byte abort the whole read.
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _is_sqlite(path: Path) -> bool:
    """Cheap magic-byte check so we route .db vs .parquet without trusting the
    extension (the artifact could be renamed)."""
    try:
        with open(path, "rb") as f:
            return f.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def available_labels(path) -> list[str]:
    """OLS labels available at *path*.

    For a report.db, reads the provenance table. For a bare parquet there is no
    label concept, so a single synthetic ``"OLS"`` is returned.
    """
    path = Path(path)
    if _is_sqlite(path):
        from DSSP2026.report.residual_artifact import list_residual_diagnostics
        # report.db keys diagnostics by experiment; gather across all experiments
        import sqlite3
        from DSSP2026.report.residual_artifact import TABLE, _has_table
        conn = sqlite3.connect(path)
        try:
            if not _has_table(conn):
                return []
            rows = conn.execute(
                f"SELECT DISTINCT experiment_id, label FROM {TABLE} "
                "ORDER BY experiment_id, label").fetchall()
        finally:
            conn.close()
        return [f"{exp} / {lab}" for exp, lab in rows]
    return ["OLS"] if path.exists() else []


def load_diag(path, *, label: Optional[str] = None):
    """Load a diagnostics frame from a report.db or a bare parquet.

    Returns ``(diag_df, meta)`` where ``meta`` carries ``formula`` (str|None)
    and ``n_obs`` (int). Returns ``None`` if nothing usable is found, mirroring
    the fail-soft contract of the artifact reader.
    """
    path = Path(path)
    if not path.exists():
        return None

    if _is_sqlite(path):
        from DSSP2026.report.residual_artifact import load_residual_diagnostics
        # label arrives as "experiment_id / label"; split it back.
        if label and " / " in label:
            exp_id, lab = label.split(" / ", 1)
        else:
            # default to the first available
            labels = available_labels(path)
            if not labels:
                return None
            exp_id, lab = (label or labels[0]).split(" / ", 1)
        loaded = load_residual_diagnostics(path, experiment_id=exp_id, label=lab)
        if loaded is None:
            return None
        diag_df, n_obs = loaded
        formula = _formula_for(path, exp_id, lab)
        return diag_df, {"formula": formula, "n_obs": n_obs}

    # Bare parquet — fully standalone.
    try:
        diag_df = pd.read_parquet(path)
    except Exception:
        return None
    missing = set(_REQUIRED) - set(diag_df.columns)
    if missing:
        return None
    return diag_df, {"formula": None, "n_obs": len(diag_df)}


def _formula_for(report_db, experiment_id, label):
    import sqlite3
    from DSSP2026.report.residual_artifact import TABLE, _has_table
    conn = sqlite3.connect(report_db)
    try:
        if not _has_table(conn):
            return None
        row = conn.execute(
            f"SELECT formula FROM {TABLE} WHERE experiment_id=? AND label=?",
            (experiment_id, label)).fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None


def lowess_overlay(x, y, frac: float = 0.45, *, max_points: int = 3000,
                   seed: int = 0) -> pd.DataFrame:
    """Sorted lowess smooth as a tidy DataFrame (``x``/``y``) for an Altair line.

    Returns an empty frame for fewer than 5 points (matching the matplotlib
    renderer's guard), so callers can layer it unconditionally.

    LOWESS is ~O(n^2); on large frames (e.g. 50k rows) computing it on every
    point hangs for minutes. Above *max_points* we fit the smooth on a random
    subsample — the curve is visually identical because LOWESS estimates a
    smooth trend, not per-point detail — keeping the panel responsive.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5:
        return pd.DataFrame({"x": [], "y": []})
    if len(x) > max_points:
        idx = np.random.default_rng(seed).choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm = lowess(y, x, frac=frac, return_sorted=True)
    return pd.DataFrame({"x": sm[:, 0], "y": sm[:, 1]})


def qq_frame(stud_resid) -> pd.DataFrame:
    """Theoretical-vs-sample quantiles for a normal Q-Q plot, plus the reference
    line endpoints. Returns ``(points_df, line_df)``-style columns in one frame:
    ``theoretical``, ``sample`` for points; the 45-ref is derived in the chart.
    """
    from scipy import stats
    r = np.asarray(stud_resid, float)
    r = r[np.isfinite(r)]
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    return pd.DataFrame({"theoretical": osm, "sample": osr})


# ===========================================================================
# Presentation — Altair builders (still Streamlit-free, unit-testable).
# ===========================================================================

def color_candidates(diag_df: pd.DataFrame) -> list[str]:
    """Carried-through columns eligible to color by.

    Excludes the diagnostic quantities themselves (fitted/residuals/etc.) and the
    bookkeeping flags, leaving the raw features + target that persist carried in.
    """
    reserved = {"fitted", "residuals", "stud_resid", "cooks", "leverage", "obs",
                "in_danger_zone", "bubble_size", "sqrt_abs_stud"}
    return [c for c in diag_df.columns if c not in reserved]


def encoding_type(series: pd.Series) -> str:
    """'nominal' for categorical/object/bool columns, 'quantitative' for numeric.

    Drives whether the color picker uses a categorical palette or a continuous
    scale, following the column's dtype.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "quantitative"
    return "nominal"


def _base_props(title: str):
    return dict(title=title, height=300)


def build_charts(diag_df: pd.DataFrame, n_obs: int, *, max_points: int = 15000,
                 color_field: Optional[str] = None):
    """Build the four Altair diagnostic panels, optionally colored by a feature.

    Returns ``charts_dict`` mapping panel name to an ``alt.Chart``.

    For large frames the *plotted* points are downsampled to *max_points* (the
    canvas renderer comfortably handles ~15k; trend lines and the Cook's 4/n line
    are still computed against the full data). Points above the Cook's threshold
    are always kept, so influential observations never get sampled away.

    *color_field*, if given, colors the three scatter panels (residuals-vs-fitted,
    scale-location, Cook's) by that carried-through column -- a categorical palette
    for nominal columns, a sequential scale for numeric ones. The Q-Q panel keeps
    a single color (its points are reordered by quantile, so a feature coloring
    there would mislead).
    """
    import altair as alt
    # Self-contained: plotted points are capped at max_points, which can exceed
    # Altair's default 5k guard, so lift it here rather than relying on the caller.
    alt.data_transformers.disable_max_rows()

    blue = ATT_COLORS["att_blue"]
    orange = ATT_COLORS["orange"]
    gray = ATT_COLORS["gray_700"]

    full = diag_df.copy()
    full["sqrt_abs_stud"] = np.sqrt(np.abs(full["stud_resid"]))

    # Trend lines use the FULL data (lowess_overlay subsamples internally).
    sm = lowess_overlay(full["fitted"], full["residuals"])
    sm2 = lowess_overlay(full["fitted"], full["sqrt_abs_stud"])

    # Downsample only the scatter points, but always retain influential rows.
    thresh = 4.0 / max(int(n_obs), 1)
    if len(full) > max_points:
        influential = full[full["cooks"] > thresh]
        n_rest = max(max_points - len(influential), 0)
        rest = full.drop(influential.index)
        if n_rest and len(rest) > n_rest:
            rest = rest.sample(n=n_rest, random_state=0)
        df = pd.concat([influential, rest]).sort_index()
    else:
        df = full

    tip = [alt.Tooltip("obs:Q", title="Obs"),
           alt.Tooltip("fitted:Q", title="Fitted", format=".3g"),
           alt.Tooltip("residuals:Q", title="Residual", format=".3g"),
           alt.Tooltip("stud_resid:Q", title="Std resid", format=".3g"),
           alt.Tooltip("cooks:Q", title="Cook's D", format=".4f")]

    # Resolve the color encoding once; reused across the three scatter panels.
    if color_field and color_field in df.columns:
        etype = encoding_type(df[color_field])
        suffix = "Q" if etype == "quantitative" else "N"
        scale = (alt.Scale(scheme="viridis") if etype == "quantitative"
                 else alt.Scale(scheme="tableau10"))
        color_enc = alt.Color(f"{color_field}:{suffix}", scale=scale,
                              legend=alt.Legend(title=color_field))
        tip = tip + [alt.Tooltip(f"{color_field}:{suffix}", title=color_field)]
    else:
        color_enc = alt.value(blue)

    # --- Residuals vs fitted -------------------------------------------------
    rvf_pts = (
        alt.Chart(df).mark_circle(size=45, opacity=0.55)
        .encode(x=alt.X("fitted:Q", title="Fitted values"),
                y=alt.Y("residuals:Q", title="Residuals"),
                color=color_enc, tooltip=tip))
    rvf_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=gray, strokeDash=[4, 4]).encode(y="y:Q")
    rvf_smooth = (alt.Chart(sm).mark_line(color=orange, strokeWidth=2.5)
                  .encode(x="x:Q", y="y:Q") if len(sm) else None)
    rvf = rvf_zero + rvf_pts
    if rvf_smooth is not None:
        rvf = rvf + rvf_smooth
    rvf = rvf.properties(**_base_props("Residuals vs fitted"))

    # --- Normal Q-Q (single color by design) ---------------------------------
    qq = qq_frame(full["stud_resid"])
    if len(qq) > max_points:
        qq = qq.iloc[np.linspace(0, len(qq) - 1, max_points).astype(int)]
    lo = float(min(qq["theoretical"].min(), qq["sample"].min()))
    hi = float(max(qq["theoretical"].max(), qq["sample"].max()))
    qq_line = (alt.Chart(pd.DataFrame({"t": [lo, hi], "s": [lo, hi]}))
               .mark_line(color=orange, strokeWidth=2)
               .encode(x="t:Q", y="s:Q"))
    qq_pts = (alt.Chart(qq).mark_circle(size=40, opacity=0.55, color=blue)
              .encode(x=alt.X("theoretical:Q", title="Theoretical quantiles"),
                      y=alt.Y("sample:Q", title="Ordered studentized resid"),
                      tooltip=[alt.Tooltip("theoretical:Q", format=".3g"),
                               alt.Tooltip("sample:Q", format=".3g")]))
    qq_chart = (qq_line + qq_pts).properties(**_base_props("Normal Q-Q"))

    # --- Scale-location ------------------------------------------------------
    sl_pts = (
        alt.Chart(df).mark_circle(size=45, opacity=0.55)
        .encode(x=alt.X("fitted:Q", title="Fitted values"),
                y=alt.Y("sqrt_abs_stud:Q", title="\u221a|studentized residual|"),
                color=color_enc, tooltip=tip))
    sl_smooth = (alt.Chart(sm2).mark_line(color=orange, strokeWidth=2.5)
                 .encode(x="x:Q", y="y:Q") if len(sm2) else None)
    sl = sl_pts if sl_smooth is None else (sl_pts + sl_smooth)
    sl = sl.properties(**_base_props("Scale-location"))

    # --- Cook's distance -----------------------------------------------------
    cook_pts = (
        alt.Chart(df).mark_circle(size=45, opacity=0.6)
        .encode(x=alt.X("obs:Q", title="Observation"),
                y=alt.Y("cooks:Q", title="Cook's D"),
                color=color_enc, tooltip=tip))
    cook_rule = (alt.Chart(pd.DataFrame({"y": [thresh]}))
                 .mark_rule(color=orange, strokeDash=[6, 4], strokeWidth=1.6)
                 .encode(y="y:Q"))
    cook = (cook_pts + cook_rule).properties(
        **_base_props(f"Cook's distance (4/n = {thresh:.4f})"))

    return {"rvf": rvf, "qq": qq_chart, "sl": sl, "cook": cook}


# ===========================================================================
# Streamlit render — presentation only.
# ===========================================================================

def compute_live_diagnostics(df: pd.DataFrame, formula: str,
                             color_columns="auto"):
    """Fit OLS on *df* with *formula* and return a diagnostics frame + meta.

    The live counterpart to the persisted-artifact path: same
    ``make_ols_diagnostics_df`` output and same carried-feature join as
    ``persist_ols_diagnostics``, but computed in-process so the dashboard can
    change the formula without round-tripping through report.db. Returns
    ``(diag_df, meta)`` with ``meta = {"formula", "n_obs"}``.

    Benchmarks (50k rows): ~0.1s for a 2-term formula, ~1.8s for a formula that
    expands to ~60 dummy columns — fast enough to drive from a Refit button.
    """
    from DSSP2026.linear_regression.fit import fit_ols, make_ols_diagnostics_df

    res = fit_ols(df, formula)
    diag_df = make_ols_diagnostics_df(res.model)

    # Carry raw columns for color encoding, aligned by position to fitted rows.
    if color_columns is not None:
        row_labels = res.model.model.data.row_labels
        if row_labels is not None:
            used = df.loc[row_labels].reset_index(drop=True)
            used["obs"] = np.arange(1, len(used) + 1)
            cols = (list(df.columns) if color_columns == "auto"
                    else [c for c in color_columns if c in used.columns])
            carry = used[["obs"] + [c for c in cols if c != "obs"]]
            diag_df = diag_df.merge(carry, on="obs", how="left")

    return diag_df, {"formula": formula, "n_obs": int(res.model.nobs)}


def render() -> None:  # pragma: no cover - UI, exercised by `streamlit run`
    import streamlit as st
    import altair as alt

    # Canvas renderer: draws all points to one bitmap instead of thousands of
    # SVG DOM nodes, which is what made ~5k-point panels sluggish. Handles ~15k
    # points smoothly and keeps color encoding fully supported.
    alt.renderers.set_embed_options(renderer="canvas")
    # build_charts downsamples plotted points, but lift the row guard so the
    # retained influential rows never trip it either.
    alt.data_transformers.disable_max_rows()

    st.set_page_config(page_title="Residual Diagnostics", layout="wide")
    st.title("OLS residual diagnostics")

    # --- Data source: raw dataframe (CSV/Parquet), computed live -------------
    default_path = os.getenv("DSSP_RESIDUALS_PATH", "")
    path_str = st.sidebar.text_input(
        "Data source (CSV or Parquet)", value=default_path)
    if not path_str or not Path(path_str).exists():
        st.warning(
            "Point the sidebar at a raw dataset (`.csv` or `.parquet`), then "
            "type an OLS formula and click **Refit**. Categoricals need `C(col)`; "
            "transforms like `log(x)` work without the `np.` prefix.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def _load_source(path: str) -> pd.DataFrame:
        return read_tabular(path)

    try:
        data = _load_source(path_str)
    except Exception as exc:  # noqa: BLE001 - surface load errors to the UI
        st.error(f"Could not read data: {exc}")
        st.stop()

    st.sidebar.caption(f"{len(data):,} rows · {len(data.columns)} columns")
    with st.sidebar.expander("Columns"):
        st.write(", ".join(map(str, data.columns)))

    # --- Formula + explicit Refit -------------------------------------------
    formula = st.sidebar.text_area(
        "OLS formula", value=st.session_state.get("resid_formula", ""),
        placeholder="price ~ x + C(cut)", height=80)
    refit = st.sidebar.button("Refit", type="primary")

    # Cache keyed on (data identity, formula): re-selecting a tried formula is a
    # cache hit (instant); only a genuinely new formula pays the refit cost.
    @st.cache_data(show_spinner="Fitting OLS…")
    def _diag_cached(path: str, n_rows: int, formula: str):
        return compute_live_diagnostics(_load_source(path), formula)

    if refit and formula.strip():
        st.session_state["resid_formula"] = formula.strip()

    active_formula = st.session_state.get("resid_formula", "")
    if not active_formula:
        st.info("Enter a formula in the sidebar and click **Refit** to begin.")
        st.stop()

    try:
        diag_df, meta = _diag_cached(path_str, len(data), active_formula)
    except Exception as exc:  # noqa: BLE001 - bad formula etc.
        st.error(f"Fit failed for `{active_formula}`:  {exc}")
        st.stop()

    st.caption(f"Formula:  `{meta['formula']}`   ·   n = {meta['n_obs']:,}")

    # Color encoding: any carried-through column, dtype-driven scale. Changing
    # this does NOT refit — it re-renders the cached diagnostics.
    candidates = color_candidates(diag_df)
    color_field = None
    if candidates:
        choice = st.sidebar.selectbox("Color points by", ["(none)"] + candidates)
        color_field = None if choice == "(none)" else choice

    MAX_POINTS = 15000
    charts = build_charts(diag_df, meta["n_obs"], max_points=MAX_POINTS,
                          color_field=color_field)

    n_plotted = min(len(diag_df), MAX_POINTS)
    if len(diag_df) > n_plotted:
        st.caption(
            f"Showing {n_plotted:,} of {len(diag_df):,} points (all rows above "
            "the Cook's threshold are kept; the rest are sampled). Trend lines "
            "and the 4/n line use the full dataset.")

    row1 = alt.hconcat(charts["rvf"], charts["qq"]).resolve_scale(
        color="independent")
    row2 = alt.hconcat(charts["sl"], charts["cook"]).resolve_scale(
        color="shared")
    st.altair_chart(alt.vconcat(row1, row2), use_container_width=True)

    st.markdown(
        "Residuals-vs-fitted (orange lowess) checks linearity; Normal Q-Q checks "
        "residual normality; scale-location checks homoscedasticity; Cook's "
        "distance flags influence against the 4/n line. Change the formula and "
        "click **Refit**; color the scatter panels by any feature without refitting.")

    # --- Influential-points drill-down --------------------------------------
    st.subheader("Most influential observations")
    thresh = 4.0 / max(int(meta["n_obs"]), 1)
    table = diag_df.copy()
    cols = [c for c in ["obs", "fitted", "residuals", "stud_resid",
                        "leverage", "cooks"] if c in table.columns]
    table = table[cols].sort_values("cooks", ascending=False)
    flagged = int((diag_df["cooks"] > thresh).sum())
    st.caption(f"{flagged} observation(s) exceed the 4/n Cook's-distance "
               f"threshold ({thresh:.4f}).")
    top_n = st.slider("Show top N by Cook's distance", 5,
                      min(50, len(table)), min(15, len(table)))
    fmt = {c: "{:.4f}" for c in ("fitted", "residuals", "stud_resid",
                                 "leverage", "cooks") if c in table.columns}
    st.dataframe(table.head(top_n).style.format(fmt),
                 use_container_width=True, hide_index=True)


if __name__ == "__main__":  # pragma: no cover
    render()