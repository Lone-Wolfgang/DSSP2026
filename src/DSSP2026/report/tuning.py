"""
report/tuning.py — report.db-backed hyperparameter-tuning views.

Three tuning plots, each suited to a different search shape, all served from a
report.db's ``trials`` table (params JSON + cv_value) plus the experiment's
persisted ``search_space`` spec. No live Optuna study is needed.

Search shape -> plot
--------------------
- **One tunable int dimension** (e.g. Decision tree ``max_depth``): the *elbow*
  curve. CV value vs. the single param, with the parsimony pick marked. This iså
  distinct from the others: it encodes a model-selection rule (prefer the
  smallest near-optimal value), not just a view of the search.
- **Grid search** (model in ``GRID_MODELS``, spec is a discrete grid): the tile
  heatmap. Any number of grid dimensions is fine, but only *two at a time* are
  shown. When 3+ dims exist, the two highest-*variance* axes (those that move
  the score most) are chosen and the others are sliced at the best trial's
  values — i.e. the 2-D grid that contains the winner. The heatmap is gated to
  grid searches: it is meaningless for a continuous TPE search whose points do
  not lie on a lattice.
- **Anything else** (continuous / mixed TPE search): parallel coordinates, which
  handle any dimensionality.

``tuning_plot(model)`` auto-routes by inspecting the spec; the three explicit
methods (``tuning_elbow_plot`` / ``tuning_heatmap`` / ``tuning_parallel_plot``)
let a caller force one and get a clear error if the search shape doesn't fit.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# Tolerance for the elbow's parsimony rule when per-trial CV standard error is
# not available in legacy report.db files. We accept any param value whose CV is
# within this fraction of the best, then take the smallest such value.
_ELBOW_RTOL = 0.01


class TuningMixin:
    """Hyperparameter-search views over a report.db ``trials`` table.

    Mixed into :class:`~DSSP2026.report.report.Report`. Relies on ``ReportBase``
    for ``_connect``, ``experiment_id``, ``models``, and ``_best_model_name``.
    """

    # ------------------------------------------------------------------
    # Loading: trials frame + search-space spec
    # ------------------------------------------------------------------

    def _model_id(self, model: str) -> int:
        from DSSP2026.report.base import ENSEMBLE_NAME
        if model == ENSEMBLE_NAME:
            raise ValueError(
                f"{ENSEMBLE_NAME!r} is a post-hoc combination of models, not a "
                f"tuned search — it has no trials to plot. Pass a real model.")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(
                f"no model {model!r} in experiment {self.experiment_id!r}.")
        return int(row[0])

    def trials_frame(self, model: str, *, completed_only: bool = True
                     ) -> pd.DataFrame:
        """Tidy trials for one model: one row per trial, params as columns.

        Columns are the union of param names across the model's trials plus
        ``cv_value``, ``cv_se``, ``trial_number``, ``rank``, ``is_best``.
        Conditional params absent from a trial are NaN. ``completed_only`` keeps
        only trials with a finite ``cv_value`` (the plottable ones).
        """
        mid = self._model_id(model)
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)")}
            cv_se_expr = "cv_se" if "cv_se" in cols else "NULL AS cv_se"
            rows = conn.execute(
                "SELECT trial_number, params, cv_value, "
                f"{cv_se_expr}, state, rank, is_best "
                "FROM trials WHERE model_id=? ORDER BY trial_number",
                (mid,)).fetchall()
        finally:
            conn.close()
        if not rows:
            raise ValueError(f"no trials persisted for model {model!r}.")

        records = []
        for r in rows:
            rec = {
                "trial_number": r["trial_number"],
                "cv_value": r["cv_value"],
                "cv_se": r["cv_se"],
                "state": r["state"],
                "rank": r["rank"],
                "is_best": bool(r["is_best"]),
            }
            params = json.loads(r["params"]) if r["params"] else {}
            # Tuples (e.g. MLP "hidden") arrive from JSON as lists; keep them
            # out of the flat axis columns — they aren't scalar plot axes.
            for k, v in params.items():
                if not isinstance(v, (list, dict)):
                    rec[k] = v
            records.append(rec)

        df = pd.DataFrame.from_records(records)
        if completed_only:
            df = df[pd.to_numeric(df["cv_value"], errors="coerce").notna()]
            df = df[np.isfinite(df["cv_value"].astype(float))]
        if df.empty:
            raise ValueError(
                f"no completed trials with a finite objective for {model!r}.")
        return df.reset_index(drop=True)

    def _param_importance(self, model: str) -> dict:
        """Persisted Optuna hyperparameter importances as ``{param: importance}``.

        Returns ``{}`` when none were stored (older report.db, too few trials, a
        single dimension, or a model with nothing to tune) — the parallel plot
        then falls back to column order. Tolerates a missing table on legacy dbs.
        """
        mid = self._model_id(model)
        conn = self._connect()
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "param_importance" not in tables:
                return {}
            rows = conn.execute(
                "SELECT param, importance FROM param_importance WHERE model_id=?",
                (mid,)).fetchall()
        finally:
            conn.close()
        return {r["param"]: float(r["importance"]) for r in rows}

    def _search_space(self, model: str) -> list:
        """The model's persisted search-space spec (list of param descriptors).

        Returns ``[]`` when the experiment predates search_space persistence or
        the model has nothing to tune; callers fall back to inferring shape from
        the trials frame in that case.
        """
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(experiments)")}
            if "search_space" not in cols:
                return []
            row = conn.execute(
                "SELECT search_space FROM experiments WHERE experiment_id=?",
                (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return []
        spaces = json.loads(row[0])
        if not isinstance(spaces, dict):
            return []
        entry = spaces.get(model, [])
        if isinstance(entry, dict):
            return entry.get("params", [])
        return entry

    def _tunable_param_cols(self, df: pd.DataFrame) -> list:
        """Param columns in a trials frame (exclude bookkeeping + feature_set).

        ``feature_set`` is sampled for every model but isn't a tuning axis in the
        elbow/heatmap sense, so it's excluded from shape detection; it remains a
        valid parallel-coordinates axis.
        """
        meta = {"trial_number", "cv_value", "cv_se", "state", "rank", "is_best"}
        return [c for c in df.columns if c not in meta and c != "feature_set"]

    # ------------------------------------------------------------------
    # Search-shape classification
    # ------------------------------------------------------------------

    def search_kind(self, model: str) -> str:
        """Classify a model's search as ``"elbow"``, ``"grid"``, or ``"parallel"``.

        Prefers the persisted spec; falls back to the trials frame. A single
        tunable int descriptor -> elbow. A grid model (``GRID_MODELS``) whose
        tunable descriptors are all discrete -> grid. Everything else ->
        parallel.
        """
        from DSSP2026.experiment.spaces import GRID_MODELS

        spec = self._search_space(model)
        tunable = [d for d in spec if d.get("type") != "group" and "name" in d]

        if spec:
            if len(tunable) == 1 and tunable[0].get("type") == "int":
                return "elbow"
            discrete = all(
                d.get("type") in ("int", "categorical") for d in tunable)
            if model in GRID_MODELS and discrete and tunable:
                return "grid"
            return "parallel"

        # No spec persisted: infer from the trials frame.
        df = self.trials_frame(model)
        cols = self._tunable_param_cols(df)
        if len(cols) == 1 and pd.api.types.is_integer_dtype(
                pd.to_numeric(df[cols[0]], errors="coerce").dropna()
                .apply(lambda x: int(x) if float(x).is_integer() else x)):
            return "elbow"
        if model in GRID_MODELS and cols:
            return "grid"
        return "parallel"

    # ------------------------------------------------------------------
    # Auto-dispatch
    # ------------------------------------------------------------------

    def tuning_plot(self, model: Optional[str] = None, **kwargs):
        """Render the tuning plot that fits ``model``'s search shape.

        Routes to the elbow curve, grid heatmap, or parallel-coordinates plot
        based on :meth:`search_kind`. ``model`` defaults to the best model.
        """
        model = model or self._best_model_name()
        kind = self.search_kind(model)
        if kind == "elbow":
            return self.tuning_elbow_plot(model, **kwargs)
        if kind == "grid":
            return self.tuning_heatmap(model, **kwargs)
        return self.tuning_parallel_plot(model, **kwargs)

    # ------------------------------------------------------------------
    # 1. Elbow (single integer dimension)
    # ------------------------------------------------------------------

    def tuning_elbow_plot(self, model: Optional[str] = None, *,
                          param: Optional[str] = None, figsize=None):
        """CV-vs-single-parameter elbow curve with the parsimony pick marked.

        Valid only for a one-dimensional integer search (e.g. Decision tree
        ``max_depth``). When per-trial ``cv_se`` is present, the marked elbow is
        the smallest param value within one standard error of the best trial,
        matching training-time selection. Legacy report.db files without
        ``cv_se`` fall back to the ``_ELBOW_RTOL`` approximation.
        """
        model = model or self._best_model_name()
        df = self.trials_frame(model)
        cols = self._tunable_param_cols(df)
        if param is None:
            if len(cols) != 1:
                raise ValueError(
                    f"elbow plot needs exactly one tunable parameter; {model!r} "
                    f"has {len(cols)} ({cols}). Pass param=... or use "
                    f"tuning_heatmap / tuning_parallel_plot.")
            param = cols[0]

        sub = df[[param, "cv_value", "cv_se", "is_best"]].copy()
        sub[param] = pd.to_numeric(sub[param], errors="coerce")
        sub = sub.dropna(subset=[param, "cv_value"])
        best_idx = sub["cv_value"].astype(float).idxmax()
        best_cv = float(sub.loc[best_idx, "cv_value"])
        best_se = pd.to_numeric(pd.Series([sub.loc[best_idx, "cv_se"]]),
                                errors="coerce").iloc[0]
        if pd.notna(best_se):
            threshold = best_cv - float(best_se)
        else:
            threshold = best_cv - _ELBOW_RTOL * abs(best_cv)
        # Collapse duplicate param values to their best CV (grid/TPE may repeat).
        agg = (sub.groupby(param, as_index=False)
               .agg(cv_value=("cv_value", "max"),
                    is_best=("is_best", "max"))
               .sort_values(param))

        qualifying = agg[agg["cv_value"] >= threshold]
        elbow_val = int(qualifying[param].min())

        from DSSP2026.report._tuning_plots import plot_elbow_curve
        return plot_elbow_curve(
            agg.rename(columns={param: "param"}), elbow=elbow_val,
            param_name=param, model=model, scoring=self._scoring_label(),
            figsize=figsize)

    # ------------------------------------------------------------------
    # 2. Grid heatmap (grid searches only)
    # ------------------------------------------------------------------

    def tuning_heatmap(self, model: Optional[str] = None, *,
                       x: Optional[str] = None, y: Optional[str] = None,
                       figsize=None):
        """Tile heatmap of CV over two grid axes — grid searches only.

        Refuses non-grid searches (a continuous TPE search's points don't lie on
        a lattice, so a heatmap would be misleading). With 3+ grid dimensions and
        no explicit ``x``/``y``, the two highest-variance axes are chosen (those
        that move the score most) and the remaining axes are sliced at the best
        trial's values — the 2-D grid slice that contains the winner.
        """
        model = model or self._best_model_name()
        if self.search_kind(model) != "grid":
            raise ValueError(
                f"tuning_heatmap is only valid for grid searches; {model!r} is "
                f"a {self.search_kind(model)!r} search. Use tuning_parallel_plot "
                f"or tuning_elbow_plot.")

        df = self.trials_frame(model)
        cols = self._tunable_param_cols(df)
        if len(cols) < 2:
            raise ValueError(
                f"heatmap needs at least two grid axes; {model!r} has {cols}.")

        # Choose the two axes. Explicit pair wins; else rank by the variance of
        # mean CV across each axis's levels (how much that axis moves the score).
        if x is not None and y is not None:
            ax_x, ax_y = x, y
        elif x is not None or y is not None:
            raise ValueError("pass both x and y, or neither.")
        else:
            ax_x, ax_y = self._rank_axes_by_influence(df, cols)[:2]

        # Slice the other axes at the best trial's values: the grid containing
        # the winner, rather than an aggregate that smears non-shown dims.
        best_row = df.loc[df["is_best"].astype(bool)]
        if best_row.empty:
            best_row = df.loc[[df["cv_value"].astype(float).idxmax()]]
        best_row = best_row.iloc[0]
        sliced = df
        for c in cols:
            if c not in (ax_x, ax_y):
                sliced = sliced[sliced[c].astype(str) == str(best_row[c])]

        # Pivot to a (y x x) grid of CV values. Duplicate cells -> max.
        grid = (sliced.pivot_table(index=ax_y, columns=ax_x,
                                   values="cv_value", aggfunc="max"))
        # Sort axes naturally (numeric where possible) for readable rails.
        grid = grid.reindex(index=_natural_sort(grid.index),
                            columns=_natural_sort(grid.columns))

        # Highlight the winner's cell (its position in the sorted grid).
        try:
            hi_r = list(grid.index).index(best_row[ax_y]) if best_row[ax_y] in \
                list(grid.index) else None
            hi_c = list(grid.columns).index(best_row[ax_x]) if best_row[ax_x] in \
                list(grid.columns) else None
            highlight = (hi_r, hi_c) if hi_r is not None and hi_c is not None \
                else None
        except Exception:
            highlight = None

        from DSSP2026.core.heatmap import plot_tile_grid
        values = grid.to_numpy(dtype=float)
        cell_text = np.where(np.isfinite(values),
                             np.vectorize(lambda v: f"{v:.3f}")(values), "")
        sliced_note = ", ".join(
            f"{c}={best_row[c]}" for c in cols if c not in (ax_x, ax_y))
        title = f"{model} — grid search CV ({self._scoring_label()})"
        if sliced_note:
            title += f"\n(sliced at best: {sliced_note})"
        return plot_tile_grid(
            values,
            row_labels=[str(v) for v in grid.index],
            col_labels=[str(v) for v in grid.columns],
            row_axis_label=ax_y, col_axis_label=ax_x,
            title=title, cell_text=cell_text,
            colorbar_label=self._scoring_label(), highlight=highlight,
            figsize=figsize)

    @staticmethod
    def _rank_axes_by_influence(df: pd.DataFrame, cols: Sequence[str]) -> list:
        """Order grid axes by descending score influence (variance of level means).

        For each axis, group by its levels and take the mean CV per level; the
        variance of those level-means measures how much moving that one axis
        moves the score. Highest-variance axes are the most informative to plot.
        """
        scored = []
        for c in cols:
            level_means = df.groupby(c)["cv_value"].mean()
            scored.append((c, float(level_means.var(ddof=0))
                           if len(level_means) > 1 else 0.0))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [c for c, _ in scored]

    # ------------------------------------------------------------------
    # 3. Parallel coordinates (any dimensionality)
    # ------------------------------------------------------------------

    def tuning_parallel_plot(self, model: Optional[str] = None, *,
                             params: Optional[Sequence[str]] = None,
                             drop: Optional[Sequence[str]] = None,
                             figsize=None, curved: bool = True):
        """Parallel-coordinates plot of the search — works for any dimensionality.

        Builds the tidy (params + ``cv_value``) frame from report.db and feeds
        the source-agnostic renderer. Without a live study, axis ordering falls
        back to DataFrame column order (no Optuna importances available offline).
        """
        model = model or self._best_model_name()
        df = self.trials_frame(model)
        cols = self._tunable_param_cols(df) + (
            ["feature_set"] if "feature_set" in df.columns else [])
        frame = df[cols + ["cv_value"]].copy()

        from DSSP2026.experiment.tuning.optuna_parallel import \
            plot_parallel_coordinates_from_frame
        return plot_parallel_coordinates_from_frame(
            frame, value_col="cv_value", params=params, drop=drop,
            importances=self._param_importance(model),
            metric_label=f"Objective (CV {self._scoring_label()})",
            title=f"{model} — hyperparameter search (parallel coordinates)",
            figsize=figsize, curved=curved)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _scoring_label(self) -> str:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT scoring FROM experiments WHERE experiment_id=?",
                (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        return (row[0] if row and row[0] else "CV score")


def _natural_sort(values):
    """Sort index/column labels numerically when all look numeric, else as str.

    Grid axes like ``max_features=[2, 3, 5, "sqrt", "log2"]`` mix ints and
    strings; pure-numeric axes (n_estimators) sort numerically, mixed ones sort
    by string so the call never raises on heterogeneous labels.
    """
    vals = list(values)
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def save_tuning_elbow_png(report, model, path, *, dpi: int = 220, **kwargs):
    from DSSP2026.core.figure import save_figure
    return save_figure(report.tuning_elbow_plot(model, **kwargs), path, dpi=dpi)


def save_tuning_heatmap_png(report, model, path, *, dpi: int = 220, **kwargs):
    from DSSP2026.core.figure import save_figure
    return save_figure(report.tuning_heatmap(model, **kwargs), path, dpi=dpi)


def save_tuning_parallel_png(report, model, path, *, dpi: int = 220, **kwargs):
    from DSSP2026.core.figure import save_figure
    return save_figure(report.tuning_parallel_plot(model, **kwargs), path, dpi=dpi)


def save_tuning_plot_png(report, model, path, *, dpi: int = 220, **kwargs):
    kind = report.search_kind(model)
    if kind == "elbow":
        return save_tuning_elbow_png(report, model, path, dpi=dpi, **kwargs)
    if kind == "grid":
        return save_tuning_heatmap_png(report, model, path, dpi=dpi, **kwargs)
    return save_tuning_parallel_png(report, model, path, dpi=dpi, **kwargs)
