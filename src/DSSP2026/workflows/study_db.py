"""
workflows/study_db.py — durable study artifact in SQLite.

Each CLI invocation appends one ``runs`` row plus one ``results`` row per
(model x feature set) actually scored — not just the best-per-model rows that
reach ``master_comparison.csv``. Per-class metrics (C1..C8 + the average rows)
and full confusion-matrix counts hang off each result so a downstream Streamlit
dashboard can reproduce every plot the workflow already renders, straight from
the database.

Design
------
- stdlib ``sqlite3`` only; no new dependencies.
- Append-only. ``CREATE TABLE IF NOT EXISTS`` is idempotent; the first run
  creates the file. History accumulates across runs.
- ``record_run`` writes everything in a single transaction and rolls back on
  error, so a partial run never lands.
- Long-format ``confusion`` (one row per true x pred cell) pivots back to an
  8x8 matrix trivially and is far easier to query than a serialized blob.

Public API
----------
    init_db(db_path)                       -> sqlite3.Connection
    record_run(db_path, run_meta, result_rows) -> int   (the new run_id)
    read_results(db_path, query=None, params=())        -> pandas.DataFrame
    read_run(db_path, run_id)              -> dict        (run row + nested results)
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    eval_kind        TEXT    NOT NULL,
    models           TEXT    NOT NULL,
    feature_sets     TEXT    NOT NULL,
    n_trials         INTEGER,
    train_file       TEXT,
    test_file        TEXT,
    random_state     INTEGER,
    cv_splits        INTEGER,
    average          TEXT,
    scoring          TEXT,
    validation_ratio REAL,
    git_commit       TEXT,
    cli_args         TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model             TEXT    NOT NULL,
    feature_set       TEXT    NOT NULL,
    detail            TEXT,
    accuracy          REAL,
    "precision"       REAL,
    recall            REAL,
    f1                REAL,
    roc_auc           REAL,
    threshold         REAL,
    hyperparams       TEXT,
    feature_list      TEXT,
    is_best_for_model INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS per_class (
    result_id    INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    class_label  TEXT    NOT NULL,
    "precision"  REAL,
    recall       REAL,
    f1           REAL,
    support      REAL
);

CREATE TABLE IF NOT EXISTS confusion (
    result_id   INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    true_label  TEXT    NOT NULL,
    pred_label  TEXT    NOT NULL,
    count       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    source          TEXT    NOT NULL,    -- 'optuna' | 'grid' | 'depth_sweep'
    trial_number    INTEGER,             -- 0-based within the search
    params          TEXT,                -- JSON of the candidate's parameters
    objective_value REAL,                -- the CV score being maximised
    objective_std   REAL,                -- across-fold std where available; else NULL
    state           TEXT,                -- 'COMPLETE'/'PRUNED'/'FAIL' (optuna) or 'COMPLETE'
    rank            INTEGER,             -- 1 = best; from grid rank or value ordering
    is_best         INTEGER NOT NULL DEFAULT 0,
    duration_sec    REAL                 -- optuna only; else NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    result_id    INTEGER PRIMARY KEY REFERENCES results(result_id) ON DELETE CASCADE,
    class_order  TEXT NOT NULL,   -- JSON list: label for each proba-matrix column
    y_true       TEXT NOT NULL,   -- JSON list: true label per eval-set sample
    y_proba      TEXT NOT NULL,   -- JSON list-of-lists: n_samples x n_classes
    n_samples    INTEGER NOT NULL,
    n_classes    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run      ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_perclass_result  ON per_class(result_id);
CREATE INDEX IF NOT EXISTS idx_confusion_result ON confusion(result_id);
CREATE INDEX IF NOT EXISTS idx_trials_result    ON trials(result_id);
"""

# Column order for the runs / results inserts (kept explicit so the dict->row
# mapping is unambiguous and reorderable without touching SQL).
_RUN_COLS = (
    "timestamp", "eval_kind", "models", "feature_sets", "n_trials",
    "train_file", "test_file", "random_state", "cv_splits", "average",
    "scoring", "validation_ratio", "git_commit", "cli_args",
)
_RESULT_COLS = (
    "model", "feature_set", "detail", "accuracy", "precision", "recall",
    "f1", "roc_auc", "threshold", "hyperparams", "feature_list",
    "is_best_for_model",
)

# Mapping from the metrics dict keys produced by ``classification_metrics``
# to the flat ``results`` columns.
_METRIC_KEY_MAP = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC-AUC": "roc_auc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Coerce numpy / pandas scalars and containers into JSON-native types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(_jsonable(obj))


def _git_commit() -> Optional[str]:
    """Best-effort short commit hash; None if git or repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def _to_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Connection / schema
# ---------------------------------------------------------------------------

def init_db(db_path: PathLike) -> sqlite3.Connection:
    """Open (creating if needed) the study DB and ensure the schema exists."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_run(
    db_path: PathLike,
    run_meta: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
) -> int:
    """Append one run and its results in a single transaction; return run_id.

    Parameters
    ----------
    run_meta : mapping
        Run-level provenance. Recognised keys mirror ``_RUN_COLS`` minus
        ``timestamp`` and ``git_commit`` (both filled in here if absent):
        eval_kind, models, feature_sets, n_trials, train_file, test_file,
        random_state, cv_splits, average, scoring, validation_ratio, cli_args.
        ``models`` / ``feature_sets`` / ``cli_args`` may be passed as native
        Python objects; they are JSON-serialised automatically.
    result_rows : sequence of mappings
        One per (model x feature set). Recognised keys:
            model, feature_set, detail, metrics (dict), threshold,
            hyperparams (dict), feature_list (list), is_best_for_model (bool),
            per_class_df (DataFrame), confusion_df (DataFrame, raw counts).
        ``metrics`` is exploded into the flat metric columns; ``per_class_df``
        and ``confusion_df`` are written to the child tables.
    """
    conn = init_db(db_path)
    try:
        with conn:  # transaction: commits on success, rolls back on exception
            run_id = _insert_run(conn, run_meta)
            for row in result_rows:
                result_id = _insert_result(conn, run_id, row)
                _insert_per_class(conn, result_id, row.get("per_class_df"))
                _insert_confusion(conn, result_id, row.get("confusion_df"))
                _insert_trials(conn, result_id, row.get("trials"))
                _insert_predictions(conn, result_id, row.get("predictions"))
        return run_id
    finally:
        conn.close()


def _insert_run(conn: sqlite3.Connection, meta: Mapping[str, Any]) -> int:
    values = {
        "timestamp": meta.get("timestamp")
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_kind": meta.get("eval_kind"),
        "models": _dumps(meta.get("models")),
        "feature_sets": _dumps(meta.get("feature_sets")),
        "n_trials": meta.get("n_trials"),
        "train_file": _str_or_none(meta.get("train_file")),
        "test_file": _str_or_none(meta.get("test_file")),
        "random_state": meta.get("random_state"),
        "cv_splits": meta.get("cv_splits"),
        "average": meta.get("average"),
        "scoring": meta.get("scoring"),
        "validation_ratio": meta.get("validation_ratio"),
        "git_commit": meta.get("git_commit", _git_commit()),
        "cli_args": _dumps(meta.get("cli_args")),
    }
    placeholders = ", ".join("?" for _ in _RUN_COLS)
    cur = conn.execute(
        f'INSERT INTO runs ({", ".join(_RUN_COLS)}) VALUES ({placeholders})',
        tuple(values[c] for c in _RUN_COLS),
    )
    return int(cur.lastrowid)


def _insert_result(conn: sqlite3.Connection, run_id: int,
                   row: Mapping[str, Any]) -> int:
    metrics = row.get("metrics") or {}
    flat = {dest: _to_float(metrics.get(src))
            for src, dest in _METRIC_KEY_MAP.items()}
    values = {
        "model": row.get("model"),
        "feature_set": row.get("feature_set"),
        "detail": row.get("detail"),
        "accuracy": flat.get("accuracy"),
        "precision": flat.get("precision"),
        "recall": flat.get("recall"),
        "f1": flat.get("f1"),
        "roc_auc": flat.get("roc_auc"),
        "threshold": _to_float(row.get("threshold")),
        "hyperparams": _dumps(row.get("hyperparams")),
        "feature_list": _dumps(row.get("feature_list")),
        "is_best_for_model": int(bool(row.get("is_best_for_model", False))),
    }
    # results columns include run_id up front.
    cols = ("run_id",) + _RESULT_COLS
    placeholders = ", ".join("?" for _ in cols)
    quoted = ", ".join(f'"{c}"' if c == "precision" else c for c in cols)
    params = (run_id,) + tuple(values[c] for c in _RESULT_COLS)
    cur = conn.execute(
        f"INSERT INTO results ({quoted}) VALUES ({placeholders})", params)
    return int(cur.lastrowid)


def _insert_per_class(conn, result_id, per_class_df):
    if per_class_df is None or len(per_class_df) == 0:
        return
    df = per_class_df
    rows = [
        (result_id, str(r["Class"]),
         _to_float(r.get("Precision")), _to_float(r.get("Recall")),
         _to_float(r.get("F1")), _to_float(r.get("Support")))
        for _, r in df.iterrows()
    ]
    conn.executemany(
        'INSERT INTO per_class (result_id, class_label, "precision", recall, '
        "f1, support) VALUES (?, ?, ?, ?, ?, ?)", rows)


def _insert_confusion(conn, result_id, confusion_df):
    if confusion_df is None or len(confusion_df) == 0:
        return
    cm = confusion_df
    rows = []
    for true_label in cm.index:
        for pred_label in cm.columns:
            rows.append((result_id, str(true_label), str(pred_label),
                         int(cm.loc[true_label, pred_label])))
    conn.executemany(
        "INSERT INTO confusion (result_id, true_label, pred_label, count) "
        "VALUES (?, ?, ?, ?)", rows)


def _insert_trials(conn, result_id, trials):
    """Insert a list of normalised trial dicts (from the trials_from_* helpers)."""
    if not trials:
        return
    rows = [
        (result_id, t.get("source"), t.get("trial_number"),
         _dumps(t.get("params")), _to_float(t.get("objective_value")),
         _to_float(t.get("objective_std")), t.get("state"),
         t.get("rank"), int(bool(t.get("is_best", False))),
         _to_float(t.get("duration_sec")))
        for t in trials
    ]
    conn.executemany(
        "INSERT INTO trials (result_id, source, trial_number, params, "
        "objective_value, objective_std, state, rank, is_best, duration_sec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_predictions(conn, result_id, predictions):
    """Insert one result's prediction block.

    ``predictions`` is a dict with keys ``class_order`` (list of labels, one per
    proba column), ``y_true`` (per-sample true labels), and ``y_proba``
    (n_samples x n_classes array/list). Stored as compact JSON so the full
    matrix round-trips for ROC curves, threshold tuning, and recomputed
    confusion matrices. Skipped if no probabilities were supplied.
    """
    if not predictions:
        return
    class_order = predictions.get("class_order")
    y_true = predictions.get("y_true")
    y_proba = predictions.get("y_proba")
    if class_order is None or y_true is None or y_proba is None:
        return
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2:
        return
    n_samples, n_classes = proba.shape
    conn.execute(
        "INSERT OR REPLACE INTO predictions "
        "(result_id, class_order, y_true, y_proba, n_samples, n_classes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (result_id,
         json.dumps([str(c) for c in class_order]),
         json.dumps([str(v) for v in (y_true.tolist()
                     if hasattr(y_true, "tolist") else list(y_true))]),
         json.dumps(proba.tolist()),
         int(n_samples), int(n_classes)))


def _str_or_none(v):
    return None if v is None else str(v)


# ---------------------------------------------------------------------------
# Trial extraction — normalise each family's search output to common rows
# ---------------------------------------------------------------------------
# A "trial" row is: {source, trial_number, params, objective_value,
# objective_std, state, rank, is_best, duration_sec}. Each extractor below maps
# one family's native tuning object into a list of these dicts. Runners call the
# matching extractor and attach the list to the result row under "trials".

def trials_from_optuna(study) -> list:
    """Every trial in an Optuna study (MLP / XGB). Source = 'optuna'.

    Pulls params, value, state, and duration directly from the study object so
    the data matches the per-family optuna_*.db exactly. The best trial (by the
    study's own ``best_trial``) is flagged; rank is by descending value among
    completed trials.
    """
    try:
        best_number = study.best_trial.number
    except Exception:
        best_number = None

    # Rank completed trials by value (direction is maximize for these searches).
    completed = [t for t in study.trials
                 if t.value is not None and str(t.state).endswith("COMPLETE")]
    order = sorted(completed, key=lambda t: t.value, reverse=True)
    rank_of = {t.number: i + 1 for i, t in enumerate(order)}

    rows = []
    for t in study.trials:
        duration = None
        if getattr(t, "datetime_start", None) and getattr(t, "datetime_complete", None):
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        rows.append({
            "source": "optuna",
            "trial_number": t.number,
            "params": dict(t.params),
            "objective_value": _to_float(t.value),
            "objective_std": None,
            "state": str(t.state).split(".")[-1],   # 'TrialState.COMPLETE' -> 'COMPLETE'
            "rank": rank_of.get(t.number),
            "is_best": (t.number == best_number),
            "duration_sec": duration,
        })
    return rows


def trials_from_rf_grid(rf_grid_result) -> list:
    """Every grid point from an RF GridSearchCV (RFGridResult). Source = 'grid'.

    Reads the sklearn ``cv_results_`` dict: one trial per param combination, with
    mean/std test score and the grid's own rank. is_best = rank 1.
    """
    cv = rf_grid_result.cv_results
    params = cv.get("params", [])
    means = cv.get("mean_test_score", [])
    stds = cv.get("std_test_score", [])
    ranks = cv.get("rank_test_score", [])
    rows = []
    for i, p in enumerate(params):
        rank = int(ranks[i]) if i < len(ranks) else None
        rows.append({
            "source": "grid",
            "trial_number": i,
            "params": dict(p),
            "objective_value": _to_float(means[i]) if i < len(means) else None,
            "objective_std": _to_float(stds[i]) if i < len(stds) else None,
            "state": "COMPLETE",
            "rank": rank,
            "is_best": (rank == 1),
            "duration_sec": None,
        })
    return rows


def trials_from_depth_sweep(depth_tune_result) -> list:
    """Every depth from a tree depth sweep (DepthTuneResult). Source = 'depth_sweep'.

    One trial per candidate max_depth, with mean/SE CV score. is_best flags the
    selected ``one_se_depth`` (the parsimonious choice the workflow actually
    fits), not merely the highest-mean depth; rank is by descending mean score.
    """
    df = depth_tune_result.results_df.copy()
    chosen = depth_tune_result.one_se_depth
    # Rank by mean_score descending (1 = highest mean).
    df = df.reset_index(drop=True)
    order = df["mean_score"].rank(ascending=False, method="min").astype(int)
    rows = []
    for i, r in df.iterrows():
        rows.append({
            "source": "depth_sweep",
            "trial_number": int(r["max_depth"]),   # depth is the natural index
            "params": {"max_depth": int(r["max_depth"])},
            "objective_value": _to_float(r["mean_score"]),
            "objective_std": _to_float(r.get("se_score")),
            "state": "COMPLETE",
            "rank": int(order.iloc[i]),
            "is_best": (int(r["max_depth"]) == int(chosen)),
            "duration_sec": None,
        })
    return rows


# ---------------------------------------------------------------------------
# Read (thin helpers for the dashboard / notebooks)
# ---------------------------------------------------------------------------

def read_results(db_path: PathLike, query: Optional[str] = None,
                 params: Sequence = ()) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Defaults to runs JOIN results."""
    if query is None:
        query = (
            "SELECT r.run_id, r.timestamp, r.eval_kind, res.* "
            "FROM results res JOIN runs r ON r.run_id = res.run_id "
            "ORDER BY r.run_id, res.f1 DESC"
        )
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def read_run(db_path: PathLike, run_id: int) -> dict:
    """Return one run's metadata plus its nested results / per_class / confusion."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise KeyError(f"no run with run_id={run_id}")
        results = conn.execute(
            "SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchall()
        out = {"run": dict(run), "results": []}
        for res in results:
            rid = res["result_id"]
            pc = conn.execute(
                "SELECT * FROM per_class WHERE result_id = ?", (rid,)).fetchall()
            cm = conn.execute(
                "SELECT * FROM confusion WHERE result_id = ?", (rid,)).fetchall()
            out["results"].append({
                **dict(res),
                "per_class": [dict(x) for x in pc],
                "confusion": [dict(x) for x in cm],
            })
        return out
    finally:
        conn.close()


def read_trials(db_path: PathLike, run_id: Optional[int] = None,
                model: Optional[str] = None) -> pd.DataFrame:
    """Tuning trials joined to their result (model / feature_set / run).

    Optionally filter by ``run_id`` and/or ``model``. ``params`` stays as JSON
    text; parse per-row as needed. Ordered by result then trial number.
    """
    sql = (
        "SELECT r.run_id, res.model, res.feature_set, t.* "
        "FROM trials t "
        "JOIN results res ON res.result_id = t.result_id "
        "JOIN runs r       ON r.run_id     = res.run_id"
    )
    clauses, params = [], []
    if run_id is not None:
        clauses.append("r.run_id = ?"); params.append(run_id)
    if model is not None:
        clauses.append("res.model = ?"); params.append(model)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY t.result_id, t.trial_number"
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def confusion_matrix(db_path: PathLike, result_id: int) -> pd.DataFrame:
    """Reconstruct the wide (true x pred) integer confusion matrix for a result."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT true_label, pred_label, count FROM confusion "
            "WHERE result_id = ?", conn, params=(result_id,))
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="true_label", columns="pred_label", values="count")
    wide.index.name = "True"
    wide.columns.name = "Predicted"
    return wide.fillna(0).astype(int)


def read_predictions(db_path: PathLike, result_id: int):
    """Return ``(class_order, y_true, y_proba)`` for a result, or None.

    ``class_order`` is a list of class labels (one per proba column); ``y_true``
    is a numpy array of per-sample true labels (as strings); ``y_proba`` is the
    n_samples x n_classes float matrix. Returns None if no prediction block was
    stored for this result.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT class_order, y_true, y_proba FROM predictions "
            "WHERE result_id = ?", (result_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    class_order = json.loads(row[0])
    y_true = np.asarray(json.loads(row[1]), dtype=object)
    y_proba = np.asarray(json.loads(row[2]), dtype=float)
    return class_order, y_true, y_proba


def has_predictions(db_path: PathLike) -> bool:
    """True if the database has a predictions table with any rows."""
    conn = sqlite3.connect(str(db_path))
    try:
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='predictions'").fetchone()
        if not tbl:
            return False
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        return n > 0
    finally:
        conn.close()


def roc_curves_for_run(db_path: PathLike, run_id: int, *, average="macro",
                       best_only=True):
    """Compute one-vs-rest ROC curves per result for a run.

    Returns a list of dicts (one per result that has stored predictions):
        {result_id, model, feature_set, label, fpr, tpr, auc}
    where ``fpr``/``tpr`` are arrays for the chosen ``average`` ('micro' or
    'macro') one-vs-rest ROC, and ``auc`` is the corresponding area. ``label``
    is a display string ("Model (feature_set)"). When ``best_only`` is True
    only the best-per-family results are included.

    Curve math lives here (data layer) so the dashboard stays presentation-only.
    """
    from sklearn.metrics import roc_curve, auc as _auc
    from sklearn.preprocessing import label_binarize

    conn = sqlite3.connect(str(db_path))
    try:
        q = ("SELECT res.result_id, res.model, res.feature_set "
             "FROM results res WHERE res.run_id = ?")
        params = [run_id]
        if best_only:
            q += " AND res.is_best_for_model = 1"
        q += " ORDER BY res.f1 DESC"
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    out = []
    for result_id, model, feature_set in rows:
        pred = read_predictions(db_path, result_id)
        if pred is None:
            continue
        class_order, y_true, y_proba = pred
        Y = label_binarize(y_true, classes=class_order)
        # Guard: binarize collapses to a single column if only 2 classes appear.
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        if average == "micro":
            fpr, tpr, _ = roc_curve(Y.ravel(), y_proba.ravel())
            roc_auc = _auc(fpr, tpr)
        else:  # macro: average per-class curves on a common FPR grid
            grid = np.linspace(0.0, 1.0, 200)
            tprs = []
            for k in range(Y.shape[1]):
                if Y[:, k].sum() == 0:
                    continue  # class absent in eval set
                fk, tk, _ = roc_curve(Y[:, k], y_proba[:, k])
                tprs.append(np.interp(grid, fk, tk))
            if not tprs:
                continue
            tpr = np.mean(tprs, axis=0)
            tpr[0] = 0.0
            fpr = grid
            roc_auc = _auc(fpr, tpr)

        out.append({
            "result_id": result_id, "model": model, "feature_set": feature_set,
            "label": f"{model} ({feature_set})",
            "fpr": fpr, "tpr": tpr, "auc": float(roc_auc),
        })
    # Sort by AUC descending so the legend leads with the strongest model.
    out.sort(key=lambda d: d["auc"], reverse=True)
    return out