"""
workflows/study_adapter.py — uniform read interface over two study schemas.

The cost dashboard reads model-comparison artifacts from a SQLite study file.
Two schemas exist in the wild:

- **study_db** (``workflows/study_db.py``): ``runs`` + ``results`` keyed by
  ``run_id`` / ``result_id``; child tables ``confusion`` / ``predictions`` /
  ``per_class`` reference ``result_id``; ``results`` carries
  ``is_best_for_model``. This is the multiclass TeleLogs artifact.

- **experiment-db**: ``experiments`` + ``models`` keyed by ``experiment_id`` /
  ``model_id``; the same child tables reference ``model_id``; no
  ``is_best_for_model`` column (each model name appears once per experiment, so
  every row is already "best"). This is what the binary report.db uses.

This adapter sniffs which schema a database file uses and exposes one canonical
interface in the ``study_db`` vocabulary (``run_id`` / ``result_id``), so the
dashboard needs no schema-specific branches. Canonical identifiers are surfaced
as integers; for the experiment schema, the string ``experiment_id`` is mapped
to a stable integer index for the run picker while ``model_id`` (already an int)
is used directly as ``result_id``.

Public interface (all take a db path):
    detect_schema(db_path)                  -> "study" | "experiment"
    list_runs(db_path)                      -> DataFrame[run_id, timestamp, eval_kind, n_results]
    list_results(db_path, run_id, best_only)-> DataFrame[result_id, model, feature_set, f1, is_best_for_model]
    confusion_matrix(db_path, result_id)    -> wide (true x pred) int DataFrame
    read_predictions(db_path, result_id)    -> (class_order, y_true, y_proba) | None
    has_predictions(db_path)                -> bool
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

SCHEMA_STUDY = "study"
SCHEMA_EXPERIMENT = "experiment"


def _tables(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def detect_schema(db_path: PathLike) -> str:
    """Return which schema the database uses.

    Prefers the study schema when both appear (it is the native artifact). Raises
    ValueError if neither ``results`` nor ``models`` is present.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        t = _tables(conn)
    finally:
        conn.close()
    if {"runs", "results"} <= t:
        return SCHEMA_STUDY
    if {"experiments", "models"} <= t:
        return SCHEMA_EXPERIMENT
    raise ValueError(
        f"{db_path} matches no known study schema (tables: {sorted(t)})")


# ---------------------------------------------------------------------------
# Run listing
# ---------------------------------------------------------------------------

def list_runs(db_path: PathLike) -> pd.DataFrame:
    """One row per run/experiment: run_id, timestamp, eval_kind, n_results.

    For the experiment schema the string ``experiment_id`` is preserved in a
    hidden ``_native_id`` column and a stable integer ``run_id`` is assigned by
    timestamp/order so the dashboard's int-keyed selectbox works unchanged.
    """
    schema = detect_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        if schema == SCHEMA_STUDY:
            df = pd.read_sql_query(
                "SELECT r.run_id AS run_id, r.timestamp AS timestamp, "
                "r.eval_kind AS eval_kind, COUNT(*) AS n_results "
                "FROM results res JOIN runs r ON r.run_id = res.run_id "
                "GROUP BY r.run_id ORDER BY r.run_id DESC", conn)
            df["_native_id"] = df["run_id"]
            return df
        # experiment schema
        df = pd.read_sql_query(
            "SELECT e.experiment_id AS _native_id, e.timestamp AS timestamp, "
            "e.eval_kind AS eval_kind, COUNT(*) AS n_results "
            "FROM models m JOIN experiments e "
            "ON e.experiment_id = m.experiment_id "
            "GROUP BY e.experiment_id ORDER BY e.timestamp DESC, "
            "e.experiment_id DESC", conn)
    finally:
        conn.close()
    # Assign stable integer run_ids (1-based, in displayed order).
    df.insert(0, "run_id", range(1, len(df) + 1))
    return df


def _native_run_id(db_path: PathLike, run_id: int):
    """Map a canonical integer run_id back to the native experiment_id (string)."""
    runs = list_runs(db_path)
    match = runs.loc[runs["run_id"] == run_id, "_native_id"]
    if match.empty:
        raise KeyError(f"no run with run_id={run_id}")
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Result listing
# ---------------------------------------------------------------------------

def list_results(db_path: PathLike, run_id: int,
                 best_only: bool = True) -> pd.DataFrame:
    """Results for a run: result_id, model, feature_set, f1, is_best_for_model.

    ``best_only`` filters to the best feature set per model. The experiment
    schema has one row per model already, so every row is treated as best and
    the flag is a no-op there.
    """
    schema = detect_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        if schema == SCHEMA_STUDY:
            q = ("SELECT res.result_id AS result_id, res.model AS model, "
                 "res.feature_set AS feature_set, res.f1 AS f1, "
                 "res.is_best_for_model AS is_best_for_model "
                 "FROM results res WHERE res.run_id = ?")
            params = [run_id]
            if best_only:
                q += " AND res.is_best_for_model = 1"
            q += " ORDER BY res.f1 DESC"
            return pd.read_sql_query(q, conn, params=params)
        # experiment schema
        native = _native_run_id(db_path, run_id)
        df = pd.read_sql_query(
            "SELECT m.model_id AS result_id, m.model AS model, "
            "m.feature_set AS feature_set, m.f1 AS f1 "
            "FROM models m WHERE m.experiment_id = ? ORDER BY m.f1 DESC",
            conn, params=[native])
    finally:
        conn.close()
    # Every experiment-schema row is its own best (one row per model name).
    df["is_best_for_model"] = 1
    return df


# ---------------------------------------------------------------------------
# Confusion / predictions  (child tables are keyed by result_id or model_id;
# the column name differs but the contents are identical in shape)
# ---------------------------------------------------------------------------

def _child_key(schema: str) -> str:
    return "result_id" if schema == SCHEMA_STUDY else "model_id"


def confusion_matrix(db_path: PathLike, result_id: int) -> pd.DataFrame:
    """Wide (true x pred) integer confusion matrix for a result/model."""
    schema = detect_schema(db_path)
    key = _child_key(schema)
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            f"SELECT true_label, pred_label, count FROM confusion "
            f"WHERE {key} = ?", conn, params=(result_id,))
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="true_label", columns="pred_label", values="count")
    # Square the matrix over the union of labels so true/pred axes match even
    # when a class never appears on one side (common in binary, imbalanced data).
    labels = sorted(set(wide.index) | set(wide.columns), key=_label_sort_key)
    wide = wide.reindex(index=labels, columns=labels)
    wide.index.name = "True"
    wide.columns.name = "Predicted"
    return wide.fillna(0).astype(int)


def read_predictions(db_path: PathLike, result_id: int):
    """Return ``(class_order, y_true, y_proba)`` for a result, or None."""
    schema = detect_schema(db_path)
    key = _child_key(schema)
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            f"SELECT class_order, y_true, y_proba FROM predictions "
            f"WHERE {key} = ?", (result_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    class_order = json.loads(row[0])
    y_true = np.asarray(json.loads(row[1]), dtype=object)
    y_proba = np.asarray(json.loads(row[2]), dtype=float)
    return class_order, y_true, y_proba


def id2label(db_path: PathLike, run_id: int) -> dict:
    """Display-name map ``{stored_label(str) -> friendly name}`` for a run.

    Only the experiment schema records this (written by ``Experiment``/
    ``report_builder`` into ``experiments.id2label``). Returns ``{}`` for the
    study schema, when the column predates this feature, or when no map was
    registered — so callers can always relabel unconditionally (identity when
    empty). Keys are normalised to str to match the stored labels, which are
    always stringified ("0"/"1", "C1"...).
    """
    schema = detect_schema(db_path)
    if schema != SCHEMA_EXPERIMENT:
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
        if "id2label" not in cols:
            return {}
        native = _native_run_id(db_path, run_id)
        row = conn.execute(
            "SELECT id2label FROM experiments WHERE experiment_id=?",
            (native,)).fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return {}
    return {str(k): str(v) for k, v in json.loads(row[0]).items()}


def has_predictions(db_path: PathLike) -> bool:
    """True if a predictions table exists and has any rows."""
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


def _label_sort_key(label):
    """Sort labels naturally: numeric where possible, else lexicographic.

    Keeps binary '0'/'1' in order and C1..C10 sensibly ordered.
    """
    s = str(label)
    if s.isdigit():
        return (0, int(s), s)
    if s and s[0] in "Cc" and s[1:].isdigit():
        return (0, int(s[1:]), s)
    return (1, 0, s)