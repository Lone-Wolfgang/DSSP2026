"""
experiment/sidecar.py — the eval sidecar (experiment_eval.db).

experiment.db is kept pure Optuna (trials, params, values, loss curves). The
held-out evaluation of each study's winning config — metrics, per-class,
confusion counts, and the full probability matrix — lives only in the in-memory
EvalRecords after a run. This sidecar persists that, in its own SQLite file next
to experiment.db, so the reporting layer can rebuild report.db from files alone
(sidecar + experiment.db) without re-running the experiment.

Keyed by (experiment_id, model). Probability matrices are stored as JSON text
(human-inspectable; matches the prior study.db convention).

Schema
------
eval_models(experiment_id, model, feature_set, detail, best_cv_value,
            accuracy, precision, recall, f1, roc_auc, hyperparams, feature_list)
eval_per_class(experiment_id, model, class_label, precision, recall, f1, support)
eval_confusion(experiment_id, model, true_label, pred_label, count)
eval_predictions(experiment_id, model, class_order, y_true, y_proba,
                 n_samples, n_classes)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_models (
    experiment_id TEXT NOT NULL,
    model         TEXT NOT NULL,
    feature_set   TEXT,
    detail        TEXT,
    best_cv_value REAL,
    accuracy      REAL,
    "precision"   REAL,
    recall        REAL,
    f1            REAL,
    roc_auc       REAL,
    hyperparams   TEXT,
    feature_list  TEXT,
    PRIMARY KEY (experiment_id, model)
);
CREATE TABLE IF NOT EXISTS eval_per_class (
    experiment_id TEXT NOT NULL,
    model         TEXT NOT NULL,
    class_label   TEXT NOT NULL,
    "precision"   REAL,
    recall        REAL,
    f1            REAL,
    support       REAL
);
CREATE TABLE IF NOT EXISTS eval_confusion (
    experiment_id TEXT NOT NULL,
    model         TEXT NOT NULL,
    true_label    TEXT NOT NULL,
    pred_label    TEXT NOT NULL,
    count         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_predictions (
    experiment_id TEXT NOT NULL,
    model         TEXT NOT NULL,
    class_order   TEXT NOT NULL,
    y_true        TEXT NOT NULL,
    y_proba       TEXT NOT NULL,
    n_samples     INTEGER NOT NULL,
    n_classes     INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, model)
);
CREATE INDEX IF NOT EXISTS idx_evalpc_key ON eval_per_class(experiment_id, model);
CREATE INDEX IF NOT EXISTS idx_evalcm_key ON eval_confusion(experiment_id, model);
"""

_METRIC_KEYS = {"Accuracy": "accuracy", "Precision": "precision",
                "Recall": "recall", "F1": "f1", "ROC-AUC": "roc_auc"}


def default_sidecar_path(experiment_db) -> str:
    """experiment.db -> experiment_eval.db (same directory)."""
    p = Path(experiment_db)
    return str(p.with_name(p.stem + "_eval.db"))


def init_sidecar(path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _to_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def write_eval(path, experiment_id, eval_record):
    """Persist one study's EvalRecord into the sidecar (idempotent per key).

    Re-writing the same (experiment_id, model) replaces its rows, so re-running
    a study doesn't duplicate.
    """
    ev = eval_record
    conn = init_sidecar(path)
    try:
        with conn:
            key = (experiment_id, ev.model)
            # Clear any prior rows for this key (idempotent).
            for tbl in ("eval_models", "eval_per_class", "eval_confusion",
                        "eval_predictions"):
                conn.execute(
                    f"DELETE FROM {tbl} WHERE experiment_id=? AND model=?", key)

            m = ev.metrics or {}
            conn.execute(
                'INSERT INTO eval_models (experiment_id, model, feature_set, '
                'detail, best_cv_value, accuracy, "precision", recall, f1, '
                'roc_auc, hyperparams, feature_list) '
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (experiment_id, ev.model, ev.feature_set, ev.detail,
                 _to_float(ev.best_cv_value),
                 _to_float(m.get("Accuracy")), _to_float(m.get("Precision")),
                 _to_float(m.get("Recall")), _to_float(m.get("F1")),
                 _to_float(m.get("ROC-AUC")),
                 json.dumps(_jsonable(ev.hyperparams)),
                 json.dumps(list(ev.feature_list))))

            # per-class
            pc = ev.per_class_df
            pc_rows = [
                (experiment_id, ev.model, str(r["Class"]),
                 _to_float(r.get("Precision")), _to_float(r.get("Recall")),
                 _to_float(r.get("F1")), _to_float(r.get("Support")))
                for _, r in pc.iterrows()
            ]
            conn.executemany(
                'INSERT INTO eval_per_class (experiment_id, model, class_label, '
                '"precision", recall, f1, support) VALUES (?,?,?,?,?,?,?)', pc_rows)

            # confusion (raw integer counts)
            cm = ev.confusion_df
            cm_rows = [
                (experiment_id, ev.model, str(t), str(p),
                 int(cm.loc[t, p]))
                for t in cm.index for p in cm.columns
            ]
            conn.executemany(
                "INSERT INTO eval_confusion (experiment_id, model, true_label, "
                "pred_label, count) VALUES (?,?,?,?,?)", cm_rows)

            # predictions (probability matrix as JSON)
            proba = np.asarray(ev.y_proba, dtype=float)
            conn.execute(
                "INSERT INTO eval_predictions (experiment_id, model, class_order, "
                "y_true, y_proba, n_samples, n_classes) VALUES (?,?,?,?,?,?,?)",
                (experiment_id, ev.model,
                 json.dumps([str(c) for c in ev.class_order]),
                 json.dumps([str(v) for v in (ev.y_true.tolist()
                             if hasattr(ev.y_true, "tolist") else list(ev.y_true))]),
                 json.dumps(proba.tolist()),
                 int(proba.shape[0]), int(proba.shape[1])))
    finally:
        conn.close()


def _jsonable(obj):
    """Coerce numpy scalars/containers to JSON-native (for hyperparams)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
