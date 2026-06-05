from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ReportBase:
    """Read-only reporting core over a report.db: connection, experiment/model
    discovery, and prediction retrieval. Mixins build analysis on top of this."""

    def __init__(self, report_db, experiment_id: Optional[str] = None):
        self.report_db = str(report_db)
        if not Path(self.report_db).exists():
            raise FileNotFoundError(f"report.db not found: {self.report_db}")
        self.experiment_id = experiment_id or self._latest_experiment_id()
        if self.experiment_id is None:
            raise ValueError("report.db has no experiments.")
        self.id2label = self._load_id2label()

    def _load_id2label(self) -> dict:
        """Display-name map {stored_label(str) -> friendly name} for this
        experiment, or ``{}`` when none was registered (or the column predates
        this feature). Keys are normalised to str to match the stored labels,
        which are always stringified ("0"/"1", "C1"...).
        """
        conn = self._connect()
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(experiments)")]
            if "id2label" not in cols:
                return {}
            row = conn.execute(
                "SELECT id2label FROM experiments WHERE experiment_id=?",
                (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        if not row or row[0] is None:
            return {}
        return {str(k): str(v) for k, v in json.loads(row[0]).items()}

    def _relabel(self, label):
        """Map one stored label to its display name (identity if unmapped)."""
        return self.id2label.get(str(label), label)

    def _relabel_seq(self, labels):
        """Map a sequence of stored labels to display names (identity if none)."""
        return [self._relabel(x) for x in labels]

    def _connect(self):
        conn = sqlite3.connect(self.report_db)
        conn.row_factory = sqlite3.Row
        return conn

    def _latest_experiment_id(self):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT experiment_id FROM experiments "
                "ORDER BY timestamp DESC, experiment_id DESC LIMIT 1").fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def experiments(self) -> pd.DataFrame:
        """All experiments in this report.db (newest first)."""
        conn = self._connect()
        try:
            return pd.read_sql_query(
                "SELECT experiment_id, timestamp, eval_kind, scoring, n_trials, "
                "n_splits FROM experiments ORDER BY timestamp DESC", conn)
        finally:
            conn.close()

    def models(self) -> list:
        """Model names present for the selected experiment."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT model FROM models WHERE experiment_id=? ORDER BY model",
                (self.experiment_id,)).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def _best_model_name(self):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT model FROM models WHERE experiment_id=? "
                "ORDER BY f1 DESC LIMIT 1", (self.experiment_id,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _read_predictions(self, model):
        """Return (class_order, y_true, y_proba) for a model, or raise.

        ``class_order`` is the list of class labels (proba columns), ``y_true``
        a numpy array of per-sample true labels (strings), ``y_proba`` the
        n_samples x n_classes float matrix.
        """
        conn = self._connect()
        try:
            mid = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid is None:
                raise ValueError(
                    f"model {model!r} not in experiment {self.experiment_id}.")
            row = conn.execute(
                "SELECT class_order, y_true, y_proba FROM predictions "
                "WHERE model_id=?", (mid[0],)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(
                f"no stored probabilities for model {model!r}; ROC/threshold "
                "plots need the predictions table.")
        class_order = [str(c) for c in json.loads(row[0])]
        y_true = np.asarray(json.loads(row[1]), dtype=object)
        y_proba = np.asarray(json.loads(row[2]), dtype=float)
        return class_order, y_true, y_proba
