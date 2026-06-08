from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ENSEMBLE_NAME = "Ensemble"


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
        return self.id2label.get(str(label), label)

    def _relabel_seq(self, labels):
        return [self._relabel(x) for x in labels]

    # ------------------------------------------------------------------
    # Training data (parquet sidecar)
    # ------------------------------------------------------------------

    def load_train_data(self, *, verify_hash: bool = True):
        """Load the training frame from the parquet sidecar, or None.

        The path is stored in report.db relative to the report.db file itself,
        so the whole output/ directory can be moved as a unit. Returns
        ``(DataFrame, target_column)`` on success, or ``None`` if the parquet
        is missing, the path isn't recorded, or (when ``verify_hash``) the
        content hash doesn't match what the experiment trained on.

        Refit-dependent features (cost decision layers, calibration) call this
        and fall back gracefully to analysis-over-stored-predictions when it
        returns None.
        """
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
            if "train_parquet" not in cols:
                return None
            row = conn.execute(
                "SELECT train_parquet, train_sha256, target FROM experiments "
                "WHERE experiment_id=?", (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        rel_path, expected_hash, target = row[0], row[1], row[2]

        # Resolve relative to report.db's own directory.
        full = Path(self.report_db).parent / rel_path
        if not full.exists():
            return None

        if verify_hash and expected_hash:
            import hashlib
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                return None   # parquet drifted from what was trained on

        try:
            df = pd.read_parquet(full)
        except Exception:
            return None
        return df, target

    def load_test_data(self, *, verify_hash: bool = True):
        """Load the held-out test frame from its parquet sidecar, or None.

        The test partition is the leak-free counterpart to the eval set: it is
        persisted (``.artifacts/test.parquet``) only when an experiment is run
        with a ``test`` frame, and is never used for fitting, threshold tuning,
        or selection — only for final scoring of a fitted ensemble/policy.

        Resolves and verifies exactly like :meth:`load_train_data` (path stored
        relative to report.db, optional sha256 check). Returns
        ``(DataFrame, target_column)`` on success, or ``None`` when the test
        sidecar is absent (the common two-way-split case), the path isn't
        recorded, or the content hash doesn't match.
        """
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
            if "test_parquet" not in cols:
                return None
            row = conn.execute(
                "SELECT test_parquet, test_sha256, target FROM experiments "
                "WHERE experiment_id=?", (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        rel_path, expected_hash, target = row[0], row[1], row[2]

        # Resolve relative to report.db's own directory.
        full = Path(self.report_db).parent / rel_path
        if not full.exists():
            return None

        if verify_hash and expected_hash:
            import hashlib
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                return None   # parquet drifted from what was recorded

        try:
            df = pd.read_parquet(full)
        except Exception:
            return None
        return df, target

    def load_validation_data(self, *, verify_hash: bool = True):
        """Load the validation frame from its parquet sidecar, or None.

        Validation is the selection set: best_fit ranks candidates on it, so the
        test set stays untouched until the final score. Resolves/verifies like
        load_test_data. Returns (DataFrame, target) or None.
        """
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
            if "validation_parquet" not in cols:
                return None
            row = conn.execute(
                "SELECT validation_parquet, validation_sha256, target FROM "
                "experiments WHERE experiment_id=?", (self.experiment_id,)).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        rel_path, expected_hash, target = row[0], row[1], row[2]
        full = Path(self.report_db).parent / rel_path
        if not full.exists():
            return None
        if verify_hash and expected_hash:
            import hashlib
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                return None
        try:
            df = pd.read_parquet(full)
        except Exception:
            return None
        return df, target

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

    def models(self, *, include_ensemble: bool = True) -> list:
        """Model names for the selected experiment.

        When ``include_ensemble=True`` (default) the virtual ``"Ensemble"``
        model is appended after the real models whenever two or more real
        models are present.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT model FROM models WHERE experiment_id=? ORDER BY model",
                (self.experiment_id,)).fetchall()
        finally:
            conn.close()
        real = [r[0] for r in rows]
        if include_ensemble and len(real) >= 2:
            return real + [ENSEMBLE_NAME]
        return real

    def _best_model_name(self):
        """Best real model by held-out F1 (Ensemble excluded from ranking)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT model FROM models WHERE experiment_id=? "
                "ORDER BY f1 DESC LIMIT 1", (self.experiment_id,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Ensemble helpers
    # ------------------------------------------------------------------

    def _ensemble_proba(self, members=None):
        """Return (class_order, y_true, mean_y_proba) for the ensemble.

        Mean probability across ``members`` (default: all real models). All
        models share the same class_order (guaranteed by the experiment layer).
        y_true is taken from the top member by F1 — identical across all models
        (same held-out set) but using the top member ensures consistent label
        ordering in the rare case of a tie.

        Parameters
        ----------
        members : sequence of str, optional
            The real models to average. ``None`` -> every real model. The
            virtual ``"Ensemble"`` name is not a valid member and is dropped if
            passed. At least two members must remain.
        """
        if members is None:
            member_list = self.models(include_ensemble=False)
        else:
            real = set(self.models(include_ensemble=False))
            member_list = [m for m in members
                           if m in real and m != ENSEMBLE_NAME]
        if len(member_list) < 2:
            raise ValueError(
                f"{ENSEMBLE_NAME} requires at least 2 real members; "
                f"got {len(member_list)}.")

        # Canonical y_true/class_order from the best available member by F1,
        # falling back to the first member if the best isn't in the subset.
        top_model = self._best_model_name()
        if top_model not in member_list:
            top_model = member_list[0]
        class_order, y_true, _ = self._read_predictions_real(top_model)

        probas = []
        for m in member_list:
            _, _, yp = self._read_predictions_real(m)
            probas.append(yp)

        return class_order, y_true, np.mean(probas, axis=0)

    def _ensemble_metrics(self, members=None):
        """Held-out metrics for the ensemble (used by compare_models row).

        ``members`` selects which real models to average (default: all).
        """
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)
        class_order, y_true, y_proba = self._ensemble_proba(members)
        y_pred = np.array([class_order[i] for i in y_proba.argmax(axis=1)],
                          dtype=object)
        binary = len(class_order) == 2
        pos = class_order[-1] if binary else None
        avg = "binary" if binary else "macro"

        def _safe(fn, **kw):
            try:
                return float(fn(y_true, y_pred, **kw))
            except Exception:
                return float("nan")

        def _roc():
            try:
                if binary:
                    ki = class_order.index(pos)
                    return float(roc_auc_score(y_true == pos, y_proba[:, ki]))
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(y_true, classes=class_order)
                return float(roc_auc_score(
                    yb, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                return float("nan")

        kw = (dict(average=avg, pos_label=pos, zero_division=0)
              if binary else dict(average=avg, zero_division=0))
        return {
            "accuracy":      float(accuracy_score(y_true, y_pred)),
            "precision":     _safe(precision_score, **kw),
            "recall":        _safe(recall_score, **kw),
            "f1":            _safe(f1_score, **kw),
            "roc_auc":       _roc(),
            "best_cv_value": float("nan"),   # no CV for ensemble
            "feature_set":   "ensemble",
        }

    # ------------------------------------------------------------------
    # Prediction retrieval
    # ------------------------------------------------------------------

    def _read_predictions_real(self, model):
        """Read predictions for a real (non-ensemble) model."""
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
                f"no stored probabilities for model {model!r}; "
                "predictions table is required.")
        class_order = [str(c) for c in json.loads(row[0])]
        y_true = np.asarray(json.loads(row[1]), dtype=object)
        y_proba = np.asarray(json.loads(row[2]), dtype=float)
        return class_order, y_true, y_proba

    def _read_predictions(self, model):
        """Return (class_order, y_true, y_proba) for a model or the Ensemble.

        Single dispatch point: every mixin that calls
        ``self._read_predictions(model)`` automatically gets ensemble support.
        """
        if model == ENSEMBLE_NAME:
            return self._ensemble_proba()
        return self._read_predictions_real(model)

    def _read_test_predictions_real(self, model):
        """Persisted test probabilities for one real model, or None if absent."""
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(test_predictions)")}
            if not cols:
                return None
            row = conn.execute(
                "SELECT class_order, y_true, y_proba FROM test_predictions "
                "WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
        except Exception:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        class_order = [str(c) for c in json.loads(row[0])]
        y_true = np.asarray(json.loads(row[1]), dtype=object)
        y_proba = np.asarray(json.loads(row[2]), dtype=float)
        return class_order, y_true, y_proba

    def _read_test_predictions(self, model, members=None):
        """Persisted test predictions for a model or Ensemble; None if missing.

        For the Ensemble, averages the persisted test probabilities of
        ``members`` (default: all real models). Returns None if the table is
        absent or any required member lacks a row, so callers can fall back to
        refitting on full train.
        """
        if model == ENSEMBLE_NAME:
            mem = (members if members is not None
                   else self.models(include_ensemble=False))
            mats, co, yt = [], None, None
            for m in mem:
                got = self._read_test_predictions_real(m)
                if got is None:
                    return None
                co, yt, p = got
                mats.append(p)
            if len(mats) < 2:
                return None
            return co, yt, np.mean(mats, axis=0)
        return self._read_test_predictions_real(model)
