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
eval_feature_importance(experiment_id, model, importance_type, feature, importance)
    importance_type values:
        "gain"        — default for tree / RF / XGBoost (sklearn feature_importances_)
        "weight"      — XGBoost split count
        "cover"       — XGBoost average cover
        "total_gain"  — XGBoost total gain across all splits
        "total_cover" — XGBoost total cover across all splits
        "abs_coef"    — logistic regression |coefficient| (mean across classes for
                        multiclass; single value for binary)
eval_trial_curves(experiment_id, model, trial_number, rank, train_curve,
                  eval_curve)
    Stores the top-10 trials by CV objective (as ranked in experiment.db).
    train_curve / eval_curve are JSON lists of per-epoch log-loss values.
    Only populated for MLP and XGBoost (the families whose objectives log curves).
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
CREATE TABLE IF NOT EXISTS eval_feature_importance (
    experiment_id   TEXT NOT NULL,
    model           TEXT NOT NULL,
    importance_type TEXT NOT NULL,
    feature         TEXT NOT NULL,
    importance      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_trial_curves (
    experiment_id TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    trial_number  INTEGER NOT NULL,
    rank          INTEGER NOT NULL,
    train_curve   TEXT    NOT NULL,
    eval_curve    TEXT    NOT NULL,
    PRIMARY KEY (experiment_id, model, trial_number)
);
CREATE INDEX IF NOT EXISTS idx_evalpc_key  ON eval_per_class(experiment_id, model);
CREATE INDEX IF NOT EXISTS idx_evalcm_key  ON eval_confusion(experiment_id, model);
CREATE INDEX IF NOT EXISTS idx_evalfi_key  ON eval_feature_importance(experiment_id, model);
CREATE INDEX IF NOT EXISTS idx_evaltc_key  ON eval_trial_curves(experiment_id, model);
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
                        "eval_predictions", "eval_feature_importance",
                        "eval_trial_curves"):
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


# ---------------------------------------------------------------------------
# Feature importance — called from experiment/study.py after refit
# ---------------------------------------------------------------------------

def write_feature_importance(path, experiment_id, model, result, model_family):
    """Persist feature importances for the winning model into the sidecar.

    Parameters
    ----------
    path : path-like
        sidecar .db path.
    experiment_id : str
    model : str
        Model name (e.g. "Decision tree", "XGBoost").
    result : ClassificationResult subclass
        The refitted winner: TreeClassifyResult, RFClassifyResult, XGBResult,
        MLPResult, or LogisticEval.
    model_family : str
        One of "tree", "rf", "xgb", "mlp", "logistic". Controls which
        importance strategy is applied.
    """
    rows = _extract_importance(result, model_family)
    if not rows:
        return

    conn = init_sidecar(path)
    try:
        with conn:
            conn.execute(
                "DELETE FROM eval_feature_importance "
                "WHERE experiment_id=? AND model=?",
                (experiment_id, model))
            conn.executemany(
                "INSERT INTO eval_feature_importance "
                "(experiment_id, model, importance_type, feature, importance) "
                "VALUES (?,?,?,?,?)",
                [(experiment_id, model, itype, feat, float(imp))
                 for itype, feat, imp in rows])
    finally:
        conn.close()


def _extract_importance(result, model_family):
    # Model feature importances from fitted estimators, not tuning parameters.
    """Return a list of (importance_type, feature, importance) triples.

    Decision tree and Random forest: sklearn ``feature_importances_`` → "gain".
    XGBoost: sklearn ``feature_importances_`` (gain) plus the four booster-level
        types (weight, cover, total_gain, total_cover).
    MLP: no intrinsic importance — returns empty list.
    Logistic: mean |coefficient| across classes, labelled "abs_coef".
        For binary the single coefficient vector is used directly.
        The intercept term is excluded.
    """
    rows = []

    if model_family in ("tree", "rf"):
        fi = result.model.feature_importances_
        for feat, imp in zip(result.features, fi):
            rows.append(("gain", feat, float(imp)))

    elif model_family == "xgb":
        # Gain via sklearn API (matches feature_importances_ convention).
        fi = result.model.feature_importances_
        for feat, imp in zip(result.features, fi):
            rows.append(("gain", feat, float(imp)))

        # Additional types from the booster.
        booster = result.model.get_booster()
        for itype in ("weight", "cover", "total_gain", "total_cover"):
            score = booster.get_score(importance_type=itype)
            for i, feat in enumerate(result.features):
                val = score.get(feat, score.get(f"f{i}", 0.0))
                rows.append((itype, feat, float(val)))

    elif model_family == "mlp":
        # MLPs have no intrinsic feature importance.
        pass

    elif model_family == "logistic":
        # LogisticEval doesn't carry a statsmodels result — importance is
        # derived from the coef_df attached to the original fit result, which
        # study._refit_winner doesn't surface directly.  Instead we re-derive
        # abs_coef from y_proba column magnitudes as a proxy, but that is
        # meaningless.  The correct approach is to pass the coef_df through.
        # For now we emit nothing; the LogisticMixin in report/ refits from
        # training data and owns the coefficient table.
        pass

    return rows


# ---------------------------------------------------------------------------
# Trial curves — called from experiment/study.py for MLP and XGBoost
# ---------------------------------------------------------------------------

_CURVE_MODELS = {"MLP", "XGBoost"}
_TOP_N_CURVES = 10


def write_trial_curves(path, experiment_id, model, study):
    """Persist the top-10 trial loss curves for MLP or XGBoost into the sidecar.

    Only called for models in ``_CURVE_MODELS``. Silently no-ops for any study
    whose completed trials don't carry loss-curve user_attrs (e.g. a study run
    with an older objective that didn't log them).

    Parameters
    ----------
    path : path-like
        sidecar .db path.
    experiment_id : str
    model : str
        "MLP" or "XGBoost".
    study : optuna.Study
        The completed Optuna study.
    """
    if model not in _CURVE_MODELS:
        return

    from DSSP2026.experiment.tuning.search import TRAIN_CURVE_KEY, EVAL_CURVE_KEY
    import optuna

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and TRAIN_CURVE_KEY in t.user_attrs
        and EVAL_CURVE_KEY in t.user_attrs
    ]
    if not completed:
        return

    top = sorted(completed, key=lambda t: t.value, reverse=True)[:_TOP_N_CURVES]

    rows = []
    for rank, trial in enumerate(top, start=1):
        rows.append((
            experiment_id,
            model,
            int(trial.number),
            rank,
            json.dumps(trial.user_attrs[TRAIN_CURVE_KEY]),
            json.dumps(trial.user_attrs[EVAL_CURVE_KEY]),
        ))

    conn = init_sidecar(path)
    try:
        with conn:
            conn.execute(
                "DELETE FROM eval_trial_curves "
                "WHERE experiment_id=? AND model=?",
                (experiment_id, model))
            conn.executemany(
                "INSERT INTO eval_trial_curves "
                "(experiment_id, model, trial_number, rank, train_curve, eval_curve) "
                "VALUES (?,?,?,?,?,?)",
                rows)
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
