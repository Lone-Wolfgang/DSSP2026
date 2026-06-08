"""
experiment/report_builder.py — build the reporting-shaped report.db.

Lifts an experiment's data from its two source files and reorganizes it for the
Report API:

  - **experiment.db** (Optuna-native) -> trials, via the Optuna *API*
    (``optuna.load_study``), NOT raw SQL. This matters: Optuna stores categorical
    params as float indices into a choices list (e.g. ``feature_set=1.0`` means
    ``choices[1]``). The API decodes that back to the real value automatically;
    hand-rolled SQL would mislabel them.
  - **experiment_eval.db** (sidecar) -> the winning configs' held-out metrics,
    per-class, confusion counts, and probability matrices.

Re-runnable from the files alone: ``build_report_db(experiment_db, sidecar, report_db)``.

report.db schema (one row per study = per best model under ``model_id``):

  experiments(experiment_id, timestamp, eval_kind, models, feature_sets,
              n_trials, n_splits, scoring, random_state, experiment_db)
  models(model_id, experiment_id, model, feature_set, detail, accuracy,
         precision, recall, f1, roc_auc, hyperparams, feature_list,
         n_trials, best_cv_value)
  trials(trial_id, model_id, trial_number, params, cv_value, cv_se, state, rank,
         is_best, duration_sec)
  per_class(model_id, class_label, precision, recall, f1, support)
  confusion(model_id, true_label, pred_label, count)
  predictions(model_id, class_order, y_true, y_proba, n_samples, n_classes)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp     TEXT,
    eval_kind     TEXT,
    models        TEXT,
    feature_sets  TEXT,
    n_trials      INTEGER,
    n_splits      INTEGER,
    scoring       TEXT,
    random_state  INTEGER,
    experiment_db TEXT,
    id2label      TEXT,
    train_parquet TEXT,
    train_sha256  TEXT,
    test_parquet  TEXT,
    test_sha256   TEXT,
    validation_parquet TEXT,
    validation_sha256  TEXT,
    target        TEXT,
    column_types  TEXT,
    search_space  TEXT
);
CREATE TABLE IF NOT EXISTS models (
    model_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    feature_set   TEXT,
    detail        TEXT,
    accuracy      REAL,
    "precision"   REAL,
    recall        REAL,
    f1            REAL,
    roc_auc       REAL,
    hyperparams   TEXT,
    feature_list  TEXT,
    n_trials      INTEGER,
    best_cv_value REAL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    trial_number  INTEGER,
    params        TEXT,
    cv_value      REAL,
    cv_se         REAL,
    state         TEXT,
    rank          INTEGER,
    is_best       INTEGER NOT NULL DEFAULT 0,
    duration_sec  REAL
);
CREATE TABLE IF NOT EXISTS per_class (
    model_id    INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    class_label TEXT NOT NULL,
    "precision" REAL,
    recall      REAL,
    f1          REAL,
    support     REAL
);
CREATE TABLE IF NOT EXISTS confusion (
    model_id   INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    true_label TEXT NOT NULL,
    pred_label TEXT NOT NULL,
    count      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    model_id    INTEGER PRIMARY KEY REFERENCES models(model_id) ON DELETE CASCADE,
    class_order TEXT NOT NULL,
    y_true      TEXT NOT NULL,
    y_proba     TEXT NOT NULL,
    n_samples   INTEGER NOT NULL,
    n_classes   INTEGER NOT NULL
);
-- Model feature importances copied from fitted/evaluated estimators.
CREATE TABLE IF NOT EXISTS feature_importance (
    model_id        INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    importance_type TEXT    NOT NULL,
    feature         TEXT    NOT NULL,
    importance      REAL    NOT NULL
);
-- Optuna hyperparameter importances computed from tuning trials.
CREATE TABLE IF NOT EXISTS param_importance (
    model_id   INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    param      TEXT    NOT NULL,
    importance REAL    NOT NULL,
    PRIMARY KEY (model_id, param)
);
CREATE TABLE IF NOT EXISTS trial_curves (
    model_id     INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    trial_number INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    train_curve  TEXT    NOT NULL,
    eval_curve   TEXT    NOT NULL,
    PRIMARY KEY (model_id, trial_number)
);
CREATE INDEX IF NOT EXISTS idx_models_exp   ON models(experiment_id);
CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model_id);
CREATE INDEX IF NOT EXISTS idx_pc_model     ON per_class(model_id);
CREATE INDEX IF NOT EXISTS idx_cm_model     ON confusion(model_id);
CREATE INDEX IF NOT EXISTS idx_fi_model     ON feature_importance(model_id);
CREATE INDEX IF NOT EXISTS idx_pi_model     ON param_importance(model_id);
CREATE INDEX IF NOT EXISTS idx_tc_model     ON trial_curves(model_id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column additions for report.dbs built before a column existed.

    ``_SCHEMA`` uses ``CREATE TABLE IF NOT EXISTS``, which never alters an
    existing table — so a DB created before a column was added would silently
    keep the old shape on rebuild. Each ``ALTER TABLE ... ADD COLUMN`` here is
    guarded by a presence check so it runs at most once and is safe to call on
    every connect.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
    if "id2label" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN id2label TEXT")
    if "train_parquet" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN train_parquet TEXT")
    if "train_sha256" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN train_sha256 TEXT")
    if "test_parquet" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN test_parquet TEXT")
    if "test_sha256" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN test_sha256 TEXT")
    if "validation_parquet" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN validation_parquet TEXT")
    if "validation_sha256" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN validation_sha256 TEXT")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS oof_predictions ("
        "experiment_id TEXT, model TEXT, class_order TEXT, y_true TEXT, "
        "y_proba TEXT, n_splits INT, random_state INT, "
        "PRIMARY KEY (experiment_id, model))")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS test_predictions ("
        "experiment_id TEXT, model TEXT, class_order TEXT, y_true TEXT, "
        "y_proba TEXT, n_samples INT, n_classes INT, "
        "PRIMARY KEY (experiment_id, model))")
    if "target" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN target TEXT")
    if "column_types" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN column_types TEXT")
    if "search_space" not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN search_space TEXT")

    trial_cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)")}
    if "cv_se" not in trial_cols:
        conn.execute("ALTER TABLE trials ADD COLUMN cv_se REAL")

    # feature_importance and trial_curves are created by _SCHEMA when the db is
    # new; for existing dbs that predate them we run CREATE TABLE IF NOT EXISTS
    # directly here rather than trying to add columns to a non-existent table.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feature_importance (
            model_id        INTEGER NOT NULL REFERENCES models(model_id)
                            ON DELETE CASCADE,
            importance_type TEXT    NOT NULL,
            feature         TEXT    NOT NULL,
            importance      REAL    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trial_curves (
            model_id     INTEGER NOT NULL REFERENCES models(model_id)
                         ON DELETE CASCADE,
            trial_number INTEGER NOT NULL,
            rank         INTEGER NOT NULL,
            train_curve  TEXT    NOT NULL,
            eval_curve   TEXT    NOT NULL,
            PRIMARY KEY (model_id, trial_number)
        );
        CREATE TABLE IF NOT EXISTS param_importance (
            model_id   INTEGER NOT NULL REFERENCES models(model_id)
                       ON DELETE CASCADE,
            param      TEXT    NOT NULL,
            importance REAL    NOT NULL,
            PRIMARY KEY (model_id, param)
        );
        CREATE INDEX IF NOT EXISTS idx_fi_model ON feature_importance(model_id);
        CREATE INDEX IF NOT EXISTS idx_tc_model ON trial_curves(model_id);
        CREATE INDEX IF NOT EXISTS idx_pi_model ON param_importance(model_id);
    """)


def _connect_report(report_db) -> sqlite3.Connection:
    Path(report_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(report_db))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _list_experiment_studies(experiment_db):
    """Return {experiment_id: {model: study_name}} from the Optuna storage.

    Study names are ``{experiment_id}:{model}``; split on the LAST colon so a
    model name containing a colon wouldn't break (model names here don't, but
    splitting from the right is the safe choice since experiment_id is a fixed
    timestamp with no colon).
    """
    import optuna

    storage = f"sqlite:///{Path(experiment_db).resolve()}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    out = {}
    for s in summaries:
        name = s.study_name
        if ":" not in name:
            continue
        exp_id, model = name.split(":", 1)
        out.setdefault(exp_id, {})[model] = name
    return out


def build_report_db(experiment_db, sidecar_db, report_db, *,
                    experiment_id: Optional[str] = None, meta: Optional[dict] = None):
    """Build report.db from experiment.db (trials) + sidecar (eval data).

    If ``experiment_id`` is given, only that experiment is built; otherwise every
    experiment found in experiment.db is built. ``meta`` (from Experiment.meta())
    supplies run-level provenance; when absent, minimal provenance is recorded
    from what the study names reveal.
    """
    import optuna
    from DSSP2026.experiment.trials import trials_from_study

    storage = f"sqlite:///{Path(experiment_db).resolve()}"
    studies_by_exp = _list_experiment_studies(experiment_db)
    if experiment_id is not None:
        studies_by_exp = {experiment_id: studies_by_exp.get(experiment_id, {})}

    side = sqlite3.connect(str(sidecar_db))
    side.row_factory = sqlite3.Row
    report = _connect_report(report_db)
    try:
        with report:
            for exp_id, model_studies in studies_by_exp.items():
                _insert_experiment(report, exp_id, meta)
                for model, study_name in model_studies.items():
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    model_id = _insert_model(report, side, exp_id, model, study)
                    _insert_trials(report, model_id, trials_from_study(study, model=model))
                    _insert_param_importance(report, model_id, study)
                    _insert_eval_children(report, side, model_id, exp_id, model)
                _copy_oof(report, side, exp_id)
                _copy_test_predictions(report, side, exp_id)
    finally:
        side.close()
        report.close()
    return str(report_db)


def _copy_oof(report, side, exp_id):
    """Copy persisted OOF predictions for an experiment from sidecar to report.

    Older runs may lack the sidecar table; then this is a no-op and the report
    layer falls back to computing OOF on the fly.
    """
    try:
        rows = side.execute(
            "SELECT experiment_id, model, class_order, y_true, y_proba, "
            "n_splits, random_state FROM oof_predictions WHERE experiment_id=?",
            (exp_id,)).fetchall()
    except sqlite3.OperationalError:
        return
    for r in rows:
        report.execute(
            "INSERT OR REPLACE INTO oof_predictions VALUES (?,?,?,?,?,?,?)",
            (r["experiment_id"], r["model"], r["class_order"], r["y_true"],
             r["y_proba"], r["n_splits"], r["random_state"]))


def _copy_test_predictions(report, side, exp_id):
    """Copy persisted test predictions for an experiment from sidecar to report.

    No-op when the sidecar lacks the table (older runs); the report layer then
    falls back to refitting on full train to regenerate test probabilities.
    """
    try:
        rows = side.execute(
            "SELECT experiment_id, model, class_order, y_true, y_proba, "
            "n_samples, n_classes FROM test_predictions WHERE experiment_id=?",
            (exp_id,)).fetchall()
    except sqlite3.OperationalError:
        return
    for r in rows:
        report.execute(
            "INSERT OR REPLACE INTO test_predictions VALUES (?,?,?,?,?,?,?)",
            (r["experiment_id"], r["model"], r["class_order"], r["y_true"],
             r["y_proba"], r["n_samples"], r["n_classes"]))


def _insert_experiment(report, exp_id, meta):
    m = meta or {}
    id2label = m.get("id2label")
    report.execute(
        "INSERT OR REPLACE INTO experiments (experiment_id, timestamp, eval_kind, "
        "models, feature_sets, n_trials, n_splits, scoring, random_state, "
        "experiment_db, id2label, train_parquet, train_sha256, "
        "test_parquet, test_sha256, validation_parquet, validation_sha256, "
        "target, column_types, search_space) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (exp_id, m.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
         m.get("eval_kind"), json.dumps(m.get("models")) if m.get("models") else None,
         json.dumps(m.get("feature_sets")) if m.get("feature_sets") else None,
         m.get("n_trials"), m.get("n_splits"), m.get("scoring"),
         m.get("random_state"), m.get("experiment_db"),
         json.dumps(id2label) if id2label else None,
         m.get("train_parquet"), m.get("train_sha256"),
         m.get("test_parquet"), m.get("test_sha256"),
         m.get("validation_parquet"), m.get("validation_sha256"), m.get("target"),
         json.dumps(m.get("column_types")) if m.get("column_types") else None,
         json.dumps(m.get("search_space")) if m.get("search_space") else None))


def _insert_model(report, side, exp_id, model, study):
    """Insert the models row from the sidecar's eval_models; return model_id."""
    row = side.execute(
        "SELECT * FROM eval_models WHERE experiment_id=? AND model=?",
        (exp_id, model)).fetchone()
    n_trials = len([t for t in study.trials if t.value is not None])
    if row is None:
        # No eval persisted (shouldn't happen in normal runs) — record a stub.
        cur = report.execute(
            "INSERT INTO models (experiment_id, model, n_trials) VALUES (?,?,?)",
            (exp_id, model, n_trials))
        return int(cur.lastrowid)
    cur = report.execute(
        'INSERT INTO models (experiment_id, model, feature_set, detail, accuracy, '
        '"precision", recall, f1, roc_auc, hyperparams, feature_list, n_trials, '
        'best_cv_value) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (exp_id, model, row["feature_set"], row["detail"], row["accuracy"],
         row["precision"], row["recall"], row["f1"], row["roc_auc"],
         row["hyperparams"], row["feature_list"], n_trials, row["best_cv_value"]))
    return int(cur.lastrowid)


def _insert_trials(report, model_id, trial_rows):
    rows = [
        (model_id, t["trial_number"], json.dumps(t["params"]), t["cv_value"],
         t.get("cv_se"), t["state"], t["rank"], int(bool(t["is_best"])),
         t["duration_sec"])
        for t in trial_rows
    ]
    report.executemany(
        "INSERT INTO trials (model_id, trial_number, params, cv_value, cv_se, "
        "state, rank, is_best, duration_sec) VALUES (?,?,?,?,?,?,?,?,?)", rows)


def _insert_param_importance(report, model_id, study):
    # Optuna hyperparameter importances for ordering tuning plot axes.
    """Persist Optuna's hyperparameter importances for a study (best-effort).

    Importances need a live study, so they're computed here at build time and
    stored for the offline parallel-coordinates plot to order its axes. The
    computation can legitimately fail or be undefined — too few completed
    trials, a single search dimension, or an all-tied objective — so any error
    is swallowed and the model simply gets no importance rows (the plot then
    falls back to column order). Models with nothing to tune (e.g. Logistic
    regression) also produce no rows.
    """
    import optuna

    completed = [t for t in study.trials
                 if t.value is not None
                 and str(t.state).endswith("COMPLETE")]
    # get_param_importances needs >=2 completed trials and >=2 distinct values
    # for at least one param; below that it raises or is meaningless.
    if len(completed) < 2:
        return
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        return
    rows = [(model_id, str(p), float(v)) for p, v in imp.items()]
    if rows:
        report.executemany(
            "INSERT OR REPLACE INTO param_importance (model_id, param, "
            "importance) VALUES (?,?,?)", rows)


def _insert_eval_children(report, side, model_id, exp_id, model):
    key = (exp_id, model)
    # per_class
    pc = side.execute(
        'SELECT class_label, "precision", recall, f1, support FROM eval_per_class '
        "WHERE experiment_id=? AND model=?", key).fetchall()
    report.executemany(
        'INSERT INTO per_class (model_id, class_label, "precision", recall, f1, '
        "support) VALUES (?,?,?,?,?,?)",
        [(model_id, r["class_label"], r["precision"], r["recall"], r["f1"],
          r["support"]) for r in pc])
    # confusion
    cm = side.execute(
        "SELECT true_label, pred_label, count FROM eval_confusion "
        "WHERE experiment_id=? AND model=?", key).fetchall()
    report.executemany(
        "INSERT INTO confusion (model_id, true_label, pred_label, count) "
        "VALUES (?,?,?,?)",
        [(model_id, r["true_label"], r["pred_label"], r["count"]) for r in cm])
    # predictions
    pr = side.execute(
        "SELECT class_order, y_true, y_proba, n_samples, n_classes FROM "
        "eval_predictions WHERE experiment_id=? AND model=?", key).fetchone()
    if pr is not None:
        report.execute(
            "INSERT INTO predictions (model_id, class_order, y_true, y_proba, "
            "n_samples, n_classes) VALUES (?,?,?,?,?,?)",
            (model_id, pr["class_order"], pr["y_true"], pr["y_proba"],
             pr["n_samples"], pr["n_classes"]))
    # feature importance
    _insert_feature_importance(report, side, model_id, key)
    # trial curves (MLP / XGBoost only; no-op for others)
    _insert_trial_curves(report, side, model_id, key)


def _insert_feature_importance(report, side, model_id, key):
    """ETL eval_feature_importance rows from sidecar into report.db."""
    rows = side.execute(
        "SELECT importance_type, feature, importance FROM eval_feature_importance "
        "WHERE experiment_id=? AND model=?", key).fetchall()
    if not rows:
        return
    report.executemany(
        "INSERT INTO feature_importance (model_id, importance_type, feature, "
        "importance) VALUES (?,?,?,?)",
        [(model_id, r["importance_type"], r["feature"], r["importance"])
         for r in rows])


def _insert_trial_curves(report, side, model_id, key):
    """ETL eval_trial_curves rows from sidecar into report.db."""
    rows = side.execute(
        "SELECT trial_number, rank, train_curve, eval_curve FROM eval_trial_curves "
        "WHERE experiment_id=? AND model=?", key).fetchall()
    if not rows:
        return
    report.executemany(
        "INSERT INTO trial_curves (model_id, trial_number, rank, train_curve, "
        "eval_curve) VALUES (?,?,?,?,?)",
        [(model_id, r["trial_number"], r["rank"], r["train_curve"], r["eval_curve"])
         for r in rows])