"""
experiment/experiment.py — the Experiment orchestrator.

An Experiment owns one run: a timestamped ``experiment_id``, the data split, the
config, and the set of model studies. ``run()`` executes each selected model as
an Optuna study (study.py), persisting all trials to a single Optuna-native
SQLite file (experiment.db). It returns the per-model StudyResults; turning
those into the reporting-shaped report.db is a separate step (report_builder.py).

Study naming inside experiment.db: ``f"{experiment_id}:{model}"`` so multiple
experiments coexist in one file and sort chronologically by id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from DSSP2026.experiment import study as study_mod
from DSSP2026.experiment.study import _MODEL_FAMILY
from DSSP2026.report import build_report_db as _build_report_db

ALL_MODELS = ["Logistic regression", "Decision tree", "Random forest",
              "MLP", "XGBoost"]


def make_experiment_id() -> str:
    """Timestamped, human-readable, chronologically sortable id (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


@dataclass
class Experiment:
    """One experiment run.

    The constructor describes *data + task*, not how any model preprocesses.

    Parameters
    ----------
    train, evaluation : DataFrame
        Pre-split training and held-out evaluation frames.
    test : DataFrame, optional
        A third, fully held-out partition used *only* for final scoring after
        model/policy selection — never for fitting, threshold tuning, or
        selection. When provided it is persisted as ``.artifacts/test.parquet``
        (mirroring the train sidecar) so the report layer can evaluate a fitted
        ensemble/policy on data nothing in the pipeline has seen. Optional:
        omit it and the test sidecar is simply not written.
    target : str
        Target column name.
    feature_sets : mapping of name -> feature columns, optional
        Named candidate column groups the search picks among. **Optional**: when
        omitted, defaults to a single ``{"all": <all columns except target>}``.
    schema : mapping of col -> {"numeric","categorical","passthrough"}
        How columns are treated by models that preprocess (only the MLP today).
        **Optional**: inferred from dtype when omitted (override only the columns
        inference would get wrong). The framework core never reads this.
    experiment_dir : path to the experiment folder (holds study.db, eval.db, report.db, config.json).
    models : which models to run (subset of ALL_MODELS); None -> all.
    """
    train: pd.DataFrame
    evaluation: pd.DataFrame
    target: str
    experiment_dir: str
    test: Optional[pd.DataFrame] = None    # required at runtime (see __post_init__)
    feature_sets: Optional[Mapping[str, Sequence[str]]] = None
    schema: Optional[Mapping[str, str]] = None
    search_space: Optional[Mapping[str, list]] = None
    eval_kind: str = "validation"     # "validation" (split from train) | "test" (passed separately)
    models: Optional[Sequence[str]] = None
    n_trials: int = 30
    n_splits: int = 5
    scoring: str = "f1_macro"
    random_state: int = 42
    experiment_id: str = field(default_factory=make_experiment_id)
    # Optional provenance for the manifest (paths the data came from).
    dataset_paths: Optional[Mapping[str, str]] = None
    # Optional display-name map for the (integer) class labels. Models train on
    # the raw encoded target; this maps each stored label to a friendly name used
    # only for *display* in the report (confusion axes, per-class tables, cost
    # views). Keys are normalised to str so {0: "..."} and {"0": "..."} both work;
    # the decision math always stays keyed on the original stored labels.
    id2label: Optional[Mapping] = None
    column_types: Optional[Mapping[str, str]] = None

    # Populated by run():
    results: dict = field(default_factory=dict)   # model -> StudyResult

    def __post_init__(self):
        # Three-way split is mandatory: train (fit + OOF threshold tuning),
        # evaluation (model/policy selection), test (final leak-free scoring).
        if self.test is None:
            raise ValueError(
                "Experiment requires a `test` split. Provide fully preprocessed "
                "train, evaluation, and test frames — train fits models and "
                "tunes thresholds (via OOF CV), evaluation selects the winner, "
                "and test is scored once for the final, unbiased number.")
        if self.train.empty or self.evaluation.empty or self.test.empty:
            raise ValueError("train, evaluation, and test frames must all be non-empty.")
        if not (set(self.train.columns) == set(self.evaluation.columns) == set(self.test.columns)):
            raise ValueError(
                "train, evaluation, and test must have identical column sets.")
        for name, frame in (
            ("train", self.train),
            ("evaluation", self.evaluation),
            ("test", self.test),
        ):
            if self.target not in frame.columns:
                raise ValueError(f"target column {self.target!r} is missing from {name}.")
        class_sets = {
            "train": set(self.train[self.target].astype(str).dropna().unique()),
            "evaluation": set(self.evaluation[self.target].astype(str).dropna().unique()),
            "test": set(self.test[self.target].astype(str).dropna().unique()),
        }
        if len({frozenset(v) for v in class_sets.values()}) != 1:
            raise ValueError(
                "target class sets must match across train, evaluation, and test.")
        # feature_sets defaults to a single "all columns" set.
        if self.feature_sets is None:
            all_cols = [c for c in self.train.columns if c != self.target]
            self.feature_sets = {"all": all_cols}
        referenced = sorted({c for cols in self.feature_sets.values() for c in cols})
        schema = self.schema if self.schema is not None else self.column_types
        if schema is None:
            raise ValueError("schema is required; provide a column role for every feature column.")
        declared = set(schema) | set(referenced)
        missing_by_frame = {
            name: sorted(c for c in declared if c not in frame.columns)
            for name, frame in (
                ("train", self.train),
                ("evaluation", self.evaluation),
                ("test", self.test),
            )
        }
        missing_by_frame = {k: v for k, v in missing_by_frame.items() if v}
        if missing_by_frame:
            raise ValueError(
                "schema/feature_sets reference columns missing from data frames: "
                f"{missing_by_frame}.")
        all_nan = [c for c in referenced if self.train[c].isna().all()]
        if all_nan:
            raise ValueError(
                "declared feature column(s) are entirely NaN in train: "
                f"{all_nan}.")
        from DSSP2026.experiment.columns import resolve_column_types, numeric_and_flag
        self._column_types = resolve_column_types(
            self.train, referenced, schema)
        self.schema = self._column_types
        self.column_types = self._column_types
        self._numeric_features, self._flag_features = numeric_and_flag(
            self._column_types)
        self._class_labels = sorted(self.train[self.target].astype(str).unique())
        # Search-space spec: built-in defaults, overlaid by any user override.
        # Whatever is actually used is recorded in the manifest.
        from DSSP2026.experiment import spaces as SP
        self._search_space = SP.normalize_search_space(self.search_space)

    # -- folder layout: report.db at top level, internals under .artifacts/ --
    @property
    def artifacts_dir(self) -> str:
        return str(Path(self.experiment_dir) / ".artifacts")

    @property
    def optuna_db(self) -> str:
        return str(Path(self.artifacts_dir) / "study.db")

    @property
    def eval_db(self) -> str:
        return str(Path(self.artifacts_dir) / "eval.db")

    @property
    def report_db(self) -> str:
        return str(Path(self.experiment_dir) / "report.db")

    @property
    def train_parquet(self) -> str:
        return str(Path(self.artifacts_dir) / "train.parquet")

    @property
    def train_parquet_rel(self) -> str:
        """Path to train.parquet relative to report.db's directory."""
        return ".artifacts/train.parquet"

    @property
    def test_parquet(self) -> str:
        return str(Path(self.artifacts_dir) / "test.parquet")

    @property
    def test_parquet_rel(self) -> str:
        """Path to test.parquet relative to report.db's directory."""
        return ".artifacts/test.parquet"

    @property
    def validation_parquet(self) -> str:
        return str(Path(self.artifacts_dir) / "validation.parquet")

    @property
    def validation_parquet_rel(self) -> str:
        """Path to validation.parquet relative to report.db's directory."""
        return ".artifacts/validation.parquet"

    @property
    def config_path(self) -> str:
        return str(Path(self.artifacts_dir) / "config.json")

    @property
    def storage_url(self) -> str:
        return f"sqlite:///{Path(self.optuna_db).resolve()}"

    # Back-compat alias (report_builder/tests referring to the sidecar path).
    @property
    def sidecar_path(self) -> str:
        return self.eval_db

    def selected_models(self):
        if self.models is None:
            return list(ALL_MODELS)
        return [m for m in ALL_MODELS if m in set(self.models)]

    def run(self, *, verbose=True, build_report=True):
        """Run all selected models' studies into the experiment folder.

        Writes study.db (Optuna trials), eval.db (held-out eval sidecar), and —
        when ``build_report`` — report.db. Finishes by stamping config.json with
        status COMPLETE. Returns self.
        """
        from DSSP2026.experiment.sidecar import write_eval
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

        # Persist the training frame as a parquet sidecar so the report layer
        # can refit models on demand (cost decision layers, calibration, etc.).
        self._write_train_parquet()
        # Persist the untouched test partition (when supplied) so the report
        # layer can score a fitted ensemble/policy on truly held-out data.
        self._write_test_parquet()
        # Persist the validation split too, so best_fit can SELECT the winner on
        # validation (keeping test untouched until the final score).
        self._write_validation_parquet()

        for model in self.selected_models():
            if verbose:
                print(f"[{self.experiment_id}] running study: {model}")
            sr = study_mod.run_study(
                model, self.train, self.evaluation,
                target=self.target, feature_sets=self.feature_sets,
                numeric_features=self._numeric_features,
                flag_features=self._flag_features,
                class_labels=self._class_labels,
                n_trials=self.n_trials, n_splits=self.n_splits,
                scoring=self.scoring, random_state=self.random_state,
                storage=self.storage_url,
                study_name=f"{self.experiment_id}:{model}",
                spec=self._search_space.get(model))
            self.results[model] = sr
            write_eval(self.eval_db, self.experiment_id, sr.eval)
            # OOF predictions from the SELECTED hyperparameters: one extra k-fold
            # pass on train, used later for leak-free threshold tuning. Stored in
            # the eval sidecar and merged into report.db by the builder.
            self._write_oof(sr)
            # Test predictions: refit on FULL train with the selected config and
            # predict the untouched test set. Persisted so the report layer and
            # dashboard read leak-free test probabilities without refitting.
            self._write_test_predictions(sr)
            from DSSP2026.experiment.sidecar import (
                write_feature_importance, write_trial_curves)
            write_feature_importance(
                self.eval_db, self.experiment_id, sr.model,
                sr.fit_result, _MODEL_FAMILY[sr.model])
            write_trial_curves(
                self.eval_db, self.experiment_id, sr.model, sr.study)
            if verbose:
                ev = sr.eval
                print(f"    best {ev.feature_set} | CV={ev.best_cv_value:.4f} | "
                      f"eval F1={ev.metrics.get('F1', float('nan')):.4f}")

        if build_report:
            _build_report_db(self.optuna_db, self.eval_db, self.report_db,
                             meta=self.meta())
            self._write_report_tuning_pngs()

        self._write_config(status="COMPLETE")
        if verbose:
            print(f"[{self.experiment_id}] COMPLETE — {self.experiment_dir}")
        return self

    def build_report_db(self, report_db, *, sidecar_db=None):
        """Build the reporting database from this experiment's outputs.

        Parameters
        ----------
        report_db : path-like
            Destination path for report.db.
        sidecar_db : path-like, optional
            Path to experiment_eval.db. Defaults to the standard sidecar
            path derived from experiment_db.
        """
        if sidecar_db is None:
            sidecar_db = self.eval_db
        _build_report_db(self.optuna_db, sidecar_db, report_db, meta=self.meta())

    def _write_report_tuning_pngs(self):
        from DSSP2026.report.report import Report
        from DSSP2026.report.tuning import save_tuning_plot_png

        report = Report(self.report_db, experiment_id=self.experiment_id)
        for model in report.models(include_ensemble=False):
            tag = model.lower().replace(" ", "_")
            try:
                save_tuning_plot_png(
                    report, model, Path(self.artifacts_dir) / f"{tag}_tuning.png")
            except Exception:
                continue

    def _write_train_parquet(self):
        """Write the training frame to .artifacts/train.parquet and record its
        hash. Parquet preserves dtypes exactly (unlike CSV), so a later refit
        sees the same columns/types the experiment trained on.
        """
        import hashlib
        path = Path(self.train_parquet)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.train.to_parquet(path, index=False)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self._train_parquet_sha256 = h.hexdigest()

    def _write_test_predictions(self, sr):
        """Refit on full train with the selected config; predict the test set.

        Stores per-model test probabilities (aligned to the canonical class
        order) so the report layer and dashboard can score on the untouched
        test partition without refitting at read time. Persisted in the eval
        sidecar and merged into report.db by the builder.
        """
        import numpy as np, json, sqlite3
        from DSSP2026.experiment.refit import refit_estimator

        ev = sr.eval
        hp = dict(ev.hyperparams)
        features = list(ev.feature_list)
        target = self.target
        class_order = [str(c) for c in self._class_labels]
        col_ix = {c: j for j, c in enumerate(class_order)}
        K = len(class_order)

        est = refit_estimator(
            sr.model, self.train, target=target, features=features,
            hyperparams=hp, column_types=self._column_types)
        p = np.asarray(est.predict_proba(self.test), dtype=float)
        aligned = np.zeros((len(self.test), K), dtype=float)
        for j, c in enumerate(est.class_order):
            if str(c) in col_ix:
                aligned[:, col_ix[str(c)]] = p[:, j]
        y_true = self.test[target].astype(str).tolist()

        conn = sqlite3.connect(self.eval_db)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS test_predictions ("
                "experiment_id TEXT, model TEXT, class_order TEXT, y_true TEXT, "
                "y_proba TEXT, n_samples INT, n_classes INT, "
                "PRIMARY KEY (experiment_id, model))")
            conn.execute(
                "INSERT OR REPLACE INTO test_predictions VALUES (?,?,?,?,?,?,?)",
                (self.experiment_id, sr.model, json.dumps(class_order),
                 json.dumps(y_true), json.dumps(aligned.tolist()),
                 len(self.test), K))
            conn.commit()
        finally:
            conn.close()

    def _write_oof(self, sr):
        """Compute out-of-fold probabilities for a study's selected config and
        persist them to the eval sidecar (later merged into report.db).

        One stratified k-fold pass over train: each fold refits the model on the
        fold-train slice using the SELECTED hyperparameters and predicts the
        fold-validation slice. The pooled OOF probabilities give every train row
        a held-out-quality prediction, enabling leak-free threshold tuning in
        the report layer without recomputing folds at report time.
        """
        import numpy as np, json, sqlite3
        from DSSP2026.experiment.cv import make_splitter
        from DSSP2026.experiment.refit import refit_estimator

        ev = sr.eval
        hp = dict(ev.hyperparams)
        features = list(ev.feature_list)
        target = self.target
        train = self.train
        class_order = [str(c) for c in self._class_labels]
        col_ix = {c: j for j, c in enumerate(class_order)}
        K = len(class_order)

        strat = train[target].astype(str).to_numpy()
        splitter = make_splitter(stratified=True, n_splits=self.n_splits,
                                 random_state=self.random_state)
        oof = np.full((len(train), K), np.nan, dtype=float)
        for tr_i, va_i in splitter.split(train, strat):
            est = refit_estimator(
                sr.model, train.iloc[tr_i], target=target, features=features,
                hyperparams=hp, column_types=self._column_types)
            p = np.asarray(est.predict_proba(train.iloc[va_i]), dtype=float)
            aligned = np.zeros((len(va_i), K), dtype=float)
            for j, c in enumerate(est.class_order):
                if str(c) in col_ix:
                    aligned[:, col_ix[str(c)]] = p[:, j]
            oof[va_i] = aligned

        oof_true = train[target].astype(str).tolist()
        conn = sqlite3.connect(self.eval_db)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS oof_predictions ("
                "experiment_id TEXT, model TEXT, class_order TEXT, y_true TEXT, "
                "y_proba TEXT, n_splits INT, random_state INT, "
                "PRIMARY KEY (experiment_id, model))")
            conn.execute(
                "INSERT OR REPLACE INTO oof_predictions VALUES (?,?,?,?,?,?,?)",
                (self.experiment_id, sr.model, json.dumps(class_order),
                 json.dumps(oof_true), json.dumps(oof.tolist()),
                 self.n_splits, self.random_state))
            conn.commit()
        finally:
            conn.close()

    def _write_validation_parquet(self):
        """Write the validation frame to .artifacts/validation.parquet + hash."""
        import hashlib
        path = Path(self.validation_parquet)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation.to_parquet(path, index=False)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self._validation_parquet_sha256 = h.hexdigest()

    def _write_test_parquet(self):
        """Write the test frame to .artifacts/test.parquet and record its hash.

        No-op when ``test`` is None (the common two-way-split case): the sidecar
        simply isn't written and ``_test_parquet_sha256`` stays unset, so the
        manifest records nulls and ``load_test_data`` returns None.
        """
        if self.test is None:
            self._test_parquet_sha256 = None
            return
        import hashlib
        path = Path(self.test_parquet)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.test.to_parquet(path, index=False)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self._test_parquet_sha256 = h.hexdigest()

    def _train_parquet_record(self) -> dict:
        """Provenance for the stored training parquet (path relative to report.db,
        content hash, and target column) — consumed by report_builder."""
        return {
            "train_parquet": self.train_parquet_rel,
            "train_sha256": getattr(self, "_train_parquet_sha256", None),
            "target": self.target,
        }

    # -- manifest --
    def _dataset_records(self) -> dict:
        """Paths + content hashes of the source datasets, when paths are known.

        Hashing reads each file's bytes; if a path is missing or unreadable the
        hash is recorded as None rather than failing the run.
        """
        import hashlib
        recs = {}
        for role, path in (self.dataset_paths or {}).items():
            entry = {"path": str(path), "sha256": None}
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                entry["sha256"] = h.hexdigest()
            except OSError:
                pass
            recs[role] = entry
        return recs

    def meta(self) -> dict:
        """Run-level provenance (also the basis of the manifest)."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_kind": self.eval_kind,
            "eval_source": ("split_from_train" if self.eval_kind == "validation"
                            else "passed_separately"),
            "models": self.selected_models(),
            "feature_sets": {k: list(v) for k, v in self.feature_sets.items()},
            "column_types": dict(self._column_types),
            "search_space": self._search_space,
            "n_trials": self.n_trials,
            "n_splits": self.n_splits,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "datasets": self._dataset_records(),
            "experiment_dir": str(Path(self.experiment_dir).resolve()),
            "id2label": ({str(k): str(v) for k, v in self.id2label.items()}
                         if self.id2label else None),
            "train_parquet": self.train_parquet_rel,
            "train_sha256": getattr(self, "_train_parquet_sha256", None),
            "test_parquet": (self.test_parquet_rel
                             if self.test is not None else None),
            "test_sha256": getattr(self, "_test_parquet_sha256", None),
            "validation_parquet": self.validation_parquet_rel,
            "validation_sha256": getattr(self, "_validation_parquet_sha256", None),
            "target": self.target,
        }

    def _write_config(self, *, status):
        """Write config.json — the self-describing experiment record."""
        import json
        manifest = self.meta()
        manifest["status"] = status
        manifest["artifacts"] = {
            "optuna_db": ".artifacts/study.db",
            "eval_db": ".artifacts/eval.db",
            "report_db": "report.db",
            "train_parquet": ".artifacts/train.parquet"}
        if self.test is not None:
            manifest["artifacts"]["test_parquet"] = ".artifacts/test.parquet"
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
