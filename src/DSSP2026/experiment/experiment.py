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
    target : str
        Target column name.
    feature_sets : mapping of name -> feature columns, optional
        Named candidate column groups the search picks among. **Optional**: when
        omitted, defaults to a single ``{"all": <all columns except target>}``.
    column_types : mapping of col -> {"numeric","categorical","passthrough"}, optional
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
    feature_sets: Optional[Mapping[str, Sequence[str]]] = None
    column_types: Optional[Mapping[str, str]] = None
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

    # Populated by run():
    results: dict = field(default_factory=dict)   # model -> StudyResult

    def __post_init__(self):
        # feature_sets defaults to a single "all columns" set.
        if self.feature_sets is None:
            all_cols = [c for c in self.train.columns if c != self.target]
            self.feature_sets = {"all": all_cols}
        referenced = sorted({c for cols in self.feature_sets.values() for c in cols})
        from DSSP2026.experiment.columns import resolve_column_types, numeric_and_flag
        self._column_types = resolve_column_types(
            self.train, referenced, self.column_types)
        self._numeric_features, self._flag_features = numeric_and_flag(
            self._column_types)
        self._class_labels = sorted(self.train[self.target].astype(str).unique())
        # Search-space spec: built-in defaults, overlaid by any user override.
        # Whatever is actually used is recorded in the manifest.
        from DSSP2026.experiment import spaces as SP
        spec = SP.default_spaces()
        if self.search_space:
            spec.update(self.search_space)
        self._search_space = spec

    # -- folder layout: fixed filenames inside experiment_dir --
    @property
    def study_db(self) -> str:
        return str(Path(self.experiment_dir) / "study.db")

    @property
    def eval_db(self) -> str:
        return str(Path(self.experiment_dir) / "eval.db")

    @property
    def report_db(self) -> str:
        return str(Path(self.experiment_dir) / "report.db")

    @property
    def config_path(self) -> str:
        return str(Path(self.experiment_dir) / "config.json")

    @property
    def storage_url(self) -> str:
        return f"sqlite:///{Path(self.study_db).resolve()}"

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
            if verbose:
                ev = sr.eval
                print(f"    best {ev.feature_set} | CV={ev.best_cv_value:.4f} | "
                      f"eval F1={ev.metrics.get('F1', float('nan')):.4f}")

        if build_report:
            from DSSP2026.experiment.report_builder import build_report_db
            build_report_db(self.study_db, self.eval_db, self.report_db,
                            meta=self.meta())

        self._write_config(status="COMPLETE")
        if verbose:
            print(f"[{self.experiment_id}] COMPLETE — {self.experiment_dir}")
        return self

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
        }

    def _write_config(self, *, status):
        """Write config.json — the self-describing experiment record."""
        import json
        manifest = self.meta()
        manifest["status"] = status
        manifest["artifacts"] = {
            "study_db": "study.db", "eval_db": "eval.db",
            "report_db": "report.db"}
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
