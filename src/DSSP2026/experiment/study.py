"""
experiment/study.py — run one model's Optuna study and evaluate its winner.

A *study* corresponds to one model. ``run_study`` builds the model's objective
(objectives.py), runs the Optuna search persisted to experiment.db, selects the
winning configuration, refits it on TRAIN, and evaluates **once** on the
held-out set — producing an ``EvalRecord`` (metrics + per-class + confusion +
probability matrix) plus the trials.

Winner selection:
  - **decision tree**: the one-standard-error rule (shallowest depth whose mean
    CV macro-F1 is within one SE of the best trial's), applied post-hoc over the
    study's trials — preserving the prior parsimony behaviour.
  - **all other models**: Optuna's ``best_trial`` (argmax CV value).

Sampler:
  - **random forest**: ``GridSampler`` over its discrete grid (exhaustive).
  - **everything else**: TPE (the default in tuning.search.run_optuna_search).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.core import metrics as CM
from DSSP2026.experiment import objectives as OBJ
from DSSP2026.experiment import spaces

# Maps model display name -> sidecar family token used by write_feature_importance.
_MODEL_FAMILY = {
    "Logistic regression": "logistic",
    "Decision tree":       "tree",
    "Random forest":       "rf",
    "MLP":                 "mlp",
    "XGBoost":             "xgb",
}


@dataclass
class EvalRecord:
    """Everything recorded for a study's winning configuration."""
    model: str
    feature_set: str
    hyperparams: dict
    feature_list: list
    best_cv_value: float
    metrics: dict                  # Accuracy/Precision/Recall/F1/ROC-AUC
    per_class_df: pd.DataFrame
    confusion_df: pd.DataFrame     # raw integer counts
    class_order: list              # labels matching y_proba columns
    y_true: np.ndarray
    y_proba: np.ndarray
    detail: str = ""


@dataclass
class StudyResult:
    """Output of run_study: the Optuna study, the eval record, and trial rows.

    ``fit_result`` holds the ClassificationResult subclass (TreeClassifyResult,
    RFClassifyResult, XGBResult, MLPResult, or LogisticEval) returned by
    _refit_winner. It is used by experiment.py to write feature importance into
    the sidecar immediately after the study completes, then discarded.
    """
    model: str
    study: object
    eval: EvalRecord
    trials: list = field(default_factory=list)
    fit_result: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------

def select_one_se_trial(study):
    """Decision-tree one-SE rule over the study's trials.

    Picks the shallowest ``max_depth`` whose mean CV value is within one
    standard error of the best trial's mean. SE per trial was stored as the
    ``cv_se`` user-attr by the objective. Falls back to best_trial if depths or
    SEs are unavailable.
    """
    completed = [t for t in study.trials
                 if t.value is not None and "max_depth" in t.params]
    if not completed:
        return study.best_trial
    best = max(completed, key=lambda t: t.value)
    best_se = best.user_attrs.get(OBJ.CV_SE_KEY, 0.0) or 0.0
    threshold = best.value - best_se
    qualifying = [t for t in completed if t.value >= threshold]
    # Shallowest depth among qualifying trials (parsimony).
    return min(qualifying, key=lambda t: t.params["max_depth"])


def _grid_from_spec(spec: list, feature_sets: Mapping[str, Sequence[str]]) -> dict:
    grid = {"feature_set": list(feature_sets)}
    for d in spec:
        if d.get("type") == "categorical":
            grid[d["name"]] = list(d["choices"])
        elif d.get("type") == "int":
            step = d.get("step", 1)
            grid[d["name"]] = list(range(int(d["low"]), int(d["high"]) + 1, int(step)))
    return grid


def _build_eval_record(model, best_params, feature_set, features, best_cv_value,
                       res, detail=""):
    """Assemble an EvalRecord from a family's ClassificationResult refit.

    ``res`` exposes ``y_true``, ``y_pred``, ``y_proba`` (cols follow
    ``classes_``), and ``classes_``. Mirrors the prior _save_common_eval bundle.
    """
    labels = list(res.classes_)
    y_proba = np.asarray(
        res.y_proba.to_numpy() if hasattr(res.y_proba, "to_numpy") else res.y_proba,
        dtype=float)
    y_true = (res.y_true.to_numpy() if hasattr(res.y_true, "to_numpy")
              else np.asarray(res.y_true))

    metrics = CM.classification_metrics(
        res.y_true, res.y_pred, y_score=res.y_proba, average="macro")
    per_class = CM.make_classification_report_df(
        res.y_true, res.y_pred, target_names=labels)
    confusion = CM.make_confusion_matrix(res.y_true, res.y_pred, labels=labels)

    return EvalRecord(
        model=model, feature_set=feature_set,
        hyperparams=dict(best_params), feature_list=list(features),
        best_cv_value=float(best_cv_value), metrics=metrics,
        per_class_df=per_class, confusion_df=confusion,
        class_order=[str(c) for c in labels],
        y_true=np.asarray([str(v) for v in y_true.tolist()], dtype=object),
        y_proba=y_proba, detail=detail)


# ---------------------------------------------------------------------------
# Per-model run + refit + evaluate
# ---------------------------------------------------------------------------

def run_study(model, train, evaluation, *, target, feature_sets,
              numeric_features, flag_features, class_labels,
              n_trials=30, n_splits=5, scoring="f1_macro", random_state=42,
              storage=None, study_name=None, spec=None):
    """Run one model's study end-to-end; return a StudyResult.

    ``storage`` is an Optuna storage URL (e.g. ``sqlite:///study.db``) so trials
    persist; ``study_name`` identifies the study inside it. ``spec`` is the
    model's search-space spec (defaults to the built-in for that model).
    """
    import optuna
    from DSSP2026.experiment.tuning.search import run_optuna_search
    from DSSP2026.experiment.trials import trials_from_study
    from DSSP2026.experiment import spaces as SP

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    entry = spec or SP.normalize_search_space(None)[model]
    if isinstance(entry, dict) and "params" in entry:
        sampler_name = entry.get("sampler", SP.DEFAULT_SAMPLERS[model])
        spec = entry["params"]
    else:
        sampler_name = SP.DEFAULT_SAMPLERS[model]
        spec = entry

    # Build the objective + choose sampler.
    sampler = None
    if model == "Logistic regression":
        objective = OBJ.logistic_objective(
            train, target, feature_sets, n_splits=n_splits, scoring=scoring,
            random_state=random_state)
        n = len(feature_sets)  # one config per feature set
    elif model == "Decision tree":
        objective = OBJ.decision_tree_objective(
            train, target, feature_sets, n_splits=n_splits, scoring=scoring,
            random_state=random_state, spec=spec)
        n = n_trials
    elif model == "Random forest":
        objective = OBJ.random_forest_objective(
            train, target, feature_sets, n_splits=n_splits, scoring=scoring,
            random_state=random_state, spec=spec)
        n = n_trials
    elif model == "MLP":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(train[target])
        objective = OBJ.mlp_objective(
            train, target, feature_sets, numeric_features, flag_features, le,
            n_splits=n_splits, random_state=random_state, spec=spec)
        n = n_trials
    elif model == "XGBoost":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(train[target])
        objective = OBJ.xgboost_objective(
            train, target, feature_sets, le, n_splits=n_splits,
            random_state=random_state, spec=spec)
        n = n_trials
    else:
        raise ValueError(f"unknown model {model!r}")

    if model != "Logistic regression" and sampler_name == "grid":
        grid = _grid_from_spec(spec, feature_sets)
        sampler = optuna.samplers.GridSampler(grid)
        n = int(np.prod([len(v) for v in grid.values()]))

    # Run the search (persisted to experiment.db if storage given).
    if sampler is not None:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction="maximize",
            sampler=sampler, load_if_exists=True)
        study.optimize(objective, n_trials=n)
    else:
        result = run_optuna_search(
            objective, n_trials=n, n_splits=n_splits, scoring=scoring,
            study_name=study_name, storage=storage, random_state=random_state)
        study = result.study

    # Select the winner.
    if model == "Decision tree":
        best_trial = select_one_se_trial(study)
    else:
        best_trial = study.best_trial
    best_params = dict(best_trial.params)
    feature_set = best_params.get("feature_set", list(feature_sets)[0])
    features = list(feature_sets[feature_set])

    # Refit the winner on TRAIN and evaluate on the held-out set.
    res, detail = _refit_winner(
        model, train, evaluation, target, features, best_params,
        numeric_features, flag_features, random_state)

    eval_record = _build_eval_record(
        model, best_params, feature_set, features, best_trial.value, res, detail)

    trials = trials_from_study(study, model=model)
    return StudyResult(model=model, study=study, eval=eval_record, trials=trials,
                       fit_result=res)


def _refit_winner(model, train, evaluation, target, features, best_params,
                  numeric_features, flag_features, random_state):
    """Refit the winning config on TRAIN, score on the held-out set."""
    if model == "Logistic regression":
        from DSSP2026.experiment import logistic_adapter as LA

        formula = f"{target} ~ " + " + ".join(features)
        binary = LA.is_binary(train, target)
        # refit_eval returns a uniform (n, K) LogisticEval for both
        # cardinalities, so _build_eval_record needs no binary special-casing.
        r = LA.refit_eval(
            train, evaluation, formula=formula, target=target, binary=binary)
        kind = "binary" if binary else "multiclass"
        return r, f"{kind}, feature_set={_fs_name(features)}"

    if model == "Decision tree":
        from DSSP2026.tree.classification.fit import fit_decision_tree_classifier
        res = fit_decision_tree_classifier(
            train, evaluation, features, target,
            max_depth=best_params["max_depth"], average="macro",
            random_state=random_state)
        return res, f"depth={best_params['max_depth']}"

    if model == "Random forest":
        from DSSP2026.tree.classification.fit import fit_random_forest_classifier
        # Pass hyperparameters (not a pre-built estimator) so the fit function
        # builds AND fits internally — the estimator= path assumes a pre-fit model.
        res = fit_random_forest_classifier(
            train, evaluation, features, target,
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            average="macro", random_state=random_state)
        return res, (f"n_est={best_params['n_estimators']}, "
                     f"max_feat={best_params['max_features']}")

    if model == "MLP":
        # Reuse the study's best config via refit_best, which reads best_params.
        # We pass a tiny shim study exposing best_params.
        from sklearn.preprocessing import LabelEncoder
        from DSSP2026.mlp.fit import refit_best
        le = LabelEncoder().fit(train[target])
        fs_map = {best_params["feature_set"]: features}

        class _S:
            pass
        s = _S(); s.best_params = best_params
        res = refit_best(
            s, train, evaluation, target, fs_map, numeric_features,
            flag_features, label_encoder=le, average="macro",
            random_state=random_state)
        return res, f"feature_set={best_params['feature_set']} (Optuna-selected)"

    if model == "XGBoost":
        from sklearn.preprocessing import LabelEncoder
        from DSSP2026.xgboost.fit import refit_best
        le = LabelEncoder().fit(train[target])
        fs_map = {best_params["feature_set"]: features}

        class _S:
            pass
        s = _S(); s.best_params = best_params
        res = refit_best(
            s, train, evaluation, target, fs_map, label_encoder=le,
            average="macro", random_state=random_state)
        return res, f"feature_set={best_params['feature_set']} (Optuna-selected)"

    raise ValueError(f"unknown model {model!r}")


def _fs_name(features):
    return f"{len(features)} features"