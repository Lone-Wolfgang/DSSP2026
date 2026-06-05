"""
experiment/objectives.py — unified Optuna objective factory.

One objective per model. Every objective:
  1. samples ``feature_set`` (categorical) + the model's hyperparameters,
  2. scores the configuration by stratified k-fold macro-F1 on TRAIN (via
     experiment.cv — never touching the held-out set),
  3. returns that macro-F1 to Optuna, stashing per-trial extras (CV std/SE for
     the one-SE rule; loss curves for MLP/XGB) as trial ``user_attrs``.

The sklearn-estimator models (decision tree, random forest) route through
``cv.cv_score``; logistic uses a statsmodels per-fold fit through
``cv.cv_fold_scores``; MLP and XGBoost reuse their existing curve-recording CV
(which itself now routes through ``cv.cv_fold_scores``).

User-attr keys written here (read later by the study/report layers):
  CV_STD_KEY, CV_SE_KEY   — fold std / standard error of the CV objective.
  feature_set is a normal Optuna param (trial.params["feature_set"]).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from DSSP2026.experiment import cv as CV
from DSSP2026.experiment import spaces

CV_STD_KEY = "cv_std"
CV_SE_KEY = "cv_se"


# ---------------------------------------------------------------------------
# sklearn-estimator objectives (decision tree, random forest)
# ---------------------------------------------------------------------------

def _sklearn_objective(estimator_builder, model_name, train, target,
                       feature_sets, *, n_splits, scoring, random_state, spec):
    """Generic objective for plain sklearn estimators scored by cv.cv_score."""
    names = list(feature_sets)
    y = train[target].to_numpy()

    def objective(trial):
        fs = spaces.suggest_feature_set(trial, names)
        features = list(feature_sets[fs])
        params = spaces.suggest_params(trial, model_name, spec)
        X = train[features]

        def factory():
            return estimator_builder(params, random_state)

        score = CV.cv_score(
            factory, X, y, n_splits=n_splits, scoring=scoring,
            stratified=True, random_state=random_state)
        trial.set_user_attr(CV_STD_KEY, score.std)
        trial.set_user_attr(CV_SE_KEY, score.se)
        return score.mean

    return objective


def decision_tree_objective(train, target, feature_sets, *, n_splits=5,
                            scoring="f1_macro", random_state=42, spec=None):
    from sklearn.tree import DecisionTreeClassifier

    def builder(params, rs):
        return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=rs)

    return _sklearn_objective(
        builder, "Decision tree", train, target, feature_sets,
        n_splits=n_splits, scoring=scoring, random_state=random_state, spec=spec)


def random_forest_objective(train, target, feature_sets, *, n_splits=5,
                            scoring="f1_macro", random_state=42, spec=None):
    from sklearn.ensemble import RandomForestClassifier

    def builder(params, rs):
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_features=params["max_features"],
            random_state=rs, n_jobs=-1)

    return _sklearn_objective(
        builder, "Random forest", train, target, feature_sets,
        n_splits=n_splits, scoring=scoring, random_state=random_state, spec=spec)


# ---------------------------------------------------------------------------
# logistic regression objective (statsmodels MNLogit, per-fold via cv_fold_scores)
# ---------------------------------------------------------------------------

def logistic_objective(train, target, feature_sets, *, n_splits=5,
                       scoring="f1_macro", random_state=42, maxiter=100):
    """One config per feature set; CV macro-F1 of a logit per fold.

    Binary vs multiclass routing is delegated to ``logistic_adapter`` so the
    class-count decision lives in exactly one place; this objective is shape-
    and cardinality-agnostic.
    """
    from DSSP2026.experiment import logistic_adapter as LA

    binary = LA.is_binary(train, target)
    names = list(feature_sets)
    y = train[target].to_numpy()

    def objective(trial):
        fs = spaces.suggest_feature_set(trial, names)
        features = list(feature_sets[fs])
        formula = f"{target} ~ " + " + ".join(features)

        def fold_fn(tr_rows, ytr, va_rows, yva):
            return LA.fold_macro_f1(
                tr_rows, va_rows, formula=formula, target=target,
                binary=binary, maxiter=maxiter)

        scores = CV.cv_fold_scores(
            fold_fn, train, y, n_splits=n_splits, stratified=True,
            random_state=random_state)
        arr = np.asarray(scores, dtype=float)
        trial.set_user_attr(CV_STD_KEY, float(arr.std(ddof=0)))
        trial.set_user_attr(CV_SE_KEY, float(arr.std(ddof=0) / np.sqrt(n_splits)))
        return float(arr.mean())

    return objective


# ---------------------------------------------------------------------------
# MLP / XGBoost objectives (reuse existing curve-recording CV)
# ---------------------------------------------------------------------------

def mlp_objective(train, target, feature_sets, numeric_features, flag_features,
                  label_encoder, *, n_splits=5, random_state=42, spec=None):
    """Reuses mlp.tune.cv_macro_f1_with_curves; stores loss curves as user_attrs."""
    from DSSP2026.mlp.tune import cv_macro_f1_with_curves
    from DSSP2026.tuning.search import TRAIN_CURVE_KEY, EVAL_CURVE_KEY

    y = label_encoder.transform(train[target])
    names = list(feature_sets)

    def objective(trial):
        fs = spaces.suggest_feature_set(trial, names)
        features = list(feature_sets[fs])
        p = spaces.suggest_params(trial, "MLP", spec)
        f1, tr_curve, ev_curve = cv_macro_f1_with_curves(
            train[features], y, features, numeric_features, flag_features,
            hidden=p["hidden"], activation=p["activation"], alpha=p["alpha"],
            lr_init=p["lr_init"], n_splits=n_splits, random_state=random_state)
        trial.set_user_attr(TRAIN_CURVE_KEY, tr_curve)
        trial.set_user_attr(EVAL_CURVE_KEY, ev_curve)
        return f1

    return objective


def xgboost_objective(train, target, feature_sets, label_encoder, *,
                      n_splits=5, random_state=42, spec=None):
    """Reuses xgboost.tune.cv_macro_f1_with_curves; stores loss curves."""
    from DSSP2026.xgboost.tune import cv_macro_f1_with_curves
    from DSSP2026.tuning.search import TRAIN_CURVE_KEY, EVAL_CURVE_KEY

    y = label_encoder.transform(train[target])
    names = list(feature_sets)

    def objective(trial):
        fs = spaces.suggest_feature_set(trial, names)
        features = list(feature_sets[fs])
        p = spaces.suggest_params(trial, "XGBoost", spec)
        f1, tr_curve, ev_curve = cv_macro_f1_with_curves(
            train[features], y, features,
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
            reg_alpha=p["reg_alpha"], min_child_weight=p["min_child_weight"],
            n_splits=n_splits, random_state=random_state)
        trial.set_user_attr(TRAIN_CURVE_KEY, tr_curve)
        trial.set_user_attr(EVAL_CURVE_KEY, ev_curve)
        return f1

    return objective