"""
experiment/refit.py — rebuild a fitted predictor from stored configuration.

The report layer (``CostDecision.fit``) needs to reconstruct a winning model
*outside* an Optuna run, from what ``report.db`` records: the model name, its
winning hyperparameters, the feature list, the target, and the column-type map.
This module is the single place that maps that record back onto each family's
own ``fit`` code — so the report layer never reimplements model construction
(which is what made MLP/logistic refit break).

``refit_estimator`` returns a uniform ``RefitEstimator`` exposing
``predict_proba(X)`` with columns following ``class_order`` (original label
space) for *every* family, including the statsmodels formula-based logistic
model, which is wrapped to present the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RefitEstimator:
    """A refit predictor with a uniform ``predict_proba`` across families.

    ``predict_proba(X)`` returns an ``(n, K)`` array whose columns follow
    ``class_order`` (string labels). ``features`` is the column list the model
    consumes; ``model`` is the underlying fitted object (sklearn pipeline,
    xgboost wrapper, or statsmodels result) for callers that want it.
    """
    model: object
    features: list
    class_order: list
    _proba_fn: object

    def predict_proba(self, X) -> np.ndarray:
        return self._proba_fn(X)


def _column_roles(train, features, column_types):
    """Resolve (numeric_features, flag_features) for the MLP pipeline.

    Uses the stored ``column_types`` map when available; otherwise re-infers
    from the training-frame dtypes (matching what the experiment layer did).
    """
    from DSSP2026.experiment.columns import (
        resolve_column_types, numeric_and_flag)
    resolved = resolve_column_types(train, features, column_types or None)
    return numeric_and_flag(resolved)


def refit_estimator(model, train, *, target, features, hyperparams,
                    column_types=None, random_state=42) -> RefitEstimator:
    """Refit one model on ``train`` from stored config; return a RefitEstimator.

    Parameters
    ----------
    model : str
        Family name ("Logistic regression", "Decision tree", "Random forest",
        "MLP", "XGBoost").
    train : DataFrame
        Full training frame (from the parquet sidecar).
    target : str
        Target column.
    features : sequence of str
        The winning feature list.
    hyperparams : dict
        The winning hyperparameters as stored in report.db (search-space form,
        e.g. ``width1..widthN`` / ``n_layers`` for the MLP — composed here the
        same way the family's ``refit_best`` composes them).
    column_types : mapping, optional
        Resolved column-type map; needed for the MLP preprocessor. Re-inferred
        from dtypes when absent.
    random_state : int
    """
    features = list(features)

    if model == "Logistic regression":
        return _refit_logistic(train, target, features)
    if model == "Decision tree":
        return _refit_tree(train, target, features, hyperparams, random_state)
    if model == "Random forest":
        return _refit_rf(train, target, features, hyperparams, random_state)
    if model == "MLP":
        return _refit_mlp(train, target, features, hyperparams,
                          column_types, random_state)
    if model == "XGBoost":
        return _refit_xgb(train, target, features, hyperparams, random_state)
    raise ValueError(f"refit not supported for model {model!r}.")


# ---------------------------------------------------------------------------
# Per-family refit (each delegates to the family's own fit code)
# ---------------------------------------------------------------------------

def _refit_tree(train, target, features, hp, random_state):
    from DSSP2026.tree.classification.fit import fit_decision_tree_classifier
    res = fit_decision_tree_classifier(
        train, train, features, target,
        max_depth=hp["max_depth"], average="macro", random_state=random_state)
    est = res.model
    class_order = [str(c) for c in res.classes_]

    def proba(X):
        return est.predict_proba(X[features] if hasattr(X, "columns") else X)
    return RefitEstimator(est, features, class_order, proba)


def _refit_rf(train, target, features, hp, random_state):
    from DSSP2026.tree.classification.fit import fit_random_forest_classifier
    res = fit_random_forest_classifier(
        train, train, features, target,
        n_estimators=hp["n_estimators"], max_features=hp["max_features"],
        average="macro", random_state=random_state)
    est = res.model
    class_order = [str(c) for c in res.classes_]

    def proba(X):
        return est.predict_proba(X[features] if hasattr(X, "columns") else X)
    return RefitEstimator(est, features, class_order, proba)


def _refit_mlp(train, target, features, hp, column_types, random_state):
    from sklearn.preprocessing import LabelEncoder
    from DSSP2026.mlp.fit import fit_mlp_classifier

    numeric_features, flag_features = _column_roles(train, features, column_types)
    hidden = tuple(hp[f"width{i}"] for i in range(1, hp["n_layers"] + 1))
    le = LabelEncoder().fit(train[target])

    res = fit_mlp_classifier(
        train, train, features, target, numeric_features, flag_features,
        hidden=hidden, activation=hp["activation"], alpha=hp["alpha"],
        lr_init=hp["lr_init"], label_encoder=le, average="macro",
        random_state=random_state)
    pipe = res.model
    class_order = [str(c) for c in res.classes_]
    # pipe.predict_proba columns follow the ENCODED order; map to label space.
    enc_order = [str(c) for c in le.inverse_transform(pipe.classes_)]

    def proba(X):
        Xf = X[features] if hasattr(X, "columns") else X
        p = pipe.predict_proba(Xf)
        # Reorder columns from encoded order to class_order (they match here,
        # since class_order was built from the same inverse_transform).
        return p
    return RefitEstimator(pipe, features, enc_order, proba)


def _refit_xgb(train, target, features, hp, random_state):
    from sklearn.preprocessing import LabelEncoder
    from DSSP2026.xgboost.fit import fit_xgb_classifier

    le = LabelEncoder().fit(train[target])
    res = fit_xgb_classifier(
        train, train, features, target,
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], subsample=hp["subsample"],
        colsample_bytree=hp.get("colsample_bytree", 1.0),
        min_child_weight=hp.get("min_child_weight", 1),
        label_encoder=le, average="macro", random_state=random_state)
    clf = res.model
    class_order = [str(c) for c in res.classes_]

    def proba(X):
        return clf.predict_proba(X[features] if hasattr(X, "columns") else X)
    return RefitEstimator(clf, features, class_order, proba)


def _refit_logistic(train, target, features):
    """Refit binary/multiclass logistic via the statsmodels formula path and
    wrap it to expose ``predict_proba`` with columns following ``class_order``.
    """
    from DSSP2026.experiment import logistic_adapter as LA

    binary = LA.is_binary(train, target)
    formula = f"{target} ~ " + " + ".join(features)

    if binary:
        from DSSP2026.logistic_regression.binary import (
            fit_logit, predict_proba as _bin_proba)
        # logistic_adapter encodes string targets; mirror that for the refit.
        enc_train, label_map = _maybe_encode(train, target)
        res = fit_logit(enc_train, formula)
        # class_order in original label space: [neg, pos].
        if label_map is not None:
            class_order = [label_map[0], label_map[1]]
        else:
            class_order = ["0", "1"]

        def proba(X):
            p_pos = np.asarray(_bin_proba(res.model, X), dtype=float).reshape(-1)
            return np.column_stack([1.0 - p_pos, p_pos])
        return RefitEstimator(res.model, features, class_order, proba)

    # Multiclass logistic.
    from DSSP2026.logistic_regression.multiclass import (
        fit_mnlogit, predict_mnlogit)
    res = fit_mnlogit(train, formula)
    pred0 = predict_mnlogit(res, train.head(1))
    class_order = [str(c) for c in pred0.proba.columns]

    def proba(X):
        return np.asarray(predict_mnlogit(res, X).proba.to_numpy(), dtype=float)
    return RefitEstimator(res, features, class_order, proba)


def _maybe_encode(train, target):
    """Encode a non-numeric binary target to 0/1; return (frame, label_map|None).

    Mirrors logistic_adapter's encoding so the refit sees the same 0/1 endog the
    experiment used, and the positive class (higher sort value) maps to 1.
    """
    import pandas as pd
    s = train[target]
    if pd.api.types.is_bool_dtype(s):
        out = train.copy(); out[target] = s.astype(int)
        return out, None
    if pd.api.types.is_numeric_dtype(s):
        uniques = set(pd.unique(s.dropna()))
        if uniques <= {0, 1}:
            out = train.copy(); out[target] = s.astype(int)
            return out, None
    levels = sorted(s.dropna().unique(), key=str)
    label_map = {0: levels[0], 1: levels[1]}
    out = train.copy()
    out[target] = s.map({levels[0]: 0, levels[1]: 1}).astype(int)
    return out, label_map
