"""
report/cost_fit.py — attach a cost-optimal decision layer to a refit model.

``CostDecision.fit()`` lands here. It loads the training parquet, refits the
winning model **through the experiment layer's** ``refit_estimator`` (the single
source of truth for model construction — no reimplementation), and wraps the
result in a :class:`CostDecisionModel` that applies the winning policy's
decision rule to fresh probabilities.

For an Ensemble winner, every member is refit and their probabilities averaged
before the decision layer is applied.
"""

from __future__ import annotations

import json

import numpy as np

from DSSP2026.report.base import ENSEMBLE_NAME
from DSSP2026.report.cost.math import expected_value_decisions


class CostDecisionModel:
    """A refit predictor (or ensemble) plus a cost-optimal decision layer.

    ``predict_proba(X)`` returns the class-probability matrix; ``predict(X)``
    applies the winning policy to produce cost-optimal class decisions.
    """

    def __init__(self, *, estimators, features, class_order, policy, threshold,
                 schedule):
        # estimators: list of RefitEstimator (1 for a single model; N averaged
        # for an ensemble). All share features/class_order.
        self._estimators = estimators
        self.features = list(features)
        self.class_order = [str(c) for c in class_order]
        self.policy = policy
        self.threshold = threshold
        self._schedule = schedule

    def predict_proba(self, X) -> np.ndarray:
        probas = [np.asarray(e.predict_proba(X), dtype=float)
                  for e in self._estimators]
        return np.mean(probas, axis=0)

    def predict(self, X) -> np.ndarray:
        return self._decide(self.predict_proba(X))

    def _decide(self, proba):
        co = self.class_order
        if self.policy == "ArgMax":
            return np.asarray([co[i] for i in proba.argmax(axis=1)], dtype=object)
        if self.policy in ("F1", "Youden's J"):
            from DSSP2026.core.threshold import decisions_from_thresholds
            return decisions_from_thresholds(proba, co, self.threshold or {})
        if self.policy == "Bayes":
            return expected_value_decisions(co, proba, self._schedule)
        raise ValueError(f"unknown policy {self.policy!r}.")


def _model_record(report, model_name):
    """Read (hyperparams, feature_list) for a model from report.db."""
    conn = report._connect()
    try:
        row = conn.execute(
            "SELECT hyperparams, feature_list FROM models "
            "WHERE experiment_id=? AND model=?",
            (report.experiment_id, model_name)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"model {model_name!r} not in report.db.")
    hp = json.loads(row["hyperparams"]) if row["hyperparams"] else {}
    features = json.loads(row["feature_list"])
    return hp, features


def _column_types(report):
    """Stored resolved column-type map for this experiment, or None."""
    conn = report._connect()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
        if "column_types" not in cols:
            return None
        row = conn.execute(
            "SELECT column_types FROM experiments WHERE experiment_id=?",
            (report.experiment_id,)).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def _refit_one(report, model_name, train_df, target, column_types):
    """Refit one model via the experiment layer's refit_estimator."""
    from DSSP2026.experiment.refit import refit_estimator
    hp, features = _model_record(report, model_name)
    return refit_estimator(
        model_name, train_df, target=target, features=features,
        hyperparams=hp, column_types=column_types)


def fit_cost_model(report, model_type, policy, threshold, schedule,
                   class_order, *, allow_ensemble=False):
    """Refit the winning model(s) and return a CostDecisionModel.

    Requires the training parquet (``ReportBase.load_train_data``); raises a
    clear error if it's missing or has drifted.
    """
    loaded = report.load_train_data()
    if loaded is None:
        raise RuntimeError(
            "cost fit requires the training parquet, which is missing or has "
            "drifted. Re-run the experiment so .artifacts/train.parquet sits "
            "next to report.db.")
    train_df, target = loaded
    column_types = _column_types(report)

    if model_type == ENSEMBLE_NAME:
        members = report.models(include_ensemble=False)
        refits = [_refit_one(report, m, train_df, target, column_types)
                  for m in members]
        # Members share the feature list and class order.
        features = refits[0].features
        co = refits[0].class_order
        return CostDecisionModel(
            estimators=refits, features=features, class_order=co,
            policy=policy, threshold=threshold, schedule=schedule)

    refit = _refit_one(report, model_type, train_df, target, column_types)
    return CostDecisionModel(
        estimators=[refit], features=refit.features,
        class_order=refit.class_order, policy=policy, threshold=threshold,
        schedule=schedule)
