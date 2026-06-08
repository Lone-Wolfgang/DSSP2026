"""
report/fit.py — metric-policy model selection and deployable fitting.

``Report.fit`` is the metric-only sibling of ``Report.cost_optimize().fit()``:
instead of choosing the (model, policy) pair that maximises dollar net benefit,
it picks the candidate model that maximises a chosen *classification* metric
under a chosen decision policy, refits it from the training parquet, and wraps
it in a deployable :class:`PolicyModel` that re-applies the same policy to fresh
probabilities.

Model construction is never reimplemented here: refitting goes through the
experiment layer's :func:`refit_estimator` (the single source of truth used by
the cost layer too). For an Ensemble winner, the selected members are refit and
their probabilities averaged before the decision layer is applied — the same
mean-probability ensemble ``compare_models`` scores.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

from DSSP2026.report.base import ENSEMBLE_NAME
from DSSP2026.report.policy import (
    validate_policy, decisions_under_policy, metrics_from_decisions)


class PolicyModel:
    """A refit predictor (or ensemble) plus a metric-policy decision layer.

    ``predict_proba(X)`` returns the class-probability matrix (mean across
    members for an ensemble); ``predict(X)`` applies the winning policy's
    decision rule. Schedule-free counterpart to ``CostDecisionModel``.

    Attributes
    ----------
    model_type : str
        The winning model name (a family name or ``"Ensemble"``).
    policy : str
        The decision policy applied ("ArgMax", "F1", "Youden's J").
    threshold : dict | None
        Per-class cutoffs for a tuned policy, else None (ArgMax).
    metric : str
        The metric the selection optimised (e.g. "f1"); the ranking metric.
    fit_score : float
        The winner's out-of-fold (train CV) value of the ranking metric.
    test_metrics : dict | None
        Held-out test-set metrics, populated by ``Report.fit`` when a test
        partition is available; None otherwise.
    """

    def __init__(self, *, estimators, features, class_order, model_type,
                 policy, threshold, metric, fit_score):
        self._estimators = estimators
        self.features = list(features)
        self.class_order = [str(c) for c in class_order]
        self.model_type = model_type
        self.policy = policy
        self.threshold = threshold
        self.metric = metric
        self.fit_score = fit_score
        self.test_metrics = None

    def predict_proba(self, X) -> np.ndarray:
        probas = [np.asarray(e.predict_proba(X), dtype=float)
                  for e in self._estimators]
        return np.mean(probas, axis=0)

    def predict(self, X) -> np.ndarray:
        return decisions_under_policy(
            self.policy, self.class_order, self.predict_proba(X),
            self.threshold)

    def score(self, test_df, target, *, decimals: int = 4):
        """Evaluate this fitted model on a held-out test frame.

        Runs the full decision pipeline (``predict_proba`` -> policy decision)
        on ``test_df`` and returns the held-out classification metrics. This is
        the leak-free evaluation: the estimators were fit on train, the policy
        thresholds were tuned on train (CV out-of-fold), and ``test_df`` is the
        untouched partition persisted as ``test.parquet`` — nothing in the
        selection/tuning pipeline has seen it.

        Parameters
        ----------
        test_df : DataFrame
            The test partition, including the target column.
        target : str
            Target column name in ``test_df``.
        decimals : int
            Rounding for the returned metric values.

        Returns
        -------
        dict
            ``{"accuracy", "precision", "recall", "f1"}`` on the test set.
        """
        y_true = np.asarray(test_df[target], dtype=object)
        y_pred = self.predict(test_df)
        m = metrics_from_decisions(y_true, y_pred, self.class_order)
        return {k: round(float(v), decimals) for k, v in m.items()}

    def __repr__(self):
        thr = "argmax" if self.threshold is None else "per-class thresholds"
        return (f"PolicyModel(model={self.model_type!r}, policy={self.policy!r}, "
                f"{thr}, {self.metric}={self.fit_score:.4f})")


class FitMixin:

    def fit(self, *, policy: str = "ArgMax",
            models: Optional[Sequence[str]] = None,
            allow_ensemble: bool = False,
            rank_by: Optional[str] = None,
            n_splits: int = 5,
            random_state: int = 42,
            evaluate_on_test: bool = True) -> PolicyModel:
        """Tune on train (CV), select the best candidate, refit, score on test.

        The leak-free pipeline:

        1. **Hyperparameters** were already tuned on train (k-fold CV) by the
           experiment that produced report.db.
        2. **Decision thresholds** are tuned here on train, via *out-of-fold*
           k-fold CV: each fold refits the candidate on the fold-train slice
           and scores the fold-validation slice, the OOF probabilities are
           pooled, and per-class thresholds are tuned on that pool. No row's
           threshold is chosen using a model that saw it.
        3. **Selection** ranks candidates by their OOF score under the policy
           (also train-only — the eval set is never touched here).
        4. **Refit** the winner on the *full* train set for deployment.
        5. **Evaluation** (optional) scores the fitted model on the untouched
           ``test.parquet`` partition.

        Parameters
        ----------
        policy : {"ArgMax", "F1", "Youden's J"}
            Decision rule. ArgMax tunes nothing (plain argmax); F1 / Youden's J
            tune per-class thresholds on the OOF pool toward that criterion.
        models : sequence of str, optional
            Candidate pool (real models). ``None`` -> all. Selection and any
            Ensemble draw only from this pool.
        allow_ensemble : bool
            Include the mean-probability Ensemble of the pool as a candidate
            (default False). Requires >= 2 real candidates.
        rank_by : str, optional
            Final-decision metric used to rank candidates: ``"f1"`` (default),
            ``"accuracy"``, ``"precision"``, ``"recall"``. Independent of the
            policy (which only sets the decision rule).
        n_splits : int
            Folds for the train-side threshold CV (default 5).
        random_state : int
            Seed for the stratified fold splitter (default 42).
        evaluate_on_test : bool
            When True (default) and ``test.parquet`` is available, attach the
            test-set metrics to the returned model's ``test_metrics``. When the
            test partition is absent, ``test_metrics`` stays None.

        Returns
        -------
        PolicyModel
            A deployable predictor with ``.predict`` / ``.predict_proba`` /
            ``.score`` and, when evaluated, ``.test_metrics``.
        """
        validate_policy(policy)

        real = self.models(include_ensemble=False)
        if models is not None:
            want = {models} if isinstance(models, str) else set(models)
            pool = [m for m in real if m in want]
        else:
            pool = list(real)
        if not pool:
            raise ValueError("no candidate models after filtering `models`.")

        candidates = list(pool)
        if allow_ensemble and len(pool) >= 2:
            candidates.append(ENSEMBLE_NAME)

        rank_metric = rank_by or "f1"
        _RANKABLE = ("accuracy", "precision", "recall", "f1")
        if rank_metric not in _RANKABLE:
            raise ValueError(
                f"rank_by must be one of {list(_RANKABLE)}; got {rank_metric!r}.")

        # Load train once; needed for OOF tuning AND the final refit.
        from DSSP2026.report.cost.fit import _column_types
        loaded = self.load_train_data()
        if loaded is None:
            raise RuntimeError(
                "fit requires the training parquet, which is missing or has "
                "drifted. Re-run the experiment so .artifacts/train.parquet "
                "sits next to report.db.")
        train_df, target = loaded
        column_types = _column_types(self)

        # Score every candidate on its OOF predictions under the policy.
        best = None   # (score, name, class_order, thresholds)
        for name in candidates:
            members = ([m for m in pool if m != ENSEMBLE_NAME]
                       if name == ENSEMBLE_NAME else [name])
            co, oof_true, oof_proba = self._oof_predictions(
                members, train_df, target, column_types,
                n_splits=n_splits, random_state=random_state)
            thr = (None if policy == "ArgMax"
                   else self._tune_thresholds_oof(
                       policy, co, oof_true, oof_proba))
            y_pred = decisions_under_policy(policy, co, oof_proba, thr)
            scored = metrics_from_decisions(oof_true, y_pred, co)
            value = scored[rank_metric]
            if best is None or value > best[0]:
                best = (value, name, co, thr)

        oof_score, win_name, win_co, win_thr = best

        # Refit the winner on the FULL train set for deployment.
        estimators, features, class_order = self._refit_full(
            win_name, pool, train_df, target, column_types)

        model = PolicyModel(
            estimators=estimators, features=features, class_order=class_order,
            model_type=win_name, policy=policy, threshold=win_thr,
            metric=rank_metric, fit_score=float(oof_score))

        # Final, leak-free evaluation on the untouched test partition.
        model.test_metrics = None
        if evaluate_on_test:
            test_loaded = self.load_test_data()
            if test_loaded is not None:
                test_df, test_target = test_loaded
                model.test_metrics = model.score(test_df, test_target)

        return model

    def best_fit(self, models: Optional[Sequence[str]] = None,
                 policies: Optional[Sequence[str]] = None, *,
                 allow_ensemble: bool = False,
                 target: str = "F1") -> "PolicyModel":
        """Pick the best (model, policy) on VALIDATION, report its TEST score.

        Sweeps every combination of the given ``models`` and ``policies``,
        scores each on the **validation** set (thresholds tuned on train OOF,
        refit on full train, applied to validation), and selects the winner by
        ``target``. The winner is then scored once on the untouched **test**
        set — that test number is leak-free because selection never touched it.

        Parameters
        ----------
        models : sequence of str, optional
            Candidate models. ``None`` -> all models in the study.
        policies : sequence of str, optional
            Candidate policies. ``None`` -> all of ArgMax, F1, Youden's J.
        allow_ensemble : bool
            Include the mean-probability Ensemble of the pool as a candidate.
        target : {"F1", "accuracy"}
            Selection metric (case-insensitive). ROC-AUC is not selectable here
            because it is threshold/policy-independent.

        Returns
        -------
        PolicyModel
            Winner refit on full train, with ``.test_metrics`` populated.
        """
        from DSSP2026.report.policy import (
            validate_policy, decisions_under_policy, metrics_from_decisions,
            POLICIES)
        from DSSP2026.report.cost.fit import _column_types

        tkey = {"f1": "f1", "accuracy": "accuracy"}.get(target.lower())
        if tkey is None:
            raise ValueError(
                f"target must be 'F1' or 'accuracy'; got {target!r} "
                "(ROC-AUC is policy-independent and not selectable here).")

        real = self.models(include_ensemble=False)
        pool = ([m for m in real if m in (
                    {models} if isinstance(models, str) else set(models))]
                if models is not None else list(real))
        if not pool:
            raise ValueError("no candidate models after filtering `models`.")
        cand_models = list(pool)
        if allow_ensemble and len(pool) >= 2:
            cand_models.append(ENSEMBLE_NAME)

        cand_policies = list(policies) if policies is not None else list(POLICIES)
        for p in cand_policies:
            validate_policy(p)

        loaded = self.load_train_data()
        val_loaded = self.load_validation_data()
        if loaded is None or val_loaded is None:
            raise RuntimeError(
                "best_fit requires train.parquet and validation.parquet. "
                "Re-run the experiment with train/validation/test splits.")
        train_df, target_col = loaded
        val_df, val_target = val_loaded
        column_types = _column_types(self)

        # Refit each candidate once on full train (shared across policies);
        # tune each model's OOF thresholds per policy.
        refit_cache, oof_cache = {}, {}
        best = None   # (val_score, model, policy, thresholds, estimators, class_order)
        for name in cand_models:
            members = ([m for m in pool if m != ENSEMBLE_NAME]
                       if name == ENSEMBLE_NAME else [name])
            if name not in refit_cache:
                refit_cache[name] = self._refit_full(
                    name, pool, train_df, target_col, column_types)
                oof_cache[name] = self._oof_predictions(
                    members, train_df, target_col, column_types,
                    n_splits=5, random_state=42)
            estimators, _, class_order = refit_cache[name]
            co, oof_true, oof_proba = oof_cache[name]
            val_proba = self._predict_proba_aligned(estimators, val_df, class_order)
            val_true = np.asarray(val_df[val_target], dtype=object)
            for policy in cand_policies:
                thr = (None if policy == "ArgMax"
                       else self._tune_thresholds_oof(policy, co, oof_true, oof_proba))
                y_pred = decisions_under_policy(policy, class_order, val_proba, thr)
                score = metrics_from_decisions(val_true, y_pred, class_order)[tkey]
                if best is None or score > best[0]:
                    best = (score, name, policy, thr, estimators, class_order)

        val_score, win_name, win_policy, win_thr, win_est, win_co = best
        model = PolicyModel(
            estimators=win_est, features=win_est[0].features, class_order=win_co,
            model_type=win_name, policy=win_policy, threshold=win_thr,
            metric=tkey, fit_score=float(val_score))
        model.validation_score = float(val_score)

        test_loaded = self.load_test_data()
        model.test_metrics = (model.score(*test_loaded)
                              if test_loaded is not None else None)
        return model

    # ------------------------------------------------------------------
    # Train-side out-of-fold prediction + threshold tuning
    # ------------------------------------------------------------------

    def _oof_predictions(self, members, train_df, target, column_types, *,
                         n_splits, random_state):
        """Pooled out-of-fold probabilities for a model (or member-averaged set).

        Stratified k-fold over ``train_df``: each fold refits every member on
        the fold-train slice (via the experiment layer's ``refit_estimator``,
        no reimplementation) and predicts the fold-validation slice; member
        probabilities are averaged for an ensemble. Returns
        ``(class_order, oof_y_true, oof_y_proba)`` with one row per training
        sample, each scored by models that did not see it.

        Fast path: when every member's OOF probabilities are already persisted
        in report.db (computed during the experiment from the selected
        hyperparameters), they are read and averaged directly — no refitting.
        Falls back to computing folds on the fly when the table is absent (old
        DBs) or a member is missing.
        """
        cached = self._read_oof_cached(members)
        if cached is not None:
            return cached

        from DSSP2026.experiment.cv import make_splitter
        from DSSP2026.experiment.refit import refit_estimator
        from DSSP2026.report.cost.fit import _model_record

        # Per-member stored config (hyperparams + feature list).
        configs = {m: _model_record(self, m) for m in members}
        # Canonical class order from a full-train refit's labels would be ideal,
        # but the stored class order is identical and cheaper; take it from the
        # training target to guarantee every class is represented.
        class_order = sorted(train_df[target].astype(str).unique())

        strat = train_df[target].astype(str).to_numpy()
        splitter = make_splitter(stratified=True, n_splits=n_splits,
                                 random_state=random_state)

        n = len(train_df)
        K = len(class_order)
        col_ix = {c: j for j, c in enumerate(class_order)}
        oof = np.full((n, K), np.nan, dtype=float)

        for tr_i, va_i in splitter.split(train_df, strat):
            fold_train = train_df.iloc[tr_i]
            fold_val = train_df.iloc[va_i]
            member_probas = []
            for m in members:
                hp, features = configs[m]
                est = refit_estimator(
                    m, fold_train, target=target, features=features,
                    hyperparams=hp, column_types=column_types)
                p = np.asarray(est.predict_proba(fold_val), dtype=float)
                # Align this member's columns to the canonical class order.
                aligned = np.zeros((len(fold_val), K), dtype=float)
                for j, c in enumerate(est.class_order):
                    if str(c) in col_ix:
                        aligned[:, col_ix[str(c)]] = p[:, j]
                member_probas.append(aligned)
            oof[va_i] = np.mean(member_probas, axis=0)

        oof_true = train_df[target].astype(str).to_numpy()
        return class_order, oof_true, oof

    def _read_oof_cached(self, members):
        """Read persisted OOF for ``members`` from report.db; None if unavailable.

        Returns ``(class_order, oof_y_true, mean_oof_proba)`` averaging the
        members (mirroring the on-the-fly ensemble), or None when the table is
        missing or any member lacks a stored row — so the caller recomputes.
        """
        import json
        conn = self._connect()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(oof_predictions)")}
            if not cols:
                return None
            rows = {}
            for m in members:
                r = conn.execute(
                    "SELECT class_order, y_true, y_proba FROM oof_predictions "
                    "WHERE experiment_id=? AND model=?",
                    (self.experiment_id, m)).fetchone()
                if r is None:
                    return None
                rows[m] = r
        except Exception:
            return None
        finally:
            conn.close()

        co = [str(c) for c in json.loads(rows[members[0]]["class_order"])]
        y_true = np.asarray(json.loads(rows[members[0]]["y_true"]), dtype=object)
        mats = [np.asarray(json.loads(rows[m]["y_proba"]), dtype=float)
                for m in members]
        return co, y_true, np.mean(mats, axis=0)

    def _tune_thresholds_oof(self, policy, class_order, oof_true, oof_proba):
        """Per-class thresholds tuned on the OOF pool for a tuned policy."""
        from DSSP2026.core.threshold import per_class_thresholds_cv
        from DSSP2026.report.policy import POLICY_METRIC
        metric = POLICY_METRIC[policy]
        return per_class_thresholds_cv(
            oof_true, oof_proba, class_order, metric=metric)

    # ------------------------------------------------------------------
    # Full-train refit for deployment (delegates to the experiment layer)
    # ------------------------------------------------------------------

    def _refit_full(self, model_name, pool, train_df, target, column_types):
        """Return (estimators, features, class_order) refit on the full train."""
        from DSSP2026.report.cost.fit import _refit_one

        if model_name == ENSEMBLE_NAME:
            members = [m for m in pool if m != ENSEMBLE_NAME]
            refits = [_refit_one(self, m, train_df, target, column_types)
                      for m in members]
            return refits, refits[0].features, refits[0].class_order

        refit = _refit_one(self, model_name, train_df, target, column_types)
        return [refit], refit.features, refit.class_order
