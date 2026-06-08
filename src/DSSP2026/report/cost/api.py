"""
report/cost_api.py — the cost_optimize sweep (CostMixin).

Sweeps every (model × policy) combination, scores each by net benefit under the
schedule, and returns a CostDecision for the winner.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.report.cost.tables import (
    CostDecision, PolicyTable, ClassBreakdownTable, POLICIES, _POLICY_METRIC)
from DSSP2026.report.cost.math import (
    per_class_counts, class_contributions, net_benefit,
    confusion_from_decisions, expected_value_decisions, no_action_baseline,
)
from DSSP2026.report.base import ENSEMBLE_NAME


class CostMixin:

    def cost_optimize(self, payoff, models: Optional[Sequence[str]] = None,
                      policies: Optional[Sequence[str]] = None, *,
                      allow_ensemble: bool = False,
                      tune_on_train: bool = True,
                      n_splits: int = 5, random_state: int = 42) -> CostDecision:
        """Find the (model, policy) combination with the highest net benefit.

        Parameters
        ----------
        payoff : DataFrame
            The per-class schedule: a frame indexed by class label with columns
            ``TP/FP``, ``TP``, ``FN`` (the same structure the dashboard edits).
            Values are signed (costs negative).
        models : sequence of str, optional
            Restrict the sweep to these models. Default: all models in the
            experiment. The virtual Ensemble is included only when
            ``allow_ensemble=True``.
        policies : sequence of str, optional
            Restrict to these policies. Default: ArgMax, F1, Youden's J, Bayes.
        allow_ensemble : bool
            Include the Ensemble in the model sweep (default False). When the
            Ensemble wins, ``CostDecision.fit()`` returns a wrapped averaging
            ensemble of refit members.
        tune_on_train : bool
            When True (default), the *decision rule* for each (model, policy) —
            per-class thresholds (F1/Youden) or the Bayes boundary — is fit on
            **train out-of-fold CV predictions** (persisted by the experiment
            when available). The winning (model, policy) is then *selected* by
            net benefit on the **validation** set, and the returned tables /
            net benefit report the winner's performance on the untouched
            **test** set. This is the leak-free three-set pipeline: tune on
            train, select on validation, report on test — nothing is tuned and
            scored on the same data, and selection never touches test. When
            False, the legacy behaviour is used: tune and score both on the
            stored eval predictions (optimistic; kept for backward comparison).
        n_splits, random_state : int
            Fold count / seed for the train-OOF tuning (used only when
            ``tune_on_train`` is True and OOF must be recomputed).

        Returns
        -------
        CostDecision
            The winning combination, its two display tables, and ``fit()``.
        """
        schedule = self._coerce_schedule(payoff)

        # Candidate models.
        real = self.models(include_ensemble=False)
        if allow_ensemble and len(real) >= 2:
            candidate_models = real + [ENSEMBLE_NAME]
        else:
            candidate_models = list(real)
        if models is not None:
            want = {models} if isinstance(models, str) else set(models)
            candidate_models = [m for m in candidate_models if m in want]
        if not candidate_models:
            raise ValueError("no models to sweep after filtering.")

        # Candidate policies.
        candidate_policies = list(policies) if policies is not None else list(POLICIES)
        bad = [p for p in candidate_policies if p not in POLICIES]
        if bad:
            raise ValueError(f"unknown policies {bad}; valid: {list(POLICIES)}.")

        # Three-set pipeline: decision rule fit on train OOF; winner SELECTED by
        # net benefit on validation; winner REPORTED on the untouched test set.
        pool = [m for m in candidate_models if m != ENSEMBLE_NAME]
        oof_cache = {}        # model -> (co, oof_true, oof_proba)   [tuning]
        val_cache = {}        # model -> (co, val_true, val_proba)   [selection]
        test_cache = {}       # model -> (co, test_true, test_proba) [reporting]
        if tune_on_train:
            from DSSP2026.report.cost.fit import _column_types
            loaded = self.load_train_data()
            val_loaded = self.load_validation_data()
            test_loaded = self.load_test_data()
            if loaded is None or val_loaded is None or test_loaded is None:
                raise RuntimeError(
                    "cost_optimize(tune_on_train=True) requires train, "
                    "validation, and test parquets. Re-run the experiment with "
                    "all three splits, or pass tune_on_train=False.")
            train_df, target = loaded
            val_df, val_target = val_loaded
            test_df, test_target = test_loaded
            column_types = _column_types(self)

        def _tuning_preds(model):
            """(class_order, y_true, y_proba) used to FIT the decision rule."""
            if not tune_on_train:
                return self._read_predictions(model)
            if model not in oof_cache:
                members = (pool if model == ENSEMBLE_NAME else [model])
                oof_cache[model] = self._oof_predictions(
                    members, train_df, target, column_types,
                    n_splits=n_splits, random_state=random_state)
            return oof_cache[model]

        def _refit_and_predict(model, frame, frame_target, cache):
            if model not in cache:
                est, _, co = self._refit_full(
                    model, candidate_models, train_df, target, column_types)
                proba = self._predict_proba_aligned(est, frame, co)
                yt = frame[frame_target].astype(str).to_numpy()
                cache[model] = (co, yt, proba)
            return cache[model]

        def _selection_preds(model):
            """Validation predictions — used to SELECT the winner."""
            if not tune_on_train:
                return self._read_predictions(model)
            return _refit_and_predict(model, val_df, val_target, val_cache)

        def _scoring_preds(model):
            """Test predictions — used to REPORT the winner (and tables)."""
            if not tune_on_train:
                return self._read_predictions(model)
            return _refit_and_predict(model, test_df, test_target, test_cache)

        def _net_benefit(co, yt, decisions):
            cm = confusion_from_decisions(yt, decisions, co)
            contrib = class_contributions(per_class_counts(cm), schedule)
            inv = float(contrib["investment"].sum())
            loss = float(contrib["loss"].sum())
            gains = float(contrib["gains"].sum())
            gross = inv + loss + gains
            return inv, loss, gains, gross

        # No Action baseline computed on the TEST set (the reporting surface).
        ref_model = candidate_models[0]
        co_ref, y_true_ref, y_proba_ref = _scoring_preds(ref_model)
        argmax_ref = [co_ref[i] for i in y_proba_ref.argmax(axis=1)]
        cm_ref = confusion_from_decisions(y_true_ref, argmax_ref, co_ref)
        na_row = no_action_baseline(cm_ref, schedule)
        gross_na = float(na_row["Gross"])

        # Sweep: fit rule on OOF, SELECT on validation, REPORT on test.
        rows = []
        winner = None   # selected by validation net benefit
        for model in candidate_models:
            co_t, yt_t, yp_t = _tuning_preds(model)
            co_v, yt_v, yp_v = _selection_preds(model)
            co_s, yt_s, yp_s = _scoring_preds(model)
            for policy in candidate_policies:
                # Fit the decision rule on the tuning (train OOF) set.
                _, threshold = self._decide(policy, co_t, yt_t, yp_t, schedule)

                # SELECT by net benefit on validation.
                val_dec = self._apply_decision(policy, co_v, yp_v, threshold, schedule)
                *_, val_gross = _net_benefit(co_v, yt_v, val_dec)
                val_nb = val_gross  # baseline-invariant for argmax selection

                # REPORT net benefit on test (what the tables show).
                test_dec = self._apply_decision(policy, co_s, yp_s, threshold, schedule)
                inv, loss, gains, gross = _net_benefit(co_s, yt_s, test_dec)
                nb = gross - gross_na
                rows.append({
                    "Model": model, "Policy": policy,
                    "Investment": inv, "Loss": loss, "Gains": gains,
                    "Gross": gross, "Net Benefit": nb,
                })
                if winner is None or val_nb > winner[0]:
                    winner = (val_nb, model, policy, threshold,
                              test_dec, co_s, yt_s, nb)

        rows_df = pd.DataFrame(rows)

        _, m_win, p_win, thr_win, dec_win, co_win, yt_win, nb_win = winner

        # Class breakdown for the winner.
        cm_win = confusion_from_decisions(yt_win, dec_win, co_win)
        counts_win = per_class_counts(cm_win)
        contrib_win = class_contributions(counts_win, schedule)
        breakdown = pd.DataFrame({
            "Class": [self._relabel(c) for c in counts_win.index],
            "TP": counts_win["tp"].astype(int).values,
            "FP": counts_win["fp"].astype(int).values,
            "FN": counts_win["fn"].astype(int).values,
            "TP Value": contrib_win["tp_value"].values,
            "FP Value": contrib_win["fp_value"].values,
            "FN Value": contrib_win["fn_value"].values,
            "Net": contrib_win["net_benefit"].values,
        })

        policy_table = PolicyTable(
            rows_df, no_action_row=na_row,
            best_model=m_win, best_policy=p_win)
        class_breakdown = ClassBreakdownTable(breakdown)

        return CostDecision(
            model_type=m_win, policy=p_win, threshold=thr_win,
            policy_table=policy_table, class_breakdown=class_breakdown,
            net_benefit=nb_win,
            _report=self, _schedule=schedule, _class_order=list(co_win),
            _allow_ensemble=allow_ensemble,
            _decisions=np.asarray(dec_win, dtype=object),
            _y_true=np.asarray(yt_win, dtype=object))

    # ------------------------------------------------------------------
    # Decision rules per policy
    # ------------------------------------------------------------------

    def _decide(self, policy, class_order, y_true, y_proba, schedule):
        """Return (decisions, threshold) for one policy.

        threshold is 0.5 (ArgMax), a {class: cutoff} dict (F1/Youden), or
        None (Bayes).
        """
        if policy == "ArgMax":
            decisions = np.asarray(
                [class_order[i] for i in y_proba.argmax(axis=1)], dtype=object)
            return decisions, 0.5

        if policy in _POLICY_METRIC:
            from DSSP2026.core.threshold import (
                per_class_thresholds, decisions_from_thresholds)
            metric = _POLICY_METRIC[policy]
            thr = per_class_thresholds(y_true, y_proba, class_order, metric=metric)
            decisions = decisions_from_thresholds(y_proba, class_order, thr)
            return decisions, thr

        if policy == "Bayes":
            decisions = expected_value_decisions(class_order, y_proba, schedule)
            return decisions, None

        raise ValueError(f"unknown policy {policy!r}.")

    def _apply_decision(self, policy, class_order, y_proba, threshold, schedule):
        """Apply an already-fit decision rule to ``y_proba``.

        Separates *applying* a rule from *fitting* it (``_decide``), so a rule
        tuned on train OOF can be applied to a held-out scoring set. ArgMax and
        Bayes carry no fitted parameters (Bayes re-derives its boundary from the
        schedule, which is user input, not learned); F1/Youden reuse the
        per-class ``threshold`` dict produced by ``_decide``.
        """
        if policy == "ArgMax":
            return np.asarray(
                [class_order[i] for i in y_proba.argmax(axis=1)], dtype=object)
        if policy in _POLICY_METRIC:
            from DSSP2026.core.threshold import decisions_from_thresholds
            return decisions_from_thresholds(y_proba, class_order, threshold or {})
        if policy == "Bayes":
            return expected_value_decisions(class_order, y_proba, schedule)
        raise ValueError(f"unknown policy {policy!r}.")

    def _predict_proba_aligned(self, estimators, frame, class_order):
        """Mean ``predict_proba`` over estimators, aligned to ``class_order``."""
        K = len(class_order)
        col_ix = {c: j for j, c in enumerate(class_order)}
        mats = []
        for e in estimators:
            p = np.asarray(e.predict_proba(frame), dtype=float)
            aligned = np.zeros((len(frame), K), dtype=float)
            for j, c in enumerate(e.class_order):
                if str(c) in col_ix:
                    aligned[:, col_ix[str(c)]] = p[:, j]
            mats.append(aligned)
        return np.mean(mats, axis=0)

    # ------------------------------------------------------------------
    # Schedule coercion
    # ------------------------------------------------------------------

    def _coerce_schedule(self, payoff) -> pd.DataFrame:
        """Validate/normalize the schedule frame (index=class, cols TP/FP,TP,FN)."""
        from DSSP2026.report.cost.math import TPFP_COL, TP_COL, FN_COL
        if not hasattr(payoff, "columns"):
            raise ValueError(
                "payoff must be a DataFrame indexed by class label with columns "
                f"'{TPFP_COL}', '{TP_COL}', '{FN_COL}'.")
        missing = [c for c in (TPFP_COL, TP_COL, FN_COL)
                   if c not in payoff.columns]
        if missing:
            raise ValueError(f"schedule missing columns {missing}.")
        out = payoff.copy()
        out.index = [str(i) for i in out.index]
        return out

    # ------------------------------------------------------------------
    # Standalone cost threshold plot (used by the dashboard's Bayes view)
    # ------------------------------------------------------------------

    def cost_threshold_plot(self, payoff, model: Optional[str] = None, *,
                            target_class=None, n_thresholds: int = 200,
                            title=None, figsize=None):
        """Expected-net-value vs. threshold sweep for one model+class.

        Builds a CostThresholdPlot directly from the stored probabilities and
        the schedule (converted to a 2-D payoff matrix internally). Independent
        of the model/policy sweep in ``cost_optimize``.
        """
        from DSSP2026.report.plots import CostThresholdPlot
        from DSSP2026.report.cost.math import TPFP_COL, TP_COL, FN_COL

        if model is None:
            model = self._best_model_name()
        schedule = self._coerce_schedule(payoff)
        class_order, y_true, y_proba = self._read_predictions(model)

        co = [str(c) for c in class_order]
        tpfp = schedule[TPFP_COL].reindex(co).fillna(0.0)
        tp = schedule[TP_COL].reindex(co).fillna(0.0)
        fn = schedule[FN_COL].reindex(co).fillna(0.0)
        K = len(co)
        V = np.zeros((K, K), dtype=float)
        for ai in range(K):
            for ki in range(K):
                if ai == ki:
                    V[ai, ki] = float(tpfp.iloc[ai]) + float(tp.iloc[ai])
                else:
                    V[ai, ki] = float(tpfp.iloc[ai]) + float(fn.iloc[ki])

        if target_class is None:
            classes = [co[0]]
        elif target_class == "all":
            classes = list(co)
        elif isinstance(target_class, str):
            classes = [str(target_class)]
        else:
            classes = [str(c) for c in target_class]

        curves = self._cost_sweep(co, y_true, y_proba, V, classes, n_thresholds)
        return CostThresholdPlot(
            curves=curves, class_order=co, model=model,
            title=title, figsize=figsize)

    def _cost_sweep(self, class_order, y_true, y_proba, V, classes, n_thresholds):
        """{class: (thresholds, expected_values)} for the cost threshold plot."""
        n = len(y_true)
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        class_ix = {c: i for i, c in enumerate(class_order)}
        true_ix = np.array([class_ix.get(str(k), 0) for k in y_true], dtype=int)
        payoff_per_action = V[:, true_ix].T            # (n, K)
        ev_scores = y_proba @ V.T                       # (n, K)
        curves = {}
        for target in classes:
            ki = class_order.index(target)
            ai = ki
            p_target = y_proba[:, ki]
            alt = ev_scores.copy()
            alt[:, ai] = -np.inf
            best_alt = np.argmax(alt, axis=1)
            payoff_alt = payoff_per_action[np.arange(n), best_alt]
            payoff_target = payoff_per_action[:, ai]
            flag = p_target[:, None] >= thresholds[None, :]
            ev = (flag * payoff_target[:, None]
                  + (~flag) * payoff_alt[:, None]).mean(axis=0)
            curves[target] = (thresholds, ev)
        return curves