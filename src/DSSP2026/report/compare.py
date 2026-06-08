from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.report._common import _METRIC_COLUMNS, _DB_METRIC
from DSSP2026.report.base import ENSEMBLE_NAME
from DSSP2026.report.tables import ReportTable


class CompareMixin:

    def compare_models(self, models: Optional[Sequence[str]] = None, *,
                       policy: str = "ArgMax",
                       allow_ensemble: bool = False,
                       sort_by: str = "F1", ascending: bool = False,
                       decimals: int = 4,
                       n_splits: int = 5,
                       random_state: int = 42) -> ReportTable:
        """Leak-free comparison: every row tuned on train, scored on test.

        For each candidate (and, when ``allow_ensemble``, the mean-probability
        Ensemble of the pool):

        1. tune per-class decision thresholds on **train out-of-fold CV**
           (nothing the test set has seen),
        2. refit on the **full train** set,
        3. predict the untouched **test partition** (``test.parquet``) and
           score it.

        Every metric in the returned table is therefore measured on data that
        played no part in fitting, threshold tuning, or selection — there is no
        tune-and-score-on-the-same-set leakage and no eval-set reuse. Requires
        both ``train.parquet`` and ``test.parquet`` (re-run the experiment with
        a ``test`` frame if the latter is absent).

        Parameters
        ----------
        models : sequence of str, optional
            Candidate pool (real models). ``None`` -> all. Selection and any
            Ensemble draw only from this pool.
        policy : {"ArgMax", "F1", "Youden's J"}
            Decision rule. ArgMax tunes nothing; F1 / Youden's J tune per-class
            thresholds on the train OOF pool toward that criterion.
        allow_ensemble : bool
            Add the mean-probability Ensemble of the pool as a row (default
            False). Requires >= 2 real candidates.
        sort_by, ascending, decimals
            Table sort column / direction / rounding (as in ``compare_models``).
        n_splits, random_state : int
            Folds / seed for the train-OOF threshold tuning.

        Returns
        -------
        ReportTable
            Columns: ``Model, Accuracy, Precision, Recall, F1`` — all on test.
            (ROC-AUC is omitted: it is a property of probabilities, and the row
            here represents a thresholded decision policy scored on test.)
        """
        from DSSP2026.report.policy import (
            validate_policy, decisions_under_policy, metrics_from_decisions)
        from DSSP2026.report.cost.fit import _column_types
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

        train_loaded = self.load_train_data()
        test_loaded = self.load_test_data()
        # train_df/test_df are only needed when a cached artifact is missing and
        # we must recompute (OOF folds or a full-train refit). The fast path uses
        # persisted OOF + test predictions and touches neither parquet.
        train_df, target = (train_loaded if train_loaded is not None else (None, None))
        test_df, test_target = (test_loaded if test_loaded is not None else (None, None))
        column_types = _column_types(self)

        records = []
        for name in candidates:
            members = ([m for m in pool if m != ENSEMBLE_NAME]
                       if name == ENSEMBLE_NAME else [name])
            # 1. tune thresholds on train OOF
            co, oof_true, oof_proba = self._oof_predictions(
                members, train_df, target, column_types,
                n_splits=n_splits, random_state=random_state)
            thr = (None if policy == "ArgMax"
                   else self._tune_thresholds_oof(policy, co, oof_true, oof_proba))
            # 2+3. prefer persisted test predictions; refit only if absent.
            cached = self._read_test_predictions(name, members)
            if cached is not None:
                class_order, y_true, test_proba = cached
            else:
                estimators, _, class_order = self._refit_full(
                    name, pool, train_df, target, column_types)
                test_proba = self._predict_proba_aligned(
                    estimators, test_df, class_order)
                y_true = np.asarray(test_df[test_target], dtype=object)
            y_pred = decisions_under_policy(policy, class_order, test_proba, thr)
            scored = metrics_from_decisions(y_true, y_pred, class_order)
            records.append({
                "Model": name,
                "Accuracy": scored["accuracy"], "Precision": scored["precision"],
                "Recall": scored["recall"], "F1": scored["f1"],
            })

        df = pd.DataFrame.from_records(records)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df[num_cols] = df[num_cols].round(decimals)

        policy_note = "" if policy == "ArgMax" else f", {policy} policy"
        title = (f"Model comparison — held-out TEST {self.experiment_id} "
                 f"(best row by {sort_by}{policy_note})")
        return ReportTable(df.reset_index(drop=True), best_by=sort_by, title=title)

    def classification_report(self, model: Optional[str] = None, *,
                              metric: Optional[str] = None,
                              models: Optional[Sequence[str]] = None,
                              decimals: int = 4,
                              include_summary: bool = True) -> ReportTable:
        """Per-class precision/recall/F1/support.

        Two modes:

        - **single-model** (``metric=None``, default): one model's full
          breakdown — rows = classes, columns = Precision/Recall/F1/Support.
          ``model=None`` selects the best model by held-out F1; pass a name to
          pick one. ``include_summary`` appends the accuracy / macro-avg /
          weighted-avg rows.
        - **cross-model** (``metric`` given, e.g. ``"f1"``): that one metric for
          every class across models — rows = classes, columns = models. The
          per-class comparison view. ``models`` restricts which models appear;
          summary rows are excluded here (only true classes compare cleanly).

        Returns a ``ReportTable`` (notebook render + CSV/PNG export).
        """
        if metric is None:
            return self._classification_report_single(
                model, decimals=decimals, include_summary=include_summary)
        return self._classification_report_cross(
            metric, models=models, decimals=decimals)

    def _summary_labels(self):
        return {"accuracy", "macro avg", "weighted avg"}

    def _classification_report_single(self, model, *, decimals, include_summary):
        if model is None:
            model = self._best_model_name()

        from DSSP2026.report.base import ENSEMBLE_NAME
        if model == ENSEMBLE_NAME:
            return self._classification_report_single_from_proba(
                model, decimals=decimals, include_summary=include_summary)

        conn = self._connect()
        try:
            mid_row = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid_row is None:
                raise ValueError(
                    f"model {model!r} not in experiment {self.experiment_id}.")
            rows = conn.execute(
                'SELECT class_label, "precision", recall, f1, support '
                "FROM per_class WHERE model_id=? ORDER BY rowid",
                (mid_row[0],)).fetchall()
        finally:
            conn.close()

        summary = self._summary_labels()
        records = []
        for r in rows:
            is_summary = str(r["class_label"]).lower() in summary
            if is_summary and not include_summary:
                continue
            records.append({
                "Class": r["class_label"] if is_summary
                         else self._relabel(r["class_label"]),
                "Precision": r["precision"], "Recall": r["recall"],
                "F1": r["f1"], "Support": r["support"],
            })
        df = pd.DataFrame.from_records(
            records, columns=["Class", "Precision", "Recall", "F1", "Support"])
        for c in ("Precision", "Recall", "F1"):
            df[c] = df[c].round(decimals)
        df["Support"] = df["Support"].round().astype("Int64")
        title = f"Classification report — {model} (held-out {self.experiment_id})"
        return ReportTable(df, best_by=None, title=title)

    def _classification_report_single_from_proba(self, model, *, decimals,
                                                  include_summary):
        """Compute per-class metrics for the Ensemble from its mean proba."""
        from sklearn.metrics import classification_report as sk_cr
        class_order, y_true, y_proba = self._read_predictions(model)
        y_pred = np.array([class_order[i] for i in y_proba.argmax(axis=1)],
                          dtype=object)
        report = sk_cr(y_true, y_pred, labels=class_order,
                       output_dict=True, zero_division=0)
        summary = self._summary_labels()
        records = []
        for label in class_order:
            d = report.get(str(label), {})
            records.append({
                "Class": self._relabel(str(label)),
                "Precision": d.get("precision"),
                "Recall":    d.get("recall"),
                "F1":        d.get("f1-score"),
                "Support":   d.get("support"),
            })
        if include_summary:
            for key in ("macro avg", "weighted avg"):
                d = report.get(key, {})
                records.append({
                    "Class": key,
                    "Precision": d.get("precision"),
                    "Recall":    d.get("recall"),
                    "F1":        d.get("f1-score"),
                    "Support":   d.get("support"),
                })
        df = pd.DataFrame.from_records(
            records, columns=["Class", "Precision", "Recall", "F1", "Support"])
        for c in ("Precision", "Recall", "F1"):
            df[c] = df[c].round(decimals)
        df["Support"] = df["Support"].round().astype("Int64")
        title = f"Classification report — {model} (held-out {self.experiment_id})"
        return ReportTable(df, best_by=None, title=title)

    def _classification_report_cross(self, metric, *, models, decimals):
        db_metric = _DB_METRIC.get(metric, metric)
        if db_metric not in ("precision", "recall", "f1", "support"):
            raise ValueError(
                "metric must be one of precision, recall, f1, support "
                f"(or their display names); got {metric!r}.")

        from DSSP2026.report.base import ENSEMBLE_NAME
        from sklearn.metrics import classification_report as sk_cr

        conn = self._connect()
        try:
            mrows = conn.execute(
                "SELECT model_id, model FROM models WHERE experiment_id=? "
                "ORDER BY f1 DESC", (self.experiment_id,)).fetchall()
            wanted = set(models) if models is not None else None
            # Real models filtered by wanted; Ensemble handled separately.
            order = [(r["model_id"], r["model"]) for r in mrows
                     if wanted is None or r["model"] in wanted]
            include_ensemble = (
                (wanted is None or ENSEMBLE_NAME in wanted)
                and len([r for r in mrows]) >= 2
            )
            summary = self._summary_labels()
            class_order_list = []
            cells = {}
            for mid, name in order:
                for r in conn.execute(
                        f'SELECT class_label, "{db_metric}" AS v FROM per_class '
                        "WHERE model_id=? ORDER BY rowid", (mid,)):
                    cl = r["class_label"]
                    if str(cl).lower() in summary:
                        continue
                    if cl not in cells:
                        cells[cl] = {}
                        class_order_list.append(cl)
                    cells[cl][name] = r["v"]
        finally:
            conn.close()

        model_names = [name for _, name in order]

        # Add Ensemble column computed from mean proba.
        if include_ensemble:
            try:
                co, y_true, y_proba = self._ensemble_proba()
                y_pred = np.array([co[i] for i in y_proba.argmax(axis=1)],
                                  dtype=object)
                report = sk_cr(y_true, y_pred, labels=co,
                               output_dict=True, zero_division=0)
                key_map = {"precision": "precision", "recall": "recall",
                           "f1": "f1-score", "support": "support"}
                sk_key = key_map[db_metric]
                for label in co:
                    cl = str(label)
                    if cl not in cells:
                        cells[cl] = {}
                        class_order_list.append(cl)
                    cells[cl][ENSEMBLE_NAME] = report.get(cl, {}).get(sk_key)
                model_names.append(ENSEMBLE_NAME)
            except Exception:
                pass

        records = []
        for cl in class_order_list:
            rec = {"Class": self._relabel(cl)}
            for name in model_names:
                rec[name] = cells[cl].get(name)
            records.append(rec)
        df = pd.DataFrame.from_records(records, columns=["Class"] + model_names)
        num_cols = [c for c in df.columns if c != "Class"]
        if db_metric != "support":
            df[num_cols] = df[num_cols].round(decimals)
        disp = {"precision": "Precision", "recall": "Recall", "f1": "F1",
                "support": "Support"}[db_metric]
        title = (f"Per-class {disp} across models "
                 f"(held-out {self.experiment_id})")
        return ReportTable(df, best_by=None, title=title)
