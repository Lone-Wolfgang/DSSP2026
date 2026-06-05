from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.experiment.report._common import _METRIC_COLUMNS, _DB_METRIC
from DSSP2026.experiment.report.tables import ReportTable


class CompareMixin:
    def compare_models(self, models: Optional[Sequence[str]] = None, *,
                       sort_by: str = "F1", ascending: bool = False,
                       decimals: int = 4, include_feature_set: bool = True,
                       include_cv: bool = True) -> ReportTable:
        """One row per model = its winning config, evaluated on the held-out set.

        Parameters
        ----------
        models : sequence of str, optional
            Restrict to these models (by name). ``None`` -> all models in the
            experiment.
        sort_by : str
            Column to sort by (default the held-out ``F1``). Falls back to no
            sort if the column is absent.
        ascending : bool
            Sort direction (default False = best first for F1-like metrics).
        decimals : int
            Rounding for the metric columns in the displayed/exported frame.
        include_feature_set : bool
            Show the winner's feature set (a hyperparameter) as a column.
        include_cv : bool
            Show ``CV F1`` (the winning CV macro-F1) beside the held-out metrics.

        Returns
        -------
        ReportTable
            Renders as a readable table in notebooks; ``.to_csv`` / ``.to_png``
            export it; ``.df`` is the underlying DataFrame.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                'SELECT model, feature_set, accuracy, "precision", recall, f1, '
                "roc_auc, best_cv_value FROM models WHERE experiment_id=?",
                (self.experiment_id,)).fetchall()
        finally:
            conn.close()

        if not rows:
            return ReportTable(pd.DataFrame(columns=["Model"] + _METRIC_COLUMNS))

        wanted = set(models) if models is not None else None
        records = []
        for r in rows:
            if wanted is not None and r["model"] not in wanted:
                continue
            rec = {"Model": r["model"]}
            if include_feature_set:
                rec["Feature set"] = r["feature_set"]
            for disp, col in _DB_METRIC.items():
                rec[disp] = r[col]
            if include_cv:
                rec["CV F1"] = r["best_cv_value"]
            records.append(rec)

        df = pd.DataFrame.from_records(records)

        # Order columns: Model, [Feature set], metrics..., [CV F1].
        ordered = ["Model"]
        if include_feature_set:
            ordered.append("Feature set")
        ordered += _METRIC_COLUMNS
        if include_cv:
            ordered.append("CV F1")
        df = df[[c for c in ordered if c in df.columns]]

        # Sort (best first by default).
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")

        # Round metric columns for display/export.
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df[num_cols] = df[num_cols].round(decimals)

        title = (f"Model comparison — held-out {self.experiment_id} "
                 f"(best row by {sort_by})")
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
        # Round metric floats; render Support as an integer count (nullable Int
        # so it stays "24" not "24.0", and the summary rows' totals stay clean).
        for c in ("Precision", "Recall", "F1"):
            df[c] = df[c].round(decimals)
        df["Support"] = df["Support"].round().astype("Int64")
        title = f"Classification report — {model} (held-out {self.experiment_id})"
        # No best-row highlight here (rows are classes, not models).
        return ReportTable(df, best_by=None, title=title)

    def _classification_report_cross(self, metric, *, models, decimals):
        db_metric = _DB_METRIC.get(metric, metric)  # accept "F1" or "f1"
        if db_metric not in ("precision", "recall", "f1", "support"):
            raise ValueError(
                "metric must be one of precision, recall, f1, support "
                f"(or their display names); got {metric!r}.")
        conn = self._connect()
        try:
            mrows = conn.execute(
                "SELECT model_id, model FROM models WHERE experiment_id=? "
                "ORDER BY f1 DESC", (self.experiment_id,)).fetchall()
            wanted = set(models) if models is not None else None
            order = [(r["model_id"], r["model"]) for r in mrows
                     if wanted is None or r["model"] in wanted]
            summary = self._summary_labels()
            # class_label -> {model -> value}, preserving class order from the
            # first model encountered.
            class_order = []
            cells = {}
            for mid, name in order:
                for r in conn.execute(
                        f'SELECT class_label, "{db_metric}" AS v FROM per_class '
                        "WHERE model_id=? ORDER BY rowid", (mid,)):
                    cl = r["class_label"]
                    if str(cl).lower() in summary:
                        continue  # true classes only in the cross view
                    if cl not in cells:
                        cells[cl] = {}
                        class_order.append(cl)
                    cells[cl][name] = r["v"]
        finally:
            conn.close()

        model_names = [name for _, name in order]
        records = []
        for cl in class_order:
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
