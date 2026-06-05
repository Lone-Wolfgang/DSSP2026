from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.experiment.report.plots import (
    ConfusionPlot, ROCPlot, ThresholdSweepPlot)


class DiagnosticsMixin:
    def confusion_matrix(self, model: Optional[str] = None, *,
                         normalize: bool = False,
                         title: Optional[str] = None,
                         true_axis: str = "True",
                         pred_axis: str = "Predicted",
                         order: Optional[Sequence[str]] = None,
                         labels: Optional[Sequence[str]] = None,
                         figsize: Optional[tuple] = None):
        """Confusion-matrix heatmap (rows = true class, cols = predicted class).

        ``model=None`` selects the best model by held-out F1; pass a name to
        pick one. Returns a ``ConfusionPlot`` that renders inline and exports via
        ``.to_png(path)``.

        Parameters
        ----------
        normalize : bool
            Color by row-normalized rate (the diagonal reads as per-class
            recall) while still printing raw counts in each cell. Default False
            colors by raw count.
        title : str, optional
            Plot title. Defaults to "Confusion matrix — {model}".
        true_axis, pred_axis : str
            Axis titles. Defaults make the orientation explicit: rows are the
            true class ("True"), columns the predicted class ("Predicted").
        order : sequence of str, optional
            Class labels (the ORIGINAL stored names) in the sequence to display.
            Reindexes both axes together. Acts as a filter too: a class omitted
            from ``order`` is dropped from the matrix (with a warning, since that
            may be intentional). A name in ``order`` that isn't a class is an
            error (a typo can't be intended). Defaults to the stored class order.
        labels : sequence of str, optional
            Display names for the classes, applied positionally over the final
            ``order`` sequence; renames both axes. Length must match the number
            of classes shown.
        figsize : tuple, optional
            Figure size; defaults to a square sized to the class count.
        """
        if model is None:
            model = self._best_model_name()
        cm_df = self._confusion_df(model)  # rows=true, cols=pred, integer counts
        stored = [str(c) for c in cm_df.index]

        # Resolve display order (and optional subsetting) over the stored classes.
        if order is not None:
            order = [str(x) for x in order]
            unknown = [c for c in order if c not in stored]
            if unknown:
                raise ValueError(
                    f"order contains labels not in the model's classes "
                    f"{stored}: {unknown}.")
            omitted = [c for c in stored if c not in order]
            if omitted:
                import warnings
                warnings.warn(
                    f"confusion_matrix: {len(omitted)} class(es) omitted from "
                    f"`order` and dropped from the matrix: {omitted}.",
                    stacklevel=2)
            seq = order
            cm_df = cm_df.reindex(index=seq, columns=seq)
        else:
            seq = stored

        class_labels = list(seq)
        if labels is not None:
            if len(labels) != len(seq):
                raise ValueError(
                    f"labels has {len(labels)} entries but the matrix shows "
                    f"{len(seq)} class(es) {seq}.")
            class_labels = [str(x) for x in labels]
        else:
            # Default to registered display names (identity if none registered).
            class_labels = self._relabel_seq(seq)

        if title is None:
            title = f"Confusion matrix — {model}"
        return ConfusionPlot(
            counts=cm_df.to_numpy(dtype=float), class_labels=class_labels,
            normalize=normalize, title=title, true_axis=true_axis,
            pred_axis=pred_axis, figsize=figsize)

    def _confusion_df(self, model):
        """Reconstruct the wide (true x pred) integer confusion matrix for a model.

        Built from the long ``confusion`` rows, ordered by the class order in the
        stored predictions so rows/cols align with the model's class set.
        """
        conn = self._connect()
        try:
            mid_row = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid_row is None:
                raise ValueError(
                    f"model {model!r} not in experiment {self.experiment_id}.")
            model_id = mid_row[0]
            long = pd.read_sql_query(
                "SELECT true_label, pred_label, count FROM confusion "
                "WHERE model_id=?", conn, params=(model_id,))
            order_row = conn.execute(
                "SELECT class_order FROM predictions WHERE model_id=?",
                (model_id,)).fetchone()
        finally:
            conn.close()
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="true_label", columns="pred_label",
                          values="count").fillna(0).astype(int)
        # Order rows/cols by the stored class order when available.
        if order_row is not None:
            order = [str(c) for c in json.loads(order_row[0])]
            wide = wide.reindex(index=order, columns=order, fill_value=0)
        wide.index.name = "True"
        wide.columns.name = "Predicted"
        return wide

    def roc_compare(self, models: Optional[Sequence[str]] = None, *,
                    average: str = "macro", title: Optional[str] = None):
        """Overlay one one-vs-rest ROC curve per model (micro or macro average).

        ``models=None`` includes every model in the experiment. ``average`` is
        "macro" (per-class curves averaged equally) or "micro" (all samples
        pooled). Returns an ``ROCPlot`` (inline render + ``.to_png``).
        """
        from sklearn.metrics import roc_curve, auc as _auc
        from sklearn.preprocessing import label_binarize

        if average not in ("macro", "micro"):
            raise ValueError("average must be 'macro' or 'micro'.")
        names = models if models is not None else self.models()
        curves = []
        for model in names:
            class_order, y_true, y_proba = self._read_predictions(model)
            Y = label_binarize(y_true, classes=class_order)
            if Y.shape[1] == 1:                       # degenerate 2-class binarize
                Y = np.hstack([1 - Y, Y])
            if average == "micro":
                fpr, tpr, _ = roc_curve(Y.ravel(), y_proba.ravel())
                roc_auc = _auc(fpr, tpr)
            else:
                grid = np.linspace(0.0, 1.0, 200)
                tprs = []
                for k in range(Y.shape[1]):
                    if Y[:, k].sum() == 0:
                        continue
                    fk, tk, _ = roc_curve(Y[:, k], y_proba[:, k])
                    tprs.append(np.interp(grid, fk, tk))
                if not tprs:
                    continue
                tpr = np.mean(tprs, axis=0)
                tpr[0] = 0.0
                fpr = grid
                roc_auc = _auc(fpr, tpr)
            curves.append({"label": model, "fpr": fpr, "tpr": tpr,
                           "auc": float(roc_auc)})
        curves.sort(key=lambda d: d["auc"], reverse=True)
        if title is None:
            title = (f"ROC comparison — {average}-averaged (one-vs-rest), "
                     f"held-out {self.experiment_id}")
        return ROCPlot(curves, average=average, title=title)

    def threshold_sweep(self, model: Optional[str] = None, *,
                        target_class,
                        optimize="f1",
                        metrics: Optional[Sequence[str]] = None,
                        title: Optional[str] = None):
        """Sweep the decision threshold for ``target_class`` (one-vs-rest).

        Plots each chosen metric as a curve vs threshold and marks a vertical
        line at the threshold that optimizes each requested ``optimize`` metric.
        ``model=None`` selects the best model by held-out F1.

        ``metrics`` — which curves to draw (default precision/recall/f1).
        ``optimize`` — a single metric name OR a list of names; one dashed
        optimum line is drawn per entry, so you can compare e.g. Youden's J and
        F1 optima on the same axes.

        Allowed names (for both ``metrics`` and ``optimize``): precision,
        recall, sensitivity (alias of recall), specificity, f1, youden,
        accuracy, false_negative_rate, false_positive_rate.

        Returns a ``ThresholdSweepPlot`` (inline render + ``.to_png``).
        """
        from DSSP2026.core.threshold import tune_threshold, _ALLOWED_METRICS

        # sensitivity == recall; accept either spelling, resolve to the column.
        alias = {"sensitivity": "recall"}

        def _resolve(name):
            key = alias.get(name, name)
            if key not in _ALLOWED_METRICS:
                raise ValueError(
                    f"unknown metric {name!r}; choose from "
                    f"{list(_ALLOWED_METRICS) + list(alias)}.")
            return key, name           # (column key, display-requested name)

        if model is None:
            model = self._best_model_name()
        class_order, y_true, y_proba = self._read_predictions(model)
        target_class = str(target_class)
        if target_class not in class_order:
            raise ValueError(
                f"target_class {target_class!r} not among classes {class_order}.")

        # One-vs-rest: positive = target class; score = that class's proba column.
        col = class_order.index(target_class)
        y_bin = (y_true.astype(str) == target_class).astype(int)
        score = y_proba[:, col]

        # Curves to draw (resolve aliases to columns, keep requested display).
        requested = list(metrics) if metrics else ["precision", "recall", "f1"]
        plot_metrics, plot_aliases = [], {}
        for name in requested:
            key, disp = _resolve(name)
            plot_metrics.append(key)
            if disp != key:
                plot_aliases[key] = disp

        # Optimize: accept a single name or a list -> one optimum line each.
        optimize_list = [optimize] if isinstance(optimize, str) else list(optimize)
        if not optimize_list:
            raise ValueError("optimize must name at least one metric.")

        marks, sweep_df = [], None
        for name in optimize_list:
            key, disp = _resolve(name)
            sweep = tune_threshold(y_bin, score, metric=key)
            sweep_df = sweep.sweep_df          # identical across calls
            marks.append({"metric": key, "display": disp,
                          "threshold": sweep.best_threshold,
                          "value": sweep.best_value})

        if title is None:
            opt_disp = ", ".join(m["display"] for m in marks)
            title = (f"Threshold tuning — {model}, {target_class} vs rest "
                     f"(optimizing {opt_disp})")
        return ThresholdSweepPlot(
            sweep_df=sweep_df, plot_metrics=plot_metrics, marks=marks,
            display_aliases=plot_aliases, target_class=target_class,
            model=model, title=title)
