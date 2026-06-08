"""
report/evaluation.py — Report methods for feature importance and training curves.

Both methods read exclusively from report.db (no re-fitting, no live Optuna
study required) and return simple plot objects with .show() / .save() interfaces.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from DSSP2026.report.base import ENSEMBLE_NAME


class EvaluationMixin:

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        model: Optional[str] = None,
        *,
        importance_type: str = "gain",
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        figsize=None,
    ):
        """Horizontal bar chart of feature importances for one model.

        Reads the ``feature_importance`` table written by
        ``report_builder.build_report_db``.

        Parameters
        ----------
        model : str, optional
            Model name. Defaults to the best model by held-out F1.
        importance_type : str
            Which importance metric to display. Default ``"gain"`` (available
            for Decision tree, Random forest, XGBoost). XGBoost also stores
            ``"weight"``, ``"cover"``, ``"total_gain"``, ``"total_cover"``.
            Logistic regression stores ``"abs_coef"`` (mean |coefficient|
            across classes). MLP has no importance data.
        top_n : int, optional
            Show only the top-N features. None shows all.
        title : str, optional
            Override the default title.
        figsize : tuple, optional

        Returns
        -------
        FeatureImportancePlot
            Call ``.show()`` to display, ``.save(path)`` to write to disk.

        Raises
        ------
        ValueError
            If the model has no importance data for the requested type, or if
            the model has no importance data at all.
        """
        from DSSP2026.report.plots import FeatureImportancePlot

        if model is None:
            model = self._best_model_name()

        if model == ENSEMBLE_NAME:
            raise ValueError(
                f"{ENSEMBLE_NAME} has no intrinsic feature importance. "
                "Use feature_importance() on individual models instead.")

        df = self._read_feature_importance(model, importance_type)
        if df.empty:
            available = self._available_importance_types(model)
            if not available:
                raise ValueError(
                    f"No feature importance data stored for model {model!r}. "
                    "MLP does not produce importance scores.")
            raise ValueError(
                f"No importance data for model {model!r} with "
                f"importance_type={importance_type!r}. "
                f"Available types: {available}.")

        return FeatureImportancePlot(
            df, model=model, importance_type=importance_type,
            top_n=top_n, title=title, figsize=figsize)

    def _read_feature_importance(self, model, importance_type):
        """Return a DataFrame(Feature, Importance) for model + type, descending."""
        conn = self._connect()
        try:
            mid = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid is None:
                raise ValueError(
                    f"model {model!r} not in experiment {self.experiment_id}.")
            rows = conn.execute(
                "SELECT feature, importance FROM feature_importance "
                "WHERE model_id=? AND importance_type=? "
                "ORDER BY importance DESC",
                (mid[0], importance_type)).fetchall()
        finally:
            conn.close()
        if not rows:
            return pd.DataFrame(columns=["Feature", "Importance"])
        return pd.DataFrame(rows, columns=["Feature", "Importance"])

    def _available_importance_types(self, model):
        """Return sorted list of importance_type values stored for a model."""
        conn = self._connect()
        try:
            mid = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid is None:
                return []
            rows = conn.execute(
                "SELECT DISTINCT importance_type FROM feature_importance "
                "WHERE model_id=? ORDER BY importance_type",
                (mid[0],)).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------

    def class_distribution(
        self,
        df: "pd.DataFrame",
        target: str,
        *,
        split: str = "all",
        title: Optional[str] = None,
        figsize=None,
    ):
        """Horizontal bar chart of class label counts in a DataFrame.

        Pass any split of the original data — the full frame, just the
        training set, or just the held-out set.  The eval set ground truth
        is also available directly from ``report.db`` via
        ``class_distribution_from_db(model)``.

        Parameters
        ----------
        df : DataFrame
            Any frame that contains ``target``.
        target : str
            The target column name.
        split : str
            Label for the title: ``"all"`` (default), ``"train"``, or
            ``"test"``.
        title : str, optional
        figsize : tuple, optional

        Returns
        -------
        ClassDistributionPlot
        """
        from DSSP2026.report.plots import ClassDistributionPlot
        counts = df[target].value_counts()
        return ClassDistributionPlot(
            counts, split=split, title=title, figsize=figsize)

    def class_distribution_from_db(
        self,
        model: Optional[str] = None,
        *,
        title: Optional[str] = None,
        figsize=None,
    ):
        """Class distribution of the eval set stored in ``report.db``.

        Reads ``y_true`` from the predictions table — no external DataFrame
        required.  This is the eval-set distribution only; for train or the
        full dataset, use ``class_distribution(df, target)``.

        Parameters
        ----------
        model : str, optional
            Defaults to the best model by F1.
        title : str, optional
        figsize : tuple, optional

        Returns
        -------
        ClassDistributionPlot
        """
        import pandas as pd
        from DSSP2026.report.plots import ClassDistributionPlot

        if model is None:
            model = self._best_model_name()

        class_order, y_true, _ = self._read_predictions(model)
        counts = pd.Series(y_true).value_counts()
        return ClassDistributionPlot(
            counts, split="test", title=title, figsize=figsize)

    def training_curves(
        self,
        model: Optional[str] = None,
        *,
        title: Optional[str] = None,
        figsize=None,
    ):
        """Aggregated training / eval loss curves for the top stored trials.

        Available only for models whose objectives log per-epoch loss curves
        (MLP and XGBoost). Reads the ``trial_curves`` table; no live Optuna
        study is required.

        The plot shows mean ± 1 std bands across the stored top-N trials
        (ranked by CV objective value) with a dashed line at the epoch of
        minimum mean eval loss.

        Parameters
        ----------
        model : str, optional
            Model name. Defaults to the best model by held-out F1.
        title : str, optional
            Override the default title.
        figsize : tuple, optional

        Returns
        -------
        TrainingCurvePlot
            Call ``.show()`` to display, ``.save(path)`` to write to disk.

        Raises
        ------
        ValueError
            If no trial curves are stored for the model (e.g. Decision tree,
            Random forest, or Logistic regression).
        """
        from DSSP2026.report.plots import TrainingCurvePlot

        if model is None:
            model = self._best_model_name()

        if model == ENSEMBLE_NAME:
            raise ValueError(
                f"{ENSEMBLE_NAME} has no training curves. "
                "Use training_curves() on individual models instead.")

        train_curves, eval_curves, ranks = self._read_trial_curves(model)
        if not train_curves:
            raise ValueError(
                f"No trial curves stored for model {model!r}. "
                "Only MLP and XGBoost log per-epoch loss curves.")

        return TrainingCurvePlot(
            train_curves, eval_curves, ranks,
            model=model, title=title, figsize=figsize)

    def _read_trial_curves(self, model):
        """Return (train_curves, eval_curves, ranks) for the stored top-N trials."""
        conn = self._connect()
        try:
            mid = conn.execute(
                "SELECT model_id FROM models WHERE experiment_id=? AND model=?",
                (self.experiment_id, model)).fetchone()
            if mid is None:
                raise ValueError(
                    f"model {model!r} not in experiment {self.experiment_id}.")
            rows = conn.execute(
                "SELECT rank, train_curve, eval_curve FROM trial_curves "
                "WHERE model_id=? ORDER BY rank",
                (mid[0],)).fetchall()
        finally:
            conn.close()
        if not rows:
            return [], [], []
        train_curves = [json.loads(r["train_curve"]) for r in rows]
        eval_curves  = [json.loads(r["eval_curve"])  for r in rows]
        ranks        = [r["rank"] for r in rows]
        return train_curves, eval_curves, ranks
