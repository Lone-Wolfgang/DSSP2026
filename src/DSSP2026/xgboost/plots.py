"""
xgboost/plots.py — XGBoost-specific plotting.

Mostly thin. ``XGBClassifier`` exposes ``.feature_importances_`` in the same
shape the shared ``tree._shared`` helpers expect, so feature-importance plots
reuse those directly rather than reimplementing them. The rest of an XGB run's
plots are already covered by shared modules:

- Training loss curves (aggregate over top-N trials):
    ``DSSP2026.tuning.optuna_train.save_training_run(study, path)``
- Parallel-coordinates over the searched hyperparameters:
    ``DSSP2026.tuning.optuna_parallel.save_optuna_parallel_coordinates(study, path)``
- Held-out confusion matrix:
    ``DSSP2026.core.heatmap.save_confusion_matrix_png(cm, path, ...)``
- Per-class classification report:
    ``DSSP2026.evaluation.tables.save_classification_report_png(report_df, path)``

The default importance here is XGBoost's gain-based ``feature_importances_``.
``importance_df_from_booster`` is offered for the other XGB importance types
(weight / cover / total_gain) when you want them.
"""

from typing import Sequence

import pandas as pd

from DSSP2026.tree._shared import feature_importance_df, save_rf_feature_importance_png


def xgb_feature_importance_df(result) -> pd.DataFrame:
    """Tidy, descending feature-importance DataFrame for an XGBResult.

    Uses the model's gain-based ``feature_importances_`` via the shared
    ``tree._shared.feature_importance_df`` (XGBClassifier matches that API).
    """
    return feature_importance_df(result.model, result.features)


def save_xgb_feature_importance_png(result, path, *, top_n=None, dpi=220):
    """Save the gain-based feature-importance bar chart for an XGBResult.

    Reuses the shared importance plotter, so the styling matches the tree/RF
    importance plots exactly.
    """
    return save_rf_feature_importance_png(
        xgb_feature_importance_df(result), path, top_n=top_n, dpi=dpi)


def importance_df_from_booster(result, importance_type: str = "gain") -> pd.DataFrame:
    """Feature importances by an explicit XGBoost ``importance_type``.

    ``feature_importances_`` is gain-based; this exposes the others
    ("weight", "cover", "total_gain", "total_cover") straight from the booster.
    Features the booster never split on are reported as 0.0.

    Parameters
    ----------
    result : XGBResult
    importance_type : {"gain", "weight", "cover", "total_gain", "total_cover"}

    Returns
    -------
    DataFrame
        Columns ``Feature`` / ``Importance``, descending — same shape the shared
        plotter consumes.
    """
    booster = result.model.get_booster()
    score = booster.get_score(importance_type=importance_type)  # keyed by f0,f1,...
    features = list(result.features)
    # XGBoost keys are positional ("f{idx}") unless feature_names were set.
    values = []
    for i, feat in enumerate(features):
        values.append(score.get(feat, score.get(f"f{i}", 0.0)))
    return pd.DataFrame({"Feature": features, "Importance": values}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)
