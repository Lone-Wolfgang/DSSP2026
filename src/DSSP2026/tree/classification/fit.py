"""
tree/classification/fit.py — fitting classification trees and forests.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from DSSP2026.core.metrics import classification_metrics
from DSSP2026.core.results import ClassificationResult
from DSSP2026.tree._shared import feature_importance_df


@dataclass
class TreeClassifyResult(ClassificationResult):
    """Container returned by fit_decision_tree_classifier."""
    max_depth: Optional[int] = None


def fit_decision_tree_classifier(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    max_depth: Optional[int] = None,
    average: str = "macro",
    random_state: int = 42,
) -> TreeClassifyResult:
    """Fit a DecisionTreeClassifier at a fixed depth and score it on Test.

    Parameters
    ----------
    max_depth : int, optional
        Tree depth (typically the one-SE depth from tune_dt_depth_cv).
    average : str
        Averaging for precision/recall/F1, passed to
        ``core.metrics.classification_metrics``. Default "macro".

    Returns
    -------
    TreeClassifyResult
        Holds the model, the test metrics, and the test predictions/probabilities
        ready for ``core.metrics.make_confusion_matrix`` /
        ``make_classification_report_df`` and ``core.plot.save_confusion_matrix_png``.
    """
    from sklearn.tree import DecisionTreeClassifier

    features = list(features)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(Train[features], Train[target])

    y_true = Test[target].to_numpy()
    y_pred = model.predict(Test[features])
    # predict_proba columns follow model.classes_; pass it for multiclass ROC-AUC.
    y_proba = model.predict_proba(Test[features])

    metrics = classification_metrics(
        y_true, y_pred, y_score=y_proba, average=average)

    return TreeClassifyResult(
        model=model, metrics=metrics, features=features, target=target,
        classes_=list(model.classes_), y_true=y_true, y_pred=y_pred,
        y_proba=y_proba, max_depth=max_depth)


def classifier_feature_importance_df(result: TreeClassifyResult) -> pd.DataFrame:
    """Tidy, descending feature-importance table for a fitted classifier.

    Thin wrapper over ``tree._shared.feature_importance_df`` so callers can pass the
    result object directly.
    """
    return feature_importance_df(result.model, result.features)


@dataclass
class RFClassifyResult(ClassificationResult):
    """Container returned by fit_random_forest_classifier."""
    best_params: Optional[dict] = None


def fit_random_forest_classifier(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    estimator=None,
    n_estimators: int = 100,
    max_features="sqrt",
    average: str = "macro",
    random_state: int = 42,
) -> RFClassifyResult:
    """Fit (or accept a pre-fit) RandomForestClassifier and score it on Test.

    Pass ``estimator`` to reuse the model already fit by the grid search
    (``RFGridResult.best_estimator``); otherwise one is built and fit here.

    Parameters
    ----------
    average : str
        Averaging for precision/recall/F1, passed to
        ``core.metrics.classification_metrics``. Default "macro".

    Returns
    -------
    RFClassifyResult
        Holds the model, test metrics, and test predictions/probabilities ready
        for the ``core`` confusion-matrix / classification-report / ROC paths.
    """
    from sklearn.ensemble import RandomForestClassifier

    features = list(features)
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_features=max_features,
            random_state=random_state, n_jobs=-1)
        estimator.fit(Train[features], Train[target])

    y_true = Test[target].to_numpy()
    y_pred = estimator.predict(Test[features])
    y_proba = estimator.predict_proba(Test[features])   # cols follow classes_

    metrics = classification_metrics(
        y_true, y_pred, y_score=y_proba, average=average)

    best_params = getattr(estimator, "_grid_best_params", None)
    return RFClassifyResult(
        model=estimator, metrics=metrics, features=features, target=target,
        classes_=list(estimator.classes_), y_true=y_true, y_pred=y_pred,
        y_proba=y_proba, best_params=best_params)


def rf_feature_importance_df(result: RFClassifyResult) -> pd.DataFrame:
    """Tidy, descending feature-importance table for a fitted RF classifier.

    Thin wrapper over ``tree._shared.feature_importance_df`` so callers can pass the
    result object directly.
    """
    return feature_importance_df(result.model, result.features)