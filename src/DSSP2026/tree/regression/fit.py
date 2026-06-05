"""
tree/fit.py — fitting tree models at chosen hyperparameters.

Depth selection lives in tree/tune.py. This module fits a tree once the depth
is decided and returns the model plus a tidy metrics row.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from DSSP2026.core.metrics import regression_metrics


@dataclass
class TreeFitResult:
    """Container returned by fit_decision_tree."""
    model: object
    metrics: dict
    features: list
    target: str
    max_depth: Optional[int] = None


def fit_decision_tree(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    max_depth: Optional[int] = None,
    random_state: int = 1,
) -> TreeFitResult:
    """Fit a DecisionTreeRegressor at a fixed depth and score it on Test."""
    from sklearn.tree import DecisionTreeRegressor

    features = list(features)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(Train[features], Train[target])
    preds = model.predict(Test[features])
    metrics = regression_metrics(Test[target], preds)
    return TreeFitResult(model=model, metrics=metrics, features=features,
                         target=target, max_depth=max_depth)


def fit_random_forest(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    estimator=None,
    n_estimators: int = 100,
    max_features=None,
    random_state: int = 42,
) -> TreeFitResult:
    """Fit (or accept a pre-fit) RandomForestRegressor and score it on Test.

    Pass `estimator` to reuse a model already fit by a grid search (e.g.
    GridSearchResult.best_estimator); otherwise one is built and fit here.
    """
    from sklearn.ensemble import RandomForestRegressor

    features = list(features)
    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features,
            random_state=random_state, n_jobs=-1)
        estimator.fit(Train[features], Train[target])

    preds = estimator.predict(Test[features])
    metrics = regression_metrics(Test[target], preds)
    return TreeFitResult(model=estimator, metrics=metrics, features=features,
                         target=target)
