"""
tree/tune.py — hyperparameter tuning for tree models.

- Decision-tree depth selection via the elbow method on a held-out test set.
- Random-forest hyperparameter search via cross-validated grid search.

These return the search results (and chosen params); fitting a model at given
hyperparameters lives in tree/fit.py.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _find_elbow(depths, mspe_vals) -> int:
    """Elbow point: max distance from the line joining first and last points."""
    x = np.array(depths, dtype=float)
    y = np.array(mspe_vals, dtype=float)

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    d = p2 - p1
    d_norm = d / (np.linalg.norm(d) + 1e-12)

    distances = np.abs((y_n - p1[1]) * d_norm[0] - (x_n - p1[0]) * d_norm[1])
    return int(depths[int(np.argmax(distances))])


def tune_dt_depth(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    depths: Optional[Sequence[int]] = None,
):
    """Sweep decision-tree max_depth and pick the elbow.

    Returns
    -------
    results_df : pd.DataFrame   columns: max_depth, MSPE, R²
    elbow : int                 selected max_depth
    """
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor

    if depths is None:
        depths = list(range(1, 11))
    depths = list(depths)
    features = list(features)

    rows = []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=1)
        m.fit(Train[features], Train[target])
        preds = m.predict(Test[features])
        rows.append({
            "max_depth": d,
            "MSPE": mean_squared_error(Test[target], preds),
            "R²": r2_score(Test[target], preds),
        })
    results_df = pd.DataFrame(rows)
    elbow = _find_elbow(results_df["max_depth"].tolist(), results_df["MSPE"].tolist())
    return results_df, elbow


@dataclass
class GridSearchResult:
    """Container returned by tune_rf_grid."""
    best_estimator: object
    best_params: dict
    cv_results: dict
    param_grid: dict


# tune_rf_grid (GridSearchCV/KFold) was retired here; regression-forest grid
# search is expressed as an Optuna GridSampler study calling experiment.cv.cv_score.