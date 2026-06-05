"""tree/classification/tune.py — result containers only.

The CV/grid/threshold tuning logic that used to live here was retired and
centralized in ``experiment/cv.py`` (depth-sweep + grid CV) and the new
Optuna-based experiment layer. These dataclasses remain as the result
containers other modules still reference.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DepthTuneResult:
    """Returned by tune_dt_depth_cv. Pure data — no figures.

    Attributes
    ----------
    results_df : pd.DataFrame
        One row per depth: max_depth, mean_score, std_score, se_score.
    best_depth : int
        Depth with the highest mean CV score.
    one_se_depth : int
        Shallowest depth whose mean CV score is within one SE of the best
        depth's mean (the selected, parsimonious depth).
    scoring : str
        The sklearn scorer used (e.g. "f1_macro").
    n_splits : int
        Number of CV folds.
    se_threshold : float
        best_mean - 1 SE; depths at/above this line qualify under the one-SE rule.
    """
    results_df: pd.DataFrame
    best_depth: int
    one_se_depth: int
    scoring: str
    n_splits: int
    se_threshold: float


@dataclass
class RFGridResult:
    """Returned by tune_rf_grid_classify.

    Mirrors tree.tune.GridSearchResult, plus the scorer name so downstream
    plots/labels can report which metric the grid was scored on.
    """
    best_estimator: object
    best_params: dict
    cv_results: dict
    param_grid: dict
    scoring: str