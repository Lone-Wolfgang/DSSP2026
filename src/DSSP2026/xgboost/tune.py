"""
xgboost/tune.py — Optuna hyperparameter search for XGBoost classifiers.

Searches the XGBoost hyperparameters and which feature set to use, scoring by
stratified k-fold **macro-F1** on the training set. Like the MLP search, each
trial records per-round train/eval log-loss curves as Optuna ``user_attrs``
(keys ``train_loss_curve`` / ``eval_loss_curve``) so the shared
``tuning.optuna_train`` plot can render the aggregate training run later.

Unlike the MLP — whose curves come from a manual per-epoch ``partial_fit`` loop
— XGBoost produces the curves natively via ``eval_set`` + ``evals_result()``,
one value per boosting round. The fold curves are averaged (NaN-padded) to a
single train/eval curve, matching the format ``plot_training_run`` expects.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.tuning.search import (
    EVAL_CURVE_KEY,
    TRAIN_CURVE_KEY,
    OptunaSearchResult,
    run_optuna_search,
)
from DSSP2026.experiment.cv import average_curves

XGBTuneResult = OptunaSearchResult


def cv_macro_f1_with_curves(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    min_child_weight: float,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Stratified CV that records per-round train/eval log-loss curves.

    For each fold an ``XGBClassifier`` is fit with the held-out fold as an
    ``eval_set``; ``evals_result()`` yields the per-round train and validation
    mlogloss, and macro-F1 is computed from the fold's final predictions. The
    per-round fold curves are averaged (NaN-padded to the longest) into single
    train/eval curves.

    Returns
    -------
    mean_f1 : float
        Mean macro-F1 across folds.
    train_curve, eval_curve : list of float
        Per-round mean train / eval log-loss aggregated across folds.
    """
    import xgboost as xgb
    from sklearn.metrics import f1_score

    from DSSP2026.experiment.cv import cv_fold_scores

    features = list(features)
    classes = np.unique(y)

    def fold_fn(Xtr, ytr, Xva, yva):
        # Objective is auto-selected by class count (binary:logistic for 2
        # classes, multi:softprob for >2). The eval metric and the key used to
        # read evals_result() differ accordingly: "logloss" vs "mlogloss".
        eval_metric = "mlogloss" if len(classes) > 2 else "logloss"
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            reg_alpha=reg_alpha, min_child_weight=min_child_weight,
            tree_method="hist", eval_metric=eval_metric,
            random_state=random_state, n_jobs=1)
        clf.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)

        ev = clf.evals_result()
        # validation_0 == train eval_set, validation_1 == held-out fold
        tr_curve = [float(v) for v in ev["validation_0"][eval_metric]]
        ev_curve = [float(v) for v in ev["validation_1"][eval_metric]]
        preds = clf.predict(Xva)
        return (f1_score(yva, preds, average="macro"), tr_curve, ev_curve)

    fold_results = cv_fold_scores(
        fold_fn, X[features], y, n_splits=n_splits, stratified=True,
        random_state=random_state)
    fold_f1 = [r[0] for r in fold_results]
    fold_train = [r[1] for r in fold_results]
    fold_eval = [r[2] for r in fold_results]

    return float(np.mean(fold_f1)), average_curves(fold_train), average_curves(fold_eval)


def make_xgb_objective(
    train: pd.DataFrame,
    target: str,
    feature_sets: Mapping[str, Sequence[str]],
    label_encoder,
    *,
    n_splits: int = 5,
    n_estimators_range: tuple = (100, 600),
    max_depth_range: tuple = (2, 10),
    lr_range: tuple = (1e-3, 3e-1),
    subsample_range: tuple = (0.5, 1.0),
    colsample_range: tuple = (0.5, 1.0),
    reg_lambda_range: tuple = (1e-3, 1e2),
    reg_alpha_range: tuple = (1e-4, 1e1),
    min_child_weight_range: tuple = (1.0, 10.0),
    random_state: int = 42,
):
    """Build an Optuna objective that searches feature set + XGB hyperparameters.

    The objective suggests a ``feature_set`` (a key of ``feature_sets``) plus
    ``n_estimators``, ``max_depth``, ``learning_rate``, ``subsample``,
    ``colsample_bytree``, ``reg_lambda``, ``reg_alpha`` and ``min_child_weight``,
    then scores via :func:`cv_macro_f1_with_curves`. Per-round loss curves are
    stored on the trial as ``user_attrs`` for the shared training-run plot.

    Returns
    -------
    objective : callable
        An ``objective(trial) -> float`` for ``study.optimize``.
    """
    y = label_encoder.transform(train[target])     # int labels
    feature_set_names = list(feature_sets)

    def objective(trial):
        feature_set = trial.suggest_categorical("feature_set", feature_set_names)
        features = list(feature_sets[feature_set])

        params = dict(
            n_estimators=trial.suggest_int(
                "n_estimators", n_estimators_range[0], n_estimators_range[1], step=50),
            max_depth=trial.suggest_int(
                "max_depth", max_depth_range[0], max_depth_range[1]),
            learning_rate=trial.suggest_float(
                "learning_rate", lr_range[0], lr_range[1], log=True),
            subsample=trial.suggest_float(
                "subsample", subsample_range[0], subsample_range[1]),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", colsample_range[0], colsample_range[1]),
            reg_lambda=trial.suggest_float(
                "reg_lambda", reg_lambda_range[0], reg_lambda_range[1], log=True),
            reg_alpha=trial.suggest_float(
                "reg_alpha", reg_alpha_range[0], reg_alpha_range[1], log=True),
            min_child_weight=trial.suggest_float(
                "min_child_weight", min_child_weight_range[0], min_child_weight_range[1]),
        )

        f1, train_curve, eval_curve = cv_macro_f1_with_curves(
            train[features], y, features, n_splits=n_splits,
            random_state=random_state, **params)

        trial.set_user_attr(TRAIN_CURVE_KEY, train_curve)
        trial.set_user_attr(EVAL_CURVE_KEY, eval_curve)
        return f1

    return objective


def run_xgb_search(
    train: pd.DataFrame,
    target: str,
    feature_sets: Mapping[str, Sequence[str]],
    label_encoder,
    *,
    n_trials: int = 30,
    n_splits: int = 5,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    random_state: int = 42,
    callbacks: Optional[Sequence] = None,
    show_progress_bar: bool = False,
    **objective_kwargs,
) -> XGBTuneResult:
    """Create/load an Optuna study and run the XGB search; return XGBTuneResult.

    Direction is ``maximize`` (macro-F1). If ``storage`` and ``study_name`` are
    given the study is persisted (and resumed if it already exists), so it stays
    inspectable in optuna-dashboard and can be re-opened for the training-run /
    parallel-coordinates plots in ``tuning``.

    **objective_kwargs are forwarded to :func:`make_xgb_objective` (the search
    ranges).

    Returns
    -------
    XGBTuneResult
    """
    objective = make_xgb_objective(
        train, target, feature_sets, label_encoder, n_splits=n_splits,
        random_state=random_state, **objective_kwargs)
    return run_optuna_search(
        objective,
        n_trials=n_trials,
        n_splits=n_splits,
        scoring="f1_macro",
        study_name=study_name,
        storage=storage,
        random_state=random_state,
        callbacks=callbacks,
        show_progress_bar=show_progress_bar,
    )