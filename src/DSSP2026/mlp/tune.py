"""
mlp/tune.py — Optuna hyperparameter search for MLP classifiers.

Searches both the MLP hyperparameters and which feature set to use, scoring by
stratified k-fold **macro-F1** on the training set. Selection records per-epoch
train/eval log-loss curves on each trial (as Optuna ``user_attrs``) so the
shared ``tuning.optuna_train`` plot can render the aggregate training run later
without refitting.

The two curve-attr keys (``train_loss_curve`` / ``eval_loss_curve``) match the
keys ``tuning.optuna_train.plot_training_run`` reads, so a study produced here
plugs straight into ``save_training_run``.
"""

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.mlp.pipeline import build_preprocessor
from DSSP2026.experiment.tuning.search import (
    EVAL_CURVE_KEY,
    TRAIN_CURVE_KEY,
    OptunaSearchResult,
    run_optuna_search,
)
from DSSP2026.experiment.cv import average_curves

MLPTuneResult = OptunaSearchResult


def cv_macro_f1_with_curves(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
    *,
    hidden,
    activation: str,
    alpha: float,
    lr_init: float,
    n_splits: int = 5,
    n_epochs: int = 150,
    n_iter_no_change: int = 8,
    min_improve: float = 1e-3,
    random_state: int = 42,
):
    """Stratified CV that records per-epoch train/eval log-loss curves.

    For each fold: fit the preprocessor on the fold's training rows only, then
    drive the MLP with ``partial_fit`` one epoch at a time, recording the
    training loss (``mlp.loss_``) and the validation log-loss on the held-out
    fold each epoch. A fold early-stops when val log-loss fails to improve by at
    least ``min_improve`` for ``n_iter_no_change`` epochs (a meaningful-
    improvement tolerance, so a flat-lined fold cuts off rather than running all
    ``n_epochs``).

    Parameters
    ----------
    X : DataFrame
        Training features (all candidate columns; ``features`` selects within).
    y : array-like of int
        Integer-encoded class labels.
    features : sequence of str
        Columns this configuration uses.
    numeric_features, flag_features : sequence of str
        Column vocabularies, forwarded to the preprocessor.
    hidden : tuple of int
        Per-layer widths.
    n_epochs : int
        Max epochs per fold.
    n_iter_no_change, min_improve : int, float
        Early-stopping patience and the minimum val-loss improvement that counts.

    Returns
    -------
    mean_f1 : float
        Mean macro-F1 across folds (at each fold's best-val-loss epoch).
    train_curve, eval_curve : list of float
        Per-epoch mean train / eval log-loss aggregated (NaN-padded) across folds.
    """
    from sklearn.metrics import f1_score, log_loss
    from sklearn.neural_network import MLPClassifier

    from DSSP2026.experiment.cv import cv_fold_scores

    features = list(features)
    classes = np.unique(y)

    def fold_fn(Xtr_rows, ytr, Xva_rows, yva):
        # Fit the preprocessor on this fold's training rows only (leakage-safe),
        # then drive the MLP epoch-by-epoch recording train/eval log-loss curves.
        pre = build_preprocessor(features, numeric_features, flag_features)
        Xtr = pre.fit_transform(Xtr_rows[features])
        Xva = pre.transform(Xva_rows[features])

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden, activation=activation, alpha=alpha,
            learning_rate_init=lr_init, random_state=random_state)

        tr_curve, ev_curve = [], []
        best_ev, best_proba, since_best = np.inf, None, 0
        for _ in range(n_epochs):
            mlp.partial_fit(Xtr, ytr, classes=classes)
            tr_curve.append(float(mlp.loss_))                       # train log-loss
            proba = mlp.predict_proba(Xva)
            ev = float(log_loss(yva, proba, labels=classes))        # eval log-loss
            ev_curve.append(ev)
            if ev < best_ev - min_improve:
                best_ev, best_proba, since_best = ev, proba, 0
            else:
                since_best += 1
                if since_best >= n_iter_no_change:
                    break

        preds = classes[np.argmax(best_proba, axis=1)]
        return (f1_score(yva, preds, average="macro"), tr_curve, ev_curve)

    # cv_fold_scores owns the stratified split; the per-fold work is the callback.
    fold_results = cv_fold_scores(
        fold_fn, X[features], y, n_splits=n_splits, stratified=True,
        random_state=random_state)
    fold_f1 = [r[0] for r in fold_results]
    fold_train = [r[1] for r in fold_results]
    fold_eval = [r[2] for r in fold_results]

    return float(np.mean(fold_f1)), average_curves(fold_train), average_curves(fold_eval)


def make_mlp_objective(
    train: pd.DataFrame,
    target: str,
    feature_sets: Mapping[str, Sequence[str]],
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
    label_encoder,
    *,
    n_splits: int = 5,
    max_layers: int = 4,
    width_range: tuple = (16, 512),
    alpha_range: tuple = (1e-6, 1e1),
    lr_range: tuple = (1e-4, 1e-1),
    activations: Sequence[str] = ("relu", "tanh"),
    random_state: int = 42,
):
    """Build an Optuna objective that searches feature set + MLP hyperparameters.

    The objective suggests a ``feature_set`` (a key of ``feature_sets``), a
    layer count and per-layer widths, an activation, ``alpha``, and
    ``learning_rate_init``, then scores the configuration with
    :func:`cv_macro_f1_with_curves`. The per-epoch loss curves are stored on the
    trial as ``user_attrs`` (keys ``train_loss_curve`` / ``eval_loss_curve``) so
    ``tuning.optuna_train.save_training_run`` can plot them later.

    Parameters
    ----------
    train : DataFrame
        Training data (features + target column).
    target : str
        Name of the target column in ``train``.
    feature_sets : mapping of str -> sequence of str
        Named candidate feature sets; the objective picks among the keys.
    numeric_features, flag_features : sequence of str
        Column vocabularies, forwarded to CV/preprocessing.
    label_encoder : fitted LabelEncoder
        Maps the string target to the integer labels the MLP trains on.
    max_layers : int
        Upper bound on the searched layer count (widths are width1..widthN).
    width_range, alpha_range, lr_range : tuple
        Search bounds (widths and alpha/lr are searched on a log scale).
    activations : sequence of str
        Candidate activations.

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

        n_layers = trial.suggest_int("n_layers", 1, max_layers)
        hidden = tuple(
            trial.suggest_int(f"width{i}", width_range[0], width_range[1], log=True)
            for i in range(1, n_layers + 1)
        )
        activation = trial.suggest_categorical("activation", list(activations))
        alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)
        lr_init = trial.suggest_float("lr_init", lr_range[0], lr_range[1], log=True)

        f1, train_curve, eval_curve = cv_macro_f1_with_curves(
            train[features], y, features, numeric_features, flag_features,
            hidden=hidden, activation=activation, alpha=alpha, lr_init=lr_init,
            n_splits=n_splits, random_state=random_state)

        # Persist per-epoch curves so the training-run plot can read them later.
        trial.set_user_attr(TRAIN_CURVE_KEY, train_curve)
        trial.set_user_attr(EVAL_CURVE_KEY, eval_curve)
        return f1

    return objective


def run_mlp_search(
    train: pd.DataFrame,
    target: str,
    feature_sets: Mapping[str, Sequence[str]],
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
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
) -> MLPTuneResult:
    """Create/load an Optuna study and run the MLP search; return MLPTuneResult.

    Direction is ``maximize`` (macro-F1). If ``storage`` and ``study_name`` are
    given the study is persisted (and resumed if it already exists), so it stays
    inspectable in optuna-dashboard and can be re-opened for the training-run /
    parallel-coordinates plots in ``tuning``.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.
    study_name, storage : str, optional
        Study name and storage URL (e.g. ``"sqlite:///.../optuna_mlp.db"``).
        Both must be given to persist; otherwise an in-memory study is used.
    callbacks : sequence, optional
        Optuna callbacks (e.g. a per-trial logger).
    **objective_kwargs
        Forwarded to :func:`make_mlp_objective` (max_layers, width_range, ...).

    Returns
    -------
    MLPTuneResult
    """
    objective = make_mlp_objective(
        train, target, feature_sets, numeric_features, flag_features,
        label_encoder, n_splits=n_splits, random_state=random_state,
        **objective_kwargs)
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