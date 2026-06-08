from dataclasses import dataclass

import numpy as np


TRAIN_CURVE_KEY = "train_loss_curve"
EVAL_CURVE_KEY = "eval_loss_curve"


@dataclass
class OptunaSearchResult:
    """Returned by Optuna hyperparameter searches. Pure data — no figures."""
    study: object
    best_params: dict
    best_value: float
    scoring: str
    n_splits: int


def run_optuna_search(
    objective,
    *,
    n_trials=30,
    n_splits,
    scoring="f1_macro",
    study_name=None,
    storage=None,
    random_state=42,
    callbacks=None,
    show_progress_bar=False,
) -> OptunaSearchResult:
    import optuna

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=list(callbacks) if callbacks else None,
    )
    return OptunaSearchResult(
        study=study,
        best_params=study.best_params,
        best_value=study.best_value,
        scoring=scoring,
        n_splits=n_splits,
    )