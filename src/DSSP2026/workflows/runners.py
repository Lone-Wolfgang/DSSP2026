"""
workflows/runners.py — per-family model runners with a uniform interface.

Each runner takes ``(train, evaluation, eval_kind, feature_sets, out_dir)`` and
returns a list of result rows, one per (model x feature set) it scored:

    {"model": str, "feature_set": str, "detail": str, "metrics": dict}

This implements CLI decision #2 (feature sets): for logistic / tree / rf, each
selected feature set is run side by side (one row each). For the MLP, the
feature sets fold into the Optuna search instead (the search picks one), so the
MLP runner returns a single row whose ``feature_set`` is the chosen one.

Every family is fit on ``train`` and scored once on ``evaluation`` (the carved
validation slice or the real test set, decided upstream in ``data``). The shared
confusion-matrix + classification-report artifacts come from ``_save_common_eval``
so all families look identical downstream.
"""

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from DSSP2026.core import metrics as CM
from DSSP2026.core.heatmap import save_confusion_matrix_png
from DSSP2026.evaluation.tables import save_classification_report_png
from DSSP2026.workflows import config as C
from DSSP2026.workflows import data as D
from DSSP2026.workflows import study_db

logger = logging.getLogger(__name__)


def _save_common_eval(y_true, y_pred, y_proba, tag, title, out_dir, eval_kind,
                      *, labels=None, target_names=None, average=None,
                      class_order=None):
    """Confusion matrix + classification report on the evaluation set.

    Renders the per-model artifacts (identically for every family) and returns
    ``(metrics, per_class_df, confusion_counts_df, predictions)`` where
    ``predictions`` is a dict {class_order, y_true, y_proba} capturing the full
    per-sample probability matrix so the study database can later rebuild ROC
    curves, tune the decision threshold, and recompute confusion matrices.
    The PNGs are unchanged: the confusion-matrix PNG is still row-normalised for
    display, while the returned matrix holds integer counts.

    ``class_order`` is the list of class labels matching the COLUMNS of
    ``y_proba`` (each model family orders proba columns by its own classes_;
    pass that order so the stored matrix is unambiguous). Falls back to
    ``labels`` if not given.

    ``labels`` / ``target_names`` / ``average`` default to the multiclass
    TeleLogs config (C1..C8, macro). Binary runs pass their own label space
    (e.g. ``labels=[0, 1]``) so the confusion matrix and report match the 0/1
    predictions instead of looking for the C1..C8 classes.
    """
    labels = labels if labels is not None else C.CLASS_LABELS
    target_names = target_names if target_names is not None else labels
    average = average if average is not None else C.AVERAGE

    # Raw integer counts (normalize=None). The PNG saver normalises for display
    # via its own normalize=True; the DataFrame returned here stays counts.
    cm = CM.make_confusion_matrix(y_true, y_pred, labels=labels)
    save_confusion_matrix_png(
        cm, out_dir / f"{tag}_confusion_matrix.png", class_labels=labels,
        normalize=True, title=f"{title}: {eval_kind} confusion matrix")
    report = CM.make_classification_report_df(
        y_true, y_pred, target_names=target_names)
    save_classification_report_png(
        report, out_dir / f"{tag}_classification_report.png")
    metrics = CM.classification_metrics(
        y_true, y_pred, y_score=y_proba, average=average)

    # Prediction block for the study DB (full probability matrix + true labels).
    proba_arr = np.asarray(
        y_proba.to_numpy() if hasattr(y_proba, "to_numpy") else y_proba,
        dtype=float)
    if class_order is None:
        # Prefer the proba DataFrame's own columns if present, else display labels.
        class_order = (list(y_proba.columns) if hasattr(y_proba, "columns")
                       else list(labels))
    true_arr = (y_true.to_numpy() if hasattr(y_true, "to_numpy")
                else np.asarray(y_true))
    predictions = {
        "class_order": [str(c) for c in class_order],
        "y_true": [str(v) for v in true_arr.tolist()],
        "y_proba": proba_arr,
    }
    return metrics, report, cm, predictions


def _row(model, feature_set, detail, metrics, *,
         hyperparams=None, feature_list=None,
         per_class_df=None, confusion_df=None, threshold=None, trials=None,
         predictions=None):
    """Assemble a result row. The first four args feed the comparison table;
    the keyword extras are persisted to the study database (study_db)."""
    return {"model": model, "feature_set": feature_set,
            "detail": detail, "metrics": metrics,
            "hyperparams": hyperparams or {},
            "feature_list": list(feature_list) if feature_list is not None else None,
            "per_class_df": per_class_df, "confusion_df": confusion_df,
            "threshold": threshold, "trials": trials,
            "predictions": predictions}


# ===========================================================================
# Logistic regression (multinomial logit)
# ===========================================================================
def run_logistic(train, evaluation, eval_kind, feature_sets, out_dir):
    from DSSP2026.logistic_regression import multiclass as MC

    tr, ev = D.standardise_numeric(train, evaluation)
    rows = []
    for name, features in feature_sets.items():
        tag = f"logistic_{name}"
        title = f"Logistic ({name})"
        logger.info("Logistic — feature set '%s' (%d features)", name, len(features))

        res = MC.fit_mnlogit(tr, f"{C.TARGET} ~ " + " + ".join(features))
        MC.save_mnlogit_coefficients(res, out_dir / f"{tag}_coefficients.png")
        MC.save_mnlogit_odds_ratios(
            res, out_dir / f"{tag}_odds_ratios.png",
            title=f"{title}: odds ratios by class (95% CI)")

        pred = MC.predict_mnlogit(res, ev)
        y_true = MC.get_endog(res, ev)
        metrics, per_class_df, confusion_df, predictions = _save_common_eval(
            y_true, pred.labels, pred.proba.to_numpy(), tag, title, out_dir,
            eval_kind, class_order=list(pred.proba.columns))
        rows.append(_row("Logistic regression", name,
                         f"feature_set={name}", metrics,
                         hyperparams={}, feature_list=features,
                         per_class_df=per_class_df, confusion_df=confusion_df,
                         predictions=predictions))
    return rows


# ===========================================================================
# Decision tree (CV depth tuning, one-SE rule)
# ===========================================================================
def run_tree(train, evaluation, eval_kind, feature_sets, out_dir):
    from DSSP2026.tree.classification import tune as TT
    from DSSP2026.tree.classification import fit as TF
    from DSSP2026.tree.classification import plots as TP
    from DSSP2026.tree._shared import save_rf_feature_importance_png

    tr, ev = D.impute_numeric(train, evaluation)
    rows = []
    for name, features in feature_sets.items():
        tag = f"tree_{name}"
        title = f"Decision tree ({name})"
        logger.info("Tree — feature set '%s' (%d features)", name, len(features))

        tune = TT.tune_dt_depth_cv(
            tr, features, C.TARGET, depths=list(range(1, 20)),
            scoring=C.SCORING, n_splits=C.CV_SPLITS, random_state=C.RANDOM_STATE)
        TP.save_cv_depth_curve_png(tune, out_dir / f"{tag}_cv_depth.png")

        res = TF.fit_decision_tree_classifier(
            tr, ev, features, C.TARGET, max_depth=tune.one_se_depth,
            average=C.AVERAGE, random_state=C.RANDOM_STATE)
        save_rf_feature_importance_png(
            TF.classifier_feature_importance_df(res),
            out_dir / f"{tag}_importance.png", top_n=C.IMPORTANCE_TOP_N)

        metrics, per_class_df, confusion_df, predictions = _save_common_eval(
            res.y_true, res.y_pred, res.y_proba, tag, title, out_dir, eval_kind,
            class_order=list(res.classes_))
        rows.append(_row("Decision tree", name,
                         f"depth={tune.one_se_depth}", metrics,
                         hyperparams={"max_depth": tune.one_se_depth},
                         feature_list=features,
                         per_class_df=per_class_df, confusion_df=confusion_df,
                         trials=study_db.trials_from_depth_sweep(tune),
                         predictions=predictions))
    return rows


# ===========================================================================
# Random forest (CV grid search over n_estimators x max_features)
# ===========================================================================
def run_rf(train, evaluation, eval_kind, feature_sets, out_dir):
    from DSSP2026.tree.classification import tune as TT
    from DSSP2026.tree.classification import fit as TF
    from DSSP2026.tree.classification import plots as TP
    from DSSP2026.tree._shared import save_rf_feature_importance_png

    tr, ev = D.impute_numeric(train, evaluation)
    param_grid = {"n_estimators": [50, 100, 200, 300],
                  "max_features": [2, 3, 5, "sqrt", "log2"]}
    rows = []
    for name, features in feature_sets.items():
        tag = f"rf_{name}"
        title = f"Random forest ({name})"
        logger.info("RF — feature set '%s' (%d features)", name, len(features))

        grid = TT.tune_rf_grid_classify(
            tr, features, C.TARGET, param_grid=param_grid, scoring=C.SCORING,
            n_splits=C.CV_SPLITS, random_state=C.RANDOM_STATE)
        TP.save_rf_grid_heatmap(
            grid.cv_results, out_dir / f"{tag}_grid.png", scoring=grid.scoring,
            best_n=grid.best_params["n_estimators"],
            best_mf=grid.best_params["max_features"])

        res = TF.fit_random_forest_classifier(
            tr, ev, features, C.TARGET, estimator=grid.best_estimator,
            average=C.AVERAGE, random_state=C.RANDOM_STATE)
        save_rf_feature_importance_png(
            TF.rf_feature_importance_df(res),
            out_dir / f"{tag}_importance.png", top_n=C.IMPORTANCE_TOP_N)

        metrics, per_class_df, confusion_df, predictions = _save_common_eval(
            res.y_true, res.y_pred, res.y_proba, tag, title, out_dir, eval_kind,
            class_order=list(res.classes_))
        bp = grid.best_params
        rows.append(_row(
            "Random forest", name,
            f"n_est={bp['n_estimators']}, max_feat={bp['max_features']}", metrics,
            hyperparams=dict(bp), feature_list=features,
            per_class_df=per_class_df, confusion_df=confusion_df,
            trials=study_db.trials_from_rf_grid(grid),
            predictions=predictions))
    return rows


# ===========================================================================
# MLP (Optuna search over architecture + feature set)
# ---------------------------------------------------------------------------
# Unlike the others, the MLP folds the feature sets INTO its search: feature_set
# is an Optuna categorical, so the search picks one. The runner therefore returns
# a single row whose feature_set is the chosen one.
# ===========================================================================
def run_mlp(train, evaluation, eval_kind, feature_sets, out_dir, *,
            n_trials=30):
    import optuna
    from sklearn.preprocessing import LabelEncoder

    from DSSP2026.mlp.tune import run_mlp_search
    from DSSP2026.mlp.fit import refit_best
    from DSSP2026.tuning.optuna_train import save_training_run
    from DSSP2026.tuning.optuna_parallel import save_optuna_parallel_coordinates

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, ev = D.impute_numeric(train, evaluation)
    le = LabelEncoder().fit(tr[C.TARGET])

    storage = f"sqlite:///{(out_dir / 'optuna_mlp.db').resolve()}"
    tune = run_mlp_search(
        tr, C.TARGET, feature_sets, C.NUMERIC_FEATURES, C.FLAG_FEATURES, le,
        n_trials=n_trials, n_splits=C.CV_SPLITS,
        study_name="workflow_mlp", storage=storage, random_state=C.RANDOM_STATE)
    logger.info("MLP best CV macro-F1 = %.4f", tune.best_value)

    save_optuna_parallel_coordinates(
        tune.study, out_dir / "mlp_parallel_coordinates.png")
    save_training_run(tune.study, out_dir / "mlp_training_run.png")

    res = refit_best(
        tune.study, tr, ev, C.TARGET, feature_sets,
        C.NUMERIC_FEATURES, C.FLAG_FEATURES, label_encoder=le, average=C.AVERAGE,
        random_state=C.RANDOM_STATE)
    chosen = tune.best_params["feature_set"]
    metrics, per_class_df, confusion_df, predictions = _save_common_eval(
        res.y_true, res.y_pred, res.y_proba, "mlp", f"MLP ({chosen})",
        out_dir, eval_kind, class_order=list(res.classes_))
    return [_row("MLP", chosen, f"feature_set={chosen} (Optuna-selected)", metrics,
                 hyperparams=dict(tune.best_params),
                 feature_list=feature_sets.get(chosen),
                 per_class_df=per_class_df, confusion_df=confusion_df,
                 trials=study_db.trials_from_optuna(tune.study),
                 predictions=predictions)]


# ===========================================================================
# XGBoost (Optuna search over hyperparameters + feature set)
# ---------------------------------------------------------------------------
# Like the MLP, XGB folds the feature sets INTO its Optuna search (feature_set
# is a categorical), so the runner returns a single row whose feature_set is the
# chosen one. XGBoost handles NaN natively, but we impute upstream for
# consistency with the other tree families.
# ===========================================================================
def run_xgb(train, evaluation, eval_kind, feature_sets, out_dir, *,
            n_trials=30):
    import optuna
    from sklearn.preprocessing import LabelEncoder

    from DSSP2026.xgboost.tune import run_xgb_search
    from DSSP2026.xgboost.fit import refit_best
    from DSSP2026.xgboost.plots import save_xgb_feature_importance_png
    from DSSP2026.tuning.optuna_train import save_training_run
    from DSSP2026.tuning.optuna_parallel import save_optuna_parallel_coordinates

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, ev = D.impute_numeric(train, evaluation)
    le = LabelEncoder().fit(tr[C.TARGET])

    storage = f"sqlite:///{(out_dir / 'optuna_xgb.db').resolve()}"
    tune = run_xgb_search(
        tr, C.TARGET, feature_sets, le,
        n_trials=n_trials, n_splits=C.CV_SPLITS,
        study_name="workflow_xgb", storage=storage, random_state=C.RANDOM_STATE)
    logger.info("XGB best CV macro-F1 = %.4f", tune.best_value)

    save_optuna_parallel_coordinates(
        tune.study, out_dir / "xgb_parallel_coordinates.png")
    save_training_run(tune.study, out_dir / "xgb_training_run.png")

    res = refit_best(
        tune.study, tr, ev, C.TARGET, feature_sets,
        label_encoder=le, average=C.AVERAGE, random_state=C.RANDOM_STATE)
    save_xgb_feature_importance_png(
        res, out_dir / "xgb_importance.png", top_n=C.IMPORTANCE_TOP_N)

    chosen = tune.best_params["feature_set"]
    metrics, per_class_df, confusion_df, predictions = _save_common_eval(
        res.y_true, res.y_pred, res.y_proba, "xgb", f"XGBoost ({chosen})",
        out_dir, eval_kind, class_order=list(res.classes_))
    return [_row("XGBoost", chosen, f"feature_set={chosen} (Optuna-selected)", metrics,
                 hyperparams=dict(tune.best_params),
                 feature_list=feature_sets.get(chosen),
                 per_class_df=per_class_df, confusion_df=confusion_df,
                 trials=study_db.trials_from_optuna(tune.study),
                 predictions=predictions)]


# Registry: family name -> runner. cli.py iterates this.
RUNNERS = {
    "logistic": run_logistic,
    "tree": run_tree,
    "rf": run_rf,
    "mlp": run_mlp,
    "xgb": run_xgb,
}