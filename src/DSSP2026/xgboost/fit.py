"""
xgboost/fit.py — fit an XGBoost classifier and score it on held-out data.

Mirrors ``mlp.fit`` / ``tree.classification.fit``: a result dataclass carrying
the model, test metrics, and test predictions/probabilities, ready to flow
through the shared evaluators (``core.metrics`` + ``core.heatmap`` +
``evaluation.tables``).

XGBoost trains on integer class labels, so a ``LabelEncoder`` is used to encode
the target for fitting and decode predictions back to the original labels — the
same pattern as the MLP. ``XGBClassifier`` exposes ``.feature_importances_``, so
the shared ``tree._shared`` importance helpers work unchanged on an XGBResult's
model.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from DSSP2026.core.metrics import classification_metrics
from DSSP2026.core.results import ClassificationResult


@dataclass
class XGBResult(ClassificationResult):
    """Container returned by fit_xgb_classifier.

    Shared fields are defined by ``core.results.ClassificationResult``.

    Attributes
    ----------
    model : xgboost.XGBClassifier
        The fitted classifier (fit on integer-encoded labels).
    metrics : dict
        Test-set metrics from ``classification_metrics``.
    features : list
        Feature columns the model used.
    target : str
        Target column name.
    classes_ : list
        Class labels in the model's native order (string labels, decoded).
    y_true, y_pred : np.ndarray
        Test labels and predictions in the original label space.
    y_proba : np.ndarray, optional
        Test class-probability matrix (columns follow ``classes_``).
    best_params : dict, optional
        The Optuna best params this model was rebuilt from, when applicable.
    """
    best_params: Optional[dict] = None


def _make_classifier(*, n_estimators, max_depth, learning_rate, subsample,
                     colsample_bytree, reg_lambda, reg_alpha, min_child_weight,
                     num_class, random_state):
    """Construct an XGBClassifier. Local import so the package is importable
    even if xgboost isn't installed until first use.

    The objective is left unset so XGBoost picks it from the class count:
    ``binary:logistic`` for 2 classes, ``multi:softprob`` for >2. Forcing
    ``multi:softprob`` would make ``predict`` return a 2-column array on a binary
    target and break label decoding. ``num_class`` is accepted for signature
    compatibility but intentionally not passed (XGBoost infers it; passing it
    alongside a binary objective conflicts).
    """
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )


def fit_xgb_classifier(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
    label_encoder=None,
    average: str = "macro",
    random_state: int = 42,
    best_params: Optional[dict] = None,
) -> XGBResult:
    """Fit an XGBoost classifier on Train and score it on Test.

    Trees are scale-invariant, so no scaling is applied. XGBoost handles NaN
    natively (it learns a default split direction), so numeric imputation is
    optional — but the workflow imputes upstream for consistency with the other
    tree families.

    Parameters
    ----------
    features : sequence of str
        Columns to train on.
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    reg_lambda, reg_alpha, min_child_weight : XGBoost hyperparameters.
    label_encoder : fitted LabelEncoder, optional
        If given, used to encode/decode the target; otherwise one is fit on
        ``Train[target]`` internally.
    average : str
        Averaging for precision/recall/F1, passed to ``classification_metrics``.

    Returns
    -------
    XGBResult
    """
    from sklearn.preprocessing import LabelEncoder

    features = list(features)
    le = label_encoder if label_encoder is not None else LabelEncoder().fit(Train[target])

    y_train = le.transform(Train[target])
    clf = _make_classifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        min_child_weight=min_child_weight, num_class=len(le.classes_),
        random_state=random_state)
    clf.fit(Train[features], y_train)

    y_true = Test[target].to_numpy()
    y_pred = le.inverse_transform(clf.predict(Test[features]))
    y_proba = clf.predict_proba(Test[features])              # cols follow encoded order
    classes_ = list(le.inverse_transform(np.arange(len(le.classes_))))

    metrics = classification_metrics(y_true, y_pred, y_score=y_proba, average=average)

    return XGBResult(
        model=clf, metrics=metrics, features=features, target=target,
        classes_=classes_, y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        best_params=best_params)


def refit_best(
    study,
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    target: str,
    feature_sets,
    *,
    label_encoder=None,
    average: str = "macro",
    random_state: int = 42,
) -> XGBResult:
    """Rebuild the best configuration from a study and refit + score it.

    Reads ``study.best_params``, resolves the chosen ``feature_set`` against
    ``feature_sets``, and calls :func:`fit_xgb_classifier` with the searched
    hyperparameters.

    Parameters
    ----------
    study : optuna.Study (or anything with ``best_params``)
        The completed search.
    feature_sets : mapping of str -> sequence of str
        Same mapping the search used; the best ``feature_set`` keys into it.

    Returns
    -------
    XGBResult
    """
    bp = dict(study.best_params)
    features = list(feature_sets[bp["feature_set"]])

    return fit_xgb_classifier(
        Train, Test, features, target,
        n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
        learning_rate=bp["learning_rate"], subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"], reg_lambda=bp["reg_lambda"],
        reg_alpha=bp["reg_alpha"], min_child_weight=bp["min_child_weight"],
        label_encoder=label_encoder, average=average, random_state=random_state,
        best_params=bp)