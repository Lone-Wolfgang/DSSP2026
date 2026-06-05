"""
mlp/fit.py — fit an MLP classifier and score it on held-out test data.

Mirrors ``tree.classification.fit``: a result dataclass carrying the model, test
metrics, and test predictions/probabilities, ready to flow through the shared
evaluators (``core.metrics.make_confusion_matrix`` /
``make_classification_report_df``, ``core.heatmap.save_confusion_matrix_png``,
``evaluation.tables.save_classification_report_png``).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from DSSP2026.core.metrics import classification_metrics
from DSSP2026.core.results import ClassificationResult
from DSSP2026.mlp.pipeline import build_mlp_pipeline


@dataclass
class MLPResult(ClassificationResult):
    """Container returned by fit_mlp_classifier.

    Shared fields are defined by ``core.results.ClassificationResult``.

    Attributes
    ----------
    model : object
        The fitted sklearn Pipeline (preprocessor + MLPClassifier).
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


def fit_mlp_classifier(
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
    *,
    hidden,
    activation: str = "relu",
    alpha: float = 1e-4,
    lr_init: float = 1e-3,
    label_encoder=None,
    average: str = "macro",
    random_state: int = 42,
    best_params: Optional[dict] = None,
) -> MLPResult:
    """Fit an MLP pipeline on Train and score it on Test.

    Numeric features are imputed + standardised and flags pass through (see
    ``mlp.pipeline``). If a ``label_encoder`` is supplied the target is encoded
    to integers for fitting and predictions are decoded back to the original
    labels, so ``y_true`` / ``y_pred`` / ``classes_`` are all in label space.

    Parameters
    ----------
    features : sequence of str
        Columns to train on.
    numeric_features, flag_features : sequence of str
        Column vocabularies, forwarded to the pipeline.
    hidden : tuple of int
        Per-layer widths (e.g. the best layout from the Optuna search).
    label_encoder : fitted LabelEncoder, optional
        If given, used to encode/decode the target; otherwise the MLP is fit on
        the raw target labels directly.
    average : str
        Averaging for precision/recall/F1, passed to
        ``core.metrics.classification_metrics``. Default "macro".

    Returns
    -------
    MLPResult
    """
    features = list(features)
    pipe = build_mlp_pipeline(
        features, numeric_features, flag_features,
        hidden=hidden, activation=activation, alpha=alpha, lr_init=lr_init,
        random_state=random_state)

    if label_encoder is not None:
        pipe.fit(Train[features], label_encoder.transform(Train[target]))
        y_true = Test[target].to_numpy()
        y_pred = label_encoder.inverse_transform(pipe.predict(Test[features]))
        y_proba = pipe.predict_proba(Test[features])     # cols follow encoded order
        classes_ = list(label_encoder.inverse_transform(pipe.classes_))
    else:
        pipe.fit(Train[features], Train[target])
        y_true = Test[target].to_numpy()
        y_pred = pipe.predict(Test[features])
        y_proba = pipe.predict_proba(Test[features])
        classes_ = list(pipe.classes_)

    metrics = classification_metrics(
        y_true, y_pred, y_score=y_proba, average=average)

    return MLPResult(
        model=pipe, metrics=metrics, features=features, target=target,
        classes_=classes_, y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        best_params=best_params)


def refit_best(
    study,
    Train: pd.DataFrame,
    Test: pd.DataFrame,
    target: str,
    feature_sets,
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
    *,
    label_encoder=None,
    average: str = "macro",
    random_state: int = 42,
) -> MLPResult:
    """Rebuild the best configuration from a study and refit + score it.

    Reads ``study.best_params``, reconstructs the layer tuple from the per-layer
    ``width1..widthN`` parameters, resolves the chosen ``feature_set`` against
    ``feature_sets``, and calls :func:`fit_mlp_classifier`.

    Parameters
    ----------
    study : optuna.Study (or anything with ``best_params``)
        The completed search.
    feature_sets : mapping of str -> sequence of str
        Same mapping the search used; the best ``feature_set`` keys into it.

    Returns
    -------
    MLPResult
    """
    bp = dict(study.best_params)
    feature_set = bp["feature_set"]
    features = list(feature_sets[feature_set])
    hidden = tuple(bp[f"width{i}"] for i in range(1, bp["n_layers"] + 1))

    return fit_mlp_classifier(
        Train, Test, features, target, numeric_features, flag_features,
        hidden=hidden, activation=bp["activation"], alpha=bp["alpha"],
        lr_init=bp["lr_init"], label_encoder=label_encoder, average=average,
        random_state=random_state, best_params=bp)