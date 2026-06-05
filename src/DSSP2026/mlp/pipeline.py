"""
mlp/pipeline.py — sklearn pipeline construction for MLP classifiers.

MLPs are scale-sensitive, so numeric features are median-imputed and
standardised, while 0/1 flag features are passed through untouched. The scaler
is part of the ``Pipeline`` (and the preprocessor is rebuilt per CV fold in
``mlp.tune``), so validation-fold statistics never leak into training.

The split between numeric and flag columns is supplied by the caller
(``numeric_features`` / ``flag_features``) rather than hard-coded, so this module
is dataset-agnostic: any script can describe its own columns.
"""

from typing import Optional, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(
    features: Sequence[str],
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
) -> ColumnTransformer:
    """The ColumnTransformer half of the MLP pipeline.

    Numeric columns (those in ``numeric_features``) are median-imputed and
    standardised; flag columns (those in ``flag_features``) are passed through.
    Columns of ``features`` that fall in neither group are dropped.

    Separated from ``build_mlp_pipeline`` because cross-validation needs to fit
    the scaler on each fold's training rows only (see ``mlp.tune``).

    Parameters
    ----------
    features : sequence of str
        The columns this model will actually use.
    numeric_features, flag_features : sequence of str
        The full numeric / flag column vocabularies; ``features`` is intersected
        against each to decide how each used column is treated.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    num_cols = [c for c in features if c in set(numeric_features)]
    flag_cols = [c for c in features if c in set(flag_features)]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ))
    if flag_cols:
        transformers.append(("flag", "passthrough", flag_cols))

    return ColumnTransformer(transformers, remainder="drop")


def build_mlp_pipeline(
    features: Sequence[str],
    numeric_features: Sequence[str],
    flag_features: Sequence[str],
    *,
    hidden,
    activation: str = "relu",
    alpha: float = 1e-4,
    lr_init: float = 1e-3,
    max_iter: int = 400,
    early_stopping: bool = True,
    n_iter_no_change: int = 15,
    random_state: int = 42,
) -> Pipeline:
    """Full preprocessing + ``MLPClassifier`` pipeline.

    The numeric branch is imputed and standardised; flags pass through. Because
    the scaler lives inside the pipeline, fitting the pipeline fits the scaler on
    exactly the rows it is trained on — the correct behaviour for both a final
    refit and any CV that clones this estimator.

    Parameters
    ----------
    features : sequence of str
        Columns the model uses.
    numeric_features, flag_features : sequence of str
        Column vocabularies, forwarded to :func:`build_preprocessor`.
    hidden : tuple of int
        ``hidden_layer_sizes`` for the MLP (per-layer widths).
    activation : {"relu", "tanh", ...}
        MLP activation.
    alpha : float
        L2 regularisation strength.
    lr_init : float
        Initial learning rate.
    max_iter : int
        Max optimiser iterations for the final fit.
    early_stopping : bool
        Hold out an internal validation split and stop when it stops improving.
    n_iter_no_change : int
        Patience for ``early_stopping``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``[("pre", ColumnTransformer), ("mlp", MLPClassifier)]``.
    """
    pre = build_preprocessor(features, numeric_features, flag_features)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=activation,
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("mlp", mlp)])
