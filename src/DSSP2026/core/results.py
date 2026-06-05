from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ClassificationResult:
    """Shared container for a fitted classifier scored on held-out data."""
    model: object
    metrics: dict
    features: list
    target: str
    classes_: list
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_proba: Optional[np.ndarray] = field(default=None, repr=False)
