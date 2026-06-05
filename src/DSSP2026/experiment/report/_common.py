from __future__ import annotations

from DSSP2026.core.style import apply_att_style

apply_att_style(base_size = 18)

_METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
_DB_METRIC = {"Accuracy": "accuracy", "Precision": "precision",
              "Recall": "recall", "F1": "f1", "ROC-AUC": "roc_auc"}
