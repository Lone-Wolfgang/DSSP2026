"""
mlp/plots.py — MLP-specific plots.

Intentionally minimal. The plots an MLP run needs are already provided by shared
modules — point at those rather than duplicating them:

- Training loss curves (aggregate over top-N trials):
    ``DSSP2026.tuning.optuna_train.save_training_run(study, path)``
- Parallel-coordinates over the searched hyperparameters:
    ``DSSP2026.tuning.optuna_parallel.save_optuna_parallel_coordinates(study, path)``
- Held-out confusion matrix:
    ``DSSP2026.core.heatmap.save_confusion_matrix_png(cm, path, ...)``
- Per-class classification report:
    ``DSSP2026.evaluation.tables.save_classification_report_png(report_df, path)``

Add a function here only for a genuinely MLP-specific view that none of the
above cover — for example, the loss curve of the *single best refit* model
(distinct from the across-trials aggregate in ``tuning.optuna_train``). Until
such a need arises, this module is deliberately empty of functions.
"""
