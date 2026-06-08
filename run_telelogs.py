"""
FLS readmission business case — logistic / random forest / MLP.

Produces every artifact the assignment asks for:
  - one comparison row per model: F1, AUC (+ accuracy/precision/recall, CV F1)
  - per-model threshold sweep (Youden's J + the cost-optimal threshold)
  - per-model business cost/benefit (argmax baseline vs cost-optimal rule)

Economics: a readmit costs $150,000; holding a patient 5 extra days costs
$20,000 and is assumed to prevent the readmit. The decision per patient is
HOLD vs DON'T HOLD, so this is a binary cost problem driven by the model's
predicted readmit probability.
"""

import pandas as pd

from DSSP2026.experiment.experiment import Experiment
from DSSP2026.report import Report
from DSSP2026.experiment.data import standardise_numeric
from sklearn.model_selection import train_test_split
import logging
from config import (
    TRAIN_FILE,
    TEST_FILE,
    OUTPUT_ROOT,
    TARGET,
    FEATURE_SETS,
    RANDOM_STATE,
    NUMERIC_FEATURES,
    SCHEMA
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

train = pd.read_parquet(TRAIN_FILE)
train, eval = train_test_split(train, test_size=0.2, random_state=RANDOM_STATE)
test = pd.read_parquet(TEST_FILE)
train, eval,test = standardise_numeric(train, eval, test, numeric_features=NUMERIC_FEATURES)
experiment = Experiment(
    train=train,
    evaluation=eval,
    test=test,
    target=TARGET,
    schema=SCHEMA,
    experiment_dir=str(OUTPUT_ROOT),
    feature_sets=FEATURE_SETS,
    n_trials=30,
    n_splits=5,
    random_state=RANDOM_STATE
)
experiment.run()

logger.info("Experiment completed. Reporting.")
report = Report(experiment.report_db)
report.compare_models(allow_ensemble=True).to_png(OUTPUT_ROOT / "compare_models_argmax.png")
report.compare_models(allow_ensemble=True, policy = "F1").to_png(OUTPUT_ROOT / "compare_models_f1.png")
report.compare_models(allow_ensemble=True, policy = "Youden's J").to_png(OUTPUT_ROOT / "compare_models_youden.png")
report.roc_compare().to_png(OUTPUT_ROOT / "roc_compare.png")


logger.info("Models: %s", report.models())

# for model in report.models():
