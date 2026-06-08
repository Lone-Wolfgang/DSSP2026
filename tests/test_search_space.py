import json
import sqlite3

import pandas as pd
import pytest

from DSSP2026.experiment import spaces
from DSSP2026.experiment.experiment import Experiment


def test_describe_lists_models_and_params():
    text = spaces.describe()
    assert text
    for model in spaces.VALID_MODELS:
        assert model in text
    xgb = spaces.describe("XGBoost")
    assert "learning_rate" in xgb
    assert "n_estimators" in xgb


def test_tunable_params_decision_tree():
    params = spaces.tunable_params("Decision tree")
    assert params == [{"name": "max_depth", "type": "int", "low": 1, "high": 19}]


def test_grid_compatible_defaults():
    assert spaces.grid_compatible("Random forest") is True
    assert spaces.grid_compatible("XGBoost") is False


def test_normalize_search_space_defaults():
    normalized = spaces.normalize_search_space(None)
    assert normalized["Random forest"]["sampler"] == "grid"
    assert normalized["XGBoost"]["sampler"] == "tpe"
    assert normalized["Random forest"]["params"] == spaces.default_space("Random forest")


def test_xgboost_grid_rejects_continuous_default():
    with pytest.raises(ValueError, match="XGBoost grid requested.*learning_rate"):
        spaces.normalize_search_space({"XGBoost": {"sampler": "grid"}})


def test_unknown_descriptor_name_rejected():
    with pytest.raises(ValueError, match="bogus"):
        spaces.normalize_search_space({
            "Decision tree": [{"name": "bogus", "type": "int", "low": 1, "high": 2}]
        })


def test_experiment_records_normalized_search_space(tmp_path):
    df = pd.DataFrame({
        "x1": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3],
        "x2": [1.0, 1.1, 1.2, 1.3, 0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3],
        "answer": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
    })
    search_space = {
        "Decision tree": {
            "params": [{"name": "max_depth", "type": "int", "low": 1, "high": 2}],
            "sampler": "grid",
        },
        "Random forest": {"sampler": "grid"},
    }
    exp = Experiment(
        train=df.copy(),
        evaluation=df.copy(),
        test=df.copy(),
        target="answer",
        experiment_dir=str(tmp_path),
        feature_sets={"all": ["x1", "x2"]},
        schema={"x1": "numeric", "x2": "numeric"},
        search_space=search_space,
        models=["Decision tree"],
        n_trials=2,
        n_splits=3,
        random_state=0,
    )
    exp.run(verbose=False, build_report=True)

    conn = sqlite3.connect(tmp_path / "report.db")
    try:
        raw = conn.execute("SELECT search_space FROM experiments WHERE experiment_id=?", (exp.experiment_id,)).fetchone()[0]
    finally:
        conn.close()
    recorded = json.loads(raw)
    assert recorded["Decision tree"]["sampler"] == "grid"
    assert recorded["Decision tree"]["params"] == search_space["Decision tree"]["params"]
    assert recorded["Random forest"]["sampler"] == "grid"
    assert recorded["XGBoost"]["sampler"] == "tpe"
