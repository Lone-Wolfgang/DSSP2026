import pytest
import pandas as pd

from DSSP2026.experiment.experiment import Experiment


def _frame():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "flag": [0, 1, 0, 1],
        "target": ["a", "b", "a", "b"],
    })


def _experiment(**kwargs):
    frame = _frame()
    params = dict(
        train=frame.copy(),
        evaluation=frame.copy(),
        test=frame.copy(),
        target="target",
        experiment_dir="/tmp/dssp-schema-guards",
        feature_sets={"all": ["x", "flag"]},
        schema={"x": "numeric", "flag": "categorical"},
    )
    params.update(kwargs)
    return Experiment(**params)


def test_no_schema_raises():
    with pytest.raises(ValueError, match="schema is required"):
        _experiment(schema=None)


def test_schema_missing_feature_column_raises_with_column_name():
    with pytest.raises(ValueError, match="flag"):
        _experiment(schema={"x": "numeric"})


def test_mismatched_train_test_columns_raises():
    test = _frame().drop(columns=["flag"])
    with pytest.raises(ValueError, match="identical column sets"):
        _experiment(test=test)


def test_well_formed_case_succeeds():
    exp = _experiment()
    assert exp._numeric_features == ["x"]
    assert "flag" in exp._flag_features
