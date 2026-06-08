"""
tests/test_smoke_leakfree.py — regression net for the leak-free report layer.

Builds a tiny synthetic 3-way-split report.db (train/validation/test, plus the
oof_predictions and test_predictions tables the experiment now persists) and
asserts the invariants that the runner-retirement reorg must not break:

  * compare_models() reports TEST scores, one row per model (+ Ensemble),
    using cached test predictions (ZERO refits).
  * best_fit() selects on validation and reports test_metrics.
  * confusion_matrix(model, policy=...) totals the TEST sample count for
    ArgMax / F1 / Youden's J, for a real model and the Ensemble.

Run: pytest -q tests/test_smoke_leakfree.py   (or: python tests/test_smoke_leakfree.py)
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold


def _build_report_db(d: Path):
    from DSSP2026.report.report_builder import _connect_report, _insert_experiment
    from DSSP2026.tree.classification.fit import (
        fit_decision_tree_classifier, fit_random_forest_classifier)
    from DSSP2026.experiment.refit import refit_estimator

    X, y = make_classification(n_samples=1200, n_features=6, n_informative=4,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    cols = [f"f{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=cols)
    df["answer"] = [f"C{c+1}" for c in y]
    train = df.iloc[:700].reset_index(drop=True)
    val = df.iloc[700:950].reset_index(drop=True)
    test = df.iloc[950:].reset_index(drop=True)        # 250 rows
    art = d / ".artifacts"
    art.mkdir()

    def wp(fr, n):
        p = art / n
        fr.to_parquet(p, index=False)
        return ".artifacts/" + n, hashlib.sha256(open(p, "rb").read()).hexdigest()

    trel, tsha = wp(train, "train.parquet")
    xrel, xsha = wp(test, "test.parquet")
    vrel, vsha = wp(val, "validation.parquet")

    dbp = str(d / "report.db")
    conn = _connect_report(dbp)
    _insert_experiment(conn, "exp1", {
        "timestamp": "2026-06-08", "eval_kind": "validation", "target": "answer",
        "train_parquet": trel, "train_sha256": tsha,
        "test_parquet": xrel, "test_sha256": xsha,
        "validation_parquet": vrel, "validation_sha256": vsha,
        "n_splits": 4, "random_state": 42})

    class_order = sorted(train["answer"].unique())
    col_ix = {c: j for j, c in enumerate(class_order)}
    K = len(class_order)
    specs = [
        ("Decision tree", {"max_depth": 5}, fit_decision_tree_classifier, dict(max_depth=5)),
        ("Random forest", {"n_estimators": 60, "max_features": "sqrt"},
         fit_random_forest_classifier, dict(n_estimators=60, max_features="sqrt")),
    ]
    for mid, (name, hp, fitfn, kw) in enumerate(specs, 1):
        res = fitfn(train, val, cols, "answer", average="macro", random_state=42, **kw)
        co = [str(c) for c in res.classes_]
        yp = res.model.predict_proba(val[cols])
        conn.execute(
            "INSERT INTO models(model_id,experiment_id,model,feature_set,accuracy,"
            "precision,recall,f1,roc_auc,hyperparams,feature_list,n_trials,"
            "best_cv_value) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, "exp1", name, "all", 0.8, 0.8, 0.8, 0.8, 0.9,
             json.dumps(hp), json.dumps(cols), 10, 0.8))
        conn.execute(
            "INSERT INTO predictions(model_id,class_order,y_true,y_proba,"
            "n_samples,n_classes) VALUES(?,?,?,?,?,?)",
            (mid, json.dumps(co), json.dumps(val["answer"].astype(str).tolist()),
             json.dumps(yp.tolist()), len(val), len(co)))
        # OOF
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        oof = np.full((len(train), K), np.nan)
        strat = train["answer"].astype(str).to_numpy()
        for tr_i, va_i in skf.split(train, strat):
            est = refit_estimator(name, train.iloc[tr_i], target="answer",
                                  features=cols, hyperparams=hp, column_types={})
            p = est.predict_proba(train.iloc[va_i])
            al = np.zeros((len(va_i), K))
            for j, c in enumerate(est.class_order):
                if str(c) in col_ix:
                    al[:, col_ix[str(c)]] = p[:, j]
            oof[va_i] = al
        conn.execute(
            "INSERT INTO oof_predictions VALUES(?,?,?,?,?,?,?)",
            ("exp1", name, json.dumps(class_order),
             json.dumps(train["answer"].astype(str).tolist()),
             json.dumps(oof.tolist()), 4, 42))
        # TEST predictions (refit full train, predict test)
        est = refit_estimator(name, train, target="answer", features=cols,
                              hyperparams=hp, column_types={})
        pt = est.predict_proba(test)
        alt = np.zeros((len(test), K))
        for j, c in enumerate(est.class_order):
            if str(c) in col_ix:
                alt[:, col_ix[str(c)]] = pt[:, j]
        conn.execute(
            "INSERT INTO test_predictions VALUES(?,?,?,?,?,?,?)",
            ("exp1", name, json.dumps(class_order),
             json.dumps(test["answer"].astype(str).tolist()),
             json.dumps(alt.tolist()), len(test), K))
    conn.commit()
    conn.close()
    return dbp, len(test)


def run_smoke():
    from DSSP2026.report.report import Report
    import DSSP2026.report.cost.fit as CF

    d = Path(tempfile.mkdtemp())
    dbp, n_test = _build_report_db(d)
    r = Report(dbp, experiment_id="exp1")

    # 1. compare_models: one row per real model + Ensemble, finite metrics.
    tbl = r.compare_models(allow_ensemble=True)
    assert set(["Model", "Accuracy", "F1"]).issubset(tbl.df.columns)
    assert len(tbl.df) == 3, f"expected 3 rows (2 models + ensemble), got {len(tbl.df)}"
    assert tbl.df["F1"].notna().all()

    # 2. compare_models uses cached test predictions -> ZERO refits.
    calls = {"n": 0}
    orig = CF._refit_one
    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    CF._refit_one = spy
    try:
        r.compare_models(allow_ensemble=True)
    finally:
        CF._refit_one = orig
    assert calls["n"] == 0, f"compare_models refit {calls['n']} times (cache unused)"

    # 3. best_fit selects on validation, reports test.
    m = r.best_fit(allow_ensemble=True, target="F1")
    assert m.test_metrics is not None and "f1" in m.test_metrics
    assert hasattr(m, "validation_score")

    # 4. confusion_matrix totals TEST size for every policy, real + ensemble.
    for policy in ["ArgMax", "F1", "Youden's J"]:
        for model in ["Random forest", "Ensemble"]:
            wide = r._confusion_df(model, policy=policy)
            total = int(wide.values.sum())
            assert total == n_test, (
                f"{model}/{policy}: confusion total {total} != test size {n_test}")

    print("SMOKE OK — leak-free invariants hold "
          f"(test size {n_test}, {len(tbl.df)} models compared).")


def test_smoke_leakfree():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
