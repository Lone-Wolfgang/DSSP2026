"""
workflows/cli.py — one CLI for the TeleLogs root-cause comparison.

Replaces the four standalone scripts (logistic / tree / rf / mlp) with a single
entry point exposing the three workflow decisions as arguments:

  --models        which model families to train (default: all)
  --feature-sets  which feature groups to use (default: all). For logistic/tree/
                  rf each selected set is a separate row; for the MLP they fold
                  into the Optuna search as a categorical hyperparameter.
  --use-test-file evaluate on the separate test.parquet instead of carving a
                  validation slice from train (the default). The validation
                  ratio and seed live in workflows/config.py.

Every family is fit on train and scored once on the evaluation set (validation
slice or test file). One master comparison table collects every (model x
feature set) row, all scored on the identical evaluation set.

Examples
--------
    python -m DSSP2026.workflows.cli
    python -m DSSP2026.workflows.cli --models tree rf
    python -m DSSP2026.workflows.cli --feature-sets both --use-test-file
    python -m DSSP2026.workflows.cli --models mlp --n-trials 50
"""

import argparse
import logging
import os

# Cap BLAS/OpenMP threads before numpy import (macOS nested-thread deadlock guard).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from DSSP2026.core.color_scales import apply_att_seaborn
from DSSP2026.evaluation.tables import save_classification_comparison_png
from DSSP2026.workflows import config as C
from DSSP2026.workflows import data as D
from DSSP2026.workflows.runners import RUNNERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _add_run_args(p):
    """Attach the model-comparison arguments to a parser (the ``run`` command)."""
    p.add_argument(
        "--models", nargs="+", choices=list(RUNNERS), default=list(RUNNERS),
        metavar="MODEL",
        help="Model families to train (default: all = %s)." % " ".join(RUNNERS))
    p.add_argument(
        "--feature-sets", nargs="+", choices=list(C.FEATURE_SETS),
        default=list(C.FEATURE_SETS), metavar="SET",
        help="Feature sets to use (default: all = %s). Many sets -> one row each "
             "for logistic/tree/rf; folded into the Optuna search for mlp."
             % " ".join(C.FEATURE_SETS))
    p.add_argument(
        "--use-test-file", action="store_true",
        help="Evaluate on the separate test.parquet. Default: carve a validation "
             "slice from train (ratio/seed in config).")
    p.add_argument(
        "--n-trials", type=int, default=50,
        help="Optuna trials for the MLP family (default: 30).")
    p.add_argument("--train-file", default=None, help="Override config TRAIN_FILE.")
    p.add_argument("--test-file", default=None, help="Override config TEST_FILE.")
    p.add_argument("--output-root", default=None, help="Override config OUTPUT_ROOT.")
    p.add_argument("--study-db", default=None,
                   help="Override config STUDY_DB (the append-only study artifact).")
    return p


def build_parser():
    p = argparse.ArgumentParser(
        description="TeleLogs root-cause classification — unified workflow CLI.")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    run_p = sub.add_parser(
        "run", help="Train the model families and record a study run (default).")
    _add_run_args(run_p)

    dash_p = sub.add_parser(
        "dashboard",
        help="Launch the interactive cost / net-benefit comparison dashboard.")
    dash_p.add_argument(
        "--study-db", default=None,
        help="Study database to read (default: config STUDY_DB).")
    dash_p.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (default: 8501).")

    return p


def main(argv=None):
    import sys as _sys
    if argv is None:
        argv = _sys.argv[1:]
    argv = list(argv)

    # Backward compatibility: legacy invocations passed run-flags with no
    # subcommand (e.g. `... cli --models tree`). If the first token isn't a
    # known subcommand (or help), inject `run` so those keep working.
    known = {"run", "dashboard", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["run"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        return _launch_dashboard(args)

    return _run(args)


def _launch_dashboard(args):
    """Shell out to ``streamlit run`` on the dashboard module."""
    import sys
    from pathlib import Path

    from DSSP2026.workflows import config as C

    db = args.study_db or C.STUDY_DB
    dashboard_module = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_module),
           "--server.port", str(args.port)]
    env = dict(os.environ)
    env["DSSP_DASHBOARD_DB"] = str(db)
    logger.info("Launching dashboard on http://localhost:%d (study-db: %s)",
                args.port, db)
    try:
        import subprocess
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        logger.error("Streamlit is not installed. Install it with: "
                     "pip install streamlit altair")
        return 1


def _run(args):
    apply_att_seaborn()

    output_root = args.output_root or C.OUTPUT_ROOT
    from pathlib import Path
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve the selected feature sets (preserve config order).
    feature_sets = {k: C.FEATURE_SETS[k] for k in C.FEATURE_SETS
                    if k in set(args.feature_sets)}

    # --- Load + split (decision #3) ----------------------------------------
    train, evaluation, eval_kind = D.load_train_eval(
        use_test_file=args.use_test_file,
        train_file=args.train_file, test_file=args.test_file)
    logger.info("Evaluating on the %s set. Models=%s  Feature sets=%s",
                eval_kind, args.models, list(feature_sets))

    # --- Run each selected family ------------------------------------------
    all_rows = []
    for name in args.models:
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 64)
        logger.info("Running family: %s", name)
        logger.info("=" * 64)
        try:
            kw = {"n_trials": args.n_trials} if name == "mlp" else {}
            rows = RUNNERS[name](
                train, evaluation, eval_kind, feature_sets, out_dir, **kw)
            all_rows.extend(rows)
            for r in rows:
                logger.info("%s [%s] metrics: %s", r["model"], r["feature_set"],
                            {k: round(v, 4) for k, v in r["metrics"].items()})
        except Exception as e:
            # One family failing shouldn't sink the rest of the comparison.
            logger.exception("Family '%s' failed: %s", name, e)

    if not all_rows:
        logger.error("No families produced results.")
        return None

    # --- Master comparison: one row per model type --------------------------
    # Every feature set was still fit and its artifacts saved in the model's
    # folder; here we collapse to the best feature set per model (by F1) so the
    # master table has a single row per model type.
    metric_keys = [k for k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
                   if k in all_rows[0]["metrics"]]

    best_per_model = {}
    for r in all_rows:
        cur = best_per_model.get(r["model"])
        if cur is None or r["metrics"]["F1"] > cur["metrics"]["F1"]:
            best_per_model[r["model"]] = r

    # Flag the rows that win their model (these become the master-table rows).
    # best_per_model holds references into all_rows, so this marks the same dicts.
    for r in all_rows:
        r["is_best_for_model"] = False
    for r in best_per_model.values():
        r["is_best_for_model"] = True

    master = pd.DataFrame([
        {"Model": r["model"], "Feature set": r["feature_set"], "Config": r["detail"],
         **{k: r["metrics"][k] for k in metric_keys}}
        for r in best_per_model.values()
    ]).sort_values("F1", ascending=False).reset_index(drop=True)

    logger.info("\n=== MASTER COMPARISON (all on identical %s set) ===\n%s",
                eval_kind, master.to_string(index=False))

    # The comparison saver formats every non-Model column as a float, so keep
    # the string columns (Feature set, Config) out of the PNG; they stay in CSV.
    master_numeric = master.drop(columns=["Feature set", "Config"])
    save_classification_comparison_png(
        master_numeric, output_root / "master_comparison.png", best_by="F1")
    master.to_csv(output_root / "master_comparison.csv", index=False)
    logger.info("Done. Master outputs in %s", output_root)

    # --- Persist the study artifact (append-only SQLite) -------------------
    # Non-fatal: the PNGs/CSV are the immediate deliverable; a DB write failure
    # is logged but doesn't sink the run.
    try:
        from datetime import datetime, timezone
        from DSSP2026.workflows import study_db

        study_db_path = Path(args.study_db or C.STUDY_DB)
        run_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_kind": eval_kind,
            "models": args.models,
            "feature_sets": list(feature_sets),
            "n_trials": args.n_trials,
            "train_file": args.train_file or C.TRAIN_FILE,
            "test_file": (args.test_file or C.TEST_FILE) if args.use_test_file else None,
            "random_state": C.RANDOM_STATE,
            "cv_splits": C.CV_SPLITS,
            "average": C.AVERAGE,
            "scoring": C.SCORING,
            "validation_ratio": None if args.use_test_file else C.VALIDATION_RATIO,
            "cli_args": vars(args),
        }
        run_id = study_db.record_run(study_db_path, run_meta, all_rows)
        logger.info("Recorded study run #%d in %s", run_id, study_db_path)
    except Exception as e:
        logger.warning("Study DB write skipped (run otherwise succeeded): %s", e)

    return master


if __name__ == "__main__":
    main()