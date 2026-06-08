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
from DSSP2026.core.color_scales import apply_att_seaborn
from DSSP2026.experiment.columns import numeric_and_flag, resolve_column_types
from DSSP2026.experiment.experiment import ALL_MODELS, Experiment
from DSSP2026.experiment import data as D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLI_MODEL_ALIASES = {
    "logistic": "Logistic regression",
    "tree": "Decision tree",
    "rf": "Random forest",
    "mlp": "MLP",
    "xgb": "XGBoost",
}
CLI_MODEL_CHOICES = list(CLI_MODEL_ALIASES) + list(ALL_MODELS)
REQUIRED_CONFIG_KEYS = ("target", "data", "schema")


def load_config(path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    from pathlib import Path
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in cfg]
    data = cfg.get("data") or {}
    for key in ("train_file", "test_file"):
        if key not in data:
            missing.append(f"data.{key}")
    if missing:
        raise ValueError(f"config missing required key(s): {missing}.")
    if not isinstance(cfg["schema"], dict) or not cfg["schema"]:
        raise ValueError("config key 'schema' must be a non-empty mapping.")
    return cfg


def _add_run_args(p):
    """Attach the model-comparison arguments to a parser (the ``run`` command)."""
    p.add_argument("--config", required=True, help="YAML run configuration file.")
    p.add_argument(
        "--models", nargs="+", choices=CLI_MODEL_CHOICES,
        default=list(CLI_MODEL_ALIASES),
        metavar="MODEL",
        help="Model families to train (default: all = %s)."
             % " ".join(CLI_MODEL_ALIASES))
    p.add_argument(
        "--feature-sets", nargs="+", default=None, metavar="SET",
        help="Feature sets to use (default: all = %s). Many sets -> one row each "
             "for logistic/tree/rf; folded into the Optuna search for mlp."
             % "configured feature_sets")
    p.add_argument(
        "--n-trials", type=int, default=None,
        help="Optuna trials for the MLP family.")
    p.add_argument(
        "--n-splits", type=int, default=None,
        help="CV folds for Optuna studies.")
    p.add_argument("--train-file", default=None, help="Override config data.train_file.")
    p.add_argument("--test-file", default=None, help="Override config data.test_file.")
    p.add_argument("--output-root", default=None, help="Override config output_root.")
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
        "--config", default=None, help="Optional YAML config file.")
    dash_p.add_argument(
        "--report-db", default=None,
        help="report.db to read. If omitted with --config, uses output_root/report.db.")
    dash_p.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (default: 8501).")

    resid_p = sub.add_parser(
        "residuals",
        help="Launch the live OLS residual-diagnostics dashboard (fit from a formula).")
    resid_p.add_argument(
        "--config", default=None, help="Optional YAML config file.")
    resid_p.add_argument(
        "--path", default=None,
        help="Raw dataset to fit against (.csv or .parquet). "
             "Type the OLS formula in the dashboard.")
    resid_p.add_argument(
        "--port", type=int, default=8502,
        help="Streamlit port (default: 8502, to coexist with the cost dashboard).")

    # Discoverability commands (read-only; print and exit).
    sub.add_parser(
        "describe-models",
        help="Print the tunable hyperparameters / samplers for each model family.")

    dr_p = sub.add_parser(
        "describe-report",
        help="Print which analyses are available for a given report.db.")
    dr_p.add_argument(
        "--report-db", default=None,
        help="report.db to inspect.")
    dr_p.add_argument(
        "--config", default=None,
        help="Optional YAML config (uses output_root/report.db if --report-db omitted).")

    return p


def main(argv=None):
    import sys as _sys
    if argv is None:
        argv = _sys.argv[1:]
    argv = list(argv)

    # Backward compatibility: legacy invocations passed run-flags with no
    # subcommand (e.g. `... cli --models tree`). If the first token isn't a
    # known subcommand (or help), inject `run` so those keep working.
    known = {"run", "dashboard", "residuals", "describe-models",
             "describe-report", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["run"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        return _launch_dashboard(args)

    if args.command == "residuals":
        return _launch_residuals(args)

    if args.command == "describe-models":
        return _describe_models(args)

    if args.command == "describe-report":
        return _describe_report(args)

    return _run(args)


def _describe_models(args):
    """Print the per-model tunable surface (spaces.describe())."""
    from DSSP2026.experiment import spaces
    if not hasattr(spaces, "describe"):
        print("Tunable-surface introspection (spaces.describe) is not available "
              "in this build. Model families: "
              + ", ".join(ALL_MODELS) + ".")
        return 0
    print(spaces.describe())
    return 0


def _describe_report(args):
    """Print available analyses for a report.db (Report.describe())."""
    from pathlib import Path
    from DSSP2026.report import Report
    db = args.report_db
    if db is None and args.config:
        cfg = load_config(args.config)
        if cfg.get("output_root"):
            db = str(Path(cfg["output_root"]) / "report.db")
    if not db:
        raise ValueError("provide --report-db or --config with output_root.")
    print(repr(Report(db).describe()))
    return 0


def _launch_dashboard(args):
    """Shell out to ``streamlit run`` on the dashboard module."""
    import sys
    from pathlib import Path
    db = args.report_db
    if db is None and args.config:
        cfg = load_config(args.config)
        if cfg.get("output_root"):
            db = str(Path(cfg["output_root"]) / "report.db")
    if db is None:
        raise ValueError("provide --report-db or --config with output_root.")
    dashboard_module = Path(__file__).parent / "dashboards" / "cost_analysis.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_module),
           "--server.port", str(args.port)]
    env = dict(os.environ)
    env["DSSP_DASHBOARD_DB"] = str(db)
    logger.info("Launching dashboard on http://localhost:%d (report-db: %s)",
                args.port, db)
    try:
        import subprocess
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        logger.error("Streamlit is not installed. Install it with: "
                     "pip install streamlit altair")
        return 1


def _launch_residuals(args):
    """Shell out to ``streamlit run`` on the standalone residuals dashboard."""
    import sys
    from pathlib import Path
    cfg = load_config(args.config) if args.config else {}
    path = args.path or (cfg.get("data") or {}).get("train_file") or ""
    module = Path(__file__).parent / "dashboards" / "residuals.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(module),
           "--server.port", str(args.port)]
    env = dict(os.environ)
    env["DSSP_RESIDUALS_PATH"] = str(path)
    logger.info("Launching residuals dashboard on http://localhost:%d%s",
                args.port, f" (data: {path})" if path else "")
    try:
        import subprocess
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        logger.error("Streamlit is not installed. Install it with: "
                     "pip install streamlit altair")
        return 1


def _run(args):
    apply_att_seaborn()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    target = cfg["target"]
    schema = cfg["schema"]
    feature_sets_cfg = cfg.get("feature_sets") or {"all": list(schema)}
    selected_feature_set_names = args.feature_sets or list(feature_sets_cfg)
    missing_sets = [name for name in selected_feature_set_names if name not in feature_sets_cfg]
    if missing_sets:
        raise ValueError(f"feature set(s) not present in config: {missing_sets}.")
    feature_sets = {k: feature_sets_cfg[k] for k in feature_sets_cfg
                    if k in set(selected_feature_set_names)}
    referenced = sorted({c for cols in feature_sets.values() for c in cols})
    import pandas as pd
    column_types = resolve_column_types(
        pd.read_parquet(args.train_file or data_cfg["train_file"]),
        referenced,
        schema)
    numeric_features, flag_features = numeric_and_flag(column_types)

    output_root = args.output_root or cfg.get("output_root") or "outputs"
    from pathlib import Path
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = [CLI_MODEL_ALIASES.get(m, m) for m in args.models]
    random_state = cfg.get("random_state", 42)
    train_file = args.train_file or data_cfg["train_file"]
    test_file = args.test_file or data_cfg["test_file"]
    train, validation, test = D.load_train_val_test(
        train_file=train_file,
        test_file=test_file,
        validation_ratio=data_cfg.get("validation_ratio", 0.30),
        seed=random_state,
        target=target,
        numeric_features=numeric_features,
        flag_features=flag_features)
    logger.info("Running Experiment. Models=%s  Feature sets=%s",
                selected, list(feature_sets))

    exp = Experiment(
        train=train, evaluation=validation, test=test,
        target=target, schema=schema, experiment_dir=str(output_root),
        feature_sets=feature_sets, models=selected,
        n_trials=args.n_trials if args.n_trials is not None else cfg.get("n_trials", 50),
        n_splits=args.n_splits if args.n_splits is not None else cfg.get("cv_splits", 5),
        scoring=cfg.get("scoring", "f1_macro"),
        random_state=random_state,
        dataset_paths={"train": str(train_file), "test": str(test_file)})
    exp.run(build_report=True)
    logger.info("Done. report.db written in %s", output_root)
    return exp


if __name__ == "__main__":
    main()