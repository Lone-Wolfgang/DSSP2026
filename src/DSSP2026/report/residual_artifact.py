"""
report/residual_artifact.py — persist and load the OLS residual-diagnostics frame.

A small, self-contained sidecar for *regression* diagnostics, kept deliberately
separate from the classification report pipeline (report.db stores y_proba /
confusion / F1; none of that applies to OLS). The contract is one parquet file
plus one provenance row, following the exact conventions already used for
``train.parquet`` in :mod:`DSSP2026.experiment.experiment`:

- the frame is written to ``.artifacts/`` and hashed with SHA-256;
- the recorded path is *relative to report.db's own directory*, so the whole
  output folder moves as a unit;
- the reader verifies the hash and returns ``None`` on any mismatch / absence,
  so the dashboard degrades gracefully instead of raising.

The frame itself is whatever ``linear_regression.fit.make_ols_diagnostics_df``
produces (columns ``fitted, residuals, stud_resid, cooks, leverage, obs`` plus
the ``in_danger_zone`` / ``bubble_size`` helpers). The renderer
``evaluation.residuals.plot_residual_diagnostics`` consumes the first five.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Standard sidecar locations, consistent with experiment.experiment.
ARTIFACT_REL = ".artifacts/diagnostics.parquet"
TABLE = "residual_diagnostics"

# The columns plot_residual_diagnostics requires; we assert these are present
# at write time so a bad frame fails loudly during the run, not silently in the UI.
_REQUIRED = ("fitted", "residuals", "stud_resid", "cooks", "obs")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} ("
        "  experiment_id TEXT,"
        "  label         TEXT,"        # which OLS model/formula this frame is for
        "  formula       TEXT,"        # the formula string, for display
        "  n_obs         INTEGER,"     # nobs used for the Cook's 4/n line
        "  parquet_path  TEXT,"        # relative to report.db's directory
        "  sha256        TEXT,"
        "  PRIMARY KEY (experiment_id, label))")


def write_residual_diagnostics(
    report_db,
    diag_df: pd.DataFrame,
    *,
    experiment_id: str,
    label: str = "OLS",
    formula: Optional[str] = None,
    n_obs: Optional[int] = None,
) -> str:
    """Persist *diag_df* next to *report_db* and record its provenance.

    Parameters
    ----------
    report_db : path-like
        Path to report.db. The parquet is written to ``.artifacts/`` *beside*
        it, and the recorded path is relative to its directory.
    diag_df : DataFrame
        As produced by ``linear_regression.fit.make_ols_diagnostics_df``.
    experiment_id, label : str
        Identify the frame. ``label`` lets several OLS specs (e.g. competing
        formulas) coexist under one experiment; defaults to ``"OLS"``.
    formula : str, optional
        The formula string, stored for display in the dashboard.
    n_obs : int, optional
        Observation count for the Cook's-distance ``4/n`` threshold line.
        Defaults to ``len(diag_df)``; persisted so the reader needn't guess.

    Returns
    -------
    str
        The relative parquet path recorded in report.db.
    """
    missing = set(_REQUIRED) - set(diag_df.columns)
    if missing:
        raise ValueError(
            f"diag_df is missing columns {sorted(missing)}; expected at least "
            f"{sorted(_REQUIRED)} (use make_ols_diagnostics_df).")

    report_db = Path(report_db)
    full = report_db.parent / ARTIFACT_REL
    full.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(full, index=False)
    digest = _sha256(full)

    conn = sqlite3.connect(report_db)
    try:
        _ensure_table(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE} "
            "(experiment_id, label, formula, n_obs, parquet_path, sha256) "
            "VALUES (?,?,?,?,?,?)",
            (experiment_id, label, formula,
             int(n_obs) if n_obs is not None else int(len(diag_df)),
             ARTIFACT_REL, digest))
        conn.commit()
    finally:
        conn.close()
    return ARTIFACT_REL


def list_residual_diagnostics(report_db, *, experiment_id: str) -> pd.DataFrame:
    """Provenance rows for an experiment (label, formula, n_obs). May be empty."""
    conn = sqlite3.connect(report_db)
    try:
        if not _has_table(conn):
            return pd.DataFrame(columns=["label", "formula", "n_obs"])
        return pd.read_sql_query(
            f"SELECT label, formula, n_obs FROM {TABLE} "
            "WHERE experiment_id=? ORDER BY label",
            conn, params=(experiment_id,))
    finally:
        conn.close()


def load_residual_diagnostics(
    report_db,
    *,
    experiment_id: str,
    label: str = "OLS",
    verify_hash: bool = True,
):
    """Load a persisted diagnostics frame, or ``None`` if unavailable.

    Returns ``(diag_df, n_obs)`` on success. Returns ``None`` when the table /
    row / parquet is missing, or (when ``verify_hash``) the content hash drifts
    from what was written — matching ``ReportBase.load_train_data``'s
    fail-soft contract so the dashboard can fall back cleanly.
    """
    report_db = Path(report_db)
    conn = sqlite3.connect(report_db)
    try:
        if not _has_table(conn):
            return None
        row = conn.execute(
            f"SELECT parquet_path, sha256, n_obs FROM {TABLE} "
            "WHERE experiment_id=? AND label=?",
            (experiment_id, label)).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return None
    rel_path, expected_hash, n_obs = row

    full = report_db.parent / rel_path
    if not full.exists():
        return None
    if verify_hash and expected_hash and _sha256(full) != expected_hash:
        return None
    try:
        diag_df = pd.read_parquet(full)
    except Exception:
        return None
    return diag_df, int(n_obs) if n_obs is not None else len(diag_df)


def _has_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (TABLE,)).fetchone()
    return row is not None
