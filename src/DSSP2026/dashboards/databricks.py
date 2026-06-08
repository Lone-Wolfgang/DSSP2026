"""
workflows/databricks.py — launch the cost dashboard from a Databricks notebook.

The dashboard itself (``workflows/dashboard.py``) is unchanged: it reads a study
DB path from the ``DSSP_DASHBOARD_DB`` env var and renders the cost / net-benefit
comparison. The only thing Databricks needs that a local Jupyter doesn't is a way
to *reach* the Streamlit server, because the cluster driver sits behind the
Databricks web proxy — ``http://localhost:8501`` is not reachable from the
browser. Databricks exposes driver ports through a proxy URL of the form::

    https://<workspace-host>/driver-proxy/o/<org-id>/<cluster-id>/<port>/

For that proxy path to serve Streamlit's assets correctly, the server must be
told it lives under that base path (``--server.baseUrlPath``) and must bind to
all interfaces (``--server.address 0.0.0.0``). This module wires both up and
returns the clickable proxy URL.

Scope: this targets a **classic single-user (dedicated) cluster**, where the
driver proxy is reachable and the driver may spawn subprocesses. Shared /
no-isolation and serverless compute generally block the proxy and/or subprocess
spawning; for those, deploy the dashboard as a Databricks App instead (see
``deploy_as_app`` notes at the bottom of this file).

Typical notebook use::

    from DSSP2026.dashboards.databricks import launch_dashboard

    url = launch_dashboard(report_db="/dbfs/FileStore/dssp/report.db")
    # -> prints and returns the proxy URL; click it.

To stop it later::

    from DSSP2026.dashboards.databricks import stop_dashboard
    stop_dashboard()
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Module-level handle so a second launch in the same notebook session can
# replace the first instead of leaking an orphaned server on the port.
_PROC: Optional[subprocess.Popen] = None
_LAST_URL: Optional[str] = None


# ---------------------------------------------------------------------------
# Databricks context discovery
# ---------------------------------------------------------------------------
def _dbutils():
    """Return the notebook's dbutils, or None if not on Databricks.

    On a Databricks notebook ``dbutils`` is injected into the user namespace but
    is not importable as a normal module. We reach it via the active Spark/REPL
    context. Returns None off-cluster so the function can be imported anywhere.
    """
    # 1) Already in the caller's globals (most notebooks).
    try:
        import builtins
        if hasattr(builtins, "dbutils"):
            return builtins.dbutils
    except Exception:
        pass
    # 2) Construct from the Spark session (works in library code too).
    try:
        from pyspark.sql import SparkSession
        from pyspark.dbutils import DBUtils  # type: ignore
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return DBUtils(spark)
    except Exception:
        pass
    return None


def _context_tags():
    """Pull (host, org_id, cluster_id) from the notebook context.

    These come from the Databricks notebook context tag bag. Tag names have been
    stable across recent runtimes but are read defensively: a missing tag yields
    None and the caller raises a clear error rather than building a broken URL.
    """
    db = _dbutils()
    if db is None:
        raise RuntimeError(
            "Not running on a Databricks cluster (dbutils unavailable). "
            "Use the plain `python -m DSSP2026.workflows.cli dashboard` path "
            "for a local Jupyter instead.")

    ctx = db.notebook.entry_point.getDbutils().notebook().getContext()

    def _opt(java_opt):
        # Databricks returns Scala Options; .get() throws if empty, so guard.
        try:
            return java_opt.get() if java_opt.isDefined() else None
        except Exception:
            try:
                return java_opt.get()
            except Exception:
                return None

    host = _opt(ctx.browserHostName())
    org_id = _opt(ctx.tags().get("orgId"))
    cluster_id = _opt(ctx.clusterId())

    missing = [n for n, v in
               (("browserHostName", host), ("orgId", org_id),
                ("clusterId", cluster_id)) if not v]
    if missing:
        raise RuntimeError(
            f"Could not read Databricks context tag(s): {missing}. "
            "The driver-proxy URL can't be built. This usually means the "
            "compute is shared/serverless rather than single-user classic; "
            "deploy the dashboard as a Databricks App in that case.")
    return host, org_id, cluster_id


def proxy_url(port: int) -> str:
    """Build the driver-proxy URL that reaches a driver-local ``port``."""
    host, org_id, cluster_id = _context_tags()
    return f"https://{host}/driver-proxy/o/{org_id}/{cluster_id}/{port}/"


def _base_url_path(port: int) -> str:
    """The path segment Streamlit must serve under, matching the proxy URL.

    Streamlit's ``--server.baseUrlPath`` wants the path WITHOUT a leading or
    trailing slash. It must equal everything after the host in ``proxy_url``.
    """
    _, org_id, cluster_id = _context_tags()
    return f"driver-proxy/o/{org_id}/{cluster_id}/{port}"


# ---------------------------------------------------------------------------
# Launch / stop
# ---------------------------------------------------------------------------
def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _resolve_db(report_db) -> str:
    """Resolve and validate the report DB path; return it as a string.

    Accepts a path or None. None falls back to config.OUTPUT_ROOT / report.db.
    On Databricks the DB usually lives on DBFS; the local FUSE
    mount is ``/dbfs/...`` and that is what a normal file read needs, so a
    ``dbfs:/foo`` URI is rewritten to ``/dbfs/foo``.
    """
    if report_db is None:
        raise ValueError("report_db is required.")
    s = str(report_db)
    if s.startswith("dbfs:/"):
        s = "/dbfs/" + s[len("dbfs:/"):].lstrip("/")
    if not Path(s).exists():
        raise FileNotFoundError(
            f"Study database not found at {s!r}. Run the workflow first "
            "(python -m DSSP2026.workflows.cli run ...) to produce a report.db, "
            "or pass report_db=<path> pointing at an existing one. On DBFS use "
            "the /dbfs/... FUSE path (e.g. /dbfs/FileStore/dssp/report.db).")
    return s


def launch_dashboard(report_db=None, *, port: int = 8501,
                     wait: bool = True, timeout: float = 30.0) -> str:
    """Start the Streamlit dashboard on the driver and return its proxy URL.

    Parameters
    ----------
    report_db : str | Path | None
        Path to the report database the dashboard reads. None -> OUTPUT_ROOT/report.db.
        DBFS paths should be the /dbfs/...
        FUSE form; a ``dbfs:/`` URI is rewritten automatically.
    port : int
        Driver-local port for the Streamlit server. Must be free.
    wait : bool
        If True, block until the server is accepting connections (or timeout)
        before returning, so the printed URL is live when you click it.
    timeout : float
        Seconds to wait for readiness when ``wait`` is True.

    Returns
    -------
    str
        The Databricks driver-proxy URL. Also printed.
    """
    global _PROC, _LAST_URL

    db = _resolve_db(report_db)
    url = proxy_url(port)                 # also validates we're on a cluster
    base_path = _base_url_path(port)

    # Replace any dashboard we started earlier in this session.
    stop_dashboard()

    if not _port_is_free(port):
        raise RuntimeError(
            f"Port {port} is already in use on the driver. Pass a different "
            "port=, or call stop_dashboard() if a previous launch is lingering.")

    dashboard_module = Path(__file__).with_name("cost_analysis.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dashboard_module),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",       # bind all interfaces for the proxy
        "--server.baseUrlPath", base_path,   # serve under the proxy path
        "--server.enableCORS", "false",      # proxy terminates on a different origin
        "--server.enableXsrfProtection", "false",
        "--server.headless", "true",         # no "open browser" attempt on the driver
        "--browser.gatherUsageStats", "false",
    ]

    env = dict(os.environ)
    env["DSSP_DASHBOARD_DB"] = db            # the dashboard reads this (line 538)

    try:
        _PROC = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Streamlit is not installed on the cluster. Install it on the "
            "cluster (Libraries tab) or in-notebook with: "
            "%pip install streamlit altair") from None

    if wait:
        _wait_until_ready(port, timeout)

    _LAST_URL = url
    print(f"Dashboard reading: {db}")
    print(f"Open: {url}")
    return url


def _wait_until_ready(port: int, timeout: float) -> None:
    """Poll the local port until the server accepts a connection, or die.

    If the subprocess exits early (bad install, import error in dashboard.py),
    surface its captured output instead of silently timing out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _PROC is not None and _PROC.poll() is not None:
            out = ""
            try:
                out = _PROC.stdout.read() if _PROC.stdout else ""
            except Exception:
                pass
            raise RuntimeError(
                "Streamlit exited before it became ready. Captured output:\n"
                + (out or "(none)"))
        if not _port_is_free(port):       # something is listening now
            time.sleep(0.5)               # small grace for full app init
            return
        time.sleep(0.4)
    # Timed out but process still alive — likely slow init, not fatal.
    print(f"(note: server not confirmed ready after {timeout:.0f}s; "
          "it may still be starting — give the URL a moment.)")


def stop_dashboard() -> None:
    """Terminate the dashboard started by ``launch_dashboard`` (if any)."""
    global _PROC
    if _PROC is None:
        return
    if _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _PROC.kill()
    _PROC = None