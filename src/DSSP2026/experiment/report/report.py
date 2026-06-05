from __future__ import annotations

from DSSP2026.experiment.report.base import ReportBase
from DSSP2026.experiment.report.compare import CompareMixin
from DSSP2026.experiment.report.diagnostics import DiagnosticsMixin
from DSSP2026.experiment.report.cost_api import CostMixin
from DSSP2026.experiment.report.logistic import LogisticMixin


class Report(ReportBase, CompareMixin, DiagnosticsMixin, CostMixin,
             LogisticMixin):
    """Read-only reporting API over a report.db.

    Composes the DB core (ReportBase) with the analysis mixins: model
    comparison, diagnostics (confusion / ROC / threshold), the cost-sensitive
    decision layer, and the logistic-regression coefficient / odds-ratio views.

    Most views are served from report.db alone. The logistic coefficient table
    and odds-ratio plot are the exception: coefficients aren't persisted, so
    those methods take the training frame and refit the winning model (see
    ``LogisticMixin``).
    """
