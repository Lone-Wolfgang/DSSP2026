from __future__ import annotations

from DSSP2026.report.base import ReportBase
from DSSP2026.report.compare import CompareMixin
from DSSP2026.report.diagnostics import DiagnosticsMixin
from DSSP2026.report.cost.api import CostMixin
from DSSP2026.report.fit import FitMixin
from DSSP2026.report.logistic import LogisticMixin
from DSSP2026.report.evaluation import EvaluationMixin
from DSSP2026.report.tuning import TuningMixin
from DSSP2026.report.describe import DescribeMixin


class Report(ReportBase, CompareMixin, DiagnosticsMixin, CostMixin, FitMixin,
             LogisticMixin, EvaluationMixin, TuningMixin, DescribeMixin):
    """Read-only reporting API over a report.db.

    Composes the DB core (ReportBase) with the analysis mixins: model
    comparison, diagnostics (confusion / ROC / threshold), the cost-sensitive
    decision layer, the logistic-regression coefficient / odds-ratio views, and
    the hyperparameter-tuning views (elbow / grid heatmap / parallel
    coordinates, auto-routed by search shape).

    Most views are served from report.db alone. The logistic coefficient table
    and odds-ratio plot are the exception: coefficients aren't persisted, so
    those methods take the training frame and refit the winning model (see
    ``LogisticMixin``).
    """