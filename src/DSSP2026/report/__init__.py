from DSSP2026.core.style import apply_att_style
from DSSP2026.report import _common  # noqa: F401

apply_att_style(base_size=18)

from DSSP2026.report.base import ENSEMBLE_NAME
from DSSP2026.report.tables import ReportTable
from DSSP2026.report.plots import (
    ConfusionPlot, ROCPlot, ThresholdSweepPlot,
    FeatureImportancePlot, TrainingCurvePlot, ClassDistributionPlot,
    CostThresholdPlot)
from DSSP2026.report.cost.tables import (
    CostDecision, PolicyTable, ClassBreakdownTable, CostConfusion)
from DSSP2026.report.cost.fit import CostDecisionModel
from DSSP2026.report.fit import PolicyModel
from DSSP2026.report.logistic import CoefficientsTable, OddsRatioPlot
from DSSP2026.report.evaluation import EvaluationMixin
from DSSP2026.report.report import Report
from DSSP2026.report.report_builder import build_report_db

__all__ = [
    "Report", "ReportTable",
    "ConfusionPlot", "ROCPlot", "ThresholdSweepPlot",
    "FeatureImportancePlot", "TrainingCurvePlot", "ClassDistributionPlot",
    "CostThresholdPlot",
    "CostDecision", "PolicyTable", "ClassBreakdownTable", "CostConfusion",
    "CostDecisionModel", "PolicyModel",
    "CoefficientsTable", "OddsRatioPlot",
    "EvaluationMixin", "build_report_db", "ENSEMBLE_NAME",
]
