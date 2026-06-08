"""
report/describe.py — `Report.describe()`: what can I do with THIS report.db?

The Report API spans ~20 methods across several mixins, and several are
conditional on what the underlying report.db actually contains:

  * logistic coefficient / odds-ratio views need a logistic model present;
  * tuning plots need a model with more than one trial;
  * best_fit / cost_optimize / leak-free test scoring need the persisted
    ``test_predictions`` / ``oof_predictions`` tables (older DBs lack them);
  * the Ensemble views need >= 2 real models.

`describe()` introspects the loaded report.db and reports, per analysis, whether
it is available here and — when not — why. It is read-only and never raises for
a normal (even old) report.db. Returns a :class:`ReportCapabilities` whose
``__repr__`` prints a readable report and which is also programmatically usable
(``.available`` / ``.unavailable`` / ``.models`` / ``.policies``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from DSSP2026.report.base import ENSEMBLE_NAME
from DSSP2026.report.policy import POLICIES


@dataclass
class ReportCapabilities:
    """Structured result of ``Report.describe()``.

    Attributes
    ----------
    experiment_id : str
    models : list[str]            real models (+ Ensemble when >=2), as Report.models()
    policies : list[str]          decision policies confusion_matrix / fit accept
    available : dict[str, str]    analysis name -> short "what it does"
    unavailable : dict[str, str]  analysis name -> why it is NOT available here
    """
    experiment_id: str
    models: List[str] = field(default_factory=list)
    policies: List[str] = field(default_factory=list)
    available: Dict[str, str] = field(default_factory=dict)
    unavailable: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [f"Report capabilities — experiment {self.experiment_id}"]
        lines.append(f"  models:   {', '.join(self.models) or '(none)'}")
        lines.append(f"  policies: {', '.join(self.policies)}")
        lines.append("  available analyses:")
        for name, what in self.available.items():
            lines.append(f"    • {name} — {what}")
        if self.unavailable:
            lines.append("  not available here:")
            for name, why in self.unavailable.items():
                lines.append(f"    • {name} — {why}")
        return "\n".join(lines)


class DescribeMixin:

    def describe(self) -> ReportCapabilities:
        """Introspect this report.db and report which analyses are available.

        Read-only; safe on any report.db (including older ones missing the
        leak-free prediction tables). See :class:`ReportCapabilities`.
        """
        real = self.models(include_ensemble=False)
        has_ensemble = len(real) >= 2
        all_models = self.models(include_ensemble=True)

        # Probe which optional tables/structures this db has.
        tables = self._describe_tables()
        has_test = "test_predictions" in tables
        has_oof = "oof_predictions" in tables
        has_logistic = any(m.lower().startswith("logistic") for m in real)
        tunable = self._describe_tunable_models()   # models with >1 trial

        cap = ReportCapabilities(
            experiment_id=self.experiment_id,
            models=all_models,
            policies=list(POLICIES),
        )
        A, U = cap.available, cap.unavailable

        # Always-available core views (need only stored predictions).
        A["compare_models"] = "per-model metrics table (test-scored when available)"
        A["confusion_matrix"] = ("confusion heatmap for a model + policy "
                                 f"(policies: {', '.join(POLICIES)})")
        A["roc_compare"] = "ROC curves across models"
        A["classification_report"] = "per-class precision/recall/F1 table"
        A["feature_importance"] = "feature-importance plot for a model"

        # Ensemble-gated.
        if has_ensemble:
            A["Ensemble views"] = (f"mean-probability ensemble of {len(real)} "
                                   "models (pass model='Ensemble')")
        else:
            U["Ensemble views"] = (f"needs >= 2 real models; this experiment has "
                                   f"{len(real)}")

        # Leak-free fit / cost (need persisted test + OOF).
        if has_test and has_oof:
            A["best_fit"] = "select on validation, report leak-free test metrics"
            A["fit"] = "tune thresholds on train OOF, deploy a PolicyModel"
            A["cost_optimize"] = "cost-sensitive (model x policy) sweep, test-scored"
        else:
            missing = []
            if not has_test:
                missing.append("test_predictions")
            if not has_oof:
                missing.append("oof_predictions")
            why = (f"needs persisted {' and '.join(missing)} table(s); re-run the "
                   "experiment with train/validation/test splits")
            U["best_fit"] = why
            U["fit"] = why
            U["cost_optimize"] = why

        # Logistic-gated.
        if has_logistic:
            A["coefficients_table"] = "logistic coefficients (refits on train frame)"
            A["odds_ratios_plot"] = "logistic odds-ratio forest plot"
        else:
            U["coefficients_table"] = "no logistic model in this experiment"
            U["odds_ratios_plot"] = "no logistic model in this experiment"

        # Tuning-gated (needs a model with >1 trial).
        if tunable:
            A["tuning_plot"] = (f"hyperparameter search plots for: "
                                f"{', '.join(tunable)} (auto-routed by search shape)")
        else:
            U["tuning_plot"] = ("no model has more than one trial (nothing to "
                                "plot — e.g. logistic-only or single-config runs)")

        return cap

    # ------------------------------------------------------------------
    # introspection helpers
    # ------------------------------------------------------------------

    def _describe_tables(self) -> set:
        """Set of table names present in the report.db."""
        conn = self._connect()
        try:
            return {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        except Exception:
            return set()
        finally:
            conn.close()

    def _describe_tunable_models(self) -> list:
        """Real models with more than one trial (i.e. something to plot)."""
        conn = self._connect()
        try:
            if "trials" not in {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}:
                return []
            rows = conn.execute(
                "SELECT m.model, COUNT(t.trial_number) AS n "
                "FROM models m LEFT JOIN trials t ON t.model_id = m.model_id "
                "WHERE m.experiment_id=? GROUP BY m.model HAVING n > 1",
                (self.experiment_id,)).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []
        finally:
            conn.close()