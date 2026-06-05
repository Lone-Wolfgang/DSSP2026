from __future__ import annotations

import numpy as np
import pandas as pd

from DSSP2026.experiment.report.plots import ConfusionPlot
from DSSP2026.experiment.report.tables import ReportTable, PolicyCostTable

class CostDecision:
    """Result of applying a cost/benefit payoff matrix as a decision layer.

    Holds the cost-weighted predictions alongside the plain-argmax baseline, and
    scores both by *realized value* — the sum over samples of V[predicted action,
    true class]. The headline number is realized value, not accuracy: optimizing
    expected value usually trades raw accuracy for better-valued mistakes, so
    accuracy/F1 may drop by design while total value rises.
    """

    def __init__(self, *, model, class_order, actions, payoff, y_true,
                 pred_cost, pred_argmax, abstain_label, experiment_id,
                 components=None, id2label=None):
        self.model = model
        self.class_order = list(class_order)
        self.actions = list(actions)
        self.payoff = payoff                 # V[action, class], aligned to class_order
        self.y_true = np.asarray(y_true, dtype=object)
        self.pred_cost = np.asarray(pred_cost, dtype=object)
        self.pred_argmax = np.asarray(pred_argmax, dtype=object)
        self.abstain_label = abstain_label
        self.experiment_id = experiment_id
        self.components = components          # {benefit, action_cost, inaction_cost} or None
        self._action_ix = {a: i for i, a in enumerate(self.actions)}
        self._class_ix = {c: k for k, c in enumerate(self.class_order)}
        # Display-name map {stored label -> friendly}; used only for plot labels.
        # The abstain action is kept verbatim (it isn't a class). All decision
        # math stays keyed on the original stored labels in class_order/actions.
        self._id2label = {str(k): str(v) for k, v in (id2label or {}).items()}

    def _disp(self, label):
        """Display name for a class/action label (identity if unmapped)."""
        if label == self.abstain_label:
            return label
        return self._id2label.get(str(label), label)

    @property
    def predictions(self) -> pd.DataFrame:
        """Per-sample true class, argmax prediction, and cost-optimal action."""
        return pd.DataFrame({
            "true": self.y_true,
            "argmax": self.pred_argmax,
            "cost_action": self.pred_cost,
        })

    def _realized_value(self, preds):
        """Total realized value of `preds` under the payoff matrix."""
        total = 0.0
        for a, k in zip(preds, self.y_true):
            ai = self._action_ix.get(str(a))
            ki = self._class_ix.get(str(k))
            if ai is None or ki is None:
                continue
            total += self.payoff[ai, ki]
        return float(total)

    def _accuracy(self, preds):
        """Accuracy over NON-abstained samples (abstain isn't a class guess)."""
        mask = preds != self.abstain_label
        if mask.sum() == 0:
            return float("nan")
        return float((preds[mask] == self.y_true[mask]).mean())

    # Monetary metric row order for the breakdown / summary table.
    _MONEY_COLS = ["Gross benefit", "Cost of intervention", "Intervention waste",
                   "Cost of no action", "Net benefit"]

    def _money_breakdown(self, preds) -> dict:
        """Decompose realized value into the monetary quantities.

        With enriched components (benefit / action_cost / inaction_cost) the
        split is exact and intervention cost is a genuine cost:
          - gross_benefit       = Σ benefit on TP cells                       (+)
          - cost_of_intervention= Σ action_cost over ALL acted cells (TP & FP)(-)
          - intervention_waste  = Σ action_cost on FP cells only              (-)
          - cost_of_no_action   = Σ inaction_cost on abstain cells            (-)
          - net_benefit         = gross + cost_of_intervention + cost_of_no_action
                                = total realized value
        Without components (a single net payoff matrix), it falls back to the
        net-based split: gross = Σ positive payoffs on TP, intervention_waste =
        Σ FP payoffs, cost_of_intervention = gross + waste, no_action = abstain.
        """
        comp = self.components
        if comp is not None:
            ben = comp["benefit"]
            acost = comp["action_cost"]
            inact = comp["inaction_cost"]
            gross = waste = interv = no_action = 0.0
            for a, k in zip(preds, self.y_true):
                a, k = str(a), str(k)
                if a == self.abstain_label:
                    if inact is not None:
                        no_action += inact.get(k, 0.0)
                    continue
                # a real action was taken -> incurs its action cost
                interv += acost.get(a, 0.0)
                if a == k:
                    gross += ben.get(k, 0.0)
                else:
                    waste += acost.get(a, 0.0)
            return {
                "Gross benefit": gross,
                "Cost of intervention": interv,            # TP+FP action cost (negative)
                "Intervention waste": waste,               # FP action cost (subset)
                "Cost of no action": no_action,
                "Net benefit": gross + interv + no_action,
            }

        # Legacy net-only fallback (single payoff matrix).
        gross = waste = no_action = 0.0
        for a, k in zip(preds, self.y_true):
            a, k = str(a), str(k)
            ai, ki = self._action_ix.get(a), self._class_ix.get(k)
            if ai is None or ki is None:
                continue
            v = self.payoff[ai, ki]
            if a == self.abstain_label:
                no_action += v
            elif a == k:
                gross += v
            else:
                waste += v
        return {
            "Gross benefit": gross,
            "Cost of intervention": gross + waste,
            "Intervention waste": waste,
            "Cost of no action": no_action,
            "Net benefit": gross + waste + no_action,
        }

    def summary(self):
        """Enriched cost/benefit breakdown: argmax baseline vs. cost-optimal.

        Returns a ``ReportTable`` — one row per rule (Baseline, Cost optimized),
        with the five monetary line items (gross benefit, cost of intervention,
        intervention waste, cost of no action, net benefit), plus accuracy and
        the abstain count; net benefit is highlighted as the best-by column.

        For a policy-comparison view with color-scaled money cells, use
        :meth:`policy_table`.
        """
        n = len(self.y_true)
        n_abstain = int((self.pred_cost == self.abstain_label).sum())
        rules = [("Baseline", self.pred_argmax, 0),
                 ("Cost optimized", self.pred_cost, n_abstain)]
        rows = []
        for name, preds, abst in rules:
            mb = self._money_breakdown(preds)
            row = {"Rule": name}
            row.update({c: mb[c] for c in self._MONEY_COLS})
            row["Accuracy"] = self._accuracy(preds)
            row["Abstained"] = abst
            rows.append(row)
        df = pd.DataFrame(rows)
        for c in self._MONEY_COLS:
            df[c] = df[c].round(2)
        df["Accuracy"] = df["Accuracy"].round(4)
        title = (f"Cost-sensitive decision — {self.model} "
                 f"(held-out {self.experiment_id})")
        return ReportTable(df, best_by="Net benefit", title=title)

    # -- policy comparison (one row per decision policy) --
    # Canonical internal keys -> default display names. ``columns`` selects and
    # renames; ``policies`` selects and reorders the rows.
    _POLICY_DEFAULT_COLS = {
        "interventions": "Interventions",
        "intervention_cost": "Intervention Cost",
        "failures": "Failures to Act",
        "failure_cost": "Failure Cost",
        "total_cost": "Total Cost",
        "savings": "Savings vs. Do Nothing",
    }
    _POLICY_MONEY_KEYS = {"intervention_cost", "failure_cost",
                          "total_cost", "savings"}
    _POLICY_DEFAULT_NAMES = {
        "do_nothing": "Do Nothing",
        "always": "Always Intervene",
        "baseline": "Baseline Model",
        "cost": "Cost Optimized",
    }
    _POLICY_DEFAULT_ORDER = ["do_nothing", "always", "baseline", "cost"]

    def _intervention_unit_cost(self, pos_label):
        """Per-intervention cost magnitude (positive number) for ``pos_label``."""
        if self.components is not None:
            ac = self.components.get("action_cost") or {}
            return abs(float(ac.get(str(pos_label), 0.0)))
        ai = self._action_ix.get(str(pos_label))
        ci = self._class_ix.get(str(pos_label))
        if ai is None or ci is None:
            return 0.0
        # FP cell value (act, wrong) is the pure action cost.
        wrong = next((k for k in range(len(self.class_order)) if k != ci), ci)
        return abs(float(self.payoff[ai, wrong]))

    def _failure_unit_cost(self, pos_label):
        """Per-missed-positive cost magnitude (positive number)."""
        if self.components is not None:
            ben = self.components.get("benefit") or {}
            return abs(float(ben.get(str(pos_label), 0.0)))
        ai = self._action_ix.get(str(pos_label))
        ci = self._class_ix.get(str(pos_label))
        not_act = next((a for a in range(len(self.actions)) if a != ai), ai)
        if not_act is None or ci is None:
            return 0.0
        return abs(float(self.payoff[not_act, ci]))

    def policy_table(self, *, pos_label=None, policies=None, columns=None,
                     color=True, title=None):
        """Four-policy cost comparison; costs negative, money cells shaded.

        Rows are decision policies — Do Nothing, Always Intervene, Baseline
        Model (argmax), Cost Optimized — and each is scored by how it would act
        on the held-out set:

          Interventions     = patients acted on (predicted positive)
          Intervention Cost  = -(interventions x per-intervention cost)
          Failures to Act    = true positives NOT acted on (misses)
          Failure Cost       = -(failures x per-miss cost)
          Total Cost         = Intervention Cost + Failure Cost   (negative)
          Savings vs Do Nothing = Total Cost - (Do Nothing total)  (positive = good)

        ``pos_label`` is the "intervene" class (defaults to the last class in
        ``class_order``, the positive class in the binary 0/1 encoding).

        ``policies`` selects/reorders rows by key: ``do_nothing``, ``always``,
        ``baseline``, ``cost`` (default: all four in that order).

        ``columns`` selects, reorders, and RENAMES columns. Pass a sequence of
        canonical keys (``interventions``, ``intervention_cost``, ``failures``,
        ``failure_cost``, ``total_cost``, ``savings``) for defaults, or a dict
        ``{key: "Display Name"}`` to rename. Default: all six.

        ``color=True`` returns a :class:`PolicyCostTable` (shaded, with
        ``.to_png`` / ``.to_csv`` / ``.df``); ``color=False`` returns the raw
        DataFrame.
        """
        if pos_label is None:
            pos_label = self.class_order[-1]
        pos = str(pos_label)

        unit_interv = self._intervention_unit_cost(pos)
        unit_fail = self._failure_unit_cost(pos)

        y_true = self.y_true.astype(str)
        is_pos = (y_true == pos)
        n_pos = int(is_pos.sum())

        def metrics_for(acted_mask):
            interventions = int(acted_mask.sum())
            failures = int((~acted_mask & is_pos).sum())  # missed positives
            interv_cost = -interventions * unit_interv
            fail_cost = -failures * unit_fail
            total = interv_cost + fail_cost
            return interventions, interv_cost, failures, fail_cost, total

        acted = {
            "do_nothing": np.zeros(len(y_true), dtype=bool),
            "always": np.ones(len(y_true), dtype=bool),
            "baseline": self.pred_argmax.astype(str) == pos,
            "cost": self.pred_cost.astype(str) == pos,
        }
        raw = {k: metrics_for(m) for k, m in acted.items()}
        do_nothing_total = raw["do_nothing"][4]

        order = list(policies) if policies else list(self._POLICY_DEFAULT_ORDER)
        rows = []
        for key in order:
            if key not in raw:
                raise ValueError(
                    f"unknown policy {key!r}; choose from "
                    f"{list(self._POLICY_DEFAULT_ORDER)}.")
            interv, ic, fail, fc, tot = raw[key]
            rows.append({
                "Policy": self._POLICY_DEFAULT_NAMES[key],
                "interventions": interv,
                "intervention_cost": ic,
                "failures": fail,
                "failure_cost": fc,
                "total_cost": tot,
                "savings": tot - do_nothing_total,   # >= 0; saving vs do-nothing
            })
        wide = pd.DataFrame(rows)

        # Resolve column selection + display names.
        if columns is None:
            sel = dict(self._POLICY_DEFAULT_COLS)
        elif isinstance(columns, dict):
            sel = {k: str(v) for k, v in columns.items()}
        else:
            sel = {k: self._POLICY_DEFAULT_COLS[k] for k in columns}
        bad = [k for k in sel if k not in self._POLICY_DEFAULT_COLS]
        if bad:
            raise ValueError(
                f"unknown column keys {bad}; choose from "
                f"{list(self._POLICY_DEFAULT_COLS)}.")

        out = {"Policy": wide["Policy"]}
        money_cols, count_cols = [], []
        for key, name in sel.items():
            out[name] = wide[key]
            (money_cols if key in self._POLICY_MONEY_KEYS else count_cols).append(name)
        df = pd.DataFrame(out)

        if title is None:
            title = f"Policy cost comparison — {self.model}"
        if not color:
            return df
        return PolicyCostTable(df, money_cols=money_cols, count_cols=count_cols,
                               policy_col="Policy", title=title)

    def confusion(self, *, by="value", normalize=False, value_text="both",
                  title=None, **kwargs):
        """Confusion matrix under the cost-optimal decision rule.

        Rows = true class, columns = the chosen action (includes an abstain
        column when any sample abstained).

        ``by="value"`` (default) colors each cell by its **total realized value**
        (count x payoff) on a diverging firebrick -> gold -> forest-green scale,
        symmetric about zero so gold = break-even; the colorbar is the monetary
        scale and ``value_text`` controls the cell text ("both" = count + dollar,
        "count", or "value"). ``by="count"`` falls back to the plain count/blue
        matrix (optionally ``normalize``).
        """
        true_labels = list(self.class_order)
        action_labels = list(self.actions)
        idx = {c: i for i, c in enumerate(true_labels)}
        adx = {a: j for j, a in enumerate(action_labels)}
        counts = np.zeros((len(true_labels), len(action_labels)), dtype=float)
        for a, k in zip(self.pred_cost, self.y_true):
            ki, aj = idx.get(str(k)), adx.get(str(a))
            if ki is not None and aj is not None:
                counts[ki, aj] += 1

        if by == "value":
            # Per-cell realized value = count x payoff. payoff is V[action, class];
            # cell (true=k, action=a) uses V[a, k].
            value = np.zeros_like(counts)
            for ki, k in enumerate(true_labels):
                for aj, a in enumerate(action_labels):
                    ai = self._action_ix.get(a)
                    cl = self._class_ix.get(k)
                    if ai is not None and cl is not None:
                        value[ki, aj] = counts[ki, aj] * self.payoff[ai, cl]
            if title is None:
                title = f"Realized value (cost-optimal) — {self.model}"
            return ConfusionPlot(
                counts=counts, class_labels=[self._disp(a) for a in action_labels],
                row_labels=[self._disp(k) for k in true_labels],
                title=title, true_axis="True", pred_axis="Action",
                value_matrix=value, value_text=value_text, **kwargs)

        if title is None:
            title = f"Confusion (cost-optimal) — {self.model}"
        return ConfusionPlot(
            counts=counts, class_labels=[self._disp(a) for a in action_labels],
            normalize=normalize,
            title=title, true_axis="True", pred_axis="Action",
            row_labels=[self._disp(k) for k in true_labels], **kwargs)
