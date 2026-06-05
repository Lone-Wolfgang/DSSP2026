from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from DSSP2026.experiment.report.cost import CostDecision


class CostMixin:
    def cost_optimize(self, payoff=None, model: Optional[str] = None, *,
                      benefit=None, action_cost=None, inaction_cost=None,
                      abstain_label: str = "abstain"):
        """Apply a cost/benefit decision layer over the model's held-out probs.

        Replaces plain ``argmax(p)`` with the expected-value-maximizing decision
        ``action*(x) = argmax_a sum_k p(C_k|x) * V[a,k]`` (``scores = p @ V.T``).

        Two ways to specify the payoffs:

        **Single net matrix** — ``payoff`` = ``V[action, true_class]`` (DataFrame
        with action-row / class-column labels, or aligned array). Each cell is
        the *net* value of acting; the cost breakdown is best-effort from signs.

        **Enriched components** (recommended for a true cost breakdown) — pass
        instead:
          - ``benefit``: per-class reward for a CORRECT action (Series {class:
            value} or scalar); placed on the diagonal.
          - ``action_cost``: per-action cost of TAKING that action (Series
            {action: cost<=0} or scalar), incurred on every acted cell (TP & FP).
          - ``inaction_cost``: per-class cost of ABSTAINING (Series {class:
            cost<=0} or scalar); supplying it adds an ``abstain`` action.
        The net matrix used for the decision is ``V[a,k] = (benefit[k] if a==k
        else 0) + action_cost[a]`` for real actions, and ``inaction_cost[k]`` for
        abstain — so the decision is identical to passing that net ``V``, but the
        components let the summary separate gross benefit, intervention cost, and
        inaction cost exactly.

        ``model=None`` selects the best model by held-out F1.
        """
        if model is None:
            model = self._best_model_name()
        class_order, y_true, y_proba = self._read_predictions(model)

        components = None
        if payoff is not None:
            actions, V = self._coerce_payoff(payoff, class_order, abstain_label)
        elif benefit is not None and action_cost is not None:
            actions, V, components = self._build_from_components(
                class_order, benefit, action_cost, inaction_cost, abstain_label)
        else:
            raise ValueError(
                "provide either `payoff` (net matrix) or both `benefit` and "
                "`action_cost` (enriched components).")

        scores = y_proba @ V.T
        chosen = scores.argmax(axis=1)
        pred_actions = np.asarray([actions[a] for a in chosen], dtype=object)
        argmax_idx = y_proba.argmax(axis=1)
        pred_argmax = np.asarray([class_order[k] for k in argmax_idx], dtype=object)

        return CostDecision(
            model=model, class_order=class_order, actions=actions,
            payoff=V, y_true=y_true.astype(str),
            pred_cost=pred_actions, pred_argmax=pred_argmax,
            abstain_label=abstain_label, experiment_id=self.experiment_id,
            components=components, id2label=getattr(self, "id2label", None))

    def _build_from_components(self, class_order, benefit, action_cost,
                               inaction_cost, abstain_label):
        """Build (actions, net V, components) from separate benefit/cost pieces.

        benefit: per-class (diagonal) reward for a correct action.
        action_cost: per-action cost of taking that action (TP & FP).
        inaction_cost: per-class cost of abstaining (optional; adds abstain row).
        Returns the action list, the net payoff matrix aligned to class_order,
        and a components dict the summary uses for the exact cost breakdown.
        """
        K = len(class_order)

        def _as_class_vec(x, name, default=0.0):
            if x is None:
                return {c: default for c in class_order}
            if np.isscalar(x):
                return {c: float(x) for c in class_order}
            # Series/dict keyed by class label.
            d = dict(x)
            missing = [c for c in class_order if c not in d]
            if missing:
                raise ValueError(f"{name} missing entries for classes {missing}.")
            return {c: float(d[c]) for c in class_order}

        ben = _as_class_vec(benefit, "benefit")
        # action_cost is per-action (per class you act on).
        acost = _as_class_vec(action_cost, "action_cost")

        actions = list(class_order)
        V = np.zeros((K, K), dtype=float)
        for ai, a in enumerate(class_order):
            for ki, k in enumerate(class_order):
                V[ai, ki] = (ben[k] if a == k else 0.0) + acost[a]

        components = {
            "benefit": ben, "action_cost": acost,
            "inaction_cost": None, "abstain_label": abstain_label,
        }
        if inaction_cost is not None:
            inact = _as_class_vec(inaction_cost, "inaction_cost")
            V = np.vstack([V, np.array([inact[k] for k in class_order])])
            actions = list(class_order) + [abstain_label]
            components["inaction_cost"] = inact
        return actions, V, components

    def _coerce_payoff(self, payoff, class_order, abstain_label):
        """Normalize ``payoff`` to (action_labels, V array aligned to class_order).

        Columns are reordered to match the model's stored class order so the
        matrix multiply lines up with the probability columns. Validates shape
        and that every class column is present.
        """
        if hasattr(payoff, "loc"):  # DataFrame: labeled rows (actions) x cols (classes)
            cols = [str(c) for c in payoff.columns]
            missing = [c for c in class_order if c not in cols]
            if missing:
                raise ValueError(
                    f"payoff matrix is missing columns for classes {missing}; "
                    f"its columns are {cols}.")
            # Reorder columns to the model's class order.
            ordered = payoff.reindex(columns=class_order)
            actions = [str(a) for a in ordered.index]
            V = np.asarray(ordered.to_numpy(), dtype=float)
        else:
            V = np.asarray(payoff, dtype=float)
            K = len(class_order)
            if V.ndim != 2 or V.shape[1] != K:
                raise ValueError(
                    f"payoff array must have {K} columns (one per class, in the "
                    f"model's class order {class_order}); got shape {V.shape}.")
            if V.shape[0] == K:
                actions = list(class_order)
            elif V.shape[0] == K + 1:
                actions = list(class_order) + [abstain_label]
            else:
                raise ValueError(
                    f"payoff array must have {K} rows (one action per class) or "
                    f"{K + 1} (with a trailing abstain row); got {V.shape[0]}.")
        return actions, V