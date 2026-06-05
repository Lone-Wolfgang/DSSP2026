"""
optuna_trainplot_att.py — ATT-styled training-run plot for the MLP study.

Reads the per-epoch train/eval log-loss curves that telelogs_mlp.py logs into
each trial's user_attrs ("train_loss_curve" / "eval_loss_curve"), takes the
top-N trials by objective, aggregates their curves (mean ± band across runs,
NaN-padded for unequal lengths), and plots train vs. eval loss together.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import optuna

from DSSP2026.core.style import ATT_COLORS
from DSSP2026.core.figure import save_figure
from DSSP2026.tuning.search import EVAL_CURVE_KEY, TRAIN_CURVE_KEY

TRAIN_KEY = TRAIN_CURVE_KEY
EVAL_KEY = EVAL_CURVE_KEY


def _stack(curves):
    """Stack ragged curves into a (n, max_len) array, NaN-padded."""
    L = max(len(c) for c in curves)
    M = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves):
        M[i, :len(c)] = c
    return M


def plot_training_run(study, *, top_n: int = 10, figsize: Optional[tuple] = None,
                      title: Optional[str] = None):
    """Aggregate the top-N trials' train/eval loss curves and plot them.

    Mean curve (solid) with a ±1 std band across the top-N runs, train in AT&T
    blue and eval in orange. The x-axis is epoch; because runs early-stop at
    different epochs, later epochs average over fewer runs (the band reflects
    that via NaN-aware stats).
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None
                 and TRAIN_KEY in t.user_attrs and EVAL_KEY in t.user_attrs]
    if not completed:
        raise ValueError(
            "No completed trials carry loss curves. Re-run the search with the "
            "curve-logging objective (telelogs_mlp.py) so trials store "
            f"'{TRAIN_KEY}'/'{EVAL_KEY}' in user_attrs.")

    top = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]
    n = len(top)

    train_M = _stack([t.user_attrs[TRAIN_KEY] for t in top])
    eval_M = _stack([t.user_attrs[EVAL_KEY] for t in top])

    epochs = np.arange(train_M.shape[1])
    tr_mean, tr_std = np.nanmean(train_M, axis=0), np.nanstd(train_M, axis=0)
    ev_epochs = np.arange(eval_M.shape[1])
    ev_mean, ev_std = np.nanmean(eval_M, axis=0), np.nanstd(eval_M, axis=0)

    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))

    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                    color=ATT_COLORS["deep_blue"], alpha=0.15, linewidth=0)
    ax.plot(epochs, tr_mean, color=ATT_COLORS["deep_blue"], linewidth=2.4,
            label="Train loss")

    ax.fill_between(ev_epochs, ev_mean - ev_std, ev_mean + ev_std,
                    color=ATT_COLORS["orange"], alpha=0.15, linewidth=0)
    ax.plot(ev_epochs, ev_mean, color=ATT_COLORS["orange"], linewidth=2.4,
            label="Eval loss")

    # Mark the mean eval-loss minimum (the effective early-stopping point).
    best_ep = int(np.nanargmin(ev_mean))
    ax.axvline(best_ep, color=ATT_COLORS["gray_500"], linestyle="--",
               linewidth=1.3, alpha=0.8,
               label=f"Min eval loss (epoch {best_ep})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title(title or f"Training run — top {n} trials (mean ± 1 std)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def save_training_run(study, path, *, dpi: int = 220, **kwargs):
    return save_figure(plot_training_run(study, **kwargs), path, dpi=dpi)


def load_and_plot(storage, study_name, path=None, **kwargs):
    study = optuna.load_study(study_name=study_name, storage=storage)
    if path:
        return save_training_run(study, path, **kwargs)
    return plot_training_run(study, **kwargs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--out", default="training_run.png")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()
    load_and_plot(args.storage, args.study_name, path=args.out, top_n=args.top_n)
    print(f"Wrote {args.out}")