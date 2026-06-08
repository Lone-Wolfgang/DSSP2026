"""
optuna_parallel_att.py — ATT-styled parallel-coordinates report for an Optuna
study, built by hand (not optuna.visualization) so the colours, axis order, and
scales follow the house style.

One polyline per completed trial across the hyperparameter axes; lines coloured
by the trial's objective value on the ATT_SEQUENTIAL ramp (darker = better).
Axes are ordered left -> right by *ascending* parameter importance, so the most
important parameter sits on the right.

Lines are drawn as smooth curves (monotone PCHIP) that still pass exactly
through each axis value, and the colourbar maps 1:1 to the line colours (W&B
style) so a line's shade can be read directly off the objective scale.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator

import optuna

from DSSP2026.core.style import ATT_COLORS, ATT_SEQUENTIAL
from DSSP2026.core.heatmap import ATT_SEQUENTIAL_CMAP   # reuse the registered ramp
from DSSP2026.core.figure import save_figure

# Optional hint of params that commonly exist across studies. The default axis
# set is inferred from the study itself (see plot logic), so this list is no
# longer required — kept only as a reference of typical MLP axes.
ALWAYS_PRESENT = ["feature_set", "n_layers", "width1", "activation",
                  "alpha", "lr_init"]
# Axes that should be spaced on a log scale (values span orders of magnitude).
# Covers both the MLP params (alpha, lr_init) and the XGBoost ones
# (learning_rate, reg_lambda, reg_alpha).
LOG_AXES = {"alpha", "lr_init", "learning_rate", "reg_lambda", "reg_alpha"}

# How many points to sample along each smoothed curve.
_CURVE_SAMPLES = 200

# Relative tolerance for treating the objective spread as "degenerate" (all
# trials effectively tied). When (max-min) is this small a fraction of the
# typical magnitude, ranking and a colour gradient are meaningless — and worse,
# misleading — so the plot switches to a uniform "tie mode" (see plot logic).
_TIE_RTOL = 1e-9


def _smooth_curve(x, y, n_samples=_CURVE_SAMPLES):
    """Monotone (PCHIP) interpolation of `y` over `x`.

    PCHIP passes through every knot without the overshoot a cubic spline would
    introduce, so each curve still reads the exact value on every axis rail —
    important for a parallel-coordinates plot. With a single axis there is
    nothing to interpolate, so we return the points unchanged.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return x, y
    xs = np.linspace(x[0], x[-1], n_samples)
    ys = PchipInterpolator(x, y)(xs)
    return xs, ys


def plot_optuna_parallel_coordinates(
    study,
    *,
    params: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
    metric_label: str = "Objective (CV macro-F1)",
    objective_label: str = "objective",
    title: str = "Hyperparameter search — parallel coordinates",
    figsize: Optional[tuple] = None,
    curved: bool = True,
):
    """ATT-styled parallel-coordinates plot of an Optuna study's trials.

    Parameters
    ----------
    study : optuna.Study
        A study with completed trials (load it from the SQLite db first).
    params : sequence of str, optional
        Which parameters to show as axes. Defaults to the always-present set.
    drop : sequence of str, optional
        Parameters to remove from the axis set (applied after `params`), so you
        can prune axes without re-listing the whole set.
    metric_label : str
        Colourbar label for the objective.
    objective_label : str
        Name shown under the rightmost metric axis (the trials' objective is
        added as the final axis, so every line terminates at its score).
    curved : bool
        If True (default), draw lines as smooth PCHIP curves; if False, fall
        back to straight segments.

    Axes are ordered by ascending importance (most important on the right),
    then the objective is appended as the final axis so each line lands at its
    score — the way Weights & Biases plots the tracked metric. Lines are
    coloured by objective on ATT_SEQUENTIAL (darker = higher), with a matching
    colourbar to the right. Categorical axes are evenly spaced; LOG_AXES are
    log-scaled.
    """
    # 1. Completed trials only, with a finite objective.
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.value is not None and np.isfinite(t.value)]
    if not trials:
        raise ValueError("No completed trials with a finite objective to plot.")

    # Optuna-native importances (best-effort); the records core orders axes by
    # this map. Computed here because it needs the live study object.
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}

    records = [dict(t.params) for t in trials]
    values = np.array([t.value for t in trials], dtype=float)
    return _plot_parallel_core(
        records, values, importances=imp, params=params, drop=drop,
        metric_label=metric_label, objective_label=objective_label,
        title=title, figsize=figsize, curved=curved)


def plot_parallel_coordinates_from_frame(
    trials_df,
    *,
    value_col: str = "cv_value",
    params: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
    importances: Optional[dict] = None,
    metric_label: str = "Objective (CV macro-F1)",
    objective_label: str = "objective",
    title: str = "Hyperparameter search — parallel coordinates",
    figsize: Optional[tuple] = None,
    curved: bool = True,
):
    """Parallel-coordinates plot driven by a tidy trials DataFrame.

    This is the report-DB entry point: it needs no live Optuna study. Each row
    is one trial; ``value_col`` holds the objective, and every *other* column is
    treated as a candidate parameter axis (subject to ``params``/``drop`` and the
    "present on all rows" rule the study path also enforces). ``importances`` is
    optional — when omitted, axes fall back to DataFrame column order. Rows whose
    objective is missing/non-finite are dropped.
    """
    import pandas as pd  # local import: keep module import light

    if value_col not in trials_df.columns:
        raise ValueError(f"trials_df has no {value_col!r} column.")
    df = trials_df[np.isfinite(pd.to_numeric(trials_df[value_col],
                                             errors="coerce"))]
    if df.empty:
        raise ValueError("No completed trials with a finite objective to plot.")

    param_cols = [c for c in df.columns if c != value_col]
    records = [{c: row[c] for c in param_cols if pd.notna(row[c])}
               for _, row in df.iterrows()]
    values = df[value_col].to_numpy(dtype=float)
    return _plot_parallel_core(
        records, values, importances=importances or {}, params=params,
        drop=drop, metric_label=metric_label, objective_label=objective_label,
        title=title, figsize=figsize, curved=curved)


def _plot_parallel_core(
    records,
    values,
    *,
    importances: dict,
    params: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
    metric_label: str = "Objective (CV macro-F1)",
    objective_label: str = "objective",
    title: str = "Hyperparameter search — parallel coordinates",
    figsize: Optional[tuple] = None,
    curved: bool = True,
):
    """Source-agnostic parallel-coordinates renderer.

    ``records`` is a list of per-trial param dicts; ``values`` the matching
    objective array; ``importances`` a ``{param: importance}`` map used only to
    order axes (missing -> 0). All callers (live study or report-DB frame) funnel
    here so the plotting logic lives in exactly one place.
    """
    values = np.asarray(values, dtype=float)
    imp = importances or {}

    # 2. Axis set: requested params, else auto-inferred — every param present on
    #    *all* selected trials (so it works for any model). Conditional params
    #    (e.g. MLP width2/width3, present only for some trials) are naturally
    #    excluded because a parallel-coordinates line needs a value on every axis.
    if params is not None:
        axes = list(params)
    else:
        common = set(records[0])
        for r in records[1:]:
            common &= set(r)
        # Preserve a stable, readable order: first-seen order across trials.
        seen, axes = set(), []
        for r in records:
            for p in r:
                if p in common and p not in seen:
                    seen.add(p); axes.append(p)
    if drop:
        axes = [p for p in axes if p not in set(drop)]
    axes = [p for p in axes if all(p in r for r in records)]
    if not axes:
        raise ValueError("No parameters left to plot after filtering/drops.")

    # 3. Order axes by ascending importance -> most important on the right.
    axes.sort(key=lambda p: imp.get(p, 0.0))   # low importance left, high right


    # 4. Build the per-axis coordinate for every trial, normalised to [0, 1].
    #    Numeric axes: min-max (log-spaced first if in LOG_AXES). Categorical:
    #    evenly spaced positions by sorted category.
    n_axes = len(axes)
    coords = np.zeros((len(records), n_axes))
    tick_info = []   # (positions, labels) per axis for annotating the rails

    for a, p in enumerate(axes):
        raw = [r[p] for r in records]
        is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                         for v in raw)

        if is_numeric:
            vals = np.array(raw, dtype=float)
            scaled = np.log10(vals) if p in LOG_AXES else vals
            lo, hi = scaled.min(), scaled.max()
            span = (hi - lo) or 1.0
            coords[:, a] = (scaled - lo) / span
            # Three ticks: min / mid / max, labelled in original units.
            tick_pos = [0.0, 0.5, 1.0]
            if p in LOG_AXES:
                tick_val = [10 ** (lo + f * span) for f in tick_pos]
                tick_lab = [f"{v:.1e}" for v in tick_val]
            else:
                tick_val = [lo + f * span for f in tick_pos]
                tick_lab = [f"{v:.3g}" for v in tick_val]
            tick_info.append((tick_pos, tick_lab))
        else:
            # Categorical: evenly spaced rows by sorted unique category.
            cats = sorted(set(map(str, raw)))
            pos = {c: (i / (len(cats) - 1) if len(cats) > 1 else 0.5)
                   for i, c in enumerate(cats)}
            coords[:, a] = [pos[str(v)] for v in raw]
            tick_info.append(([pos[c] for c in cats], cats))

    # 4b. Append the objective itself as the final (rightmost) axis, so every
    #     line terminates at its score — this is what visually "connects" the
    #     lines to the metric (W&B style), rather than relying on colour alone.
    #     Min-max normalised like any numeric axis; ticked in real units.
    lo_v, hi_v = values.min(), values.max()
    raw_span = hi_v - lo_v
    # Degenerate case: every trial scored (near-)identically. A min-max mapping
    # would divide by ~0, collapsing colour to one ramp end and inventing a rank
    # order out of float noise. Detect it and switch to a flat "tie mode".
    tie_mode = raw_span <= _TIE_RTOL * max(abs(lo_v), abs(hi_v), 1.0)
    span_v = raw_span or 1.0
    if tie_mode:
        # Pin every line to the middle of the metric axis; a 0..1 spread of pure
        # float noise would otherwise scatter them meaninglessly top-to-bottom.
        metric_col = np.full(len(values), 0.5)
        tick_pos = [0.5]
        tick_lab = [f"{lo_v:.3f} (all trials tied)"]
    else:
        metric_col = (values - lo_v) / span_v
        tick_pos = [0.0, 0.5, 1.0]
        tick_lab = [f"{lo_v + f * span_v:.3f}" for f in tick_pos]
    coords = np.column_stack([coords, metric_col])
    axes = axes + [objective_label]
    tick_info.append((tick_pos, tick_lab))
    n_axes = len(axes)

    # 5. Colour each line by objective on ATT_SEQUENTIAL, with a matching
    #    colourbar on the right. The metric axis above carries the readable
    #    mapping; colour reinforces it (darker = higher) and the rank-based
    #    width/alpha (below) de-tangles the bundle. In tie mode there is no
    #    spread to encode, so colour and weight are held uniform instead.
    norm = Normalize(vmin=lo_v, vmax=hi_v)
    cmap = ATT_SEQUENTIAL_CMAP

    fig, ax = plt.subplots(figsize=figsize or (1.7 * n_axes + 3, 6))
    x = np.arange(n_axes)

    # Rank trials by objective -> percentile in [0, 1]. Width and alpha scale
    # with rank: faint thin "losers", bold opaque "winners". This de-tangles the
    # plot far more than colour alone — the eye lands on the high-rank lines.
    # In tie mode every trial is genuinely equal, so we give them all the same
    # mid-rank weight rather than inventing an order from float noise.
    order = np.argsort(values)                 # worst -> best (best drawn last)
    rank = np.empty(len(values))
    if tie_mode:
        rank[:] = 0.5
    else:
        rank[order] = np.linspace(0.0, 1.0, len(values))

    # Tie mode: one fixed mid-ramp colour for every line (legible, not the washed
    # ramp end); otherwise colour by objective as usual.
    tie_colour = cmap(0.55)

    segs, cols, widths, alphas = [], [], [], []
    for i in order:
        if curved:
            cx, cy = _smooth_curve(x, coords[i])
        else:
            cx, cy = x, coords[i]
        segs.append(np.column_stack([cx, cy]))
        if tie_mode:
            cols.append(tie_colour)
            widths.append(1.4)                         # uniform medium weight
            alphas.append(0.55)                        # all clearly visible
        else:
            cols.append(cmap(norm(values[i])))
            widths.append(0.8 + 2.4 * rank[i] ** 2)    # 0.8 (worst) -> 3.2 (best)
            alphas.append(0.12 + 0.83 * rank[i] ** 1.5)  # 0.12 -> ~0.95
    # LineCollection takes one alpha; bake per-line alpha into the RGBA colours.
    rgba = np.array([(c[0], c[1], c[2], a) for c, a in zip(cols, alphas)])
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=widths))

    # The single best trial gets the ATT orange accent at full width, drawn on
    # top — one line that unmistakably reads "this is the winner" (mirrors how
    # the heatmaps ring the best cell). Same curve treatment as the rest. In tie
    # mode there is no winner to highlight, so we annotate the tie instead.
    if tie_mode:
        ax.legend([], [], title=f"All {len(values)} trials tied "
                  f"(objective = {lo_v:.3f})",
                  loc="lower left", fontsize=9, framealpha=0.9)
    else:
        best_i = int(order[-1])
        if curved:
            bx, by = _smooth_curve(x, coords[best_i])
        else:
            bx, by = x, coords[best_i]
        ax.plot(bx, by, color=ATT_COLORS["orange"], linewidth=3.4,
                zorder=5, solid_capstyle="round",
                label=f"Best trial (f1 = {values[best_i]:.3f})")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # 6. Vertical axis rails + per-axis ticks/labels. The final (metric) rail
    #    is drawn in the ATT orange accent so it reads as "the score axis".
    for a, p in enumerate(axes):
        is_metric = (a == n_axes - 1)
        ax.axvline(a, color=ATT_COLORS["orange"] if is_metric
                   else ATT_COLORS["gray_300"],
                   linewidth=1.6 if is_metric else 1.0, zorder=0)
        positions, labels = tick_info[a]
        for pos, lab in zip(positions, labels):
            ax.text(a, pos, f" {lab}", fontsize=8, va="center", ha="left",
                    color=ATT_COLORS["gray_700"])

    ax.set_xlim(-0.3, n_axes - 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_facecolor(ATT_COLORS["gray_100"])   # faint panel so lines read crisper
    ax.set_xticks(x)
    # Importance rank under each param axis; the metric axis gets its label only.
    ax.set_xticklabels(
        [(f"{p}\n(imp {imp.get(p, 0.0):.2f})" if a < n_axes - 1 else f"{p}")
         for a, p in enumerate(axes)], fontsize=10)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title)
    # Importance ordering applies to the param axes only (left of the metric).
    ax.text(n_axes - 1.3, 1.08, "more important \u2192", fontsize=9,
            ha="right", color=ATT_COLORS["gray_500"])

    # Colourbar maps 1:1 to the line colours; tick it at real objective values
    # so a line can be matched to its score on the bar (W&B style). In tie mode
    # there is no range to show, so we draw a swatch of the single tie colour
    # with one labelled tick at the tied value.
    if tie_mode:
        tie_norm = Normalize(vmin=lo_v - 0.5, vmax=lo_v + 0.5)
        from matplotlib.colors import ListedColormap
        cbar = fig.colorbar(ScalarMappable(norm=tie_norm,
                            cmap=ListedColormap([tie_colour])), ax=ax,
                            fraction=0.046, pad=0.02)
        cbar.set_label(metric_label)
        cbar.set_ticks([lo_v])
        cbar.ax.set_yticklabels([f"{lo_v:.3f}"])
    else:
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            fraction=0.046, pad=0.02)
        cbar.set_label(metric_label)
        cbar_ticks = np.linspace(lo_v, hi_v, 5)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_yticklabels([f"{v:.3f}" for v in cbar_ticks])

    fig.tight_layout()
    return fig


def save_optuna_parallel_coordinates(study, path, *, dpi: int = 220, **kwargs):
    """Render and save the ATT parallel-coordinates plot."""
    return save_figure(plot_optuna_parallel_coordinates(study, **kwargs), path, dpi=dpi)


def load_and_plot(storage, study_name, path=None, **kwargs):
    """Convenience: load a study from a storage URL and plot it.

    storage : e.g. "sqlite:///.../optuna_mlp.db"
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    if path:
        return save_optuna_parallel_coordinates(study, path, **kwargs)
    return plot_optuna_parallel_coordinates(study, **kwargs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--out", default="optuna_parallel_coordinates.png")
    ap.add_argument("--drop", nargs="*", default=None)
    ap.add_argument("--straight", action="store_true",
                    help="Use straight segments instead of curved lines.")
    args = ap.parse_args()
    load_and_plot(args.storage, args.study_name, path=args.out,
                  drop=args.drop, curved=not args.straight)
    print(f"Wrote {args.out}")