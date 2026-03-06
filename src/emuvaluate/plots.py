"""
plotting/extremes.py
---------------------
Visualise the best and worst emulated regions according to a chosen metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .metrics import compute_metric_all_regions, METRIC_LABELS
from .transforms import preprocess

import random
import numpy as np
import matplotlib.pyplot as plt


def plot_random_timeseries(
    scenario_data: np.ndarray,
    y_pred_ensemble: np.ndarray,
    n_rows: int = 10,
    figsize: tuple = (14, 22),
    save_path: str = None,
    dpi: int = 300,
    seed: int = None,
) -> plt.Figure:
    """
    Plot n_rows randomly sampled (ensemble, region) pairs side-by-side:
    true timeseries on the left, predicted on the right.

    Parameters
    ----------
    scenario_data   : (n_ensembles, n_timesteps, n_regions)
    y_pred_ensemble : (n_ensembles, n_timesteps, n_regions)  — same shape
    n_rows          : number of random pairs to plot (default 10)
    figsize         : figure size tuple
    save_path       : file path to save the figure (None to skip saving)
    dpi             : resolution for saved figure
    seed            : optional random seed for reproducibility

    Returns
    -------
    fig : matplotlib Figure
    """
    assert scenario_data.shape == y_pred_ensemble.shape, (
        f"Shape mismatch: scenario_data {scenario_data.shape} vs "
        f"y_pred_ensemble {y_pred_ensemble.shape}"
    )

    n_ensembles, n_timesteps, n_regions = scenario_data.shape

    if seed is not None:
        random.seed(seed)

    total = n_ensembles * n_regions
    n_rows = min(n_rows, total)  # can't sample more than available pairs
    indices = random.sample(range(total), n_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing works for single row

    for row, idx in enumerate(indices):
        i = idx // n_regions   # ensemble index
        j = idx  % n_regions   # region index

        true_ts = scenario_data[i, :, j]
        pred_ts = y_pred_ensemble[i, :, j]

        # --- Left: True ---
        ax_true = axes[row, 0]
        ax_true.plot(true_ts, color="steelblue", linewidth=0.8)
        ax_true.set_ylabel(f"e={i}, r={j}", fontsize=8, rotation=0, labelpad=40)
        ax_true.spines["top"].set_visible(False)
        ax_true.spines["right"].set_visible(False)
        if row == 0:
            ax_true.set_title("True", fontsize=11, fontweight="bold")

        # --- Right: Predicted ---
        ax_pred = axes[row, 1]
        ax_pred.plot(pred_ts, color="tomato", linewidth=0.8)
        ax_pred.spines["top"].set_visible(False)
        ax_pred.spines["right"].set_visible(False)
        if row == 0:
            ax_pred.set_title("Predicted", fontsize=11, fontweight="bold")

        # Shared y-axis range per row for fair comparison
        y_min = min(true_ts.min(), pred_ts.min())
        y_max = max(true_ts.max(), pred_ts.max())
        pad   = (y_max - y_min) * 0.05 or 0.1  # avoid zero pad for flat series
        ax_true.set_ylim(y_min - pad, y_max + pad)
        ax_pred.set_ylim(y_min - pad, y_max + pad)

        # Hide x tick labels except on the bottom row
        if row < n_rows - 1:
            ax_true.set_xticklabels([])
            ax_pred.set_xticklabels([])

    fig.suptitle(
        f"True vs Predicted — {n_rows} Random Timeseries "
        f"({n_ensembles} ensembles, {n_timesteps} timesteps, {n_regions} regions)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig

# ── Ranking ────────────────────────────────────────────────────────────────────

def rank_regions(
    metric_scores: dict[int, float],
    n_examples: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Split region scores into the *n_examples* best and worst.

    Parameters
    ----------
    metric_scores : dict mapping region index → score (lower = better)
    n_examples    : how many regions to return at each extreme

    Returns
    -------
    best_pairs  : list of (region_idx, score), ascending by score
    worst_pairs : list of (region_idx, score), descending by score
    """
    sorted_pairs = sorted(metric_scores.items(), key=lambda x: x[1])
    best_pairs  = sorted_pairs[:n_examples]
    worst_pairs = sorted_pairs[-n_examples:][::-1]
    return best_pairs, worst_pairs


# ── Single-row plotting helper ────────────────────────────────────────────────

def _plot_row(
    axes_row: np.ndarray,
    region_idx: int,
    score: float,
    true_ts: np.ndarray,
    pred_ts: np.ndarray,
    metric: str,
    label: str,
    is_first_row: bool,
    is_last_row: bool,
) -> None:
    """
    Fill one pair of axes (simulations | emulations) for a single region.

    Parameters
    ----------
    axes_row    : array of two Axes objects
    region_idx  : region index (used in ylabel)
    score       : metric value for this region
    true_ts     : (n_members, T) — ground-truth ensemble
    pred_ts     : (n_members, T) — predicted ensemble
    metric      : metric key (for ylabel label)
    label       : annotation string, e.g. "Best #1"
    is_first_row: whether to draw column titles
    is_last_row : whether to show x-tick labels
    """
    y_min = min(true_ts.min(), pred_ts.min())
    y_max = max(true_ts.max(), pred_ts.max())
    pad   = (y_max - y_min) * 0.05

    panels = [
        (true_ts, "steelblue", "Simulations"),
        (pred_ts, "tomato",    "Emulations"),
    ]

    for col, (ens, color, name) in enumerate(panels):
        ax = axes_row[col]
        for member in ens:
            ax.plot(member, color=color, linewidth=0.4, alpha=0.2)
        ax.plot(np.median(ens, axis=0), color=color, linewidth=1.5, label="Median")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(
            f"r={region_idx}\n{metric}={score:.4f}",
            fontsize=8,
            rotation=0,
            labelpad=55,
        )
        if is_first_row:
            ax.set_title(name, fontsize=11, fontweight="bold")
        if not is_last_row:
            ax.set_xticklabels([])

    # Side annotation
    axes_row[1].annotate(
        label,
        xy=(1.02, 0.5),
        xycoords="axes fraction",
        fontsize=9,
        color="grey",
        va="center",
        rotation=270,
    )


# ── Main plot function ────────────────────────────────────────────────────────

def plot_metric_extremes(
    y_pred_ensemble: np.ndarray,
    scenario_data: np.ndarray,
    n_examples: int = 5,
    yearly_average: bool = False,
    month_selection: int | None = None,
    metric: str = "crps",
    save_path: str | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
) -> dict[int, float]:
    """
    Preprocess ensembles, compute a metric for every region, then plot the
    *n_examples* best and worst regions side-by-side.

    Parameters
    ----------
    y_pred_ensemble : (n_members, T, n_regions) — emulated ensemble
    scenario_data   : (n_members, T, n_regions) — ground-truth ensemble
    n_examples      : number of best / worst regions to display
    yearly_average  : aggregate months → annual means before scoring
    month_selection : restrict to a single calendar month (1–12)
    metric          : one of 'crps', 'mean', 'sigma', 'psd'
    save_path       : if given, save figure to this path (300 dpi)
    detrend         : remove Gaussian-smoothed trend before scoring
    detrend_tau     : smoothing sigma for detrending (timesteps)
    deseasonalise   : subtract mean seasonal cycle before scoring

    Returns
    -------
    dict mapping region index → metric score
    """
    # 1. Preprocess
    preprocess_kwargs = dict(
        apply_yearly_average=yearly_average,
        month_selection=month_selection,
        apply_deseasonalise=deseasonalise,
        apply_detrend=detrend,
        detrend_tau=detrend_tau,
    )
    obs_p  = preprocess(scenario_data,   **preprocess_kwargs)
    pred_p = preprocess(y_pred_ensemble, **preprocess_kwargs)

    # 2. Score every region
    metric_scores = compute_metric_all_regions(obs_p, pred_p, metric)

    # 3. Rank
    best_pairs, worst_pairs = rank_regions(metric_scores, n_examples)

    # 4. Build figure
    n_rows = n_examples * 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_examples * 4 + 2))

    for k, (region_idx, score) in enumerate(best_pairs):
        _plot_row(
            axes_row=axes[k],
            region_idx=region_idx,
            score=score,
            true_ts=obs_p[:, :, region_idx],
            pred_ts=pred_p[:, :, region_idx],
            metric=metric,
            label=f"Best #{k + 1}",
            is_first_row=(k == 0),
            is_last_row=(k == n_rows - 1),
        )

    # Divider between best and worst
    fig.add_artist(
        mlines.Line2D(
            [0.05, 0.95],
            [1 - (n_examples / n_rows) - 0.01] * 2,
            transform=fig.transFigure,
            color="grey",
            linewidth=1,
            linestyle="--",
        )
    )

    for k, (region_idx, score) in enumerate(worst_pairs):
        row = n_examples + k
        _plot_row(
            axes_row=axes[row],
            region_idx=region_idx,
            score=score,
            true_ts=obs_p[:, :, region_idx],
            pred_ts=pred_p[:, :, region_idx],
            metric=metric,
            label=f"Worst #{k + 1}",
            is_first_row=False,
            is_last_row=(row == n_rows - 1),
        )

    # 5. Title
    if yearly_average:
        mode_str = "Yearly Average"
    elif month_selection is not None:
        mode_str = f"Month {month_selection} Only"
    else:
        mode_str = "Monthly"

    fig.suptitle(
        f"{METRIC_LABELS[metric]} Extremes [{mode_str}]"
        f" — {n_examples} Best & {n_examples} Worst Regions",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return metric_scores