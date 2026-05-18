"""
plotting/extremes.py
---------------------
Visualise the best and worst emulated regions according to a chosen metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .metrics import compute_metric_all_regions, METRIC_LABELS, compute_qq_scores_all_regions, QQ_METRIC_LABELS, ERROR_METRIC_REGISTRY, linearity_score, compute_psd_scores, compute_rank_histogram, compute_eof_scores, lagged_correlation
from .transforms import preprocess, weighted_linear_smoothing

import random
import numpy as np
import matplotlib.pyplot as plt

def plot_gmt_vs_regional(
    gmt: np.ndarray,
    regional_data: dict[str, np.ndarray],
    region_indices: list[int] | None = None,
    sample_indices: list[int] | None = None,
    n_random: int | None = None,
    region_names: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    seed: int = 42,
) -> plt.Figure:
    """
    Plot regional values on the y-axis against GMT on the x-axis for
    multiple data sources.

    Parameters
    ----------
    gmt             : (T,) or (n_samples, T) — GMT timeseries
    regional_data   : dict of name → (n_samples, T, n_regions) or (T, n_regions)
    region_indices  : list of region indices to plot. If None, uses n_random
    sample_indices  : list of sample indices to plot. If None, uses n_random
                      or all samples for 2-D inputs
    n_random        : if region_indices or sample_indices is None, sample
                      this many randomly instead
    region_names    : list of region label strings. If len matches
                      len(region_indices), treated as already-resolved labels
                      for the selected regions. Otherwise treated as full list.
    title           : optional figure title
    save_path       : path to save figure
    dpi             : resolution for saved figure
    seed            : random seed for reproducibility

    Returns
    -------
    matplotlib Figure
    """
    rng = np.random.default_rng(seed)
    gmt = np.asarray(gmt)

    # ── Normalise all inputs to (n_samples, T, n_regions) ────────────────────
    normalised = {}
    for name, data in regional_data.items():
        data = np.asarray(data)
        if data.ndim == 2:
            normalised[name] = data[np.newaxis, :, :]
        elif data.ndim == 3:
            normalised[name] = data
        else:
            raise ValueError(f"'{name}': expected 2-D or 3-D array, got shape {data.shape}")

    ref_data                  = next(iter(normalised.values()))
    n_samples_ref, T, n_regions = ref_data.shape

    gmt_3d = np.tile(gmt, (n_samples_ref, 1)) if gmt.ndim == 1 else gmt

    # ── Resolve which regions and samples to plot ─────────────────────────────
    if region_indices is None:
        n_pick         = n_random if n_random is not None else min(3, n_regions)
        region_indices = rng.choice(n_regions, size=min(n_pick, n_regions), replace=False).tolist()

    if sample_indices is None:
        n_pick         = n_random if n_random is not None else min(3, n_samples_ref)
        sample_indices = rng.choice(n_samples_ref, size=min(n_pick, n_samples_ref), replace=False).tolist()

    n_reg = len(region_indices)

    # ── Resolve labels ────────────────────────────────────────────────────────
    if region_names is not None and len(region_names) == n_reg:
        # Already-resolved labels for the selected regions
        plot_labels = list(region_names)
    else:
        full_labels = region_names if region_names is not None else [str(i) for i in range(n_regions)]
        plot_labels = [full_labels[i] for i in region_indices]

    # ── Colour per source ─────────────────────────────────────────────────────
    source_colors = {
        name: plt.get_cmap("tab10").colors[k % 10]
        for k, name in enumerate(normalised.keys())
    }

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_reg, figsize=(5 * n_reg, 4), sharey=False)
    if n_reg == 1:
        axes = [axes]

    for col, region_idx in enumerate(region_indices):
        ax = axes[col]

        for name, data in normalised.items():
            color = source_colors[name]
            n_s   = data.shape[0]

            for s_idx in sample_indices:
                if s_idx >= n_s:
                    continue
                g = gmt_3d[s_idx] if gmt_3d.shape[0] > 1 else gmt_3d[0]
                y = data[s_idx, :, region_idx]
                ax.scatter(
                    g, y,
                    color=color, s=4, alpha=0.3,
                    label=name if s_idx == sample_indices[0] else "_nolegend_",
                )

            valid_samples = [s for s in sample_indices if s < n_s]
            if valid_samples:
                g_mean   = gmt_3d[valid_samples].mean(axis=0) if gmt_3d.shape[0] > 1 else gmt_3d[0]
                y_median = np.median(data[valid_samples, :, region_idx], axis=0)
                ax.scatter(g_mean, y_median, color=color, s=12, alpha=0.9, zorder=3)

                sort_idx = np.argsort(g_mean)
                m, b     = np.polyfit(g_mean, y_median, 1)
                ax.plot(
                    g_mean[sort_idx],
                    m * g_mean[sort_idx] + b,
                    color=color, linewidth=1.5, linestyle="--", alpha=0.8,
                )

        ax.set_xlabel("GMT", fontsize=10)
        ax.set_title(plot_labels[col], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Regional value", fontsize=10)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen[l] = h
    axes[0].legend(seen.values(), seen.keys(), frameon=False, fontsize=9)

    fig.suptitle(title or "Regional values vs GMT", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_gmt_vs_regional_linearity_extremes(
    gmt: np.ndarray,
    regional_data: dict[str, np.ndarray],
    n_examples: int = 5,
    sample_indices: list[int] | None = None,
    n_random_samples: int = 5,
    region_names: list[str] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    seed: int = 42,
) -> dict:
    """
    Rank regions by linearity of their GMT–regional relationship and plot
    the n most and least linear regions using plot_gmt_vs_regional.

    Linearity is measured using the R² of a linear fit between GMT and the
    regional values of the first source in regional_data.

    Parameters
    ----------
    gmt              : (T,) or (n_samples, T)
    regional_data    : dict of name → (n_samples, T, n_regions) or (T, n_regions)
                       linearity is ranked using the first entry
    n_examples       : number of most / least linear regions to show
    sample_indices   : fixed sample indices to use; if None, uses n_random_samples
    n_random_samples : number of random samples to draw if sample_indices is None
    region_names     : list of region label strings
    save_path        : base path for saving — '_most_linear' and '_least_linear'
                       are appended before the extension
    dpi              : resolution for saved figure
    seed             : random seed

    Returns
    -------
    dict with keys:
        'linearity_scores' : np.ndarray (n_regions,)
        'most_linear'      : list of (region_idx, score)
        'least_linear'     : list of (region_idx, score)
    """
    from .metrics import linearity_score

    rng = np.random.default_rng(seed)
    gmt = np.asarray(gmt)

    ref_name, ref_data = next(iter(regional_data.items()))
    ref_data = np.asarray(ref_data)
    if ref_data.ndim == 2:
        ref_data = ref_data[np.newaxis, :, :]

    n_samples, T, n_regions = ref_data.shape
    gmt_2d = np.tile(gmt, (n_samples, 1)) if gmt.ndim == 1 else gmt

    # ── Score linearity per region ────────────────────────────────────────────
    scores = np.array([
        linearity_score(ref_data[:, :, r], gmt_2d)
        for r in range(n_regions)
    ])

    # ── Rank ──────────────────────────────────────────────────────────────────
    sorted_idx   = np.argsort(scores)
    least_linear = [(int(i), float(scores[i])) for i in sorted_idx[:n_examples]]
    most_linear  = [(int(i), float(scores[i])) for i in sorted_idx[-n_examples:][::-1]]

    if sample_indices is None:
        sample_indices = rng.choice(
            n_samples, size=min(n_random_samples, n_samples), replace=False
        ).tolist()

    def _save_path(suffix):
        if save_path is None:
            return None
        base, ext = (save_path.rsplit(".", 1) if "." in save_path else (save_path, "png"))
        return f"{base}_{suffix}.{ext}"

    # ── Plot most linear ──────────────────────────────────────────────────────
    most_region_indices = [i for i, _ in most_linear]
    most_region_labels  = [
        f"{region_names[i] if region_names else i} (R²={s:.2f})"
        for i, s in most_linear
    ]
    plot_gmt_vs_regional(
        gmt=gmt,
        regional_data=regional_data,
        region_indices=most_region_indices,
        sample_indices=sample_indices,
        region_names=most_region_labels,
        title=f"Most linear regions — ranked by R² of {ref_name} vs GMT",
        save_path=_save_path("most_linear"),
        dpi=dpi,
        seed=seed,
    )

    # ── Plot least linear ─────────────────────────────────────────────────────
    least_region_indices = [i for i, _ in least_linear]
    least_region_labels  = [
        f"{region_names[i] if region_names else i} (R²={s:.2f})"
        for i, s in least_linear
    ]
    plot_gmt_vs_regional(
        gmt=gmt,
        regional_data=regional_data,
        region_indices=least_region_indices,
        sample_indices=sample_indices,
        region_names=least_region_labels,
        title=f"Least linear regions — ranked by R² of {ref_name} vs GMT",
        save_path=_save_path("least_linear"),
        dpi=dpi,
        seed=seed,
    )

    return {
        "linearity_scores": scores,
        "most_linear":      most_linear,
        "least_linear":     least_linear,
    }

def plot_gmt_phases(
    gmt: np.ndarray,
    split_points: list[int],
    phases: list[str] | None = None,
    tau: float = 20,
    smooth: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a GMT timeseries with phase splits highlighted.

    Parameters
    ----------
    gmt          : (T,) GMT timeseries
    split_points : list of int — split indices along T
    phases       : optional list of len(split_points) + 1 phase label strings
                   e.g. ['stable', 'ramp-up', 'ramp-down', 'stable']
    tau          : smoothing bandwidth (only used if smooth=True)
    smooth       : whether to overlay the smoothed GMT
    title        : optional figure title
    save_path    : path to save figure
    dpi          : resolution for saved figure

    Returns
    -------
    matplotlib Figure
    """
    gmt = np.asarray(gmt).ravel()
    T   = len(gmt)

    PHASE_COLORS = {
        "ramp-up":   "#d62728",
        "ramp-down": "#1f77b4",
        "stable":    "#2ca02c",
    }
    DEFAULT_COLORS = plt.get_cmap("tab10").colors

    n_segments = len(split_points) + 1
    if phases is not None and len(phases) != n_segments:
        raise ValueError(
            f"len(phases)={len(phases)} must equal len(split_points)+1={n_segments}"
        )

    boundaries = [0] + list(split_points) + [T]

    y_min = gmt.min()
    y_max = gmt.max()
    y_pad = (y_max - y_min) * 0.05

    fig, ax = plt.subplots(figsize=(12, 4))

    # ── Shaded phase regions ──────────────────────────────────────────────────
    for k in range(n_segments):
        t_start = boundaries[k]
        t_end   = boundaries[k + 1]
        label   = phases[k] if phases is not None else f"Segment {k + 1}"
        color   = (
            PHASE_COLORS.get(label, DEFAULT_COLORS[k % len(DEFAULT_COLORS)])
            if phases is not None
            else DEFAULT_COLORS[k % len(DEFAULT_COLORS)]
        )
        ax.axvspan(t_start, t_end, alpha=0.12, color=color, label=label, zorder=0)

    # ── Raw GMT ───────────────────────────────────────────────────────────────
    ax.plot(
        np.arange(T), gmt,
        color="grey", linewidth=0.8, alpha=0.6, zorder=1, label="GMT (raw)"
    )

    # ── Smoothed GMT ──────────────────────────────────────────────────────────
    if smooth:
        gmt_smooth = weighted_linear_smoothing(gmt, tau=tau)
        ax.plot(
            np.arange(T), gmt_smooth,
            color="black", linewidth=1.8, zorder=2, label=f"GMT (smoothed, τ={tau})"
        )

    # ── Split point vertical lines + labels ───────────────────────────────────
    for sp in split_points:
        ax.axvline(x=sp, color="black", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)
        ax.text(
            sp, y_max + y_pad,
            f"  t={sp}",
            fontsize=8, color="black", va="top", rotation=90,
        )

    # ── Phase labels centred in each band ─────────────────────────────────────
    y_label = y_min + (y_max - y_min) * 0.05
    for k in range(n_segments):
        t_mid = (boundaries[k] + boundaries[k + 1]) / 2
        label = phases[k] if phases is not None else f"Segment {k + 1}"
        color = (
            PHASE_COLORS.get(label, DEFAULT_COLORS[k % len(DEFAULT_COLORS)])
            if phases is not None
            else DEFAULT_COLORS[k % len(DEFAULT_COLORS)]
        )
        ax.text(
            t_mid, y_label, label,
            ha="center", va="bottom", fontsize=9,
            color=color, fontweight="bold",
        )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlim(0, T - 1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 4)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("GMT", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=9, loc="upper left")

    ax.set_title(title or "GMT Timeseries — Phase Splits", pad=8, fontsize=12)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_error_metrics_bar(
    scenario_data: np.ndarray,
    y_pred_ensemble: np.ndarray,
    baseline_emulations: dict[str, np.ndarray] | None = None,
    yearly_average: bool = False,
    month_selection: int | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
    metrics: list[str] = ("mae", "rmse", "max_ae"),
    region_names: list[str] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> dict:
    """
    Plot per-region error metrics as bar charts comparing the emulator
    against one or more baseline emulation methods.

    Parameters
    ----------
    scenario_data        : (n_members, T, n_regions) — ground truth
    y_pred_ensemble      : (n_members, T, n_regions) — emulator output
    baseline_emulations  : dict of name → (n_members, T, n_regions)
    yearly_average       : aggregate months → annual means
    month_selection      : restrict to a single calendar month (1–12)
    detrend              : remove Gaussian-smoothed trend
    detrend_tau          : smoothing sigma for detrending
    deseasonalise        : subtract mean seasonal cycle
    metrics              : which metrics to plot — any of
                           'mae', 'rmse', 'max_ae', 'mse', 'nmae', 'nmse'
    region_names         : list of region label strings
    save_path            : path to save figure
    dpi                  : resolution for saved figure

    Returns
    -------
    dict mapping method name → {metric → np.ndarray of shape (n_regions,)}
    """

    for m in metrics:
        if m not in ERROR_METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{m}'. Choose from {list(ERROR_METRIC_REGISTRY)}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocess_kwargs = dict(
        apply_yearly_average=yearly_average,
        month_selection=month_selection,
        apply_deseasonalise=deseasonalise,
        apply_detrend=detrend,
        detrend_tau=detrend_tau,
    )
    obs_p  = preprocess(scenario_data,   **preprocess_kwargs)
    pred_p = preprocess(y_pred_ensemble, **preprocess_kwargs)

    baselines_p = {}
    if baseline_emulations:
        for name, data in baseline_emulations.items():
            baselines_p[name] = preprocess(data, **preprocess_kwargs)

    n_regions = obs_p.shape[2]
    labels    = region_names if region_names is not None else [str(i) for i in range(n_regions)]

    # ── Compute scores per method per metric per region ───────────────────────
    all_methods = {"Emulator": pred_p, **baselines_p}
    results     = {}

    for method_name, pred in all_methods.items():
        results[method_name] = {}
        for metric_key in metrics:
            metric_label, score_fn = ERROR_METRIC_REGISTRY[metric_key]
            results[method_name][metric_key] = np.array([
                score_fn(obs_p[:, :, j], pred[:, :, j])
                for j in range(n_regions)
            ])

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(max(12, n_regions * 0.4), 4 * n_metrics),
        sharex=True,
    )
    if n_metrics == 1:
        axes = [axes]

    method_colors = plt.get_cmap("tab10").colors
    x      = np.arange(n_regions)
    n_meth = len(all_methods)
    width  = 0.8 / n_meth

    for ax, metric_key in zip(axes, metrics):
        metric_label, _ = ERROR_METRIC_REGISTRY[metric_key]

        for k, (method_name, method_results) in enumerate(results.items()):
            offset = (k - n_meth / 2 + 0.5) * width
            scores = method_results[metric_key]
            ax.bar(
                x + offset, scores,
                width=width * 0.9,
                color=method_colors[k % len(method_colors)],
                alpha=0.85,
                label=method_name,
            )

        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5, linestyle="--")

    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    parts = []
    if yearly_average:               parts.append("Yearly Avg")
    if month_selection is not None:  parts.append(f"Month {month_selection}")
    if deseasonalise:                parts.append("Deseasonalised")
    if detrend:                      parts.append(f"Detrended (τ={detrend_tau})")
    mode_str = " | ".join(parts) if parts else "Monthly"

    fig.suptitle(
        f"Per-Region Error Metrics [{mode_str}]",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return results


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

 
    return metric_scores

def plot_spatial_correlations(
    y_pred_ensemble: np.ndarray,
    scenario_data: np.ndarray,
    yearly_average: bool = False,
    month_selection: int | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
    region_names: list[str] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> dict:
    """
    Preprocess ensembles, compute spatial correlation matrices for simulations
    and emulations, and plot them side-by-side with their difference.

    Parameters
    ----------
    y_pred_ensemble : (n_members, T, n_regions) — emulated ensemble
    scenario_data   : (n_members, T, n_regions) — ground-truth ensemble
    yearly_average  : aggregate months → annual means before scoring
    month_selection : restrict to a single calendar month (1–12)
    detrend         : remove Gaussian-smoothed trend before scoring
    detrend_tau     : smoothing sigma for detrending (timesteps)
    deseasonalise   : subtract mean seasonal cycle before scoring
    region_names    : list of M region label strings (optional)
    save_path       : if given, save figure to this path
    dpi             : resolution for saved figure

    Returns
    -------
    dict with keys 'sim_corr', 'emu_corr', 'diff', 'mae', 'rmse'
    """
    from .metrics import spatial_correlation_scores
    from .transforms import preprocess

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

    # 2. Compute correlation matrices
    results  = spatial_correlation_scores(obs_p, pred_p)
    sim_corr = results["sim_corr"]
    emu_corr = results["emu_corr"]
    diff     = results["diff"]
    mae      = results["mae"]
    rmse     = results["rmse"]

    M      = sim_corr.shape[0]
    labels = region_names if region_names is not None else [str(i) for i in range(M)]
    tick_fontsize = max(4, min(8, 80 // M))   # scale tick labels to number of regions

    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    diff_abs = max(np.abs(diff).max(), 1e-9)   # avoid zero-range colormap

    panels = [
        (sim_corr, "Simulation Correlations", "RdBu_r", -1,        1       ),
        (emu_corr, "Emulation Correlations",  "RdBu_r", -1,        1       ),
        (diff,     "Difference (Emu − Sim)",  "PiYG",   -diff_abs, diff_abs),
    ]

    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, pad=8)
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels(labels, fontsize=tick_fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 4. Mode string for title
    parts = []
    if yearly_average:                parts.append("Yearly Avg")
    if month_selection is not None:   parts.append(f"Month {month_selection}")
    if deseasonalise:                 parts.append("Deseasonalised")
    if detrend:                       parts.append(f"Detrended (τ={detrend_tau})")
    mode_str = " | ".join(parts) if parts else "Monthly"

    fig.suptitle(
        f"Spatial Correlations [{mode_str}]  —  MAE: {mae:.4f}  |  RMSE: {rmse:.4f}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return results

def _plot_qq_row(
    axes_row: np.ndarray,
    region_idx: int,
    score: float,
    obs_q: np.ndarray,
    pred_q: np.ndarray,
    quantiles: np.ndarray,
    metric: str,
    label: str,
    is_first_row: bool,
    is_last_row: bool,
) -> None:
    """
    Fill one pair of axes for a single region:
      left  — QQ scatter (sim quantiles vs emu quantiles)
      right — quantile values vs quantile level for sim and emu
    """
    ax_qq   = axes_row[0]
    ax_dist = axes_row[1]

    # ── Left: QQ scatter ──────────────────────────────────────────────────────
    q_min = min(obs_q.min(), pred_q.min())
    q_max = max(obs_q.max(), pred_q.max())
    pad   = (q_max - q_min) * 0.05

    # 1:1 reference line
    ax_qq.plot(
        [q_min - pad, q_max + pad],
        [q_min - pad, q_max + pad],
        color="grey", linewidth=0.8, linestyle="--", zorder=1, label="1:1"
    )
    
    sc = ax_qq.scatter(
        obs_q, pred_q,
        c=quantiles, cmap="plasma",
        s=12, zorder=2, linewidths=0
    )
    if is_first_row:
        ax_qq.set_title("QQ Plot (Sim vs Emu)", fontsize=11)

    ax_qq.set_xlim(q_min - pad, q_max + pad)
    ax_qq.set_ylim(q_min - pad, q_max + pad)
    ax_qq.set_xlabel("Simulated quantile", fontsize=10)
    ax_qq.set_ylabel(
        f"{label}\n{QQ_METRIC_LABELS.get(metric, metric)}={score:.4f}",
        fontsize=10, rotation=90, labelpad=15,
    )
    ax_qq.set_aspect("equal", adjustable="box")
    ax_qq.spines["top"].set_visible(False)
    ax_qq.spines["right"].set_visible(False)

    # ── Right: quantile curves ────────────────────────────────────────────────
    ax_dist.plot(quantiles, obs_q,  color="steelblue", linewidth=1.5, label="Simulations")
    ax_dist.plot(quantiles, pred_q, color="tomato",    linewidth=1.5, label="Emulations",  linestyle="--")
    ax_dist.fill_between(quantiles, obs_q, pred_q, alpha=0.15, color="grey", label="Gap")

    if is_first_row:
        ax_dist.set_title("Quantile Curves", fontsize=11)
        ax_dist.legend(frameon=False, fontsize=8)

    ax_dist.set_xlabel("Quantile level", fontsize=10)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    if not is_last_row:
        ax_qq.set_xticklabels([])
        ax_dist.set_xticklabels([])

    # Side annotation
    axes_row[1].annotate(
        label,
        xy=(1.02, 0.5), xycoords="axes fraction",
        fontsize=10, color="grey", va="center", rotation=270,
    )


def plot_qq_extremes(
    y_pred_ensemble: np.ndarray,
    scenario_data: np.ndarray,
    n_examples: int = 5,
    yearly_average: bool = False,
    month_selection: int | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
    metric: str = "mae",
    n_quantiles: int = 99,
    save_path: str | None = None,
    dpi: int = 300,
) -> dict:
    """
    Preprocess ensembles, score every region by a QQ metric, then plot the
    n_examples best and worst regions with QQ scatter and quantile curves.

    Parameters
    ----------
    y_pred_ensemble : (n_members, T, n_regions)
    scenario_data   : (n_members, T, n_regions)
    n_examples      : number of best / worst regions to display
    yearly_average  : aggregate months → annual means
    month_selection : restrict to a single calendar month (1–12)
    detrend         : remove Gaussian-smoothed trend
    detrend_tau     : smoothing sigma for detrending
    deseasonalise   : subtract mean seasonal cycle
    metric          : one of 'mae', 'nmae', 'max_ae', 'mse', 'nmse', 'ks', 'tail_mae'
    n_quantiles     : number of evenly spaced quantile levels (default 99)
    save_path       : path to save figure
    dpi             : resolution for saved figure

    Returns
    -------
    dict with keys 'scores', 'obs_qq', 'pred_qq', 'quantiles'
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

    # 2. Quantile grid + scores
    quantiles = np.linspace(0.01, 0.99, n_quantiles)
    scores, obs_qq, pred_qq = compute_qq_scores_all_regions(obs_p, pred_p, metric, quantiles)

    # 3. Rank
    best_pairs, worst_pairs = rank_regions(scores, n_examples)

    # 4. Build figure
    n_rows = n_examples * 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_examples * 5 + 2))

    # Create a shared colorbar axis at the top right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.92, 0.08, 0.015])  # [left, bottom, width, height]

    # Dummy scatter just for the colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=quantiles.min(), vmax=quantiles.max()))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Quantile level")

    for k, (region_idx, score) in enumerate(best_pairs):
        _plot_qq_row(
            axes_row=axes[k],
            region_idx=region_idx,
            score=score,
            obs_q=obs_qq[region_idx],
            pred_q=pred_qq[region_idx],
            quantiles=quantiles,
            metric=metric,
            label=f"Best #{k + 1}",
            is_first_row=(k == 0),
            is_last_row=(k == n_rows - 1),
        )

    # Divider
    fig.add_artist(
        mlines.Line2D(
            [0.05, 0.95],
            [1 - (n_examples / n_rows) - 0.01] * 2,
            transform=fig.transFigure,
            color="grey", linewidth=1, linestyle="--",
        )
    )

    for k, (region_idx, score) in enumerate(worst_pairs):
        row = n_examples + k
        _plot_qq_row(
            axes_row=axes[row],
            region_idx=region_idx,
            score=score,
            obs_q=obs_qq[region_idx],
            pred_q=pred_qq[region_idx],
            quantiles=quantiles,
            metric=metric,
            label=f"Worst #{k + 1}",
            is_first_row=False,
            is_last_row=(row == n_rows - 1),
        )

    # 5. Title
    parts = []
    if yearly_average:               parts.append("Yearly Avg")
    if month_selection is not None:  parts.append(f"Month {month_selection}")
    if deseasonalise:                parts.append("Deseasonalised")
    if detrend:                      parts.append(f"Detrended (τ={detrend_tau})")
    mode_str = " | ".join(parts) if parts else "Monthly"

    fig.suptitle(
        f"QQ Extremes [{mode_str}] — {QQ_METRIC_LABELS[metric]}"
        f" — {n_examples} Best & {n_examples} Worst Regions",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return {
        "scores":    scores,
        "obs_qq":    obs_qq,
        "pred_qq":   pred_qq,
        "quantiles": quantiles,
    }


def plot_qq_regions(
    y_pred_ensemble: np.ndarray,
    scenario_data: np.ndarray,
    regions_to_plot: list[str],
    region_names: list[str],
    yearly_average: bool = False,
    month_selection: int | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
    metric: str = "mae",
    n_quantiles: int = 99,
    save_path: str | None = None,
    dpi: int = 300,
) -> dict:
    """
    Plot QQ curves for a specified list of regions.

    Parameters
    ----------
    y_pred_ensemble : (n_members, T, n_regions)
    scenario_data   : (n_members, T, n_regions)
    regions_to_plot : list of region abbreviations e.g. ["EPO", "NAO", "SAM"]
    region_names    : full ordered list of region names (sorted(ar6_regions.abbrevs))
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

    # 2. Quantile grid + scores
    quantiles = np.linspace(0.01, 0.99, n_quantiles)
    scores, obs_qq, pred_qq = compute_qq_scores_all_regions(obs_p, pred_p, metric, quantiles)

    # 3. Resolve region indices
    region_indices = [region_names.index(r) for r in regions_to_plot]
    n_regions_plot = len(region_indices)

    # 4. Build figure
    fig, axes = plt.subplots(n_regions_plot, 2, figsize=(9, n_regions_plot * 2.5))
    if n_regions_plot == 1:
        axes = axes[np.newaxis, :]

    # colorbar
    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=quantiles.min(), vmax=quantiles.max())
    )
    sm.set_array([])

    for k, (region_abbrev, region_idx) in enumerate(zip(regions_to_plot, region_indices)):
        score = scores[region_idx]
        _plot_qq_row(
            axes_row=axes[k],
            region_idx=region_idx,
            score=score,
            obs_q=obs_qq[region_idx],
            pred_q=pred_qq[region_idx],
            quantiles=quantiles,
            metric=metric,
            label=region_abbrev,
            is_first_row=(k == 0),
            is_last_row=(k == n_regions_plot - 1),
        )

    # 5. Title
    parts = []
    if yearly_average:               parts.append("Yearly Avg")
    if month_selection is not None:  parts.append(f"Month {month_selection}")
    if deseasonalise:                parts.append("Deseasonalised")
    if detrend:                      parts.append(f"Detrended (τ={detrend_tau})")
    mode_str = " | ".join(parts) if parts else "Monthly"

    fig.suptitle(
        f"QQ Plot [{mode_str}]",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
   # fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Quantile level")
    
    # then add colorbar above the plots
    cbar_ax = fig.add_axes([0.15, -0.03, 0.3, 0.02])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Quantile level")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        
    plt.show()

    return {
        "scores":    scores,
        "obs_qq":    obs_qq,
        "pred_qq":   pred_qq,
        "quantiles": quantiles,
    }


def plot_region_ensemble_extremes(
    scenario_data: np.ndarray,
    y_pred_ensemble: np.ndarray,
    baseline_emulations: dict[str, np.ndarray] | None = None,
    n_examples: int = 5,
    yearly_average: bool = False,
    month_selection: int | None = None,
    detrend: bool = False,
    detrend_tau: float = 20,
    deseasonalise: bool = False,
    metric: str = "mae",
    ranking_mode: str = "emulator",
    region_names: list[str] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> dict:
    """
    Plot the n best and n worst regions side by side, showing simulation,
    emulation and baseline ensemble timeseries in the same panel, ranked
    by a chosen error metric.

    Parameters
    ----------
    scenario_data        : (n_members, T, n_regions) — ground truth
    y_pred_ensemble      : (n_members, T, n_regions) — emulator output
    baseline_emulations  : dict of name → (n_members, T, n_regions)
    n_examples           : number of best / worst regions to show
    yearly_average       : aggregate months → annual means
    month_selection      : restrict to a single calendar month (1–12)
    detrend              : remove Gaussian-smoothed trend
    detrend_tau          : smoothing sigma for detrending
    deseasonalise        : subtract mean seasonal cycle
    metric               : error metric key from ERROR_METRIC_REGISTRY
    ranking_mode         : how to rank regions, one of:
        'emulator'   — rank by emulator vs simulation error
        'baseline'   — rank by baseline vs simulation error
                        (uses first baseline if multiple given)
        'difference' — rank by |baseline_error - emulator_error|
                        (how much better/worse emulator is vs baseline)
    region_names         : list of region label strings
    save_path            : path to save figure
    dpi                  : resolution for saved figure

    Returns
    -------
    dict with keys 'scores', 'best_pairs', 'worst_pairs'
    """
    if metric not in ERROR_METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(ERROR_METRIC_REGISTRY)}")
    if ranking_mode not in ("emulator", "baseline", "difference"):
        raise ValueError("ranking_mode must be 'emulator', 'baseline', or 'difference'")
    if ranking_mode in ("baseline", "difference") and not baseline_emulations:
        raise ValueError(f"ranking_mode='{ranking_mode}' requires at least one baseline_emulation")

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocess_kwargs = dict(
        apply_yearly_average=yearly_average,
        month_selection=month_selection,
        apply_deseasonalise=deseasonalise,
        apply_detrend=detrend,
        detrend_tau=detrend_tau,
    )
    obs_p  = preprocess(scenario_data,   **preprocess_kwargs)
    pred_p = preprocess(y_pred_ensemble, **preprocess_kwargs)

    baselines_p = {}
    if baseline_emulations:
        for name, data in baseline_emulations.items():
            baselines_p[name] = preprocess(data, **preprocess_kwargs)

    n_regions  = obs_p.shape[2]
    labels     = region_names if region_names is not None else [str(i) for i in range(n_regions)]
    _, score_fn = ERROR_METRIC_REGISTRY[metric]

    # ── Compute scores ────────────────────────────────────────────────────────
    emu_scores  = np.array([score_fn(obs_p[:, :, j], pred_p[:, :, j]) for j in range(n_regions)])

    base_scores = {}
    for name, base_p in baselines_p.items():
        base_scores[name] = np.array([
            score_fn(obs_p[:, :, j], base_p[:, :, j]) for j in range(n_regions)
        ])

    # ── Rank ──────────────────────────────────────────────────────────────────
    if ranking_mode == "emulator":
        rank_scores = emu_scores

    elif ranking_mode == "baseline":
        first_baseline = next(iter(base_scores.values()))
        rank_scores    = first_baseline

    elif ranking_mode == "difference":
        first_baseline = next(iter(base_scores.values()))
        # Positive = emulator is worse than baseline, negative = emulator is better
        rank_scores = emu_scores - first_baseline

    best_pairs, worst_pairs = rank_regions(
        {j: float(rank_scores[j]) for j in range(n_regions)},
        n_examples
    )

    # ── Colour scheme ─────────────────────────────────────────────────────────
    METHOD_STYLES = {
        "Simulation": ("steelblue", "-",  2.0),
        "Emulator":   ("tomato",    "--", 1.8),
    }
    BASE_COLORS = plt.get_cmap("tab10").colors[2:]  # reserve first 2 for sim/emu

    # ── Plot helper ───────────────────────────────────────────────────────────
    def _plot_row(ax, region_idx, is_first_row, label):
        true_ts = obs_p[:, :, region_idx]
        pred_ts = pred_p[:, :, region_idx]

        all_data = [true_ts, pred_ts] + [b[:, :, region_idx] for b in baselines_p.values()]
        y_min    = min(d.min() for d in all_data)
        y_max    = max(d.max() for d in all_data)
        pad      = (y_max - y_min) * 0.05

        # Simulations
        color, ls, lw = METHOD_STYLES["Simulation"]
        for member in true_ts:
            ax.plot(member, color=color, linewidth=0.3, alpha=0.15)
        ax.plot(
            np.median(true_ts, axis=0), color=color,
            linewidth=lw, linestyle=ls,
            label="Simulation" if is_first_row else "_nolegend_"
        )

        # Emulator
        color, ls, lw = METHOD_STYLES["Emulator"]
        for member in pred_ts:
            ax.plot(member, color=color, linewidth=0.3, alpha=0.15)
        ax.plot(
            np.median(pred_ts, axis=0), color=color,
            linewidth=lw, linestyle=ls,
            label="Emulator" if is_first_row else "_nolegend_"
        )

        # Baselines
        for b_idx, (name, base_p) in enumerate(baselines_p.items()):
            base_ts = base_p[:, :, region_idx]
            bcolor  = BASE_COLORS[b_idx % len(BASE_COLORS)]
            for member in base_ts:
                ax.plot(member, color=bcolor, linewidth=0.3, alpha=0.15)
            ax.plot(
                np.median(base_ts, axis=0), color=bcolor,
                linewidth=1.5, linestyle=":",
                label=name if is_first_row else "_nolegend_"
            )

        # Score annotations
        score_lines = [f"emu {metric}={emu_scores[region_idx]:.3f}"]
        for name, bs in base_scores.items():
            score_lines.append(f"{name} {metric}={bs[region_idx]:.3f}")
        if ranking_mode == "difference":
            first_bs = next(iter(base_scores.values()))
            score_lines.append(f"Δ={rank_scores[region_idx]:.3f}")

        ax.text(
            0.02, 0.97,
            "\n".join(score_lines),
            transform=ax.transAxes,
            fontsize=7, va="top", color="grey"
        )

        region_label = labels[region_idx]
        ax.set_ylabel(f"{region_label}\n{label}", fontsize=8, rotation=0, labelpad=60)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = n_examples * 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_examples * 3 + 2))
    if n_rows == 1:
        axes = [axes]

    for k, (region_idx, score) in enumerate(best_pairs):
        _plot_row(axes[k], region_idx, is_first_row=(k == 0), label=f"Best #{k+1}")
        if k < n_rows - 1:
            axes[k].set_xticklabels([])

    # Divider
    fig.add_artist(
        mlines.Line2D(
            [0.05, 0.95],
            [1 - (n_examples / n_rows) - 0.01] * 2,
            transform=fig.transFigure,
            color="grey", linewidth=1, linestyle="--",
        )
    )

    for k, (region_idx, score) in enumerate(worst_pairs):
        row = n_examples + k
        _plot_row(axes[row], region_idx, is_first_row=False, label=f"Worst #{k+1}")
        if row < n_rows - 1:
            axes[row].set_xticklabels([])

    # ── Legend + title ────────────────────────────────────────────────────────
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    ranking_labels = {
        "emulator":   "Emulator vs Simulation",
        "baseline":   f"Baseline vs Simulation",
        "difference": "Δ(Baseline error − Emulator error)",
    }
    parts = []
    if yearly_average:               parts.append("Yearly Avg")
    if month_selection is not None:  parts.append(f"Month {month_selection}")
    if deseasonalise:                parts.append("Deseasonalised")
    if detrend:                      parts.append(f"Detrended (τ={detrend_tau})")
    mode_str = " | ".join(parts) if parts else "Monthly"

    metric_label, _ = ERROR_METRIC_REGISTRY[metric]
    fig.suptitle(
        f"Best & Worst Regions [{mode_str}]\n"
        f"Ranked by {metric_label} — {ranking_labels[ranking_mode]}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return {
        "scores":      rank_scores,
        "emu_scores":  emu_scores,
        "base_scores": base_scores,
        "best_pairs":  best_pairs,
        "worst_pairs": worst_pairs,
    }



def plot_psd_comparison(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    regions_to_plot: list[str] | None = None,
    fs: float = 1.0,
    nperseg: int = 256,
) -> plt.Figure:
    """
    Plot PSD comparison between simulation and emulation for selected regions.
    Shading shows 10-90th percentile spread across members.

    Parameters
    ----------
    obs_data        : (n_members, T, n_regions)
    pred_data       : (n_members, T, n_regions)
    region_names    : list of region name strings, length n_regions
    regions_to_plot : which regions to plot (default: first 6)
    fs              : sampling frequency
    nperseg         : Welch segment length
    """
    if regions_to_plot is None:
        regions_to_plot = region_names[:6]

    psd = compute_psd_scores(obs_data, pred_data, fs=fs, nperseg=nperseg)
    freqs    = psd["freqs"]
    obs_psd  = psd["obs_psd"]
    pred_psd = psd["pred_psd"]

    # convert to period in years, mask zero frequency
    mask = freqs > 0
    periods = 1 / (freqs[mask] * 12)

    n_plots = len(regions_to_plot)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, region in zip(axes, regions_to_plot):
        r = region_names.index(region)
        for data_psd, label, color in [
            (obs_psd,  "Simulation", "steelblue"),
            (pred_psd, "Emulation",  "darkorange"),
        ]:
            median = np.median(data_psd[:, mask, r], axis=0)
            low    = np.percentile(data_psd[:, mask, r], 10, axis=0)
            high   = np.percentile(data_psd[:, mask, r], 90, axis=0)
            ax.fill_between(periods, low, high, alpha=0.25, color=color)
            ax.plot(periods, median, color=color, label=label)

        # add reference period lines — inside region loop, outside data loop
        for period, linelabel, lcolor in [
            (3.0, "~ENSO (3yr)", "green"),
            (1.0, "Annual",       "grey"),
            (0.5, "Semi-annual",  "lightgrey"),
        ]:
            ax.axvline(period, color=lcolor, linestyle=":", linewidth=1, label=linelabel)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("PSD")
        ax.set_title(region)
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig

def plot_rank_histogram(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    regions_to_plot: list[str] | None = None,
) -> plt.Figure:
    if regions_to_plot is None:
        regions_to_plot = region_names[:6]

    n_pred  = pred_data.shape[0]
    ranks   = compute_rank_histogram(obs_data, pred_data)  # (n_obs, T, n_regions)

    n_plots = len(regions_to_plot)
    ncols   = 3
    nrows   = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, region in zip(axes, regions_to_plot):
        r      = region_names.index(region)
        r_ranks = ranks[:, :, r].flatten()  # (n_obs * T,)
        ax.hist(r_ranks, bins=n_pred + 1, range=(-0.5, n_pred + 0.5),
                density=True, color="steelblue", edgecolor="white")
        ax.axhline(1 / (n_pred + 1), color="red", linestyle="--", label="Uniform (ideal)")
        ax.set_title(region)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig



def plot_eof_comparison(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    n_eofs: int = 3,
) -> plt.Figure:
    """
    Plot leading EOFs side by side for simulation and emulation,
    with explained variance fractions in the title.

    Parameters
    ----------
    obs_data     : (n_members, T, n_regions)
    pred_data    : (n_members, T, n_regions)
    region_names : list of region name strings
    n_eofs       : number of EOFs to plot
    """
    eof = compute_eof_scores(obs_data, pred_data, n_eofs=n_eofs)

    fig, axes = plt.subplots(n_eofs, 2, figsize=(14, 4 * n_eofs))
    if n_eofs == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_eofs):
        for ax, eofs, var_ratio, label, color in zip(
            axes[i],
            [eof["obs_eofs"],  eof["pred_eofs"]],
            [eof["obs_var_ratio"], eof["pred_var_ratio"]],
            ["Simulation", "Emulation"],
            ["steelblue",  "darkorange"],
        ):
            ax.bar(range(len(region_names)), eofs[i], color=color, alpha=0.7)
            ax.set_xticks(range(len(region_names)))
            ax.set_xticklabels(region_names, rotation=90, fontsize=6)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{label} EOF{i+1} ({var_ratio[i]*100:.1f}% variance explained)")

    plt.tight_layout()
    return fig


    
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
import regionmask
import xarray as xr

def plot_eof_comparison_map(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    n_eofs: int = 3,
    cmap: str = "RdBu_r",
    projection=None,
) -> plt.Figure:
    eof = compute_eof_scores(obs_data, pred_data, n_eofs=n_eofs)

    if projection is None:
        projection = ccrs.Robinson()

    ar6 = regionmask.defined_regions.ar6.all
    abbrevs = sorted(ar6.abbrevs)

    # Enforce that your region_names = sorted(ar6.abbrevs)
    if list(region_names) != abbrevs:
        raise ValueError(
            "region_names must be exactly sorted(ar6.abbrevs).\n"
            f"Got: {region_names}\n"
            f"Expected: {abbrevs}"
        )

    # map abbrev -> index in your data
    abbrev_to_idx = {abbr: i for i, abbr in enumerate(region_names)}
    # map abbrev -> AR6 region number (not loop index)
    abbrev_to_number = {abbr: ar6[abbr].number for abbr in abbrevs}

    # dummy grid
    ds = xr.Dataset(
        coords={
            "lon": np.linspace(-179.5, 179.5, 720),
            "lat": np.linspace(-89.5, 89.5, 360),
        }
    )
    mask = ar6.mask(ds)

    fig, axes = plt.subplots(
        n_eofs, 2,
        figsize=(16, 5 * n_eofs),
        subplot_kw={"projection": projection},
    )
    if n_eofs == 1:
        axes = np.array([axes])

    last_im = None

    for i in range(n_eofs):
        for ax, eofs, var_ratio, title_prefix in zip(
            axes[i],
            [eof["obs_eofs"], eof["pred_eofs"]],
            [eof["obs_var_ratio"], eof["pred_var_ratio"]],
            ["Simulation", "Emulation"],
        ):
            vals = np.array([eofs[i, abbrev_to_idx[a]] for a in abbrevs], dtype=float)
            vmax = np.nanmax(np.abs(vals))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            # initialize with NaNs
            region_values = xr.full_like(mask, np.nan, dtype=float)

            # assign each region by its true AR6 number, not loop index
            for abbr, val in zip(abbrevs, vals):
                region_values = region_values.where(mask != abbrev_to_number[abbr], val)

            last_im = ax.pcolormesh(
                ds.lon,
                ds.lat,
                region_values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="auto",
            )

            ar6.plot(
                ax=ax,
                add_label=False,
                add_coastlines=True,
                add_ocean=False,
                add_land=False,
                line_kws={"color": "black", "linewidth": 0.35},
            )
            ax.set_title(f"{title_prefix} EOF{i+1} ({var_ratio[i]*100:.1f}% variance explained)")
            ax.set_global()

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # bottom 12%; minimal top
    
    pos = axes[-1, 0].get_position()
    
    cbar_height = 0.02
    cbar_width  = pos.width
    cbar_bottom = 0.05
    cbar_left   = pos.x0 + cbar_width/2
    
    cbar_ax = fig.add_axes([
        cbar_left,
        cbar_bottom,
        cbar_width,
        cbar_height,
    ])
    fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal", label="EOF loading")
    
    return fig



def plot_teleconnection_comparison(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    index_regions: list[str],
    target_regions: list[str] | None = None,
) -> plt.Figure:
    """
    Compare instantaneous teleconnection patterns between simulation and emulation,
    after removing the ensemble mean (forced response) to isolate internal variability.

    obs_data, pred_data: (n_members, T, n_regions)
    index_regions: driving regions (e.g. ["EPO", "EAO"])
    target_regions: regions to correlate against (default: all)
    """
    # remove ensemble mean to isolate internal variability
    obs_anom  = obs_data  - obs_data.mean(axis=0, keepdims=True)
    pred_anom = pred_data - pred_data.mean(axis=0, keepdims=True)

    # pool across members
    n_members, T, n_regions = obs_data.shape
    obs_flat  = obs_anom.reshape(n_members * T, n_regions)
    pred_flat = pred_anom.reshape(n_members * T, n_regions)

    if target_regions is None:
        target_regions = region_names

    n_plots = len(index_regions)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, index_region in zip(axes, index_regions):
        x = np.arange(len(target_regions))
        idx = region_names.index(index_region)

        obs_corrs  = np.array([
            np.corrcoef(obs_flat[:, idx], obs_flat[:, region_names.index(t)])[0, 1]
            for t in target_regions
            if t != index_region  # exclude self-correlation
        ])
        pred_corrs = np.array([
            np.corrcoef(pred_flat[:, idx], pred_flat[:, region_names.index(t)])[0, 1]
            for t in target_regions
            if t != index_region
        ])

        target_regions_plot = [t for t in target_regions if t != index_region]
        x = np.arange(len(target_regions_plot))

        ax.bar(x - 0.2, obs_corrs,  0.4, label="Simulation", color="steelblue",  alpha=0.7)
        ax.bar(x + 0.2, pred_corrs, 0.4, label="Emulation",  color="darkorange", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(target_regions_plot, rotation=90, fontsize=6)
        ax.set_title(f"Teleconnections from {index_region} (ensemble-mean removed)")
        ax.set_ylabel("Correlation")
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig
    


def plot_lagged_correlations(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    region_names: list[str],
    index_region: str,
    target_regions: list[str] | None = None,
    max_lag: int = 24,
) -> plt.Figure:
    # remove ensemble mean
    obs_anom  = obs_data  - obs_data.mean(axis=0, keepdims=True)
    pred_anom = pred_data - pred_data.mean(axis=0, keepdims=True)
    n_members, T, n_regions = obs_data.shape
    
    if target_regions is None:
        target_regions = region_names[:6]

    idx  = region_names.index(index_region)
    lags = np.arange(max_lag + 1)

    n_plots = len(target_regions)
    ncols   = 3
    nrows   = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, target in zip(axes, target_regions):
        t = region_names.index(target)

        obs_lc  = np.mean([
            lagged_correlation(obs_anom[m, :, idx], obs_anom[m, :, t], max_lag)
            for m in range(obs_anom.shape[0])
        ], axis=0)

        pred_lc = np.mean([
            lagged_correlation(pred_anom[m, :, idx], pred_anom[m, :, t], max_lag)
            for m in range(pred_anom.shape[0])
        ], axis=0)

        ax.plot(lags, obs_lc,  color="steelblue",  label="Simulation")
        ax.plot(lags, pred_lc, color="darkorange", label="Emulation")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline( 1.96 / np.sqrt(T), color="steelblue", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(-1.96 / np.sqrt(T), color="steelblue", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"{index_region} → {target} (ensemble-mean removed)")
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig

