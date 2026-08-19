"""
metrics.py
----------
Every *number* in this package is computed here. Nothing in this module
draws anything, and nothing in `plots.py` computes a score — the two
communicate exclusively through the `ErrorData` objects (and the two plain
dicts) built at the bottom of this file.

Layout
------
1. Scalar per-unit scorers  — one score from one unit's
   ``(n_members, T)`` simulation/comparison pair: MAE, RMSE, NMAE, ...,
   plus CRPS.
2. Curve-based metric families — the per-unit quantity is a *curve*, and
   the score is the distance between the simulated and the emulated curve:
     * quantile curves            -> ``qq_mae``, ``qq_nmae``, ``qq_ks``, ``qq_tail_mae``
     * autocorrelation curves     -> ``temporal_corr_nmae``
     * power-spectral-density     -> ``psd_nmae``, ``psd_log_nmae``, ``psd_wasserstein``
3. `ErrorData` — the single container every map / timeseries / ranking plot
   consumes: per-unit scores for every comparison and every indicator, a
   shared colour-scale max, an optional Emulator-minus-baseline difference,
   and a full ranking of units.
4. `build_error_data_*` — the builders that produce it.

"Unit" means an AR6 region (regional data) or a gridpoint (gridded data);
"indicator" means a climate variable such as ``tas`` or ``pr``. Every
builder is indicator-generic: pass ``{indicator: array}`` dicts and you get
back an `ErrorData` with one column/row per indicator, in the order the dict
gives them. The defaults label ``tas`` and ``pr`` nicely; any other key is
labelled by its own uppercased name unless you override it.

Lower is better for every metric registered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    # scalar scorers
    "mae_score", "rmse_score", "max_ae_score", "mse_score", "nmae_score", "nmse_score",
    "crps_pairwise", "crps_map_score", "crps_timeseries_score",
    # curve helpers
    "compute_quantiles", "compute_temporal_correlation_curve", "compute_psd_curves",
    # registries
    "ERROR_METRIC_REGISTRY", "QQ_VARIABILITY_METRIC_REGISTRY",
    "TEMPORAL_CORRELATION_METRIC_REGISTRY", "PSD_METRIC_REGISTRY",
    "available_metrics",
    # correlation
    "spatial_correlation_matrix", "spatial_correlation_scores",
    # containers + builders
    "RankingResult", "ErrorData",
    "build_error_data_regional", "build_error_data_gridded",
    "build_crps_timeseries_data_gridded", "build_correlation_data",
    "build_error_data_intervariable_correlation",
    "build_error_data_intervariable_correlation_values",
    # defaults
    "DEFAULT_INDICATORS", "DEFAULT_QQ_QUANTILES",
    "DEFAULT_TEMPORAL_CORR_N_LAGS", "DEFAULT_PSD_NPERSEG",
]


# ═════════════════════════════════════════════════════════════════════════════
# Indicator defaults
# ═════════════════════════════════════════════════════════════════════════════
#
# Two indicators named "tas" and "pr" are the default everywhere, matching
# what the paper figures expect. Any other indicator name works too — it is
# labelled by its uppercased key and gets no unit string unless you pass one.

DEFAULT_INDICATORS: tuple[str, ...] = ("tas", "pr")

BUILTIN_INDICATOR_LABELS: dict[str, str] = {
    "tas": "Temperature (TAS)",
    "pr": "Precipitation (PR)",
}

BUILTIN_INDICATOR_UNITS: dict[str, str] = {
    "tas": "K",
    "pr": "mm day\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}",
}


def resolve_indicators(
    sim: dict,
    comparisons: dict[str, dict],
    indicators: list[str] | None,
    indicator_labels: dict[str, str] | None,
    indicator_units: dict[str, str] | None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Work out which indicators to score, and how to label them.

    Parameters
    ----------
    sim         : ``{indicator: array}`` — the simulation side
    comparisons : ``{comparison_name: {indicator: array}}`` — Emulator + baselines
    indicators  : explicit subset/order, or None to use ``list(sim)``
    indicator_labels : per-indicator long titles, overriding the built-in ones
    indicator_units  : per-indicator unit strings, overriding the built-in ones

    Returns
    -------
    (indicators, labels, units)
    """
    if not isinstance(sim, dict):
        raise TypeError(
            "sim must be a {indicator: array} dict, e.g. {'tas': tas_sim, 'pr': pr_sim}"
        )
    inds = list(sim) if indicators is None else list(indicators)
    if not inds:
        raise ValueError("at least one indicator is required")

    for name, arrays in comparisons.items():
        missing = [i for i in inds if i not in arrays]
        if missing:
            raise ValueError(f"comparison '{name}' is missing indicator(s) {missing}")
    missing_sim = [i for i in inds if i not in sim]
    if missing_sim:
        raise ValueError(f"sim is missing indicator(s) {missing_sim}")

    labels = {i: BUILTIN_INDICATOR_LABELS.get(i, i.upper()) for i in inds}
    labels.update({k: v for k, v in (indicator_labels or {}).items() if k in labels})
    units = {i: BUILTIN_INDICATOR_UNITS.get(i, "") for i in inds}
    units.update({k: v for k, v in (indicator_units or {}).items() if k in units})
    return inds, labels, units


# ═════════════════════════════════════════════════════════════════════════════
# 1. Scalar per-unit scorers
# ═════════════════════════════════════════════════════════════════════════════
#
# Calling convention for all of them:
#     f(obs, pred) -> float,  obs/pred both (n_members, T) for one unit.


def mae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error, over every member and timestep."""
    return float(np.mean(np.abs(obs - pred)))


def rmse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error between the two ensemble means."""
    return float(np.sqrt(np.mean((obs.mean(axis=0) - pred.mean(axis=0)) ** 2)))


def max_ae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Maximum absolute error between the two ensemble means."""
    return float(np.max(np.abs(obs.mean(axis=0) - pred.mean(axis=0))))


def mse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error between the two ensemble means."""
    return float(np.mean((obs.mean(axis=0) - pred.mean(axis=0)) ** 2))


def nmae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """MAE normalised by the full range of the observed values."""
    obs_range = np.ptp(obs)
    return float(mae_score(obs, pred) / obs_range) if obs_range > 0 else 0.0


def nmse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """MSE normalised by the variance of the observed ensemble mean."""
    obs_var = obs.mean(axis=0).var()
    return float(mse_score(obs, pred) / obs_var) if obs_var > 0 else 0.0


def mean_bias_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Absolute difference between the two grand means."""
    return float(np.abs(obs.mean() - pred.mean()))


def sigma_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Absolute difference between the two overall standard deviations."""
    return float(np.abs(obs.std() - pred.std()))


# ── CRPS ─────────────────────────────────────────────────────────────────────

def crps_pairwise(pred_members: np.ndarray, obs_members: np.ndarray) -> np.ndarray:
    """
    Empirical, multi-truth-generalised ensemble CRPS, vectorised over any
    number of trailing axes.

    There is no single deterministic "truth" when both the simulation and
    the emulator (or a baseline) are ensembles, so every simulation member
    is treated as an independent verification draw and averaged over. For a
    forecast/emulator ensemble {X_i} and a "truth" ensemble {Y_k}:

        CRPS = mean_{i,k} |X_i - Y_k|  -  0.5 * mean_{i,i'} |X_i - X_i'|

    This reduces to the standard empirical ensemble CRPS when there is a
    single true observation, and is the natural multi-member generalisation.

    Parameters
    ----------
    pred_members : (n_pred, ...) — forecast/emulator ensemble, ensemble axis first
    obs_members  : (n_obs, ...)  — "truth" ensemble, ensemble axis first

    Returns
    -------
    np.ndarray with the shape of the trailing axes. Clipped at 0 (the
    finite-sample estimator can dip very slightly negative; true CRPS never
    is).
    """
    term1 = np.abs(pred_members[:, None, ...] - obs_members[None, :, ...]).mean(axis=(0, 1))
    term2 = 0.5 * np.abs(pred_members[:, None, ...] - pred_members[None, :, ...]).mean(axis=(0, 1))
    return np.clip(term1 - term2, 0, None)


def crps_map_score(sim_unit: np.ndarray, pred_unit: np.ndarray) -> float:
    """
    CRPS for one unit, averaged over time. Same ``(obs, pred) -> float``
    convention as every other scalar scorer, so ``metric="crps"`` works in
    `build_error_data_regional` / `build_error_data_gridded` — and therefore
    in `plots.plot_map_regional` / `plot_map_gridded` — exactly like "mae".

    Parameters
    ----------
    sim_unit, pred_unit : (n_members, T)
    """
    return float(crps_pairwise(pred_unit, sim_unit).mean())


def crps_timeseries_score(sim_slice: np.ndarray, pred_slice: np.ndarray) -> float:
    """
    CRPS for one timestep, averaged over all units.

    Parameters
    ----------
    sim_slice, pred_slice : (n_members, n_units)
    """
    return float(crps_pairwise(pred_slice, sim_slice).mean())


ERROR_METRIC_REGISTRY: dict[str, tuple[str, callable]] = {
    "mae": ("MAE", mae_score),
    "rmse": ("RMSE", rmse_score),
    "max_ae": ("Max AE", max_ae_score),
    "mse": ("MSE", mse_score),
    "nmae": ("NMAE", nmae_score),
    "nmse": ("NMSE", nmse_score),
    "mean_bias": ("Mean Difference", mean_bias_score),
    "sigma": ("Sigma Difference", sigma_score),
    "crps": ("CRPS", crps_map_score),
}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Curve-based metric families
# ═════════════════════════════════════════════════════════════════════════════
#
# Each family has: a ``compute_*_curve`` function turning one unit's
# (n_members, T) ensemble into a curve, and one or more scorers taking two
# such curves and returning a distance. `_score_all_units` dispatches on
# which registry the metric name lives in.


# ── Quantile ("QQ") curves ───────────────────────────────────────────────────

DEFAULT_QQ_QUANTILES = np.linspace(0, 1, 101)[1:-1]  # 99 evenly spaced percentiles


def compute_quantiles(data: np.ndarray, quantiles: np.ndarray = DEFAULT_QQ_QUANTILES) -> np.ndarray:
    """
    Quantile curve for one unit, pooling every member and timestep.

    Parameters
    ----------
    data      : (n_members, T) — one unit's ensemble
    quantiles : (Q,) quantile levels in [0, 1]

    Returns
    -------
    np.ndarray, shape (Q,)
    """
    return np.quantile(np.asarray(data).ravel(), quantiles)


def qq_mae(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Mean absolute error between two quantile curves."""
    return float(np.mean(np.abs(obs_q - pred_q)))


def qq_nmae(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """QQ-MAE normalised by the span (range) of the observed quantiles."""
    span = np.ptp(obs_q)
    return float(np.mean(np.abs(obs_q - pred_q)) / span) if span > 0 else 0.0


def qq_ks(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """
    Maximum absolute difference between the two quantile curves — the
    Kolmogorov–Smirnov statistic evaluated on the quantile grid. A
    distribution-free measure of the largest distributional shift.
    """
    return float(np.max(np.abs(obs_q - pred_q)))


def qq_tail_mae(obs_q: np.ndarray, pred_q: np.ndarray, tail_frac: float = 0.1) -> float:
    """
    QQ-MAE restricted to the upper and lower *tail_frac* of the quantile
    grid — highlights errors in the extremes rather than the bulk.
    """
    q = len(obs_q)
    n_tail = max(1, int(q * tail_frac))
    idx = list(range(n_tail)) + list(range(q - n_tail, q))
    return float(np.mean(np.abs(obs_q[idx] - pred_q[idx])))


QQ_VARIABILITY_METRIC_REGISTRY: dict[str, tuple[str, callable]] = {
    "qq_mae": ("QQ-MAE", qq_mae),
    "qq_nmae": ("QQ-NMAE", qq_nmae),
    "qq_ks": ("QQ KS statistic", qq_ks),
    "qq_tail_mae": ("QQ Tail MAE", qq_tail_mae),
}


# ── Temporal autocorrelation curves ──────────────────────────────────────────

DEFAULT_TEMPORAL_CORR_N_LAGS = 24


def compute_temporal_correlation_curve(
    unit_series: np.ndarray,
    n_lags: int = DEFAULT_TEMPORAL_CORR_N_LAGS,
    window: int | None = None,
) -> np.ndarray:
    """
    Autocorrelation curve for one unit: for every lag 0..n_lags, the
    correlation between a timestep's value and the value `lag` steps
    earlier — computed per ensemble member using only the last `window`
    timesteps, then averaged across members.

    Parameters
    ----------
    unit_series : (n_members, T) — one unit's ensemble timeseries
    n_lags      : number of lags (curve has n_lags + 1 points, lag 0..n_lags)
    window      : restrict to the last `window` timesteps (default: all of T)

    Returns
    -------
    np.ndarray, shape (n_lags + 1,)
    """
    from statsmodels.tsa.stattools import acf

    series = unit_series if window is None else unit_series[:, -window:]
    t = series.shape[1]
    if n_lags >= t:
        # statsmodels' acf() silently returns fewer than n_lags + 1 points
        # instead of raising when there isn't enough data — fail loudly here
        # instead of letting a wrong-length curve propagate.
        raise ValueError(
            f"n_lags ({n_lags}) must be smaller than the series length ({t}); "
            "reduce n_lags or increase window."
        )
    curves = [acf(series[m], nlags=n_lags, fft=True) for m in range(series.shape[0])]
    return np.mean(curves, axis=0)


def temporal_corr_nmae(obs_curve: np.ndarray, pred_curve: np.ndarray) -> float:
    """NMAE between two autocorrelation curves, normalised by the observed span."""
    span = np.ptp(obs_curve)
    return float(np.mean(np.abs(obs_curve - pred_curve)) / span) if span > 0 else 0.0


TEMPORAL_CORRELATION_METRIC_REGISTRY: dict[str, tuple[str, callable]] = {
    "temporal_corr_nmae": ("Temporal-Corr NMAE", temporal_corr_nmae),
}


# ── Power spectral density (PSD) curves ──────────────────────────────────────

DEFAULT_PSD_NPERSEG = 256


def compute_psd_curves(
    unit_series: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-member power spectral density for one unit, via Welch's method.

    Every member is transformed separately, so the spread across members is
    a genuine sampling uncertainty that `plots.plot_psd_curves` can shade.
    The *score* (see `psd_nmae` and friends) uses the member-mean curve.

    Parameters
    ----------
    unit_series : (n_members, T) — one unit's ensemble timeseries
    fs          : sampling frequency, in samples per time unit. With monthly
                  data and fs=1 the frequency axis is cycles/month; pass
                  fs=12 to get cycles/year.
    nperseg     : Welch segment length. Longer = finer frequency resolution,
                  noisier estimate. Defaults to ``min(DEFAULT_PSD_NPERSEG, T)``.
    detrend     : passed through to `scipy.signal.welch` ("constant" removes
                  each segment's mean, which is what you want for a
                  variability spectrum).

    Returns
    -------
    (freqs, psd) : freqs is (n_freqs,), psd is (n_members, n_freqs)
    """
    from scipy.signal import welch

    series = np.asarray(unit_series)
    if series.ndim == 1:
        series = series[np.newaxis, :]
    t = series.shape[1]
    nperseg = min(DEFAULT_PSD_NPERSEG, t) if nperseg is None else min(nperseg, t)
    freqs, psd = welch(series, fs=fs, nperseg=nperseg, detrend=detrend, axis=-1)
    return freqs, np.atleast_2d(psd)


def compute_psd_curve(
    unit_series: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> np.ndarray:
    """Member-mean PSD curve for one unit — the curve the PSD metrics score."""
    _, psd = compute_psd_curves(unit_series, fs=fs, nperseg=nperseg)
    return psd.mean(axis=0)


def psd_nmae(obs_psd: np.ndarray, pred_psd: np.ndarray) -> float:
    """
    NMAE between two PSD curves, normalised by the span of the observed
    curve. Because a PSD usually spans orders of magnitude, this is
    dominated by whichever frequencies carry the most power — use
    `psd_log_nmae` if you want every frequency band weighted comparably.
    """
    span = np.ptp(obs_psd)
    return float(np.mean(np.abs(obs_psd - pred_psd)) / span) if span > 0 else 0.0


def psd_log_nmae(obs_psd: np.ndarray, pred_psd: np.ndarray) -> float:
    """
    NMAE between the log10 PSD curves, normalised by the observed log-span.
    Weights every frequency band comparably, so an emulator that gets the
    low-frequency power right but the high-frequency tail wrong is still
    penalised. Usually the more informative of the two PSD NMAEs.
    """
    eps = 1e-30
    obs_l = np.log10(np.maximum(obs_psd, eps))
    pred_l = np.log10(np.maximum(pred_psd, eps))
    span = np.ptp(obs_l)
    return float(np.mean(np.abs(obs_l - pred_l)) / span) if span > 0 else 0.0


def psd_wasserstein(obs_psd: np.ndarray, pred_psd: np.ndarray) -> float:
    """
    Wasserstein ("earth mover's") distance between the two PSD curves
    treated as distributions of power. Sensitive to power being placed at
    the *wrong frequency* rather than merely being the wrong size.
    """
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(obs_psd, pred_psd))


PSD_METRIC_REGISTRY: dict[str, tuple[str, callable]] = {
    "psd_nmae": ("PSD NMAE", psd_nmae),
    "psd_log_nmae": ("log-PSD NMAE", psd_log_nmae),
    "psd_wasserstein": ("PSD Wasserstein distance", psd_wasserstein),
}


def available_metrics() -> dict[str, list[str]]:
    """Every metric name accepted by the `build_error_data_*` builders, by family."""
    return {
        "scalar": list(ERROR_METRIC_REGISTRY),
        "quantile_curve": list(QQ_VARIABILITY_METRIC_REGISTRY),
        "autocorrelation_curve": list(TEMPORAL_CORRELATION_METRIC_REGISTRY),
        "psd_curve": list(PSD_METRIC_REGISTRY),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Spatial correlation (regional only)
# ═════════════════════════════════════════════════════════════════════════════

def spatial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Mean across-unit correlation matrix, averaged over ensemble members.

    Parameters
    ----------
    data : (n_members, T, n_units)

    Returns
    -------
    np.ndarray, shape (n_units, n_units)
    """
    return np.array([np.corrcoef(data[n].T) for n in range(data.shape[0])]).mean(axis=0)


def spatial_correlation_scores(obs_data: np.ndarray, pred_data: np.ndarray) -> dict:
    """
    Simulated and predicted spatial correlation matrices plus their
    difference and summary errors.

    Returns
    -------
    dict with keys 'sim_corr', 'emu_corr', 'diff', 'mae', 'rmse'
    """
    sim_corr = spatial_correlation_matrix(obs_data)
    emu_corr = spatial_correlation_matrix(pred_data)
    diff = emu_corr - sim_corr
    return {
        "sim_corr": sim_corr,
        "emu_corr": emu_corr,
        "diff": diff,
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. Shared scoring dispatch
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CurveOptions:
    """Extra knobs the curve-based metric families need. Ignored by scalar metrics."""
    quantiles: np.ndarray = field(default_factory=lambda: DEFAULT_QQ_QUANTILES)
    n_lags: int = DEFAULT_TEMPORAL_CORR_N_LAGS
    window: int | None = None
    psd_fs: float = 1.0
    psd_nperseg: int | None = None


def _score_all_units(sim: np.ndarray, pred: np.ndarray, metric: str, n_units: int,
                     opts: CurveOptions) -> np.ndarray:
    """
    Score every unit along the last axis, dispatching to whichever metric
    family *metric* belongs to.

    Parameters
    ----------
    sim, pred : (n_members, T, n_units)
    metric    : a key in any of the four registries
    n_units   : sim.shape[2]
    opts      : curve-family options

    Returns
    -------
    np.ndarray, shape (n_units,)
    """
    if metric in ERROR_METRIC_REGISTRY:
        _, fn = ERROR_METRIC_REGISTRY[metric]
        return np.array([fn(sim[:, :, j], pred[:, :, j]) for j in range(n_units)])

    if metric in QQ_VARIABILITY_METRIC_REGISTRY:
        _, fn = QQ_VARIABILITY_METRIC_REGISTRY[metric]
        return np.array([
            fn(compute_quantiles(sim[:, :, j], opts.quantiles),
               compute_quantiles(pred[:, :, j], opts.quantiles))
            for j in range(n_units)
        ])

    if metric in TEMPORAL_CORRELATION_METRIC_REGISTRY:
        _, fn = TEMPORAL_CORRELATION_METRIC_REGISTRY[metric]
        return np.array([
            fn(compute_temporal_correlation_curve(sim[:, :, j], opts.n_lags, opts.window),
               compute_temporal_correlation_curve(pred[:, :, j], opts.n_lags, opts.window))
            for j in range(n_units)
        ])

    if metric in PSD_METRIC_REGISTRY:
        _, fn = PSD_METRIC_REGISTRY[metric]
        return np.array([
            fn(compute_psd_curve(sim[:, :, j], opts.psd_fs, opts.psd_nperseg),
               compute_psd_curve(pred[:, :, j], opts.psd_fs, opts.psd_nperseg))
            for j in range(n_units)
        ])

    raise ValueError(f"Unknown metric '{metric}'. Available: {available_metrics()}")


def _metric_label(metric: str) -> str:
    for registry in (ERROR_METRIC_REGISTRY, QQ_VARIABILITY_METRIC_REGISTRY,
                     TEMPORAL_CORRELATION_METRIC_REGISTRY, PSD_METRIC_REGISTRY):
        if metric in registry:
            return registry[metric][0]
    raise ValueError(f"Unknown metric '{metric}'. Available: {available_metrics()}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. ErrorData
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RankingResult:
    """
    Units ordered best (lowest score) to worst, for one indicator.

    `order` is the full ranking, so a plot can show the best/median/worst
    three units (the default) *or* any explicit subset — e.g. "ranks 1-5",
    via `unit_at_rank`.
    """
    scores: np.ndarray        # (n_units,) — the scores that were ranked
    order: np.ndarray         # (n_units,) unit indices, ascending score
    ranking_label: str        # e.g. "Emulator" or a baseline name

    @property
    def best(self) -> int:
        return int(self.order[0])

    @property
    def median(self) -> int:
        return int(self.order[len(self.order) // 2])

    @property
    def worst(self) -> int:
        return int(self.order[-1])

    @property
    def n_units(self) -> int:
        return int(self.order.size)

    def unit_at_rank(self, rank: int) -> int:
        """Unit index at 1-based *rank* (1 = best, n_units = worst)."""
        if not 1 <= rank <= self.n_units:
            raise ValueError(f"rank {rank} out of range 1..{self.n_units}")
        return int(self.order[rank - 1])

    def rank_of(self, unit_idx: int) -> int:
        """1-based rank of *unit_idx*."""
        return int(np.flatnonzero(self.order == unit_idx)[0]) + 1


def _rank_units(scores: np.ndarray, ranking_label: str) -> RankingResult:
    # ascending = best first (lower is better for every metric here); NaNs sort last
    order = np.argsort(np.where(np.isnan(scores), np.inf, scores), kind="stable")
    return RankingResult(scores=scores, order=order, ranking_label=ranking_label)


@dataclass
class ErrorData:
    """
    Everything the map / bar / timeseries / ranking plots in `plots.py` need
    for one metric, across every comparison (Emulator + baselines) and every
    indicator.

    `unit_labels` is set for regional data (AR6 region abbreviations);
    `lat`/`lon` are set for gridded data instead. Use `label_for(idx)` rather
    than reading either directly.

    Attributes
    ----------
    metric, metric_label : the metric key and its human-readable name
    comparisons  : ordered comparison names, "Emulator" first
    indicators   : ordered indicator keys, e.g. ["tas", "pr"]
    indicator_labels / indicator_units : display strings per indicator
    scores       : scores[comparison][indicator] -> (n_units,), always flat
    vmax         : vmax[indicator] -> shared colour-scale maximum
    ranking      : ranking[indicator] -> RankingResult
    diff         : diff[indicator] -> (n_units,), Emulator minus diff_baseline
    """
    metric: str
    metric_label: str
    comparisons: list[str]
    indicators: list[str]
    scores: dict[str, dict[str, np.ndarray]]
    vmax: dict[str, float]
    ranking: dict[str, RankingResult]
    indicator_labels: dict[str, str]
    indicator_units: dict[str, str]
    unit_labels: list[str] | None = None
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    diff: dict[str, np.ndarray] | None = None
    diff_baseline: str | None = None
    diff_vmax: dict[str, float] | None = None

    @property
    def n_lat(self) -> int | None:
        return None if self.lat is None else self.lat.size

    @property
    def n_lon(self) -> int | None:
        return None if self.lon is None else self.lon.size

    @property
    def n_units(self) -> int:
        return int(self.scores[self.comparisons[0]][self.indicators[0]].size)

    @property
    def is_gridded(self) -> bool:
        return self.lat is not None

    def label_for(self, idx: int) -> str:
        """Region abbreviation (regional) or ``(lat°, lon°)`` string (gridded)."""
        if self.unit_labels is not None:
            return self.unit_labels[idx]
        lat_idx, lon_idx = np.unravel_index(idx, (self.n_lat, self.n_lon))
        return f"({self.lat[lat_idx]:.2f}\N{DEGREE SIGN}, {self.lon[lon_idx]:.2f}\N{DEGREE SIGN})"

    def short_label(self, indicator: str) -> str:
        """Compact indicator label used in panel titles and axis labels."""
        return indicator.upper()

    def index_of_unit(self, name) -> int:
        """
        Unit index for a region abbreviation (regional), a ``(lat, lon)``
        pair or a ``label_for``-style string (gridded).
        """
        if isinstance(name, (int, np.integer)):
            return int(name)
        if self.unit_labels is not None:
            if name not in self.unit_labels:
                raise ValueError(f"unknown unit '{name}'; expected one of {self.unit_labels}")
            return self.unit_labels.index(name)
        if isinstance(name, (tuple, list)) and len(name) == 2:
            lat_idx = int(np.argmin(np.abs(self.lat - float(name[0]))))
            lon_idx = int(np.argmin(np.abs(self.lon - float(name[1]))))
            return int(np.ravel_multi_index((lat_idx, lon_idx), (self.n_lat, self.n_lon)))
        for j in range(self.n_units):
            if self.label_for(j) == name:
                return j
        raise ValueError(
            f"unknown gridpoint '{name}'; pass a (lat, lon) pair or a label like "
            f"'{self.label_for(0)}'"
        )


def _resolve_ranking_strategy(ranking_strategy: str, baseline_emulations: dict | None) -> None:
    valid = {"emulator"} | (set(baseline_emulations) if baseline_emulations else set())
    if ranking_strategy not in valid:
        raise ValueError(
            f"ranking_strategy='{ranking_strategy}' is not valid. "
            f"Choose 'emulator' or one of: {sorted(valid - {'emulator'})}"
        )


def _assemble_error_data(
    *,
    sim_arrays: dict[str, np.ndarray],
    comparison_arrays: dict[str, dict[str, np.ndarray]],
    indicators: list[str],
    indicator_labels: dict[str, str],
    indicator_units: dict[str, str],
    metric: str,
    opts: CurveOptions,
    n_units: int,
    ranking_strategy: str,
    baseline_emulations: dict | None,
    diff_baseline: str | None,
    unit_labels: list[str] | None,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
) -> ErrorData:
    """Shared body of `build_error_data_regional` / `build_error_data_gridded`."""
    comparisons = list(comparison_arrays)

    scores = {
        name: {
            ind: _score_all_units(sim_arrays[ind], arrays[ind], metric, n_units, opts)
            for ind in indicators
        }
        for name, arrays in comparison_arrays.items()
    }
    vmax = {
        ind: float(max(np.nanmax(scores[c][ind]) for c in comparisons))
        for ind in indicators
    }

    _resolve_ranking_strategy(ranking_strategy, baseline_emulations)
    rank_key = "Emulator" if ranking_strategy == "emulator" else ranking_strategy
    ranking = {ind: _rank_units(scores[rank_key][ind], rank_key) for ind in indicators}

    diff = diff_vmax = resolved_diff_baseline = None
    if baseline_emulations:
        resolved_diff_baseline = diff_baseline or next(iter(baseline_emulations))
        if resolved_diff_baseline not in baseline_emulations:
            raise ValueError(
                f"diff_baseline '{resolved_diff_baseline}' not found in "
                f"baseline_emulations (available: {list(baseline_emulations)})."
            )
        diff = {
            ind: scores["Emulator"][ind] - scores[resolved_diff_baseline][ind]
            for ind in indicators
        }
        diff_vmax = {ind: float(np.nanmax(np.abs(diff[ind]))) for ind in indicators}

    return ErrorData(
        metric=metric,
        metric_label=_metric_label(metric),
        comparisons=comparisons,
        indicators=indicators,
        scores=scores,
        vmax=vmax,
        ranking=ranking,
        indicator_labels=indicator_labels,
        indicator_units=indicator_units,
        unit_labels=list(unit_labels) if unit_labels is not None else None,
        lat=lat,
        lon=lon,
        diff=diff,
        diff_baseline=resolved_diff_baseline,
        diff_vmax=diff_vmax,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Builders
# ═════════════════════════════════════════════════════════════════════════════

def build_error_data_regional(
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    region_names: list[str],
    metric: str = "nmae",
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    indicators: list[str] | None = None,
    indicator_labels: dict[str, str] | None = None,
    indicator_units: dict[str, str] | None = None,
    ranking_strategy: str = "emulator",
    diff_baseline: str | None = None,
    quantiles: np.ndarray = DEFAULT_QQ_QUANTILES,
    n_lags: int = DEFAULT_TEMPORAL_CORR_N_LAGS,
    window: int | None = None,
    psd_fs: float = 1.0,
    psd_nperseg: int | None = None,
) -> ErrorData:
    """
    Score every AR6 region, for every indicator and every comparison, with
    one metric — the input to `plots.plot_map_regional`,
    `plots.bar_plot_regional`, `plots.plot_timeseries_regional`,
    `plots.plot_qq_scatter`, `plots.plot_temporal_correlation_curves` and
    `plots.plot_psd_curves`.

    Parameters
    ----------
    sim, emulator : ``{indicator: (n_members, T, n_regions)}``
        Already preprocessed with `preprocessing.preprocess_regional` (which
        accepts and returns exactly this dict shape). Two indicators named
        "tas" and "pr" is the usual case, but any number and any names work.
    region_names : labels for the last axis, length n_regions.
    metric : any key from `available_metrics()` — a scalar metric ("nmae",
        "mae", "crps", ...), a quantile-curve metric ("qq_mae", "qq_nmae",
        "qq_ks", "qq_tail_mae"), "temporal_corr_nmae", or a PSD-curve metric
        ("psd_nmae", "psd_log_nmae", "psd_wasserstein").
    baseline_emulations : ``{name: {indicator: array}}``, same preprocessing
        and same indicators as the main arrays.
    indicators : explicit indicator subset/order. Defaults to ``list(sim)``.
    indicator_labels : override the column titles, e.g.
        ``{"hurs": "Relative humidity"}``. Defaults: nice names for "tas"
        and "pr", the uppercased key for anything else.
    indicator_units : override the unit strings shown on axes and colourbars.
    ranking_strategy : "emulator" (default) or a key in baseline_emulations
        — which comparison's error drives the ranking used by the
        best/median/worst plots.
    diff_baseline : baseline to subtract from the Emulator scores for the
        map's optional difference panel. Defaults to the first baseline.
    quantiles : quantile grid for the quantile-curve metric family.
    n_lags, window : lag count / trailing-window length for
        "temporal_corr_nmae" (see `compute_temporal_correlation_curve`).
    psd_fs, psd_nperseg : sampling frequency and Welch segment length for the
        PSD metric family (see `compute_psd_curves`). Pass the same values to
        `plots.plot_psd_curves` so the plotted curves match the scores.

    Returns
    -------
    ErrorData
    """
    comparison_arrays = {"Emulator": emulator, **(baseline_emulations or {})}
    inds, labels, units = resolve_indicators(
        sim, comparison_arrays, indicators, indicator_labels, indicator_units
    )
    n_regions = np.asarray(sim[inds[0]]).shape[2]
    if len(region_names) != n_regions:
        raise ValueError(
            f"region_names has {len(region_names)} entries but the data has {n_regions} regions"
        )
    return _assemble_error_data(
        sim_arrays={i: np.asarray(sim[i]) for i in inds},
        comparison_arrays={
            n: {i: np.asarray(a[i]) for i in inds} for n, a in comparison_arrays.items()
        },
        indicators=inds,
        indicator_labels=labels,
        indicator_units=units,
        metric=metric,
        opts=CurveOptions(quantiles, n_lags, window, psd_fs, psd_nperseg),
        n_units=n_regions,
        ranking_strategy=ranking_strategy,
        baseline_emulations=baseline_emulations,
        diff_baseline=diff_baseline,
        unit_labels=list(region_names),
        lat=None,
        lon=None,
    )


def build_error_data_gridded(
    sim: dict,
    emulator: dict,
    metric: str = "nmae",
    baseline_emulations: dict | None = None,
    *,
    indicators: list[str] | None = None,
    indicator_labels: dict[str, str] | None = None,
    indicator_units: dict[str, str] | None = None,
    ranking_strategy: str = "emulator",
    diff_baseline: str | None = None,
    quantiles: np.ndarray = DEFAULT_QQ_QUANTILES,
    n_lags: int = DEFAULT_TEMPORAL_CORR_N_LAGS,
    window: int | None = None,
    psd_fs: float = 1.0,
    psd_nperseg: int | None = None,
) -> ErrorData:
    """
    Gridded analogue of `build_error_data_regional` — scores every gridpoint
    instead of every AR6 region.

    Parameters
    ----------
    sim, emulator : ``{indicator: preprocessing.GriddedArray}``
        Already preprocessed with `preprocessing.preprocess_gridded` (which
        accepts and returns exactly this dict shape). All arrays, including
        every baseline's, must share the same grid.
    baseline_emulations : ``{name: {indicator: GriddedArray}}``
    (all other parameters: see `build_error_data_regional`)

    Returns
    -------
    ErrorData, with `lat`/`lon` set instead of `unit_labels`
    """
    comparison_arrays = {"Emulator": emulator, **(baseline_emulations or {})}
    inds, labels, units = resolve_indicators(
        sim, comparison_arrays, indicators, indicator_labels, indicator_units
    )
    ref = sim[inds[0]]
    return _assemble_error_data(
        sim_arrays={i: sim[i].values for i in inds},
        comparison_arrays={
            n: {i: a[i].values for i in inds} for n, a in comparison_arrays.items()
        },
        indicators=inds,
        indicator_labels=labels,
        indicator_units=units,
        metric=metric,
        opts=CurveOptions(quantiles, n_lags, window, psd_fs, psd_nperseg),
        n_units=ref.n_grid,
        ranking_strategy=ranking_strategy,
        baseline_emulations=baseline_emulations,
        diff_baseline=diff_baseline,
        unit_labels=None,
        lat=ref.lat,
        lon=ref.lon,
    )


def build_crps_timeseries_data_gridded(
    sim: dict,
    emulator: dict,
    baseline_emulations: dict | None = None,
    *,
    indicators: list[str] | None = None,
    indicator_labels: dict[str, str] | None = None,
    indicator_units: dict[str, str] | None = None,
) -> dict:
    """
    CRPS(t), averaged over all gridpoints, for every comparison and
    indicator — feeds `plots.plot_crps_timeseries_gridded`. Unlike
    `build_error_data_gridded`, this aggregates over gridpoints (not time),
    so there is no per-unit ranking.

    Parameters
    ----------
    sim, emulator : ``{indicator: preprocessing.GriddedArray}``
    baseline_emulations : ``{name: {indicator: GriddedArray}}``
    indicators, indicator_labels, indicator_units : see `build_error_data_regional`

    Returns
    -------
    dict with keys:
        "indicators"       : ordered indicator keys
        "indicator_labels" : display titles per indicator
        "indicator_units"  : unit strings per indicator
        "series"           : series[comparison][indicator] -> (T,) array
    """
    comparison_arrays = {"Emulator": emulator, **(baseline_emulations or {})}
    inds, labels, units = resolve_indicators(
        sim, comparison_arrays, indicators, indicator_labels, indicator_units
    )
    n_time = sim[inds[0]].values.shape[1]

    def _crps_all_times(sim_v: np.ndarray, pred_v: np.ndarray) -> np.ndarray:
        return np.array([
            crps_pairwise(pred_v[:, t, :], sim_v[:, t, :]).mean() for t in range(n_time)
        ])

    return {
        "indicators": inds,
        "indicator_labels": labels,
        "indicator_units": units,
        "series": {
            name: {i: _crps_all_times(sim[i].values, arrays[i].values) for i in inds}
            for name, arrays in comparison_arrays.items()
        },
    }


def build_correlation_data(
    sim: np.ndarray,
    pred: np.ndarray,
    baseline_emulations: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """
    Spatial correlation matrices (sim, pred, diff, MAE, RMSE) for the
    Emulator and every baseline — feeds `plots.plot_correlation_comparison`.

    One indicator at a time (the correlation matrix is across *units*, not
    indicators), and only meaningful for a modest number of units: the
    matrix is (n_units, n_units), which is why there is no gridded
    equivalent — a 10,000+ gridpoint correlation matrix isn't tractable.

    Parameters
    ----------
    sim  : (n_members, T, n_regions) — already preprocessed
    pred : (n_members, T, n_regions) — the Emulator's output
    baseline_emulations : {name: (n_members, T, n_regions)}

    Returns
    -------
    dict[comparison] -> spatial_correlation_scores(...) result
    """
    comparisons = {"Emulator": pred, **(baseline_emulations or {})}
    return {name: spatial_correlation_scores(sim, arr) for name, arr in comparisons.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Intervariable (between-indicator) correlation
# ═════════════════════════════════════════════════════════════════════════════

def _pearson_same_timestep(a_unit: np.ndarray, b_unit: np.ndarray) -> float:
    """
    Pearson correlation between two indicators at the same (member,
    timestep) pairs, for one unit — pools every member and every timestep
    before correlating.

    Parameters
    ----------
    a_unit, b_unit : (n_members, T)
    """
    x, y = a_unit.ravel(), b_unit.ravel()
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _indicator_pair(indicators: list[str], pair: tuple[str, str] | None) -> tuple[str, str]:
    if pair is not None:
        for p in pair:
            if p not in indicators:
                raise ValueError(f"indicator '{p}' not among {indicators}")
        return tuple(pair)
    if len(indicators) < 2:
        raise ValueError(
            "an intervariable correlation needs at least two indicators; "
            "pass indicator_pair=('a', 'b') explicitly"
        )
    return indicators[0], indicators[1]


def _intervariable_error_data(
    *,
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None,
    indicator_pair: tuple[str, str] | None,
    unit_labels: list[str] | None,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    ranking_strategy: str,
    diff_baseline: str | None,
    signed_values: bool,
) -> ErrorData:
    """
    Shared body of the two intervariable-correlation builders.

    `signed_values=False` scores |comparison correlation − simulated
    correlation| (an error); `signed_values=True` reports |correlation|
    itself for the simulation and every comparison (a magnitude).
    """
    comparison_arrays = {"Emulator": emulator, **(baseline_emulations or {})}
    all_inds, labels, units = resolve_indicators(sim, comparison_arrays, None, None, None)
    a, b = _indicator_pair(all_inds, indicator_pair)

    sim_a, sim_b = np.asarray(sim[a]), np.asarray(sim[b])
    n_units = sim_a.shape[2]
    sim_corr = np.array([
        _pearson_same_timestep(sim_a[:, :, j], sim_b[:, :, j]) for j in range(n_units)
    ])

    pair_key = f"{a}-{b}"
    pair_title = f"{labels[a]} vs {labels[b]}"

    def _corr_of(arrays) -> np.ndarray:
        pa, pb = np.asarray(arrays[a]), np.asarray(arrays[b])
        return np.array([
            _pearson_same_timestep(pa[:, :, j], pb[:, :, j]) for j in range(n_units)
        ])

    if signed_values:
        comparisons = ["Emulator", "Simulations"] + list(baseline_emulations or {})
        scores = {"Simulations": {pair_key: np.abs(sim_corr)}}
        for name, arrays in comparison_arrays.items():
            scores[name] = {pair_key: np.abs(_corr_of(arrays))}
        metric, metric_label = "intervar_corr_abs", f"|corr({a}, {b})|"
        valid = {"emulator", "simulations"} | set(baseline_emulations or {})
        if ranking_strategy not in valid:
            raise ValueError(
                f"ranking_strategy='{ranking_strategy}' is not valid. Choose "
                f"'emulator', 'simulations', or one of: {sorted(valid - {'emulator', 'simulations'})}"
            )
        rank_key = {"emulator": "Emulator", "simulations": "Simulations"}.get(
            ranking_strategy, ranking_strategy
        )
        diff_default = "Simulations"
        diff_pool = comparisons
    else:
        comparisons = list(comparison_arrays)
        scores = {
            name: {pair_key: np.abs(_corr_of(arrays) - sim_corr)}
            for name, arrays in comparison_arrays.items()
        }
        metric, metric_label = "intervar_corr_abs_diff", f"|Δ corr({a}, {b})|"
        _resolve_ranking_strategy(ranking_strategy, baseline_emulations)
        rank_key = "Emulator" if ranking_strategy == "emulator" else ranking_strategy
        diff_default = next(iter(baseline_emulations)) if baseline_emulations else None
        diff_pool = list(baseline_emulations or {})

    vmax = {pair_key: float(max(scores[c][pair_key].max() for c in comparisons))}
    ranking = {pair_key: _rank_units(scores[rank_key][pair_key], rank_key)}

    diff = diff_vmax = resolved = None
    resolved = diff_baseline if diff_baseline is not None else diff_default
    if resolved is not None and diff_pool:
        if resolved not in diff_pool:
            raise ValueError(
                f"diff_baseline '{resolved}' not found among {diff_pool}."
            )
        diff = {pair_key: scores["Emulator"][pair_key] - scores[resolved][pair_key]}
        diff_vmax = {pair_key: float(np.abs(diff[pair_key]).max())}
    else:
        resolved = None

    return ErrorData(
        metric=metric,
        metric_label=metric_label,
        comparisons=comparisons,
        indicators=[pair_key],
        scores=scores,
        vmax=vmax,
        ranking=ranking,
        indicator_labels={pair_key: pair_title},
        indicator_units={pair_key: ""},
        unit_labels=list(unit_labels) if unit_labels is not None else None,
        lat=lat,
        lon=lon,
        diff=diff,
        diff_baseline=resolved,
        diff_vmax=diff_vmax,
    )


def build_error_data_intervariable_correlation(
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    indicator_pair: tuple[str, str] | None = None,
    unit_labels: list[str] | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    ranking_strategy: str = "emulator",
    diff_baseline: str | None = None,
) -> ErrorData:
    """
    Per-unit error in the relationship *between two indicators*.

    For every unit, the Pearson correlation between the two indicators is
    computed once from the simulation and once from each comparison,
    pooling all members and timesteps at that unit — i.e. pairing indicator
    A at time t with indicator B at the *same* t. The score is the absolute
    difference between the comparison's correlation and the simulation's.

    One function for both regional and gridded data — pass plain
    ``{indicator: (n_members, T, n_units)}`` arrays either way (for gridded,
    that's ``mesh_array.values``), plus `unit_labels` for regional or
    `lat`/`lon` for gridded so `plots.plot_map_regional` /
    `plot_map_gridded` / `bar_plot_regional` can render it.

    The result has a single "indicator" — the pair itself, keyed ``"a-b"`` —
    so the map/bar plots draw one column, not one per input indicator.

    Parameters
    ----------
    sim, emulator : ``{indicator: (n_members, T, n_units)}``
    baseline_emulations : ``{name: {indicator: array}}``
    indicator_pair : which two indicators to correlate. Defaults to the
        first two in `sim` — i.e. ("tas", "pr") for the usual inputs.
    unit_labels : region abbreviations, for regional data
    lat, lon : grid coordinates, for gridded data
    ranking_strategy, diff_baseline : as in `build_error_data_regional`

    Returns
    -------
    ErrorData
    """
    return _intervariable_error_data(
        sim=sim, emulator=emulator, baseline_emulations=baseline_emulations,
        indicator_pair=indicator_pair, unit_labels=unit_labels, lat=lat, lon=lon,
        ranking_strategy=ranking_strategy, diff_baseline=diff_baseline,
        signed_values=False,
    )


def build_error_data_intervariable_correlation_values(
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    indicator_pair: tuple[str, str] | None = None,
    unit_labels: list[str] | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    ranking_strategy: str = "emulator",
    diff_baseline: str | None = "Simulations",
) -> ErrorData:
    """
    Per-unit relationship *strength* between two indicators, not error.

    Same inputs and same `ErrorData` output as
    `build_error_data_intervariable_correlation`, but instead of scoring how
    far each comparison's correlation is from the simulation's, this reports
    the |Pearson correlation| itself for "Simulations", "Emulator", and every
    baseline. Use it to see how strong the coupling actually is in each data
    source, side by side.

    The absolute value is taken (rather than the signed correlation) so the
    scores stay non-negative, matching every other `ErrorData` producer here
    — the colour scales and bar charts in `plots.py` all assume ``vmin=0``.

    Parameters
    ----------
    (see `build_error_data_intervariable_correlation`)
    ranking_strategy : "emulator" (default), "simulations", or a baseline
        name. Note this just orders units by that comparison's correlation
        *magnitude*; unlike an error metric there is no inherent
        better/worse direction, so treat `.ranking` as a convenient ordering
        rather than a quality judgement.
    diff_baseline : comparison to subtract from the Emulator's correlation
        for the optional difference panel. Defaults to "Simulations", giving
        the signed version of what
        `build_error_data_intervariable_correlation` reports as an absolute
        error. Pass None to skip the difference entirely.

    Returns
    -------
    ErrorData
    """
    return _intervariable_error_data(
        sim=sim, emulator=emulator, baseline_emulations=baseline_emulations,
        indicator_pair=indicator_pair, unit_labels=unit_labels, lat=lat, lon=lon,
        ranking_strategy=ranking_strategy, diff_baseline=diff_baseline,
        signed_values=True,
    )
