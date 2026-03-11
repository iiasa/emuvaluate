"""
metrics/scores.py
-----------------
Per-region scalar scores comparing an observed ensemble to a predicted ensemble.

Each function accepts:
    obs  : np.ndarray, shape (n_members, T)
    pred : np.ndarray, shape (n_members, T)

and returns a single float (lower = better for all metrics).
"""

import numpy as np
from statsmodels.tsa.stattools import acf

def linearity_score(obs: np.ndarray, gmt: np.ndarray) -> float:
    """
    Score how linear the relationship between a regional timeseries and GMT is.
    Uses the R² of a linear regression of the regional values on GMT.
    Higher = more linear.

    Parameters
    ----------
    obs : (n_members, T) — regional timeseries
    gmt : (n_members, T) or (T,) — GMT timeseries

    Returns
    -------
    float — R² score (0 to 1, higher = more linear)
    """
    from sklearn.linear_model import LinearRegression

    gmt = np.asarray(gmt)
    if gmt.ndim == 1:
        gmt = np.tile(gmt, (obs.shape[0], 1))

    # Pool all members
    X = gmt.ravel().reshape(-1, 1)
    y = obs.ravel()

    model = LinearRegression().fit(X, y)
    return float(model.score(X, y))

def acf_mae(obs: np.ndarray, pred: np.ndarray, n_lags: int = 48) -> float:
    """
    MAE between mean ACF curves of simulation and emulation ensembles.
    
    obs  : (n_members, T)
    pred : (n_members, T)
    """
    # Mean ACF across ensemble members
    obs_acf  = np.mean([acf(obs[m],  nlags=n_lags, fft=True) for m in range(obs.shape[0])],  axis=0)
    pred_acf = np.mean([acf(pred[m], nlags=n_lags, fft=True) for m in range(pred.shape[0])], axis=0)
    return float(np.mean(np.abs(obs_acf - pred_acf)))


def crps_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Mean Continuous Ranked Probability Score (CRPS) over all timesteps.

    At each timestep every observed member is treated as a verifying
    observation against the full predicted ensemble.

    Parameters
    ----------
    obs  : (n_members, T)
    pred : (n_members, T)

    Returns
    -------
    float — mean CRPS (lower = better)
    """
    import properscoring as ps

    n_members, T = obs.shape
    scores = [
        np.mean(
            ps.crps_ensemble(
                obs[:, t],
                pred[np.newaxis, :, t].repeat(n_members, axis=0),
            )
        )
        for t in range(T)
    ]
    return float(np.mean(scores))


def mean_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Absolute difference between the grand means of the two ensembles.

    Parameters
    ----------
    obs  : (n_members, T)
    pred : (n_members, T)

    Returns
    -------
    float
    """
    return float(np.abs(obs.mean() - pred.mean()))


def sigma_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Absolute difference between the overall standard deviations of the
    two ensembles.

    Parameters
    ----------
    obs  : (n_members, T)
    pred : (n_members, T)

    Returns
    -------
    float
    """
    return float(np.abs(obs.std() - pred.std()))


def psd_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Wasserstein distance between the mean power spectral densities of the
    two ensembles (computed via Welch's method on the ensemble-mean series).

    Parameters
    ----------
    obs  : (n_members, T)
    pred : (n_members, T)

    Returns
    -------
    float
    """
    from scipy.signal import welch
    from scipy.stats import wasserstein_distance

    T = obs.shape[1]
    nperseg = min(256, T)
    _, psd_obs  = welch(obs.mean(axis=0),  nperseg=nperseg)
    _, psd_pred = welch(pred.mean(axis=0), nperseg=nperseg)
    return float(wasserstein_distance(psd_obs, psd_pred))


# Registry — keeps the rest of the codebase decoupled from metric names
METRIC_REGISTRY: dict[str, callable] = {
    "crps":  crps_score,
    "mean":  mean_score,
    "sigma": sigma_score,
    "psd":   psd_score,
    "acf_mae": acf_mae,
}

METRIC_LABELS: dict[str, str] = {
    "crps":  "CRPS",
    "mean":  "Mean Difference",
    "sigma": "Sigma Difference",
    "psd":   "PSD Wasserstein Distance",
    "acf_mae": "Mean Absolute Error between Autocorrelation Functions"
}


def compute_metric_all_regions(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    metric: str,
) -> dict[int, float]:
    """
    Compute a metric score for every region.

    Parameters
    ----------
    obs_data  : (n_members, T, n_regions)
    pred_data : (n_members, T, n_regions)
    metric    : key in METRIC_REGISTRY

    Returns
    -------
    dict mapping region index → score
    """
    if metric not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRIC_REGISTRY)}")

    score_fn = METRIC_REGISTRY[metric]
    n_regions = obs_data.shape[2]
    return {j: score_fn(obs_data[:, :, j], pred_data[:, :, j]) for j in range(n_regions)}

def spatial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the mean spatial correlation matrix across ensemble members.

    At each member the correlation is computed across the T timesteps,
    producing an (n_regions, n_regions) matrix per member; these are then
    averaged over all members.

    Parameters
    ----------
    data : (n_members, T, n_regions)

    Returns
    -------
    np.ndarray, shape (n_regions, n_regions)
    """
    corr_matrices = np.array([
        np.corrcoef(data[n].T)   # (n_regions, n_regions)
        for n in range(data.shape[0])
    ])
    return corr_matrices.mean(axis=0)


def spatial_correlation_scores(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
) -> dict:
    """
    Compute spatial correlation matrices for simulations and emulations
    and their difference.

    Parameters
    ----------
    obs_data  : (n_members, T, n_regions)
    pred_data : (n_members, T, n_regions)

    Returns
    -------
    dict with keys:
        'sim_corr'  : (n_regions, n_regions) — simulation correlation matrix
        'emu_corr'  : (n_regions, n_regions) — emulation correlation matrix
        'diff'      : (n_regions, n_regions) — emu_corr - sim_corr
        'mae'       : float — mean absolute error of the difference
        'rmse'      : float — root mean squared error of the difference
    """
    sim_corr = spatial_correlation_matrix(obs_data)
    emu_corr = spatial_correlation_matrix(pred_data)
    diff     = emu_corr - sim_corr

    return {
        "sim_corr": sim_corr,
        "emu_corr": emu_corr,
        "diff":     diff,
        "mae":      float(np.abs(diff).mean()),
        "rmse":     float(np.sqrt((diff ** 2).mean())),
    }

def compute_quantiles(data: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Compute quantiles across all members and timesteps for a single region.

    Parameters
    ----------
    data      : (n_members, T) — ensemble for one region
    quantiles : (Q,) — quantile levels in [0, 1]

    Returns
    -------
    np.ndarray, shape (Q,)
    """
    return np.quantile(data.flatten(), quantiles)


def qq_mae(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Mean Absolute Error between quantiles."""
    return float(np.mean(np.abs(obs_q - pred_q)))


def qq_nmae(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Normalised MAE — divided by the range of observed quantiles."""
    obs_range = obs_q.max() - obs_q.min()
    if obs_range == 0:
        return 0.0
    return float(np.mean(np.abs(obs_q - pred_q)) / obs_range)


def qq_max_ae(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Maximum Absolute Error between quantiles."""
    return float(np.max(np.abs(obs_q - pred_q)))


def qq_mse(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Mean Squared Error between quantiles."""
    return float(np.mean((obs_q - pred_q) ** 2))


def qq_nmse(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """Normalised MSE — divided by the variance of observed quantiles."""
    obs_var = np.var(obs_q)
    if obs_var == 0:
        return 0.0
    return float(np.mean((obs_q - pred_q) ** 2) / obs_var)


def qq_ks(obs_q: np.ndarray, pred_q: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic — maximum absolute difference between
    the empirical CDFs, equivalent to max-AE on the quantile grid.
    Useful as a distribution-free measure of distributional shift.
    """
    return float(np.max(np.abs(obs_q - pred_q)))


def qq_tail_mae(obs_q: np.ndarray, pred_q: np.ndarray, tail_frac: float = 0.1) -> float:
    """
    MAE restricted to the upper and lower *tail_frac* of the quantile grid.
    Highlights errors in the extremes rather than the bulk of the distribution.

    Parameters
    ----------
    tail_frac : fraction of quantiles considered 'tail' at each end (default 0.1)
    """
    Q = len(obs_q)
    n_tail = max(1, int(Q * tail_frac))
    tail_idx = list(range(n_tail)) + list(range(Q - n_tail, Q))
    return float(np.mean(np.abs(obs_q[tail_idx] - pred_q[tail_idx])))


QQ_METRIC_REGISTRY: dict[str, callable] = {
    "mae":      qq_mae,
    "nmae":     qq_nmae,
    "max_ae":   qq_max_ae,
    "mse":      qq_mse,
    "nmse":     qq_nmse,
    "ks":       qq_ks,
    "tail_mae": qq_tail_mae,
}

QQ_METRIC_LABELS: dict[str, str] = {
    "mae":      "MAE (Quantiles)",
    "nmae":     "Normalised MAE (Quantiles)",
    "max_ae":   "Max Absolute Error (Quantiles)",
    "mse":      "MSE (Quantiles)",
    "nmse":     "Normalised MSE (Quantiles)",
    "ks":       "KS Statistic",
    "tail_mae": "Tail MAE (Quantiles)",
}


def compute_qq_scores_all_regions(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    metric: str,
    quantiles: np.ndarray,
) -> tuple[dict[int, float], np.ndarray, np.ndarray]:
    """
    Compute QQ metric scores for every region.

    Parameters
    ----------
    obs_data  : (n_members, T, n_regions)
    pred_data : (n_members, T, n_regions)
    metric    : key in QQ_METRIC_REGISTRY
    quantiles : (Q,) quantile levels

    Returns
    -------
    scores    : dict mapping region index → score
    obs_qq    : (n_regions, Q) — observed quantiles per region
    pred_qq   : (n_regions, Q) — predicted quantiles per region
    """
    if metric not in QQ_METRIC_REGISTRY:
        raise ValueError(
            f"Unknown QQ metric '{metric}'. Choose from {list(QQ_METRIC_REGISTRY)}"
        )
    score_fn  = QQ_METRIC_REGISTRY[metric]
    n_regions = obs_data.shape[2]

    obs_qq  = np.array([compute_quantiles(obs_data[:, :, j],  quantiles) for j in range(n_regions)])
    pred_qq = np.array([compute_quantiles(pred_data[:, :, j], quantiles) for j in range(n_regions)])
    scores  = {j: score_fn(obs_qq[j], pred_qq[j]) for j in range(n_regions)}

    return scores, obs_qq, pred_qq


def mae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error between ensemble means."""
    return float(np.mean(np.abs(obs.mean(axis=0) - pred.mean(axis=0))))


def rmse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error between ensemble means."""
    return float(np.sqrt(np.mean((obs.mean(axis=0) - pred.mean(axis=0)) ** 2)))


def max_ae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Maximum Absolute Error between ensemble means."""
    return float(np.max(np.abs(obs.mean(axis=0) - pred.mean(axis=0))))


def mse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean Squared Error between ensemble means."""
    return float(np.mean((obs.mean(axis=0) - pred.mean(axis=0)) ** 2))


def nmae_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Normalised MAE — divided by the range of the observed ensemble mean."""
    obs_range = obs.mean(axis=0).ptp()
    return float(mae_score(obs, pred) / obs_range) if obs_range > 0 else 0.0


def nmse_score(obs: np.ndarray, pred: np.ndarray) -> float:
    """Normalised MSE — divided by the variance of the observed ensemble mean."""
    obs_var = obs.mean(axis=0).var()
    return float(mse_score(obs, pred) / obs_var) if obs_var > 0 else 0.0


ERROR_METRIC_REGISTRY: dict[str, tuple[str, callable]] = {
    "mae":    ("MAE",            mae_score),
    "rmse":   ("RMSE",          rmse_score),
    "max_ae": ("Max AE",        max_ae_score),
    "mse":    ("MSE",            mse_score),
    "nmae":   ("Normalised MAE", nmae_score),
    "nmse":   ("Normalised MSE", nmse_score),
}