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
    "tail_mae": "Tail MAE",
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



from scipy.signal import welch
from sklearn.decomposition import PCA


def compute_psd_scores(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    fs: float = 1.0,  # 1 sample per month
    nperseg: int = 256,
) -> dict:
    """
    Compute power spectral density for each region and member using Welch's method,
    after removing the ensemble mean to isolate internal variability.

    Parameters
    ----------
    obs_data  : (n_members, T, n_regions)
    pred_data : (n_members, T, n_regions)
    fs        : sampling frequency (1/month by default)
    nperseg   : Welch segment length — larger = finer frequency resolution

    Returns
    -------
    dict with keys:
        'freqs'    : (n_freqs,) — frequency axis in cycles/month
        'obs_psd'  : (n_members, n_freqs, n_regions)
        'pred_psd' : (n_members, n_freqs, n_regions)
    """
    n_members, T, n_regions = obs_data.shape

    # remove ensemble mean to isolate internal variability
    obs_anom  = obs_data  - obs_data.mean(axis=0, keepdims=True)
    pred_anom = pred_data - pred_data.mean(axis=0, keepdims=True)

    freqs, _ = welch(obs_anom[0, :, 0], fs=fs, nperseg=nperseg)
    n_freqs  = len(freqs)

    obs_psd  = np.zeros((n_members, n_freqs, n_regions))
    pred_psd = np.zeros((n_members, n_freqs, n_regions))

    for m in range(n_members):
        for r in range(n_regions):
            _, obs_psd[m, :, r]  = welch(obs_anom[m, :, r],  fs=fs, nperseg=nperseg)
            _, pred_psd[m, :, r] = welch(pred_anom[m, :, r], fs=fs, nperseg=nperseg)

    return {"freqs": freqs, "obs_psd": obs_psd, "pred_psd": pred_psd}



def compute_rank_histogram(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
) -> np.ndarray:
    """
    Compute rank histogram (Talagrand diagram) ranks.
    At each (t, r), each obs member is ranked within the pred ensemble.

    Memory-efficient: loops over regions to avoid large intermediate arrays.

    Parameters
    ----------
    obs_data  : (n_obs_members, T, n_regions)
    pred_data : (n_pred_members, T, n_regions)

    Returns
    -------
    ranks : (n_obs_members, T, n_regions) — rank of each obs within pred ensemble
    """
    n_obs, T, n_regions = obs_data.shape
    ranks       = np.zeros((n_obs, T, n_regions), dtype=int)
    pred_sorted = np.sort(pred_data, axis=0)  # (n_pred, T, n_regions)

    for r in range(n_regions):
        # (n_obs, 1, T) > (1, n_pred, T) -> (n_obs, n_pred, T) -> sum -> (n_obs, T)
        ranks[:, :, r] = (
            obs_data[:, np.newaxis, :, r] > pred_sorted[np.newaxis, :, :, r]
        ).sum(axis=1)

    return ranks



def compute_eof_scores(
    obs_data: np.ndarray,
    pred_data: np.ndarray,
    n_eofs: int = 5,
) -> dict:
    """
    Compute EOFs of the ensemble-mean-removed data for simulation and emulation.

    Parameters
    ----------
    obs_data  : (n_members, T, n_regions)
    pred_data : (n_members, T, n_regions)
    n_eofs    : number of EOFs to retain
    """
    def prepare(data):
        # remove ensemble mean (forced response) to isolate internal variability
        anom = data - data.mean(axis=0, keepdims=True)
        n_members, T, n_regions = anom.shape
        return anom.reshape(n_members * T, n_regions)

    obs_flat  = prepare(obs_data)
    pred_flat = prepare(pred_data)

    pca_obs  = PCA(n_components=n_eofs).fit(obs_flat)
    pca_pred = PCA(n_components=n_eofs).fit(pred_flat)

    obs_eofs  = pca_obs.components_.copy()
    pred_eofs = pca_pred.components_.copy()
    for i in range(n_eofs):
        if obs_eofs[i, np.argmax(np.abs(obs_eofs[i]))]  < 0:
            obs_eofs[i]  *= -1
        if pred_eofs[i, np.argmax(np.abs(pred_eofs[i]))] < 0:
            pred_eofs[i] *= -1

    return {
        "obs_eofs":       obs_eofs,
        "pred_eofs":      pred_eofs,
        "obs_var_ratio":  pca_obs.explained_variance_ratio_,
        "pred_var_ratio": pca_pred.explained_variance_ratio_,
    }

def lagged_correlation(x, y, max_lag=24):
    """Correlation of x with y at lags 0, 1, ..., max_lag months."""
    return np.array([
        np.corrcoef(x[lag:], y[:len(x)-lag])[0,1] if lag > 0 
        else np.corrcoef(x, y)[0,1]
        for lag in range(max_lag + 1)
    ])