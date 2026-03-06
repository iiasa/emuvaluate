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
}

METRIC_LABELS: dict[str, str] = {
    "crps":  "CRPS",
    "mean":  "Mean Difference",
    "sigma": "Sigma Difference",
    "psd":   "PSD Wasserstein Distance",
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