"""
preprocessing/transforms.py
----------------------------
Stateless transforms applied to ensemble arrays of shape (n_members, T, n_regions).

All functions return a new array and leave the input unchanged.
"""

import numpy as np


def yearly_average(data: np.ndarray) -> np.ndarray:
    """
    Replace every 12 consecutive monthly timesteps with their mean,
    producing an array of shape (n_members, n_years, n_regions).

    Trailing months that do not form a complete year are dropped.

    Parameters
    ----------
    data : (n_members, T, n_regions)

    Returns
    -------
    np.ndarray, shape (n_members, n_years, n_regions)
    """
    n_members, T, n_regions = data.shape
    n_years = T // 12
    return (
        data[:, : n_years * 12, :]
        .reshape(n_members, n_years, 12, n_regions)
        .mean(axis=2)
    )


def select_month(data: np.ndarray, month: int) -> np.ndarray:
    """
    Retain only the timesteps corresponding to a given calendar month.

    Parameters
    ----------
    data  : (n_members, T, n_regions)  — assumed to start at month 1
    month : int in [1, 12]

    Returns
    -------
    np.ndarray, shape (n_members, T // 12, n_regions)
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in [1, 12], got {month}")
    return data[:, (month - 1) :: 12, :]


def deseasonalise(data: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Remove the mean seasonal cycle from each member × region time series.

    The climatology is computed per member so that it cannot leak
    information between ensemble members.

    Parameters
    ----------
    data   : (n_members, T, n_regions)
    period : length of the seasonal cycle in timesteps (default 12 for monthly)

    Returns
    -------
    np.ndarray, same shape as *data*
    """
    n_members, T, n_regions = data.shape
    season_index = np.arange(T) % period
    out = np.zeros_like(data)

    for m in range(n_members):
        for j in range(n_regions):
            ts = data[m, :, j]
            climatology = np.array(
                [ts[season_index == s].mean() for s in range(period)]
            )
            out[m, :, j] = ts - climatology[season_index]

    return out


def detrend_gaussian(data: np.ndarray, tau: float = 20) -> np.ndarray:
    """
    Remove a Gaussian-smoothed trend from each member × region time series.

    Parameters
    ----------
    data : (n_members, T, n_regions)
    tau  : Gaussian smoothing sigma in timesteps (default 20)

    Returns
    -------
    np.ndarray, same shape as *data*
    """
    from scipy.ndimage import gaussian_filter1d

    out = np.zeros_like(data)
    for m in range(data.shape[0]):
        for j in range(data.shape[2]):
            ts = data[m, :, j]
            trend = gaussian_filter1d(ts, sigma=tau, mode="nearest")
            out[m, :, j] = ts - trend
    return out


def preprocess(
    data: np.ndarray,
    *,
    apply_yearly_average: bool = False,
    month_selection: int | None = None,
    apply_deseasonalise: bool = False,
    apply_detrend: bool = False,
    detrend_tau: float = 20,
) -> np.ndarray:
    """
    Convenience wrapper that applies the preprocessing steps in the
    canonical order used by the original ``plot_metric_extremes``.

    Steps (each optional):
        1. Temporal aggregation  — yearly_average *or* select_month (mutually exclusive)
        2. Deseasonalise
        3. Detrend

    Parameters
    ----------
    data                 : (n_members, T, n_regions)
    apply_yearly_average : collapse months → years
    month_selection      : keep only this calendar month (1–12)
    apply_deseasonalise  : subtract mean seasonal cycle
    apply_detrend        : subtract Gaussian-smoothed trend
    detrend_tau          : smoothing sigma for detrending

    Returns
    -------
    np.ndarray — preprocessed copy of *data*
    """
    if apply_yearly_average and month_selection is not None:
        raise ValueError("apply_yearly_average and month_selection are mutually exclusive")

    out = data.copy()

    # 1. Temporal aggregation
    if apply_yearly_average:
        out = yearly_average(out)
    elif month_selection is not None:
        out = select_month(out, month_selection)

    # 2. Deseasonalise
    if apply_deseasonalise:
        if apply_yearly_average:
            print("Warning: deseasonalise has no effect after yearly_average — skipping")
        else:
            out = deseasonalise(out)

    # 3. Detrend
    if apply_detrend:
        out = detrend_gaussian(out, tau=detrend_tau)

    return out
