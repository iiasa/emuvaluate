"""
preprocessing/transforms.py
----------------------------
Stateless transforms applied to ensemble arrays of shape (n_members, T, n_regions).

All functions return a new array and leave the input unchanged.
"""

import numpy as np
import xarray as xr

def yearly_average_gridded(
    data: xr.DataArray | xr.Dataset,
    time_dim: str = "time",
    drop_incomplete_years: bool = True,
) -> xr.DataArray | xr.Dataset:
    """
    Annual mean of MESH gridded data — analogous to the SCALES `yearly_average`
    helper used e.g. as `yearly_average(scenario_data_ssp245_tas)`.
 
    Groups by calendar year and averages over `time_dim`, then relabels the
    resulting "year" axis back to a "time" dim (one datetime64 timestamp per
    year, Jan 1st). The output keeps a "time" dim so it plugs directly into
    `plot_error_metrics_map_mesh`, `compute_gmt_area_weighted`, and
    `fit_gridpoint_regressions_monthly` / `predict_pattern_scaling_mesh`
    without any renaming. All other dims (ensemble, lat, lon) and their
    order are left untouched.
 
    Works on a single DataArray (e.g. `mesh_ds["tas"]`) or a full Dataset
    (e.g. `mesh_ds` with both "tas" and "pr").
 
    Parameters
    ----------
    data                  : xr.DataArray or xr.Dataset with a datetime64
                             `time_dim`
    time_dim              : name of the time dimension (default "time")
    drop_incomplete_years : if True (default), calendar years with fewer
                             than 12 monthly timesteps (e.g. the partial
                             final year in a 2015-01 .. 2098-04 series) are
                             dropped rather than averaged over fewer months
                             — mirroring the "trailing timesteps ignored"
                             behaviour of `fit_regional_regressions_monthly`.
                             If False, a warning is printed instead and
                             those years are averaged over however many
                             months are present.
 
    Returns
    -------
    Same type as `data`, with `time_dim` collapsed to one value per
    calendar year.
    """
    month_counts = data[time_dim].groupby(f"{time_dim}.year").count()
    incomplete_years = month_counts["year"].values[month_counts.values < 12]
 
    if len(incomplete_years):
        if drop_incomplete_years:
            print(
                f"Warning: dropping incomplete year(s) with <12 months: "
                f"{list(incomplete_years)}"
            )
            keep = ~data[f"{time_dim}.year"].isin(incomplete_years)
            data = data.sel({time_dim: keep})
        else:
            print(
                f"Warning: year(s) {list(incomplete_years)} have <12 months; "
                f"annual mean for those years will be computed over fewer months."
            )
 
    annual = data.groupby(f"{time_dim}.year").mean(dim=time_dim)
    years = annual["year"].values
    annual = annual.rename({"year": time_dim}).assign_coords(
        {time_dim: [np.datetime64(f"{int(y)}-01-01") for y in years]}
    )
    return annual

def yearly_average(data: np.ndarray) -> np.ndarray:
    """
    Replace every 12 consecutive monthly timesteps with their mean.
    Trailing months that do not form a complete year are dropped.

    Parameters
    ----------
    data : (T,), (T, n_regions), or (n_members, T, n_regions)

    Returns
    -------
    np.ndarray
        (n_years,)                    if input is (T,)
        (n_years, n_regions)          if input is (T, n_regions)
        (n_members, n_years, n_regions) if input is (n_members, T, n_regions)
    """
    if data.ndim == 1:
        # (T,)
        T       = len(data)
        n_years = T // 12
        return data[: n_years * 12].reshape(n_years, 12).mean(axis=1)

    elif data.ndim == 2:
        # (T, n_regions)
        T, n_regions = data.shape
        n_years      = T // 12
        return (
            data[: n_years * 12, :]
            .reshape(n_years, 12, n_regions)
            .mean(axis=1)
        )

    elif data.ndim == 3:
        # (n_members, T, n_regions)
        n_members, T, n_regions = data.shape
        n_years                 = T // 12
        return (
            data[:, : n_years * 12, :]
            .reshape(n_members, n_years, 12, n_regions)
            .mean(axis=2)
        )

    else:
        raise ValueError(
            f"Expected array of shape (T,), (T, n_regions) or "
            f"(n_members, T, n_regions), got shape {data.shape}"
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









def weighted_linear_smoothing(data: np.ndarray, tau: float = 20, monthly: bool = False) -> np.ndarray:
    """
    Apply local weighted regression smoothing along the T dimension.

    Accepts arrays of shape (T,), (T, n_regions) or (n_samples, T, n_regions)
    and smooths along the T axis in all cases.

    If monthly=True, the T dimension is assumed to be monthly data and each
    calendar month is smoothed independently across years.

    Parameters
    ----------
    data    : np.ndarray
        Shape (T,), (T, n_regions), or (n_samples, T, n_regions)
    tau     : float
        Smoothing bandwidth (default 20)
    monthly : bool
        If True, smooth each calendar month independently (default False).
        T must be divisible by 12 when monthly=True.

    Returns
    -------
    np.ndarray, same shape as input
    """
    def weights_calculate(x0, X, tau):
        return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau ** 2)))

    def local_weighted_regression(x0, X, Y, tau):
        x0    = np.r_[1, x0]
        X     = np.c_[np.ones(len(X)), X]
        xw    = X.T * weights_calculate(x0, X, tau)
        theta = np.linalg.pinv(xw @ X) @ xw @ Y
        return x0 @ theta

    def _smooth_1d(ts: np.ndarray) -> np.ndarray:
        """Smooth a single 1-D timeseries of length T."""
        T       = len(ts)
        array_x = np.arange(T).reshape(-1, 1)
        return np.array([
            local_weighted_regression(np.array([t]), array_x, ts, tau)
            for t in range(T)
        ])

    def _smooth_1d_monthly(ts: np.ndarray) -> np.ndarray:
        """
        Smooth a single 1-D monthly timeseries by smoothing each
        calendar month independently across years.
        T must be divisible by 12.
        """
        T = len(ts)
        if T % 12 != 0:
            raise ValueError(
                f"monthly=True requires T divisible by 12, got T={T}"
            )
        n_years = T // 12
        array_x = np.arange(n_years).reshape(-1, 1)
        out     = np.zeros_like(ts)
        for month in range(12):
            month_vals = ts[month::12]
            smoothed   = np.array([
                local_weighted_regression(np.array([t]), array_x, month_vals, tau)
                for t in range(n_years)
            ])
            out[month::12] = smoothed
        return out

    # Pick the right 1-D smoother
    _smooth = _smooth_1d_monthly if monthly else _smooth_1d

    # ── Dispatch by shape ─────────────────────────────────────────────────────
    if data.ndim == 1:
        # (T,)
        return _smooth(data)

    elif data.ndim == 2:
        # (T, n_regions)
        T, n_regions = data.shape
        out = np.zeros_like(data)
        for r in range(n_regions):
            out[:, r] = _smooth(data[:, r])
        return out

    elif data.ndim == 3:
        # (n_samples, T, n_regions)
        n_samples, T, n_regions = data.shape
        out = np.zeros_like(data)
        for n in range(n_samples):
            for r in range(n_regions):
                out[n, :, r] = _smooth(data[n, :, r])
        return out

    else:
        raise ValueError(
            f"Expected array of shape (T,), (T, n_regions) or "
            f"(n_samples, T, n_regions), got shape {data.shape}"
        )



def ensemble_mean_regional(data):
    """Collapse the member axis to the ensemble mean, keeping it as one member.

    {indicator: (n_members, T, n_regions)} -> {indicator: (1, T, n_regions)}

    Keeping the axis (rather than dropping it) means the result is still the
    shape every builder and plotting function expects — it just describes a
    one-member "ensemble" that is the mean. MAE between two of these is
    therefore the MAE between the two ensemble means, and the timeseries plot
    draws one line per source instead of a spread.
    """
    return {k: np.asarray(v).mean(axis=0, keepdims=True) for k, v in data.items()}


