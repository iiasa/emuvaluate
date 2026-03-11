"""
preprocessing/transforms.py
----------------------------
Stateless transforms applied to ensemble arrays of shape (n_members, T, n_regions).

All functions return a new array and leave the input unchanged.
"""

import numpy as np


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


def split_at_indices(
    data: np.ndarray,
    split_points: list[int],
) -> list[np.ndarray]:
    """
    Split an ensemble array along the T dimension at given indices.

    Parameters
    ----------
    data         : (n_members, T, n_regions)
    split_points : list of int — indices along T at which to split
                   e.g. [200, 750, 1400] produces 4 segments:
                   [:200], [200:750], [750:1400], [1400:]

    Returns
    -------
    list of np.ndarray, length len(split_points) + 1,
    each of shape (n_members, segment_T, n_regions)
    """
    boundaries = [0] + list(split_points) + [data.shape[1]]
    return [
        data[:, boundaries[i]:boundaries[i + 1], :]
        for i in range(len(boundaries) - 1)
    ]

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



def _detect_phase(
    gmt_smooth: np.ndarray,
    phase: str,
    stable_window: int = 30,
    stable_threshold: float = 0.1,
    ramp_tolerance: float = 0.15,
    min_phase_length: int = 10,
    window_size: int | None = None,
) -> tuple[int, int]:
    """
    Find the best-fitting segment for a given phase using a rolling
    scored window rather than hard binary criteria.

    For ramp-up/down: scores each window by the net direction signal
    (mean signed diff) minus a penalty for wrong-direction diffs.
    For stable: scores by negative total variation in the window.

    Always returns a result — picks the highest-scoring window.
    """
    T    = len(gmt_smooth)
    diff = np.diff(gmt_smooth)

    if phase not in ("ramp-up", "ramp-down", "stable"):
        raise ValueError(f"Unknown phase '{phase}'. Choose from 'ramp-up', 'ramp-down', 'stable'.")

    # Default window: half the series length, at least min_phase_length
    if window_size is None:
        window_size = max(min_phase_length, T // 3)
    window_size = min(window_size, T - 1)

    best_score = -np.inf
    best_start = 0
    best_end   = window_size

    for t in range(T - window_size):
        window_diff    = diff[t:t + window_size]
        window_segment = gmt_smooth[t:t + window_size + 1]

        if phase == "ramp-up":
            # Net rise + fraction correct, penalise wrong-direction magnitude
            net_change     = window_segment[-1] - window_segment[0]
            frac_correct   = np.mean(window_diff > 0)
            wrong_mag      = np.mean(np.abs(window_diff[window_diff < 0])) if np.any(window_diff < 0) else 0
            score = net_change + frac_correct - wrong_mag

        elif phase == "ramp-down":
            # Net drop + fraction correct, penalise wrong-direction magnitude
            net_change     = window_segment[0] - window_segment[-1]
            frac_correct   = np.mean(window_diff < 0)
            wrong_mag      = np.mean(np.abs(window_diff[window_diff > 0])) if np.any(window_diff > 0) else 0
            score = net_change + frac_correct - wrong_mag

        elif phase == "stable":
            # Low total variation + small overall change
            total_var      = np.sum(np.abs(window_diff))
            overall_change = abs(window_segment[-1] - window_segment[0])
            score          = -(total_var + overall_change)

        if score > best_score:
            best_score = score
            best_start = t
            best_end   = t + window_size

    return best_start, best_end

def find_phase_split_points(
    gmt: np.ndarray,
    phases: list[str],
    tau: float = 20,
    smooth: bool = True,
    stable_window: int = 30,
    stable_threshold: float = 0.1,
    min_phase_length: int = 10,
    window_size: int | None = None,
) -> list[int]:
    gmt        = np.asarray(gmt).ravel()
    gmt_smooth = weighted_linear_smoothing(gmt, tau=tau) if smooth else gmt.copy()
    T          = len(gmt_smooth)

    split_points = []
    offset       = 0

    for k, phase in enumerate(phases):
        segment = gmt_smooth[offset:]

        start, end = _detect_phase(
            segment,
            phase=phase,
            stable_window=stable_window,
            stable_threshold=stable_threshold,
            min_phase_length=min_phase_length,
            window_size=window_size,
        )

        abs_end = offset + end
        if k < len(phases) - 1:
            split_points.append(abs_end)
        offset = abs_end

    return split_points