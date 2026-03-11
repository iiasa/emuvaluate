"""
baseline_methods/pattern_scaling.py
------------------------------------
Pattern scaling baseline emulation methods.

Fits per-region linear regressions between a global mean temperature (GMT)
series and regional climate indicators, optionally handling ramp-down phases
and monthly stratification.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_regional_regressions(
    global_series: np.ndarray,
    regional_series: np.ndarray,
    train_ramp_down: bool = False,
) -> dict:
    """
    Fits linear regressions per region.

    If train_ramp_down=False:
        region = a + b * global
    If train_ramp_down=True:
        - Fit regression on ramp-up phase
        - Fit regression on residuals during ramp-down

    Parameters
    ----------
    global_series    : (T,) or (T, 1) — GMT timeseries
    regional_series  : (T, n_regions)
    train_ramp_down  : whether to fit a separate ramp-down correction

    Returns
    -------
    dict with keys:
        slopes_up       : (n_regions,)
        intercepts_up   : (n_regions,)
        slopes_down     : (n_regions,) or None
        intercepts_down : (n_regions,) or None
    """
    global_series   = np.asarray(global_series).reshape(-1, 1)
    regional_series = np.asarray(regional_series)
    n_regions = regional_series.shape[1]
    n_time    = len(global_series)

    peak_idx      = np.argmax(global_series[:, 0])
    ramp_up_idx   = np.arange(0, peak_idx + 1)
    ramp_down_idx = np.arange(peak_idx + 1, n_time)

    slopes_up      = np.zeros(n_regions)
    intercepts_up  = np.zeros(n_regions)
    slopes_down    = np.zeros(n_regions) if train_ramp_down else None
    intercepts_down = np.zeros(n_regions) if train_ramp_down else None

    for i in range(n_regions):
        model_up = LinearRegression()
        model_up.fit(global_series[ramp_up_idx], regional_series[ramp_up_idx, i])
        slopes_up[i]     = model_up.coef_[0]
        intercepts_up[i] = model_up.intercept_

        if train_ramp_down and len(ramp_down_idx) > 0:
            pred_up   = intercepts_up[i] + slopes_up[i] * global_series[ramp_down_idx, 0]
            residuals = regional_series[ramp_down_idx, i] - pred_up
            model_down = LinearRegression(fit_intercept=False)
            model_down.fit(
                (global_series[peak_idx] - global_series[ramp_down_idx]),
                residuals,
            )
            slopes_down[i]     = model_down.coef_[0]
            intercepts_down[i] = model_down.intercept_

    return {
        "slopes_up":      slopes_up,
        "intercepts_up":  intercepts_up,
        "slopes_down":    slopes_down,
        "intercepts_down": intercepts_down,
    }


def fit_regional_regressions_monthly(
    global_series: np.ndarray,
    regional_series: np.ndarray,
    train_ramp_down: bool = False,
    monthly: bool = True,
) -> list[dict] | dict:
    """
    Fit per-region regressions, optionally stratified by calendar month.

    For monthly data (monthly=True) fits one set of regression coefficients
    per calendar month (12 dicts). For annual data (monthly=False) wraps
    fit_regional_regressions and returns a single dict.

    Parameters
    ----------
    global_series   : (T,) — GMT timeseries (monthly or annual)
    regional_series : (T, n_regions)
    train_ramp_down : whether to fit a ramp-down correction
    monthly         : if True, stratify by calendar month

    Returns
    -------
    list of 12 dicts (monthly=True) or a single dict (monthly=False),
    each dict has keys: slopes_up, intercepts_up, slopes_down, intercepts_down
    """
    global_series   = np.asarray(global_series).ravel()
    regional_series = np.asarray(regional_series)
    T = len(global_series)

    if not monthly:
        return fit_regional_regressions(
            global_series, regional_series, train_ramp_down=train_ramp_down
        )

    if T % 12 != 0:
        print(
            f"Warning: T={T} is not divisible by 12. "
            f"Trailing {T % 12} timestep(s) will be ignored."
        )

    monthly_fits = []
    for month in range(12):
        month_idx       = np.arange(month, T, 12)
        global_m        = global_series[month_idx]
        regional_m      = regional_series[month_idx]
        monthly_fits.append(
            fit_regional_regressions(
                global_m, regional_m, train_ramp_down=train_ramp_down
            )
        )

    return monthly_fits   # list of 12 dicts, index 0 = January


def predict_pattern_scaling(
    fit: list[dict] | dict,
    global_series: np.ndarray,
    monthly: bool = True,
    n_members: int = 1,
) -> np.ndarray:
    """
    Emulate regional climate from a GMT timeseries using fitted pattern
    scaling coefficients.

    Parameters
    ----------
    fit           : output of fit_regional_regressions_monthly —
                    list of 12 dicts (monthly) or a single dict (annual)
    global_series : (T,) — GMT timeseries to emulate
    monthly       : must match how fit was produced
    n_members     : number of output ensemble members (all identical —
                    pattern scaling is deterministic, but we replicate to
                    match the (n_members, T, n_regions) interface)

    Returns
    -------
    np.ndarray, shape (n_members, T, n_regions)
    """
    global_series = np.asarray(global_series).ravel()
    T = len(global_series)

    # ── Annual ────────────────────────────────────────────────────────────────
    if not monthly:
        assert isinstance(fit, dict), \
            "For annual data, fit must be a single dict (monthly=False)"
        n_regions  = len(fit["slopes_up"])
        prediction = np.zeros((T, n_regions))
        peak_gmt   = global_series.max()

        for i in range(n_regions):
            base = fit["intercepts_up"][i] + fit["slopes_up"][i] * global_series

            if fit["slopes_down"] is not None:
                # Identify ramp-down timesteps (after peak)
                peak_idx      = np.argmax(global_series)
                ramp_down_mask = np.arange(T) > peak_idx
                correction     = np.zeros(T)
                correction[ramp_down_mask] = (
                    fit["slopes_down"][i]
                    * (peak_gmt - global_series[ramp_down_mask])
                )
                base += correction

            prediction[:, i] = base

        return np.stack([prediction] * n_members, axis=0)  # (n_members, T, n_regions)

    # ── Monthly ───────────────────────────────────────────────────────────────
    assert isinstance(fit, list) and len(fit) == 12, \
        "For monthly data, fit must be a list of 12 dicts (monthly=True)"

    n_regions  = len(fit[0]["slopes_up"])
    prediction = np.zeros((T, n_regions))

    for month in range(12):
        month_idx = np.arange(month, T, 12)
        if len(month_idx) == 0:
            continue
        global_m  = global_series[month_idx]
        peak_gmt  = global_m.max()
        peak_idx_m = np.argmax(global_m)

        for i in range(n_regions):
            base = fit[month]["intercepts_up"][i] + fit[month]["slopes_up"][i] * global_m

            if fit[month]["slopes_down"] is not None:
                ramp_down_mask = np.arange(len(month_idx)) > peak_idx_m
                correction     = np.zeros(len(month_idx))
                correction[ramp_down_mask] = (
                    fit[month]["slopes_down"][i]
                    * (peak_gmt - global_m[ramp_down_mask])
                )
                base += correction

            prediction[month_idx, i] = base

    return np.stack([prediction] * n_members, axis=0)  # (n_members, T, n_regions)