"""
baseline_methods/pattern_scaling.py
------------------------------------
Pattern scaling baseline emulation methods.

Fits per-region linear regressions between a global mean temperature (GMT)
series and regional climate indicators, optionally handling ramp-down phases
and monthly stratification.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
import xarray as xr



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
    global_series : (T,) or (num_ensemble, T) — GMT timeseries to emulate
    monthly       : must match how fit was produced
    n_members     : number of output ensemble members (all identical —
                    pattern scaling is deterministic, but we replicate to
                    match the (n_members, T, n_regions) interface)
    Returns
    -------
    np.ndarray, shape (n_members, T, n_regions)
                 or  (num_ensemble, n_members, T, n_regions)
    """
    global_series = np.asarray(global_series)
    batched = global_series.ndim == 2  # (num_ensemble, T)

    if batched:
        results = [
            predict_pattern_scaling(fit, global_series[i], monthly=monthly, n_members=n_members)
            for i in range(global_series.shape[0])
        ]
        return np.stack(results, axis=0)  # (num_ensemble, n_members, T, n_regions)

    global_series = global_series.ravel()  # (T,)
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
                peak_idx       = np.argmax(global_series)
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
        global_m   = global_series[month_idx]
        peak_gmt   = global_m.max()
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



# ─────────────────────────────────────────────────────────────────────────────
# GMT — computed separately from the fit/predict functions, on purpose
# ─────────────────────────────────────────────────────────────────────────────
def compute_gmt_area_weighted(tas: xr.DataArray) -> np.ndarray:
    """
    Area-weighted (cos-latitude) global-mean-temperature timeseries from a
    gridded simulation field.

    Kept standalone (not called internally by the fit/predict functions
    below) so you can compute it once from the simulation and feed the
    result into both `fit_gridpoint_regressions_monthly` and
    `predict_pattern_scaling_mesh`.

    Parameters
    ----------
    tas : xr.DataArray
        Field with dims including "lat" and "lon", plus optionally "time"
        and/or "ensemble" (any order — dims are matched by name).

    Returns
    -------
    np.ndarray
        Shape (n_ensemble, T) if `tas` has an "ensemble" dim, else (T,).
    """
    weights = np.cos(np.deg2rad(tas["lat"]))
    gmt = tas.weighted(weights).mean(dim=("lat", "lon"))
    if "ensemble" in gmt.dims:
        gmt = gmt.transpose("ensemble", "time")
    elif "time" in gmt.dims:
        gmt = gmt.transpose("time")
    return gmt.values


# ─────────────────────────────────────────────────────────────────────────────
# Fit — one linear (ramp-up/ramp-down) model per gridpoint
# ─────────────────────────────────────────────────────────────────────────────
def fit_gridpoint_regressions_monthly(
    global_series: np.ndarray,
    gridded_series: xr.DataArray,
    train_ramp_down: bool = False,
    monthly: bool = True,
) -> list[dict] | dict:
    """
    Gridpoint analogue of `fit_regional_regressions_monthly`.

    `fit_regional_regressions_monthly` already fits one independent
    regression per column of a (T, n_regions) array — a gridpoint is
    nothing more than another kind of column, so this just flattens the
    (lat, lon) grid into a single axis and calls the existing SCALES
    function unchanged.

    Parameters
    ----------
    global_series  : (T,) — GMT timeseries, e.g. from
                     `compute_gmt_area_weighted(sim_ds["tas"])`
    gridded_series : xr.DataArray, dims (time, lat, lon) — the field to fit
                     against (e.g. a single simulation ensemble member, or
                     an ensemble mean already reduced over "ensemble").
    train_ramp_down, monthly : same semantics as in
                     `fit_regional_regressions_monthly`.

    Returns
    -------
    list of 12 dicts (monthly=True) or a single dict (monthly=False), same
    keys as `fit_regional_regressions_monthly` (slopes_up, intercepts_up,
    slopes_down, intercepts_down), except each coefficient array has length
    n_lat * n_lon (flattened row-major over the grid). Reshape any of them
    with `.reshape(len(lat), len(lon))` if you need coefficient maps
    directly; `predict_pattern_scaling_mesh` handles this for you for the
    predicted field itself.
    """
    gridded_series = gridded_series.transpose("time", "lat", "lon")
    n_time = gridded_series.sizes["time"]
    n_lat = gridded_series.sizes["lat"]
    n_lon = gridded_series.sizes["lon"]
    flat = gridded_series.values.reshape(n_time, n_lat * n_lon)

    return fit_regional_regressions_monthly(
        global_series, flat, train_ramp_down=train_ramp_down, monthly=monthly
    )


# ─────────────────────────────────────────────────────────────────────────────
# Predict — takes a single GMT timeseries, returns a gridded field
# ─────────────────────────────────────────────────────────────────────────────
def predict_pattern_scaling_gridded(
    fit: list[dict] | dict,
    global_series: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    monthly: bool = True,
    n_members: int = 1,
    time_coord: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Gridpoint analogue of `predict_pattern_scaling`.

    Takes a single (non-batched) GMT timeseries — call this once per
    ensemble member if you need to emulate several — and returns a
    properly-shaped, labelled gridded DataArray instead of a flat
    (n_members, T, n_grid) array.

    Parameters
    ----------
    fit           : output of `fit_gridpoint_regressions_monthly`
    global_series : (T,) — a single GMT timeseries
    lat, lon      : 1-D coordinate arrays of the grid `fit` was computed on
                    (must match the (lat, lon) used in
                    `fit_gridpoint_regressions_monthly`)
    monthly       : must match how `fit` was produced
    n_members     : number of (identical) output ensemble members —
                    pattern scaling is deterministic, replicated only to
                    match the SCALES interface
    time_coord    : optional datetime64 array of length T to label the
                    output "time" dim; defaults to an integer index

    Returns
    -------
    xr.DataArray, dims (ensemble, time, lat, lon)
    """
    global_series = np.asarray(global_series).ravel()
    T = len(global_series)
    n_lat, n_lon = len(lat), len(lon)

    prediction = predict_pattern_scaling(
        fit, global_series, monthly=monthly, n_members=n_members
    )  # (n_members, T, n_lat * n_lon)
    prediction = prediction.reshape(n_members, T, n_lat, n_lon)

    return xr.DataArray(
        prediction,
        dims=("ensemble", "time", "lat", "lon"),
        coords={
            "ensemble": np.arange(n_members),
            "time": time_coord if time_coord is not None else np.arange(T),
            "lat": lat,
            "lon": lon,
        },
        name="pattern_scaling_prediction",
    )
