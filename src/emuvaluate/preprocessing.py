"""
preprocessing.py
----------------
Turns raw simulation / emulator ensembles into the arrays the scoring
(`metrics.py`) and plotting (`plots.py`) functions expect — and nothing
else. No metric scoring and no plotting happens here.

Two entry points, one per data flavour:

    preprocess_regional(...)  AR6-region ensembles: plain numpy arrays of
                              shape (n_members, T, n_regions).
    preprocess_gridded(...)   Gridded simulation ensembles: xarray
                              DataArrays with dims (ensemble, time, lat, lon)
                              in any order.

Both accept either a single array or a ``{indicator: array}`` dict — pass a
dict and you get a dict back with the same keys, which is exactly the shape
the `metrics.build_error_data_*` builders and every plotting function want:

    sim = preprocess_regional({"tas": tas_raw, "pr": pr_raw}, yearly_average=True)
    # -> {"tas": ..., "pr": ...}

Both apply the same optional steps, in the same order, so regional and
gridded results stay directly comparable:

    1. temporal aggregation    (yearly_average  *or*  month_selection)
    2. remove_ensemble_mean    (subtract the member-axis mean)
    3. smooth                  (locally-weighted-regression "forced
                                 response" smoothing, per member)

`remove_ensemble_mean` subtracts the multi-ensemble mean at every
timestep/unit. Whatever trend and seasonal cycle are common across ensemble
members is removed by construction, leaving internal variability around the
ensemble's forced response. Apply it *before* slicing to an ensemble subset
if two subsets (e.g. the main ensemble and a "Simulations vs Simulations"
baseline drawn from the same pool) need to share the same ensemble-mean
reference.

`preprocess_gridded` additionally canonicalises the dimension order/grid and
flattens (lat, lon) into a single axis, returning a `GriddedArray` that keeps
the grid metadata needed later to reshape per-gridpoint scores back into
(lat, lon) maps and to label an individual gridpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .transforms import yearly_average as _yearly_average
from .transforms import select_month as _select_month
from .transforms import weighted_linear_smoothing as _weighted_linear_smoothing

__all__ = [
    "preprocess_regional",
    "preprocess_gridded",
    "select_members",
    "scale_indicators",
    "GriddedArray",
]


def _remove_ensemble_mean(data: np.ndarray) -> np.ndarray:
    """Subtract the member-axis (axis=0) mean at every timestep/unit."""
    return data - data.mean(axis=0, keepdims=True)


def _apply_pipeline(
    data: np.ndarray,
    *,
    yearly_average: bool,
    month_selection: int | None,
    remove_ensemble_mean: bool,
    smooth: bool,
    smooth_tau: float,
    smooth_monthly: bool,
) -> np.ndarray:
    """Shared step sequence for `preprocess_regional` and `preprocess_gridded`."""
    if yearly_average and month_selection is not None:
        raise ValueError("yearly_average and month_selection are mutually exclusive")

    out = data.copy()

    # 1. Temporal aggregation
    if yearly_average:
        out = _yearly_average(out)
    elif month_selection is not None:
        out = _select_month(out, month_selection)

    # 2. Remove the ensemble (member-axis) mean
    if remove_ensemble_mean:
        out = _remove_ensemble_mean(out)

    # 3. Locally-weighted-regression smoothing ("forced response")
    if smooth:
        out = _weighted_linear_smoothing(out, tau=smooth_tau, monthly=smooth_monthly)

    return out


def preprocess_regional(
    data: np.ndarray | dict[str, np.ndarray],
    *,
    yearly_average: bool = False,
    month_selection: int | None = None,
    remove_ensemble_mean: bool = False,
    smooth: bool = False,
    smooth_tau: float = 20,
    smooth_monthly: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Preprocess a regional (AR6-region) ensemble before scoring/plotting.

    Parameters
    ----------
    data : (n_members, T, n_regions) array, or ``{indicator: array}``.
        A dict in gives a dict out, with the same keys and order — the shape
        `metrics.build_error_data_regional` expects for `sim` / `emulator`.
    yearly_average       : collapse months -> annual means
    month_selection      : restrict to a single calendar month (1-12), mutually
                            exclusive with yearly_average
    remove_ensemble_mean : subtract the ensemble-mean (member-axis) signal at
                            every timestep/region — see module docstring.
                            Apply before slicing to an ensemble subset if the
                            subsets need to share the same ensemble-mean
                            reference.
    smooth                : replace each member's time series with its
                            locally-weighted-regression-smoothed ("forced
                            response") curve.
    smooth_tau            : smoothing bandwidth (in timesteps) used by `smooth`.
    smooth_monthly        : if True, `smooth` smooths each calendar month
                            independently across years (only meaningful for
                            still-monthly data).

    Returns
    -------
    np.ndarray (or dict of them) — a preprocessed copy of *data*, same layout.
    """
    kwargs = dict(
        yearly_average=yearly_average,
        month_selection=month_selection,
        remove_ensemble_mean=remove_ensemble_mean,
        smooth=smooth,
        smooth_tau=smooth_tau,
        smooth_monthly=smooth_monthly,
    )
    if isinstance(data, dict):
        return {k: _apply_pipeline(np.asarray(v), **kwargs) for k, v in data.items()}
    return _apply_pipeline(np.asarray(data), **kwargs)


@dataclass
class GriddedArray:
    """
    A preprocessed gridded ensemble, flattened to (ensemble, time, n_grid),
    plus the grid metadata needed to reshape per-gridpoint results back to
    (lat, lon) or to label an individual gridpoint.
    """

    values: np.ndarray  # (n_ensemble, T, n_grid)
    lat: np.ndarray  # (n_lat,)
    lon: np.ndarray  # (n_lon,)

    @property
    def n_lat(self) -> int:
        return self.lat.size

    @property
    def n_lon(self) -> int:
        return self.lon.size

    @property
    def n_grid(self) -> int:
        return self.n_lat * self.n_lon

    def gridpoint_label(self, flat_idx: int) -> str:
        """Human-readable ``(lat°, lon°)`` label for a flattened gridpoint index."""
        lat_idx, lon_idx = np.unravel_index(flat_idx, (self.n_lat, self.n_lon))
        return f"({self.lat[lat_idx]:.2f}\N{DEGREE SIGN}, {self.lon[lon_idx]:.2f}\N{DEGREE SIGN})"

    def isel_ensemble(self, member_slice) -> "GriddedArray":
        """Return a copy restricted to a subset of ensemble members (axis=0)."""
        return GriddedArray(values=self.values[member_slice], lat=self.lat, lon=self.lon)


def _canonicalize_gridded(
    da: xr.DataArray,
    name: str,
    ref_lat: np.ndarray | None,
    ref_lon: np.ndarray | None,
) -> np.ndarray:
    """Transpose to (ensemble, time, lat, lon) and validate the grid, returning plain values."""
    missing = {"ensemble", "time", "lat", "lon"} - set(da.dims)
    if missing:
        raise ValueError(f"'{name}' is missing dims {missing}; has {da.dims}")
    da = da.transpose("ensemble", "time", "lat", "lon")
    if ref_lat is not None and ref_lon is not None:
        if not (
            np.allclose(da["lat"].values, ref_lat)
            and np.allclose(da["lon"].values, ref_lon)
        ):
            raise ValueError(f"'{name}' is not on the expected lat/lon grid")
    return da.values


def _preprocess_gridded_one(
    data: xr.DataArray,
    name: str,
    ref_lat,
    ref_lon,
    pipeline_kwargs: dict,
) -> GriddedArray:
    lat = data["lat"].values
    lon = data["lon"].values
    values = _canonicalize_gridded(data, name, ref_lat, ref_lon)
    n_ens, n_time = values.shape[0], values.shape[1]
    flat = values.reshape(n_ens, n_time, lat.size * lon.size)
    flat = _apply_pipeline(flat, **pipeline_kwargs)
    return GriddedArray(values=flat, lat=lat, lon=lon)


def preprocess_gridded(
    data: xr.DataArray | dict[str, xr.DataArray],
    *,
    yearly_average: bool = False,
    month_selection: int | None = None,
    remove_ensemble_mean: bool = False,
    smooth: bool = False,
    smooth_tau: float = 20,
    smooth_monthly: bool = False,
    ref_lat: np.ndarray | None = None,
    ref_lon: np.ndarray | None = None,
) -> GriddedArray | dict[str, GriddedArray]:
    """
    Preprocess a gridded ensemble before scoring/plotting.

    Parameters
    ----------
    data : xr.DataArray, or ``{indicator: xr.DataArray}``
        Dims {"ensemble", "time", "lat", "lon"} in any order (transposed
        internally). A dict in gives a dict of `GriddedArray` out — the shape
        `metrics.build_error_data_gridded` expects. When a dict is passed and
        no `ref_lat`/`ref_lon` is given, the first entry's grid becomes the
        reference every other entry is validated against.
    yearly_average, month_selection, remove_ensemble_mean, smooth,
    smooth_tau, smooth_monthly :
        Same semantics as `preprocess_regional`. Applied along the time axis
        *after* the grid has been flattened, so results are numerically
        identical to running the regional pipeline on a
        (ensemble, time, n_grid) array. `remove_ensemble_mean` uses
        whichever ensemble members are present in *data* — pass the full
        ensemble (before slicing to a member subset) if two subsets need
        to share the same ensemble-mean reference; `GriddedArray.isel_ensemble`
        can then split the result.
    ref_lat, ref_lon : optional
        Reference grid to validate *data* against, e.g. the simulation's
        grid — so an emulator/baseline field can't silently be on a
        different or misaligned grid:

            sim = preprocess_gridded({"tas": ..., "pr": ...})
            emu = preprocess_gridded({"tas": ..., "pr": ...},
                                     ref_lat=sim["tas"].lat, ref_lon=sim["tas"].lon)

    Returns
    -------
    GriddedArray, or ``{indicator: GriddedArray}``
    """
    pipeline_kwargs = dict(
        yearly_average=yearly_average,
        month_selection=month_selection,
        remove_ensemble_mean=remove_ensemble_mean,
        smooth=smooth,
        smooth_tau=smooth_tau,
        smooth_monthly=smooth_monthly,
    )
    if not isinstance(data, dict):
        return _preprocess_gridded_one(data, "data", ref_lat, ref_lon, pipeline_kwargs)

    out: dict[str, GriddedArray] = {}
    for key, da in data.items():
        res = _preprocess_gridded_one(da, key, ref_lat, ref_lon, pipeline_kwargs)
        if ref_lat is None and ref_lon is None:
            ref_lat, ref_lon = res.lat, res.lon
        out[key] = res
    return out


def select_members(data, members):
    """
    Slice the ensemble (member) axis, whatever the container.

    Works on a plain ``(n_members, T, n_units)`` array, a `GriddedArray`, or a
    ``{indicator: ...}`` dict of either — so splitting a preprocessed ensemble
    into a main half and a "Simulations vs Simulations" baseline half stays a
    one-liner regardless of data flavour:

        sim = select_members(all_sim, slice(20, None))
        baseline = select_members(all_sim, slice(0, 20))

    Parameters
    ----------
    data    : array, GriddedArray, or dict of either
    members : anything numpy accepts on axis 0 — a slice, a list of indices,
              a boolean mask

    Returns
    -------
    The same container type as *data*.
    """
    if isinstance(data, dict):
        return {k: select_members(v, members) for k, v in data.items()}
    if isinstance(data, GriddedArray):
        return data.isel_ensemble(members)
    return np.asarray(data)[members]


def scale_indicators(data: dict, factors: dict[str, float]):
    """
    Multiply selected indicators by a constant — unit conversion, mostly.

    Indicators not named in *factors* are passed through untouched, so
    converting precipitation from kg m⁻² s⁻¹ to mm day⁻¹ while leaving
    temperature alone is:

        sim = scale_indicators(sim, {"pr": 86400})

    Parameters
    ----------
    data    : ``{indicator: array}`` or ``{indicator: GriddedArray}``
    factors : ``{indicator: multiplier}``

    Returns
    -------
    A new dict with the same keys.
    """
    unknown = set(factors) - set(data)
    if unknown:
        raise ValueError(f"factors names unknown indicator(s) {sorted(unknown)}")

    def _scaled(value, factor):
        if isinstance(value, GriddedArray):
            return GriddedArray(values=value.values * factor, lat=value.lat, lon=value.lon)
        return np.asarray(value) * factor

    return {k: _scaled(v, factors[k]) if k in factors else v for k, v in data.items()}

def ensemble_mean_regional(data):
    """
    Collapse the ensemble (member) axis to its mean, keeping it as one member.
 
    ``(n_members, T, n_units)`` -> ``(1, T, n_units)``, or the same for every
    entry of a ``{indicator: array}`` dict.
 
    Keeping the member axis rather than dropping it means the result is still
    the shape every builder and plotting function expects — it just describes
    a one-member "ensemble" that happens to be the mean. So
    ``metric="mae"`` between two of these is the MAE between the two ensemble
    means, and `plots.plot_timeseries_regional` draws one line per source
    instead of a spread:
 
        mean_sim = ensemble_mean(sim)
        mean_emu = ensemble_mean(emu)
        ed = build_error_data_regional(mean_sim, mean_emu, region_names, metric="mae")
 
    Use this instead of ``preprocess_regional(..., smooth=True)`` when you
    want the forced response as the ensemble mean rather than as a
    locally-smoothed per-member curve.
 
    Parameters
    ----------
    data : array, or ``{indicator: array}``
 
    Returns
    -------
    The same container type as *data*.
    """
    if isinstance(data, dict):
        return {k: ensemble_mean_regional(v) for k, v in data.items()}
    return np.asarray(data).mean(axis=0, keepdims=True)
 
 
def ensemble_mean_gridded(data):
    """
    Gridded analogue of `ensemble_mean`: collapse the member axis of a
    `GriddedArray` to its mean, keeping it as one member and preserving the
    lat/lon metadata.
 
    Accepts a `GriddedArray` or a ``{indicator: GriddedArray}`` dict — the
    shape `preprocess_gridded` returns — so it drops straight into the gridded
    pipeline:
 
        sim = preprocess_gridded({"tas": ..., "pr": ...}, yearly_average=True)
        emu = preprocess_gridded({"tas": ..., "pr": ...}, yearly_average=True, **REF)
        ed = build_error_data_gridded(
            ensemble_mean_gridded(sim), ensemble_mean_gridded(emu), metric="mae",
        )
 
    Plain arrays are passed through to `ensemble_mean`, so this also works on
    ``mesh_array.values`` if that is what you have in hand.
 
    Parameters
    ----------
    data : GriddedArray, array, or ``{indicator: ...}`` dict of either
 
    Returns
    -------
    The same container type as *data*.
    """
    if isinstance(data, dict):
        return {k: ensemble_mean_gridded(v) for k, v in data.items()}
    if isinstance(data, GriddedArray):
        return GriddedArray(
            values=data.values.mean(axis=0, keepdims=True),
            lat=data.lat,
            lon=data.lon,
        )
    return ensemble_mean_regional(data)
