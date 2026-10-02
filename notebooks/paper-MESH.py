# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: emuvaluate-test-19082026
#     language: python
#     name: emuvaluate-test-19082026
# ---

# %% [markdown]
# # MESH paper figures

# %% [markdown]
# Imports from the emuvaluate package: https://github.com/iiasa/emuvaluate 
#
# This notebook only works if the notebook is run with a kernel that has emuvaluate installed. See README.md for installation instructions. You additionally need to install ipykernel (`pip install ipykernel`) and install the environment with emuvaluate in it as a kernel (`python -m ipykernel install --user --name=name_of_kernal`). 

# %%
import numpy as np
import joblib
import xarray as xr

from emuvaluate.baseline_methods import (
    predict_pattern_scaling_gridded,
    fit_gridpoint_regressions_monthly,
    compute_gmt_area_weighted,
)
from emuvaluate.transforms import yearly_average, yearly_average_gridded
from emuvaluate.preprocessing import preprocess_gridded, select_members, ensemble_mean_gridded
from emuvaluate.metrics import (
    build_error_data_gridded, build_crps_timeseries_data_gridded,
    build_error_data_intervariable_correlation,build_spatial_correlation_curve_data_gridded, build_error_data_spatial_correlation_gridded
)
from emuvaluate.plots import (
    plot_map_gridded, plot_timeseries_gridded, plot_qq_scatter,
    plot_crps_timeseries_gridded, plot_psd_curves, plot_spatial_correlation_curves_gridded
)

# %% [markdown]
# # Configs
# - `DATA_PATH`: path to the MESH data on your local machine
# - `REFERENCE_PATH`: path to the reference NetCDF file (containing the correct coordinates and so on) on your local machine
#
# The data needed to reproduce the MESH evaluation figures in the study are archived on Zenodo: https://doi.org/10.5281/zenodo.23034786
#
# Set `DATA_PATH` to the path of "ssp245_mesh_emulations_and_simulations.pkl" and `REFERENCE_PATH` to the path of "reference_file.nc".

# %%
DATA_PATH = '/path/to/ssp245_mesh_emulations_and_simulations.pkl'
REFERENCE_PATH = '/path/to/reference_file.nc'


# %% [markdown]
# ## Data loading & unscaling
#
# Loads the gridded emulator output and the corresponding simulation ensemble, and rescales both to physical units.

# %%
def unscale(da_scaled, var_max, var_min):
    return (da_scaled + 1.0) / 2.0 * (var_max - var_min) + var_min


def build_dataset(pred_or_sim, tas_max, tas_min, pr_max, pr_min, coords):
    """
    pred_or_sim: array shaped (time, ensemble, channel, lat, lon), channel 0 = tas, 1 = pr
    coords: dict of 1-D coordinate arrays, dims = (time, ensemble, lat, lon)
    """
    dims = ("time", "ensemble", "lat", "lon")
    expected_shape = tuple(len(coords[d]) for d in dims)
    full_expected = expected_shape[:1] + (expected_shape[1],) + (2,) + expected_shape[2:]
    assert pred_or_sim.shape == full_expected, (
        f"got {pred_or_sim.shape}, expected {full_expected} "
        f"(time, ensemble, channel, lat, lon)"
    )

    tas_scaled = xr.DataArray(pred_or_sim[:, :, 0, :, :], dims=dims, coords=coords)
    pr_scaled = xr.DataArray(pred_or_sim[:, :, 1, :, :], dims=dims, coords=coords)

    ds = xr.Dataset(
        {
            "tas": unscale(tas_scaled, tas_max, tas_min).astype("float32"),
            "pr": unscale(pr_scaled, pr_max, pr_min).astype("float32"),
        }
    )
    ds["tas"].attrs.update(standard_name="air_temperature", units="K")
    ds["pr"].attrs.update(standard_name="precipitation_flux", units="kg m-2 s-1")
    return ds


data = joblib.load(f"{DATA_PATH}")

with xr.open_dataset(
    f"{REFERENCE_PATH}"
) as _ref:
    coords = {
        "time": _ref["time"].values,
        "lat": _ref["lat"].values,
        "lon": _ref["lon"].values,
        "ensemble": (
            _ref["ensemble"].values
            if "ensemble" in _ref.coords or "ensemble" in _ref.variables
            else np.arange(_ref.sizes["ensemble"])
        ),
    }

data_emu = build_dataset(
    data["pred"], data["tas_max"], data["tas_min"], data["pr_max"], data["pr_min"], coords
)
data_sim = build_dataset(
    data["sim"][: len(coords["time"])],
    data["tas_max"],
    data["tas_min"],
    data["pr_max"],
    data["pr_min"],
    coords,
)

# %%
emu_ds = data_emu.transpose("ensemble", "time", "lat", "lon").astype("float64")
emu_ds.isel(ensemble=0, time=0).pr.plot()

# %%
sim_ds = data_sim.transpose("ensemble", "time", "lat", "lon").astype("float64")
sim_ds.isel(ensemble=0, time=0).pr.plot()
print(sim_ds)


# %% [markdown]
# ## Error maps — annual means (MAE)
#
# Calculates the MAE between the first 20 ensemble members of the original simulation and:
#
# 1. the last 20 (independent) ensemble members of the simulation
# 2. the 20 emulated ensemble members, forced by the regional temperature and precipitation values of the same last 20 simulation ensemble members
#
# This compares natural variability and trend combined. The better the emulator, the closer the emulation–simulation MAE is to the simulation–simulation MAE.

# %%
def as_dict(ds, members):
    """Return a dict {'tas': DataArray, 'pr': DataArray} for the selected ensemble members.
    pr is multiplied by 86400."""
    out = {v: ds[v].isel(ensemble=members) for v in ("tas", "pr")}
    out["pr"] = out["pr"] * 86400
    return out


sim_mo = preprocess_gridded(as_dict(sim_ds, slice(0, 20)), yearly_average=True)
REF = dict(ref_lat=sim_mo["tas"].lat, ref_lon=sim_mo["tas"].lon)

emu_mo = preprocess_gridded(as_dict(emu_ds, slice(20, 40)), yearly_average=True, **REF)
baseline_simsim_mo = preprocess_gridded(as_dict(sim_ds, slice(20, 40)), yearly_average=True, **REF)

error_data_mo = build_error_data_gridded(
    sim_mo, emu_mo, metric="mae",
    baseline_emulations={"Simulations vs Simulations": baseline_simsim_mo},
)
fig = plot_map_gridded(
    error_data_mo,
    save_path="../plots/error_maps_ssp245_gridded.png",
)

# %% [markdown]
# ## Gridpoint pattern scaling baseline
#
# Fits a gridpoint-wise pattern scaling regression against GMT on the simulation ensemble, then applies it to the emulator's ensemble members.

# %%
gmt_sim = compute_gmt_area_weighted(sim_ds["tas"])              # (n_ensemble, T)

gmt_sim_yearly = yearly_average(gmt_sim[:, :, np.newaxis])
print(gmt_sim_yearly.shape)
sim_ds_yearly_tas = yearly_average_gridded(
    data=sim_ds["tas"],
    time_dim="time",
    drop_incomplete_years=True,
)

sim_ds_yearly_pr = yearly_average_gridded(
    data=sim_ds["pr"],
    time_dim="time",
    drop_incomplete_years=True,
)

gmt_emu = compute_gmt_area_weighted(emu_ds["tas"])              # (n_ensemble, T)

gmt_emu_yearly = yearly_average(gmt_emu[:, :, np.newaxis])

emu_ds_yearly_tas = yearly_average_gridded(
    data=emu_ds["tas"],
    time_dim="time",
    drop_incomplete_years=True,
)

emu_ds_yearly_pr = yearly_average_gridded(
    data=emu_ds["pr"],
    time_dim="time",
    drop_incomplete_years=True,
)

# fit against one member (or an ensemble mean) on the native grid
fit_tas = fit_gridpoint_regressions_monthly(
    global_series=gmt_sim_yearly[0],
    gridded_series=sim_ds_yearly_tas.isel(ensemble=0),
    monthly=False,
)
fit_pr = fit_gridpoint_regressions_monthly(
    global_series=gmt_sim_yearly[0],
    gridded_series=sim_ds_yearly_pr.isel(ensemble=0),
    monthly=False,
)

# predict for one ensemble member's GMT trajectory
gmt_emu_yearly_sq = gmt_emu_yearly[:, :, 0]  # drop the dummy axis -> (40, 83)

ps_tas_da = xr.concat(
    [
        predict_pattern_scaling_gridded(
            fit=fit_tas,
            global_series=gmt_emu_yearly_sq[i],   # (83,) — one member's GMT
            lat=emu_ds_yearly_tas["lat"].values,
            lon=emu_ds_yearly_tas["lon"].values,
            monthly=False,
            time_coord=emu_ds_yearly_tas["time"].values,
        ).isel(ensemble=0)
        for i in range(20)
    ],
    dim="ensemble",
).assign_coords(ensemble=np.arange(20))

ps_pr_da = xr.concat(
    [
        predict_pattern_scaling_gridded(
            fit=fit_pr,
            global_series=gmt_emu_yearly_sq[i],   # (83,) — one member's GMT
            lat=emu_ds_yearly_pr["lat"].values,
            lon=emu_ds_yearly_pr["lon"].values,
            monthly=False,
            time_coord=emu_ds_yearly_pr["time"].values,
        ).isel(ensemble=0)
        for i in range(20)
    ],
    dim="ensemble",
).assign_coords(ensemble=np.arange(20))

# %% [markdown]
# ## Error timeseries ranked by NMAE
#
# Compares the annual averages of the first 20 simulations with the ensemble mean of the 20 emulations, forced by the regional temperature and precipitation values of the last 20 independent simulations. It then plots the 5th percentile, median and 95th percentile grid points of the error distribution.
#
# This illustrates how well the emulator reproduces the trend and natural variability over time. The more closely the emulator ensemble resembles the simulation ensemble, the better the emulator. A good match even at the grid point with the highest error is especially convincing.

# %%
sim_yr = preprocess_gridded(
    {"tas": sim_ds_yearly_tas.isel(ensemble=slice(0, 20)),
     "pr": sim_ds_yearly_pr.isel(ensemble=slice(0, 20)) * 86400},
    **REF,
)
emu_yr = preprocess_gridded(
    {"tas": emu_ds_yearly_tas.isel(ensemble=slice(0, 20)),
     "pr": emu_ds_yearly_pr.isel(ensemble=slice(0, 20))* 86400},
    **REF,
)

ps_yr = preprocess_gridded({"tas": ps_tas_da, "pr": ps_pr_da* 86400}, **REF)

baseline_ps_yr = {'Pattern Scaling': ps_yr}

#error_data_yr_ps = build_error_data_gridded(
#    sim_yr, emu_yr, metric="nmae",
#    baseline_emulations=baseline_ps_yr,
#    diff_baseline='Simulation vs. Simulation',
#)

years = emu_ds_yearly_tas["time"].dt.year.values

#fig = plot_timeseries_gridded(
#    error_data_yr_ps, sim_yr, emu_yr,
#    baseline_emulations=baseline_ps_yr,
#    share_y_per_row = False,
#    years=years,
#    save_path="../plots/timeseries_ssp245_gridded.png",
#)

# Same data, but ranked by the Pattern Scaling baseline's error instead of
# the Emulator's — a separate ErrorData, since ranking_strategy affects
# which gridpoints are "best"/"median"/"worst".
error_data_yr_ps_ranked = build_error_data_gridded(
    sim_yr, emu_yr, metric="nmae",
    baseline_emulations=baseline_ps_yr,
    #ranking_strategy="Pattern Scaling",
    #diff_baseline="Pattern Scaling",
)
fig = plot_timeseries_gridded( # Choice and NMAE
    error_data_yr_ps_ranked, sim_yr, emu_yr,
    baseline_emulations=baseline_ps_yr,
    years=years,
    selection=[518, 5180, 9850],
    share_y_per_row = False,
    save_path="../plots/timeseries_ssp245_gridded_errorpercentiles.png",
)

# %% [markdown]
# ## Natural variability error map and ranked Q-Q plots (QQ-MAE map and QQ-NMAE ranked Q-Q plots)
#
# Isolates the natural variability in the simulations and emulations by subtracting the ensemble mean from each. Then, at each grid point, it calculates the distance between the quantiles of the natural variability distributions of all 40 simulation ensemble members and all 40 emulated ensemble members. Each emulated member is forced by the regional temperature and precipitation values of the corresponding simulation member. Finally, it plots Q-Q plots of the 5th percentile, median and 95th percentile grid points of the error distribution.
#
# The Q-Q plots illustrate how well the emulator reproduces the natural variability. The more closely the emulator ensemble resembles the simulation ensemble, the better the emulator. A good match even at the grid point with the highest error is especially convincing.

# %%
sim_det_full = preprocess_gridded(
    {"tas": sim_ds["tas"], "pr": sim_ds["pr"] * 86400}, remove_ensemble_mean=True, **REF,
)
emu_det_full = preprocess_gridded(
    {"tas": emu_ds["tas"], "pr": emu_ds["pr"] * 86400}, remove_ensemble_mean=True, **REF,
)

sim_det = select_members(sim_det_full, slice(0, 40))
emu_det = select_members(emu_det_full, slice(0, 40))
#baseline_simsim_det = {"Simulations vs Simulations": select_members(sim_det_full, slice(20, 40))}

qq_error_data_map = build_error_data_gridded(
    sim_det, emu_det, metric="qq_mae", #baseline_emulations=baseline_simsim_det,
)
fig1 = plot_map_gridded(
    qq_error_data_map,
    save_path="../plots/error_variability_map_gridded.png",
)

qq_error_data_scatter = build_error_data_gridded(
    sim_det, emu_det, metric="qq_nmae", #baseline_emulations=baseline_simsim_det,
)
# plot_qq_scatter works for regional and gridded data alike, so it takes
# plain arrays — pass `.values` off the GriddedArrays.
def values(d):
    return {k: v.values for k, v in d.items()}


#fig2 = plot_qq_scatter(
#    qq_error_data_scatter, values(sim_det), values(emu_det),
    #baseline_emulations={n: values(a) for n, a in baseline_simsim_det.items()},
#    save_path="../plots/error_variability_gridded.png",
#)
fig2 = plot_qq_scatter(
    qq_error_data_scatter, values(sim_det), values(emu_det),
    #baseline_emulations={n: values(a) for n, a in baseline_simsim_det.items()},
    selection=[518, 5180, 9850],
    save_path="../plots/error_variability_gridded.png",
)

# %% [markdown]
# ## CRPS map
#
# Calculates the CRPS (continuous ranked probability score) at each timestep and grid point:
#
# 1. between the first 20 simulations and the last 20 (independent) simulations
# 2. between the last 20 simulations and the emulations forced by the regional temperature and precipitation values of the first 20 simulations
#
# It then averages each CRPS over all timesteps at each grid point.
#
# The better the emulator, the closer the emulation–simulation CRPS is to the simulation–simulation CRPS. This metric jointly evaluates how well the emulator reproduces the natural variability and the trend.

# %%
from emuvaluate.preprocessing import scale_indicators

PR_MM_DAY = {"pr": 1}   # kg m-2 s-1 -> mm day-1

sim_crps = scale_indicators(
    preprocess_gridded(as_dict(sim_ds, slice(0, 20)), **REF), PR_MM_DAY)
emu_crps = scale_indicators(
    preprocess_gridded(as_dict(emu_ds, slice(20, 40)), **REF), PR_MM_DAY)
baseline_simsim_crps = scale_indicators(
    preprocess_gridded(as_dict(sim_ds, slice(20, 40)), **REF), PR_MM_DAY)

crps_map_error_data = build_error_data_gridded(
    sim_crps, emu_crps, metric="crps",
    baseline_emulations={"Simulations vs Simulations": baseline_simsim_crps},
)
fig_crps_map = plot_map_gridded(
    crps_map_error_data,
    units={"tas": "K", "pr": "mm per day"},
    save_path="../plots/crps_map_ssp245_gridded.png",
)


# %% [markdown]
# ## Great circle distance spatial correlation
#
# Isolates the natural variability in the simulations and emulations by subtracting the ensemble mean from each. At each grid point, it then calculates the average spatial correlation with other grid points at different great circle distances. Next, it calculates the MAE between the resulting distance–correlation curves:
#
# 1. between the first 20 simulations and the last 20 (independent) simulations
# 2. between the last 20 simulations and the emulations forced by the regional temperature and precipitation values of the first 20 simulations
#
# It plots this MAE on a map and plots the distance–correlation curves of the best (5th percentile), median and worst (95th percentile) grid points of the error distribution.
#
# The better the emulator, the closer the emulation–simulation MAE is to the simulation–simulation MAE. This metric evaluates how well the emulator reproduces the spatial correlations in the natural variability. The distance–correlation curves illustrate this at individual grid points. The more closely the emulator ensemble resembles the simulation ensemble, the better the emulator. A good match even at the grid point with the highest error is especially convincing.

# %%
def as_dict(ds, members):
    """{'tas': DataArray, 'pr': DataArray} for a slice of ensemble members."""
    return {v: ds[v].isel(ensemble=members) for v in ("tas", "pr")}

sim = preprocess_gridded(as_dict(sim_ds, slice(0, 20)), remove_ensemble_mean=True)
REF = dict(ref_lat=sim["tas"].lat, ref_lon=sim["tas"].lon)
emu = preprocess_gridded(as_dict(emu_ds, slice(20, None)), remove_ensemble_mean=True, **REF)
#ss  = preprocess_gridded(as_dict(sim_ds, slice(20, None)), remove_ensemble_mean=True, **REF)

cd = build_spatial_correlation_curve_data_gridded(
    sim, emu,#, baseline_emulations={"Simulations vs Simulations": ss},
    n_partners=2000, max_samples=None,
)
ed = build_error_data_spatial_correlation_gridded(cd)

_ = plot_map_gridded(ed, show_difference=False, save_path = '../plots/great_circle_distance_map.png')
_ = plot_spatial_correlation_curves_gridded(ed, cd, share_y_per_row=True,selection=[518, 5180, 9850],save_path = '../plots/great_circle_distance_timeseries.png')

# %%
