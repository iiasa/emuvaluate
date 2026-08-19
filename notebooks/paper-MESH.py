# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (climate-data-processing)
#     language: python
#     name: climate-data-processing
# ---

# %% [markdown]
# # Gridded paper figures
#
# Gridded analogue of `paper.py`, following the same three-step pattern
# throughout, for every figure:
#
# 1. **Preprocess** — `emuvaluate.preprocessing.preprocess_gridded` turns raw
#    `xr.DataArray` ensembles into `GriddedArray`s: flattened
#    `(ensemble, time, n_grid)` arrays that keep the lat/lon grid metadata
#    needed to reshape scores back into maps. Like its regional counterpart
#    it takes and returns `{indicator: ...}` dicts, and validates every
#    entry against the first one's grid. `remove_ensemble_mean=True` (for
#    the QQ-variability figures) subtracts the multi-ensemble mean,
#    isolating internal variability.
# 2. **Build error data** — `emuvaluate.metrics.build_error_data_gridded` (or
#    `build_crps_timeseries_data_gridded`) scores every gridpoint for a
#    chosen metric and ranks them.
# 3. **Plot** — `emuvaluate.plots.plot_map_gridded`,
#    `plot_timeseries_gridded`, `plot_qq_scatter`,
#    `plot_crps_timeseries_gridded` or `plot_psd_curves` render the figure.
#    None of the plotting functions preprocess data or compute scores
#    themselves. `plot_map_gridded` also produces the CRPS map, by building
#    `error_data` with `metric="crps"` — there is no separate CRPS-map
#    plotting function.

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
from emuvaluate.preprocessing import preprocess_gridded, select_members
from emuvaluate.metrics import (
    build_error_data_gridded, build_crps_timeseries_data_gridded,
    build_error_data_intervariable_correlation,
)
from emuvaluate.plots import (
    plot_map_gridded, plot_timeseries_gridded, plot_qq_scatter,
    plot_crps_timeseries_gridded, plot_psd_curves,
)

# %% [markdown]
# ## Data loading & unscaling
#
# Unchanged from the previous version of this notebook — loads the gridded
# emulator output and the matching simulation ensemble from disk and
# rescales both back to physical units.

# %%
def unscale(da_scaled, var_max, var_min):
    return (da_scaled + 1.) / 2. * (var_max - var_min) + var_min


data = joblib.load("/pdrive/projects/icigroup/SCALES-MESH/MESH/ssp245_0_1000_41epochs_flow_10steps_19Aug26.pkl")
print(data.keys())

gridded_input = xr.open_dataset("/pdrive/projects/icigroup/SCALES-MESH/MESH/ssp245_0_1000_41epochs_flow_10steps_19Aug26.pkl")
sim_data = xr.open_dataset("/pdrive/projects/icigroup/SCALES-MESH/MESH/ssp245_0_1000_41epochs_flow_10steps_19Aug26.pkl")

gridded_input["tas"].values[:] = data['pred'][:, :, 0, :, :]   # shape must exactly match (time, ensemble, lat, lon)
gridded_input["pr"].values[:] = data['pred'][:, :, 1, :, :]

sim_data["tas"].values[:] = data['sim'][:1000, :, 0, :, :]  # shape must exactly match (time, ensemble, lat, lon)
sim_data["pr"].values[:] = data['sim'][:1000, :, 1, :, :]

lol = xr.Dataset({
    "tas": unscale(gridded_input["tas"], data["tas_max"], data["tas_min"]),
    "pr": unscale(gridded_input["pr"], data["pr_max"], data["pr_min"]),
})
loli = xr.Dataset({
    "tas": unscale(sim_data["tas"], data["tas_max"], data["tas_min"]),
    "pr": unscale(sim_data["pr"], data["pr_max"], data["pr_min"]),
})

# %%
emu_ds = lol.transpose("ensemble", "time", "lat", "lon").astype("float64")
emu_ds.isel(ensemble=0, time=0).pr.plot()

# %%
sim_ds = loli.transpose("ensemble", "time", "lat", "lon").astype("float64")
sim_ds.isel(ensemble=0, time=0).pr.plot()

# %% [markdown]
# ## Error maps — annual means (MAE)
#
# `preprocess_gridded` canonicalises the grid, flattens `(lat, lon)` into
# `n_grid` and applies the temporal aggregation. Passing a dict does all
# indicators at once; `ref_lat`/`ref_lon` from the simulation's grid then
# guard every other array against a silent grid mismatch.

# %%
def as_dict(ds, members):
    """{'tas': DataArray, 'pr': DataArray} for a slice of ensemble members."""
    return {v: ds[v].isel(ensemble=members) for v in ("tas", "pr")}


sim_mo = preprocess_gridded(as_dict(sim_ds, slice(0, 20)), yearly_average=True)
REF = dict(ref_lat=sim_mo["tas"].lat, ref_lon=sim_mo["tas"].lon)

emu_mo = preprocess_gridded(as_dict(emu_ds, slice(0, 20)), yearly_average=True, **REF)
baseline_simsim_mo = preprocess_gridded(as_dict(sim_ds, slice(20, 40)), yearly_average=True, **REF)

error_data_mo = build_error_data_gridded(
    sim_mo, emu_mo, metric="mae",
    baseline_emulations={"Simulations vs Simulations": baseline_simsim_mo},
)
fig = plot_map_gridded(
    error_data_mo,
    save_path="/home/schwind/emuvaluate/plots/error_maps_ssp245_gridded.png",
)

# %% [markdown]
# ## Gridpoint pattern scaling baseline
#
# Unchanged from the previous version of this notebook — fits a
# gridpoint-wise Pattern Scaling regression against GMT on the simulation
# ensemble, then predicts it for the emulator's ensemble members.

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
print(gmt_emu_yearly.shape)
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

print(gmt_emu_yearly.shape)
print(emu_ds_yearly_tas["time"].values.shape)

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
# ## Error maps & timeseries — annual, vs. Pattern Scaling (MAE)
#
# `sim_ds_yearly_*` / `emu_ds_yearly_*` / `ps_*_da` are already annual, so
# `preprocess_gridded` is called with no temporal flags here — it only
# canonicalises the grid and flattens it.

# %%
sim_yr = preprocess_gridded(
    {"tas": sim_ds_yearly_tas.isel(ensemble=slice(0, 20)),
     "pr": sim_ds_yearly_pr.isel(ensemble=slice(0, 20))},
    **REF,
)
emu_yr = preprocess_gridded(
    {"tas": emu_ds_yearly_tas.isel(ensemble=slice(0, 20)),
     "pr": emu_ds_yearly_pr.isel(ensemble=slice(0, 20))},
    **REF,
)
ps_yr = preprocess_gridded({"tas": ps_tas_da, "pr": ps_pr_da}, **REF)

baseline_ps_yr = {"Pattern Scaling": ps_yr}

error_data_yr_ps = build_error_data_gridded(
    sim_yr, emu_yr, metric="mae",
    baseline_emulations=baseline_ps_yr,
    diff_baseline="Pattern Scaling",
)
fig, diff_fig = plot_map_gridded(
    error_data_yr_ps,
    show_difference=True,
    save_path="/home/schwind/emuvaluate/plots/error_maps_ssp119_gridded.png",
)

# %%
years = emu_ds_yearly_tas["time"].dt.year.values

fig = plot_timeseries_gridded(
    error_data_yr_ps, sim_yr, emu_yr,
    baseline_emulations=baseline_ps_yr,
    years=years,
    save_path="/home/schwind/emuvaluate/plots/timeseries_ssp245_gridded.png",
)

# Same data, but ranked by the Pattern Scaling baseline's error instead of
# the Emulator's — a separate ErrorData, since ranking_strategy affects
# which gridpoints are "best"/"median"/"worst".
error_data_yr_ps_ranked = build_error_data_gridded(
    sim_yr, emu_yr, metric="mae",
    baseline_emulations=baseline_ps_yr,
    ranking_strategy="Pattern Scaling",
    diff_baseline="Pattern Scaling",
)
fig = plot_timeseries_gridded(
    error_data_yr_ps_ranked, sim_yr, emu_yr,
    baseline_emulations=baseline_ps_yr,
    years=years,
    save_path="/home/schwind/emuvaluate/plots/timeseries_ssp119_gridded.png",
)

# %% [markdown]
# ## QQ variability — ensemble-mean removed, monthly (QQ-MAE)
#
# `preprocess_gridded(..., remove_ensemble_mean=True)` subtracts the
# multi-ensemble mean on the *full* raw ensemble first, so the
# `[0:20]`/`[20:40]` split below (main vs. "Simulations vs Simulations"
# baseline) shares the same ensemble-mean reference; `select_members` then
# splits the already-preprocessed result.

# %%
sim_det_full = preprocess_gridded(
    {"tas": sim_ds["tas"], "pr": sim_ds["pr"] * 86400}, remove_ensemble_mean=True, **REF,
)
emu_det_full = preprocess_gridded(
    {"tas": emu_ds["tas"], "pr": emu_ds["pr"] * 86400}, remove_ensemble_mean=True, **REF,
)

sim_det = select_members(sim_det_full, slice(0, 20))
emu_det = select_members(emu_det_full, slice(0, 20))
baseline_simsim_det = {"Simulations vs Simulations": select_members(sim_det_full, slice(20, 40))}

qq_error_data_map = build_error_data_gridded(
    sim_det, emu_det, metric="qq_mae", baseline_emulations=None,
)
fig1 = plot_map_gridded(
    qq_error_data_map,
    save_path="/home/schwind/emuvaluate/plots/error_variability_map_gridded.png",
)

qq_error_data_scatter = build_error_data_gridded(
    sim_det, emu_det, metric="qq_mae", baseline_emulations=baseline_simsim_det,
)
# plot_qq_scatter works for regional and gridded data alike, so it takes
# plain arrays — pass `.values` off the GriddedArrays.
def values(d):
    return {k: v.values for k, v in d.items()}


fig2 = plot_qq_scatter(
    qq_error_data_scatter, values(sim_det), values(emu_det),
    baseline_emulations={n: values(a) for n, a in baseline_simsim_det.items()},
    save_path="/home/schwind/emuvaluate/plots/error_variability_gridded.png",
)

# %% [markdown]
# ## CRPS map — monthly (vs. Simulations vs Simulations)
#
# `plot_map_gridded` also renders the CRPS map — build `error_data` with
# `metric="crps"` and everything else (ranking, colour scaling, the
# optional difference panel) works exactly like any other error metric.
#
# NOTE: the Pattern Scaling baseline (`ps_tas_da`/`ps_pr_da`) is
# *annual*-resolution (`T=83`), while the arrays here are monthly
# (`T≈1000`) — CRPS needs matching timesteps between the two sides of the
# comparison, so mixing them isn't meaningful. Only the time-compatible
# "Simulations vs Simulations" baseline is used for this monthly CRPS map;
# the CRPS-vs-Pattern-Scaling comparison is made in the annual CRPS
# timeseries below instead, where everything already shares the same annual
# grid.

# %%
sim_crps = preprocess_gridded(as_dict(sim_ds, slice(0, 20)), **REF)
emu_crps = preprocess_gridded(as_dict(emu_ds, slice(0, 20)), **REF)
baseline_simsim_crps = preprocess_gridded(as_dict(sim_ds, slice(20, 40)), **REF)

crps_map_error_data = build_error_data_gridded(
    sim_crps, emu_crps, metric="crps",
    baseline_emulations={"Simulations vs Simulations": baseline_simsim_crps},
)
fig_crps_map = plot_map_gridded(
    crps_map_error_data,
    save_path="/home/schwind/emuvaluate/plots/crps_map_ssp245_gridded.png",
)

# %% [markdown]
# ## CRPS timeseries — annual (vs. Pattern Scaling & Simulations vs Simulations)
#
# Everything here is already annual (`sim_yr`, `emu_yr`, `ps_yr`, ...), so
# the Pattern Scaling baseline is time-compatible and both baselines can be
# included.

# %%
baseline_simsim_yr = preprocess_gridded(
    {"tas": sim_ds_yearly_tas.isel(ensemble=slice(20, 40)),
     "pr": sim_ds_yearly_pr.isel(ensemble=slice(20, 40))},
    **REF,
)

crps_timeseries_data = build_crps_timeseries_data_gridded(
    sim_yr, emu_yr,
    baseline_emulations={
        "Pattern Scaling": ps_yr,
        "Simulations vs Simulations": baseline_simsim_yr,
    },
)
fig_crps_timeseries = plot_crps_timeseries_gridded(
    crps_timeseries_data,
    years=years,
    save_path="/home/schwind/emuvaluate/plots/crps_timeseries_ssp245_gridded.png",
)

# %% [markdown]
# ## Intervariable correlation — TAS vs PR relationship, annual means
#
# Gridded analogue of the regional intervariable-correlation map:
# `build_error_data_intervariable_correlation` scores every gridpoint by
# how much the comparison's TAS–PR correlation (pooling all members and
# timesteps at that gridpoint) differs from the simulation's. It's the one
# builder shared by both data flavours, so it takes plain arrays plus
# `lat`/`lon`. The result carries a single `tas-pr` "indicator", so the map
# draws one column rather than two identical ones.

# %%
intervar_error_data_gridded = build_error_data_intervariable_correlation(
    values(sim_yr), values(emu_yr),
    baseline_emulations={
        "Pattern Scaling": values(ps_yr),
        "Simulations vs Simulations": values(baseline_simsim_yr),
    },
    lat=sim_yr["tas"].lat, lon=sim_yr["tas"].lon,
)
fig = plot_map_gridded(
    intervar_error_data_gridded, show_difference=True,
    save_path="/home/schwind/emuvaluate/plots/intervariable_correlation_map_gridded.png",
)

# %% [markdown]
# ## Power spectral density — monthly, ensemble-mean removed
#
# `metric="psd_log_nmae"` scores every gridpoint by the NMAE between the
# simulated and the emulated log-PSD curve, so it drops straight into
# `plot_map_gridded`. `plot_psd_curves` is the accompanying ranking plot —
# period on the x-axis, spectral power on the y-axis, with a shaded 10–90 %
# band across ensemble members. Pass the same `nperseg` to the builder and
# the plot so the curves match the scores.

# %%
psd_error_data = build_error_data_gridded(
    sim_det, emu_det, metric="psd_log_nmae",
    baseline_emulations=baseline_simsim_det,
    psd_nperseg=256,
)
fig = plot_map_gridded(
    psd_error_data,
    save_path="/home/schwind/emuvaluate/plots/psd_log_nmae_map_gridded.png",
)
fig = plot_psd_curves(
    psd_error_data, values(sim_det), values(emu_det),
    baseline_emulations={n: values(a) for n, a in baseline_simsim_det.items()},
    nperseg=256,
    save_path="/home/schwind/emuvaluate/plots/psd_curves_gridded.png",
)

# %% [markdown]
# ### Choosing which gridpoints to show, and locking the y-axis
#
# Every ranking plot takes the same two options:
#
# * `selection` — `None` (default) gives best / median / worst. A list of
#   1-based ranks, e.g. `[1, 2, 3]`, shows the three best gridpoints. A list
#   of `(lat, lon)` pairs shows exactly those gridpoints (nearest match),
#   with each panel reporting where it falls in the ranking.
# * `share_y_per_row` — `False` by default. Set `True` to give every panel
#   in a row the same y-limits, making the columns directly comparable.

# %%
fig = plot_timeseries_gridded(
    error_data_yr_ps, sim_yr, emu_yr,
    baseline_emulations=baseline_ps_yr,
    years=years, selection=[1, 2, 3], share_y_per_row=True,
    save_path="/home/schwind/emuvaluate/plots/timeseries_top3_gridded.png",
)
fig = plot_timeseries_gridded(
    error_data_yr_ps, sim_yr, emu_yr,
    baseline_emulations=baseline_ps_yr,
    years=years, selection=[(0.0, 0.0), (50.0, 10.0)], share_y_per_row=True,
    save_path="/home/schwind/emuvaluate/plots/timeseries_named_gridpoints.png",
)
