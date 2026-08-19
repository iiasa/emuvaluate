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
# # Regional figures — MischMasch emulator
#
# Same three-step pattern as `paper.py`, run against the MischMasch
# emulator's output instead:
#
# 1. **Preprocess** — `emuvaluate.preprocessing.preprocess_regional` turns
#    raw ensembles into the arrays everything else needs. It takes and
#    returns `{indicator: array}` dicts. `remove_ensemble_mean=True` (for
#    the QQ-variability / correlation figures) subtracts the multi-ensemble
#    mean, isolating internal variability; `smooth=True` (for the
#    "locally-smoothed forced response" figures) applies
#    `weighted_linear_smoothing` instead of using the raw annual series.
# 2. **Build error data** — `emuvaluate.metrics.build_error_data_regional`
#    (or `build_correlation_data`) scores every region for a chosen metric
#    and ranks them.
# 3. **Plot** — `emuvaluate.plots.plot_map_regional`,
#    `plot_timeseries_regional`, `plot_qq_scatter`,
#    `plot_correlation_comparison`, `plot_temporal_correlation_curves`,
#    `plot_psd_curves` or `bar_plot_regional` render the figure. None of the
#    plotting functions preprocess data or compute scores themselves.

# %%
import numpy as np
import matplotlib.pyplot as plt

from emuvaluate.data_preparation import load_scenarios
from emuvaluate.baseline_methods import fit_regional_regressions_monthly, predict_pattern_scaling
from emuvaluate.transforms import yearly_average
from emuvaluate.preprocessing import preprocess_regional, select_members, scale_indicators
from emuvaluate.metrics import (
    build_error_data_regional, build_correlation_data,
    build_error_data_intervariable_correlation_values,
)
from emuvaluate.plots import (
    plot_map_regional, plot_timeseries_regional, plot_qq_scatter,
    plot_correlation_comparison, plot_temporal_correlation_curves,
    plot_psd_curves, bar_plot_regional, plot_error_matrix_regional,
)

# %% [markdown]
# ## Data loading & pattern-scaling baseline
#
# Unchanged from the previous version of this notebook — loads the
# simulation ensemble and the emulator's predictions, and fits the Pattern
# Scaling baseline used throughout the figures below.

# %%
MODEL = 'ACCESS-ESM1-5'

if MODEL == 'MPI-ESM1-2-LR':
    members = 30
else:
    members = 40

SCENARIO = 'ssp245'
scenario_data_ssp245 = load_scenarios(
    model=MODEL,
    indicators=['tas', 'pr'],
    scenarios=[SCENARIO],
    model_path=f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name=SCENARIO,
)

if SCENARIO == 'ssp126' or SCENARIO == 'ssp534-over':
    months = 5416
    ensembles = 39
else:
    months = 3012
    ensembles = 40

gmt_data_ssp245 = np.stack([lol[0, 120:months].transpose() for lol in scenario_data_ssp245], axis=0)
scenario_data_ssp245_tas = np.stack([lol[1:59, 120:months].transpose() for lol in scenario_data_ssp245], axis=0)[0:min(members,ensembles),:,:]
scenario_data_ssp245_pr = np.stack([lol[59:, 120:months].transpose() for lol in scenario_data_ssp245], axis=0)[0:min(members,ensembles),:,:]

scenario_data_ssp585 = load_scenarios(
    model=MODEL,
    indicators=['tas', 'pr'],
    scenarios=['ssp585'],
    model_path=f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp585',
)

gmt_data_ssp585 = np.stack([lol[0, 120:months].transpose() for lol in scenario_data_ssp585], axis=0)
scenario_data_ssp585_tas = np.stack([lol[1:59, 120:months].transpose() for lol in scenario_data_ssp585], axis=0)
scenario_data_ssp585_pr = np.stack([lol[59:, 120:months].transpose() for lol in scenario_data_ssp585], axis=0)

sims = np.array(list(np.load(f'/hdrive/all_users/schwind/MischMasch_old_runs/MischMasch_001/test_data/{SCENARIO}_{MODEL}_emulated.npy', allow_pickle=True)))

y_pred_ensemble_ssp245_tas =sims.transpose(0,2,1)[0:min(members,ensembles),120:,1:59]
y_pred_ensemble_ssp245_pr = sims.transpose(0,2,1)[0:min(members,ensembles),120:,59:]

if SCENARIO == 'ssp126':
    y_pred_ensemble_ssp245_tas = np.concatenate([y_pred_ensemble_ssp245_tas, y_pred_ensemble_ssp245_tas[-1:]], axis=0)
    y_pred_ensemble_ssp245_pr = np.concatenate([y_pred_ensemble_ssp245_pr, y_pred_ensemble_ssp245_pr[-1:]], axis=0)
    scenario_data_ssp245_tas = np.concatenate([scenario_data_ssp245_tas, scenario_data_ssp245_tas[-1:]], axis=0)
    scenario_data_ssp245_pr = np.concatenate([scenario_data_ssp245_pr, scenario_data_ssp245_pr[-1:]], axis=0)
elif MODEL == 'MIROC6':
    scenario_data_ssp245_tas = np.concatenate([scenario_data_ssp245_tas, scenario_data_ssp245_tas[-7:]], axis=0)
    scenario_data_ssp245_pr = np.concatenate([scenario_data_ssp245_pr, scenario_data_ssp245_pr[-7:]], axis=0)
print(yearly_average(y_pred_ensemble_ssp245_tas).shape)
print(yearly_average(y_pred_ensemble_ssp245_pr).shape)
print(yearly_average(scenario_data_ssp245_tas).shape)
print(yearly_average(scenario_data_ssp245_pr).shape)

fit_tas = fit_regional_regressions_monthly(
    global_series=yearly_average(gmt_data_ssp585[0, :]),
    regional_series=yearly_average(scenario_data_ssp585_tas[0, :]),
    monthly=False,
    train_ramp_down=False,
)
fit_pr = fit_regional_regressions_monthly(
    global_series=yearly_average(gmt_data_ssp585[0, :]),
    regional_series=yearly_average(scenario_data_ssp585_pr[0, :]),
    monthly=False,
    train_ramp_down=False,
)

ps_emulation_tas = predict_pattern_scaling(
    fit=fit_tas, global_series=yearly_average(gmt_data_ssp245[0, :]), monthly=False, n_members=members,
)
ps_emulation_pr = predict_pattern_scaling(
    fit=fit_pr, global_series=yearly_average(gmt_data_ssp245[0, :]), monthly=False, n_members=members,
)
print(ps_emulation_pr.shape)
print(ps_emulation_tas.shape)

region_list = sorted(['ARO', 'ARP', 'ARS', 'BOB', 'CAF', 'CAR', 'CAU', 'CNA', 'EAN', 'EAO', 'EAS', 'EAU', 'ECA',
                       'EEU', 'EIO', 'ENA', 'EPO', 'ESAF', 'ESB', 'GIC', 'MDG', 'MED', 'NAO', 'NAU', 'NCA',
                       'NEAF', 'NEN', 'NES', 'NEU', 'NPO', 'NSA', 'NWN', 'NWS', 'NZ', 'RAR', 'RFE', 'SAH',
                       'SAM', 'SAO', 'SAS', 'SAU', 'SCA', 'SEA', 'SEAF', 'SES', 'SIO', 'SOO', 'SPO', 'SSA',
                       'SWS', 'TIB', 'WAF', 'WAN', 'WCA', 'WCE', 'WNA', 'WSAF', 'WSB'])

# Raw ensembles, keyed by indicator — this dict shape is what every
# preprocessing / metric / plotting call below takes.
raw_sim = {'tas': scenario_data_ssp245_tas, 'pr': scenario_data_ssp245_pr}
raw_emu = {'tas': y_pred_ensemble_ssp245_tas, 'pr': y_pred_ensemble_ssp245_pr}
raw_ps = {'tas': ps_emulation_tas, 'pr': ps_emulation_pr}

# %%
index_SOO = region_list.index('SOO')


plt.plot(preprocess_regional(y_pred_ensemble_ssp245_tas[:,:,index_SOO].mean(axis = 0), yearly_average=True))
plt.plot(preprocess_regional(scenario_data_ssp245_tas[:,:,index_SOO].mean(axis = 0), yearly_average=True))
plt.show()

# %% [markdown]
# ## Error maps — monthly-derived annual means (NMAE)
#
# `[20:]` are the emulator's 20 test members; `[:20]` are held out purely to
# give a "Simulations vs Simulations" baseline — how much two disjoint
# halves of the simulation ensemble disagree with each other.
# `select_members` does that split for every indicator at once.

# %%
# 1. Preprocess
sim_annual = preprocess_regional(raw_sim, yearly_average=True)
emu_annual = preprocess_regional(raw_emu, yearly_average=True)
ps_annual = preprocess_regional(raw_ps, smooth=True)

sim = select_members(sim_annual, slice(20, None))
emu = select_members(emu_annual, slice(20, None))
ps = select_members(ps_annual, slice(20, None))
sim_only = select_members(sim_annual, slice(None, 20))

# 2. Build error data (one per baseline comparison)
error_data_vs_ps = build_error_data_regional(
    sim, emu, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': ps},
)
error_data_vs_simsim = build_error_data_regional(
    sim, emu, region_list, metric="nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only},
)

# 3. Plot
polli = plot_map_regional(
    error_data_vs_ps, show_difference=True,
    save_path='/home/schwind/emuvaluate/plots/error_maps_ssp126.png',
)
loli = plot_map_regional(
    error_data_vs_simsim,
    save_path='/home/schwind/emuvaluate/plots/error_maps_ssp245.png',
)

# %% [markdown]
# ## Error maps & timeseries — locally-smoothed forced response (MAE)
#
# `preprocess_regional(..., smooth=True)` removes most of the internal
# variability so the maps/timeseries below emphasise the forced-response
# error rather than noise. `ps` was already smoothed once above, so this is
# a second smoothing pass for it.

# %%
# 1. Preprocess
smoothed_sim = scale_indicators(preprocess_regional(sim, smooth=True), {'pr': 86400})
smoothed_emu = scale_indicators(preprocess_regional(emu, smooth=True), {'pr': 86400})
smoothed_ps = scale_indicators(preprocess_regional(ps, smooth=True), {'pr': 86400})
smoothed_sim_only = scale_indicators(preprocess_regional(sim_only, smooth=True), {'pr': 86400})
sim_only_scaled = scale_indicators(sim_only, {'pr': 86400})

# 2. Build error data
smoothed_vs_ps = build_error_data_regional(
    smoothed_sim, smoothed_emu, region_list, metric="mae",
    baseline_emulations={'Pattern Scaling': smoothed_ps},
)
smoothed_vs_simsim = build_error_data_regional(
    smoothed_sim, smoothed_emu, region_list, metric="mae",
    baseline_emulations={'Simulations vs Simulations': sim_only_scaled},
)
smoothed_vs_ps_ranked_by_ps = build_error_data_regional(
    smoothed_sim, smoothed_emu, region_list, metric="mae",
    baseline_emulations={'Pattern Scaling': smoothed_ps},
    ranking_strategy='Pattern Scaling',
)

# 3. Plot
years = np.array([1860 + i for i in range(sim['tas'].shape[1])])
polli = plot_map_regional(smoothed_vs_ps, show_difference=True, save_path='/home/schwind/emuvaluate/plots/error_maps_ssp126.png')
loli = plot_map_regional(smoothed_vs_simsim, save_path='/home/schwind/emuvaluate/plots/error_maps_ssp245.png')
lol = plot_timeseries_regional(
    smoothed_vs_ps_ranked_by_ps, smoothed_sim, smoothed_emu,
    baseline_emulations={'Pattern Scaling': smoothed_ps},
    years=years, save_path='/home/schwind/emuvaluate/plots/error_timeseries_ssp126.png',
)
lol = plot_timeseries_regional(
    smoothed_vs_ps, smoothed_sim, smoothed_emu,
    baseline_emulations={'Pattern Scaling': smoothed_ps},
    years=years, save_path='/home/schwind/emuvaluate/plots/error_timeseries_ssp245.png',
)

# %% [markdown]
# ## Emulation timeseries — annual means, no smoothing (NMAE)

# %%
years = np.array([1860 + i for i in range(sim['tas'].shape[1])])
sim_mm = scale_indicators(sim, {'pr': 86400})
emu_mm = scale_indicators(emu, {'pr': 86400})
ps_mm = scale_indicators(ps, {'pr': 86400})

ts_error_data_ranked_by_ps = build_error_data_regional(
    sim_mm, emu_mm, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': ps_mm},
    ranking_strategy='Pattern Scaling',
)
ts_error_data = build_error_data_regional(
    sim_mm, emu_mm, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': ps_mm},
)

lol = plot_timeseries_regional(
    ts_error_data_ranked_by_ps, sim_mm, emu_mm,
    baseline_emulations={'Pattern Scaling': ps_mm},
    years=years, save_path='/home/schwind/emuvaluate/plots/error_timeseries_ssp126.png',
)
lol = plot_timeseries_regional(
    ts_error_data, sim_mm, emu_mm,
    baseline_emulations={'Pattern Scaling': ps_mm},
    years=years, save_path='/home/schwind/emuvaluate/plots/error_timeseries_ssp245.png',
)

# %% [markdown]
# ## QQ variability — ensemble-mean removed, monthly (QQ-NMAE)
#
# `preprocess_regional(..., remove_ensemble_mean=True)` subtracts the
# multi-ensemble mean so these figures isolate internal variability. It's
# applied to the *full* raw ensemble first (before the `[20:]`/`[:20]`
# split below), so both halves share the same ensemble-mean reference —
# the same full arrays are reused for the spatial-correlation figures
# further down.

# %%
# 1. Preprocess
sim_full_det = preprocess_regional(raw_sim, remove_ensemble_mean=True)
emu_full_det = preprocess_regional(raw_emu, remove_ensemble_mean=True)

sim_det = scale_indicators(select_members(sim_full_det, slice(20, None)), {'pr': 86400})
emu_det = scale_indicators(select_members(emu_full_det, slice(20, None)), {'pr': 86400})
sim_only_det = scale_indicators(select_members(sim_full_det, slice(None, 20)), {'pr': 86400})

# 2. Build error data
qq_error_data = build_error_data_regional(sim_det, emu_det, region_list, metric="qq_nmae")

# 3. Plot
lol = plot_qq_scatter(
    qq_error_data, sim_det, emu_det,
    save_path='/home/schwind/emuvaluate/plots/error_variability.png',
)

# %%
lol = plot_map_regional(qq_error_data, save_path='/home/schwind/emuvaluate/plots/error_variability_map.png')

# %% [markdown]
# ## Spatial correlations — ensemble-mean removed, monthly
#
# Reuses `sim_full_det` / `emu_full_det` from the QQ-variability section
# above — same ensemble-mean-removed arrays, no need to preprocess again.
# `build_correlation_data` takes one indicator at a time, because the
# correlation is *across regions*.

# %%
tas_correlation_data = build_correlation_data(
    sim=sim_full_det['tas'][20:, :, :], pred=emu_full_det['tas'][20:, :, :],
    baseline_emulations={'Simulation vs Simulation': sim_full_det['tas'][:20, :, :]},
)
pr_correlation_data = build_correlation_data(
    sim=sim_full_det['pr'][20:, :, :], pred=emu_full_det['pr'][20:, :, :],
    baseline_emulations={'Simulation vs Simulation': sim_full_det['pr'][:20, :, :]},
)

lol = plot_correlation_comparison(tas_correlation_data, region_names=region_list, save_path='/home/schwind/emuvaluate/plots/spatial_correlations_tas.png')
lol = plot_correlation_comparison(pr_correlation_data, region_names=region_list, save_path='/home/schwind/emuvaluate/plots/spatial_correlations_pr.png')

# %%

# %% [markdown]
# ## Intervariable correlation — TAS vs PR relationship
#
# `build_error_data_intervariable_correlation_values` reports the |TAS–PR
# correlation| *itself* for the simulation and every comparison, rather than
# the emulator's error in reproducing it — so the bars show how strong the
# coupling actually is in each data source, side by side. Use
# `build_error_data_intervariable_correlation` (no `_values`) for the error
# version. The result carries a single `tas-pr` "indicator", so
# `bar_plot_regional` draws one panel rather than two identical ones.

# %%
# 2. Build error data
intervar_error_data = build_error_data_intervariable_correlation_values(
    sim_det, emu_det,
    # baseline_emulations={'Simulations vs Simulations': sim_only_det},
    unit_labels=region_list,
    diff_baseline=None,
)

# 3. Plot
lol = bar_plot_regional(
    intervar_error_data, show_difference=False,
    figsize_per_panel=(16, 4),
    save_path='/home/schwind/emuvaluate/plots/intervariable_correlation_bar_plot.png',
)

# %% [markdown]
# ## Temporal correlations — autocorrelation curves, annual means (NMAE)
#
# `metric="temporal_corr_nmae"` scores every region by the NMAE distance
# between its simulated and emulated temporal-autocorrelation curves (lag
# 0..`n_lags`) — same `build_error_data_regional`/`plot_map_regional`
# machinery as every other metric. `plot_temporal_correlation_curves` is the
# accompanying curve plot: lag on the x-axis, autocorrelation on the
# y-axis, for the best/median/worst region by that same NMAE ranking.

# %%
# 2. Build error data
temporal_corr_error_data = build_error_data_regional(
    sim_det, emu_det, region_list, metric="temporal_corr_nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=200,
)

# 3. Plot
lol = plot_map_regional(
    temporal_corr_error_data,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_nmae_map_long.png',
)
lol = plot_temporal_correlation_curves(
    temporal_corr_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=200,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_curves_long.png',
)

# %%
# 2. Build error data
temporal_corr_error_data = build_error_data_regional(
    sim_det, emu_det, region_list, metric="temporal_corr_nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=20,
)

# 3. Plot
lol = plot_map_regional(
    temporal_corr_error_data,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_nmae_map_short.png',
)
lol = plot_temporal_correlation_curves(
    temporal_corr_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=20,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_curves_short.png',
)

# %%
raw_sim_test = select_members(raw_sim, slice(20, None))
raw_emu_test = select_members(raw_emu, slice(20, None))
raw_sim_base = scale_indicators(select_members(raw_sim, slice(None, 20)), {'pr': 86400})

# 2. Build error data
temporal_corr_error_data = build_error_data_regional(
    raw_sim_test, raw_emu_test, region_list, metric="temporal_corr_nmae",
    baseline_emulations={'Simulations vs Simulations': raw_sim_base},
    n_lags=12,
)

# 3. Plot
lol = plot_map_regional(
    temporal_corr_error_data,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_nmae_map.png',
)
lol = plot_temporal_correlation_curves(
    temporal_corr_error_data, raw_sim_test, raw_emu_test,
    baseline_emulations={'Simulations vs Simulations': raw_sim_base},
    n_lags=12,
    save_path='/home/schwind/emuvaluate/plots/temporal_correlation_curves.png',
)

# %% [markdown]
# ## Power spectral density — monthly, ensemble-mean removed
#
# Same machinery again, with a PSD curve as the per-region quantity:
# `metric="psd_log_nmae"` scores every region by the NMAE between the
# simulated and the emulated log-PSD curve, so the result drops straight
# into `plot_map_regional`. `plot_psd_curves` is the accompanying ranking
# plot — period on the x-axis, spectral power on the y-axis, with a shaded
# 10–90 % band across ensemble members so you can see whether the
# emulator/simulation gap is bigger than the sampling spread.

# %%
psd_error_data = build_error_data_regional(
    sim_det, emu_det, region_list, metric="psd_log_nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    psd_nperseg=256,
)

lol = plot_map_regional(
    psd_error_data,
    save_path='/home/schwind/emuvaluate/plots/psd_log_nmae_map.png',
)
lol = plot_psd_curves(
    psd_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    nperseg=256,
    save_path='/home/schwind/emuvaluate/plots/psd_curves.png',
)

# %% [markdown]
# ### Choosing which regions to show, and locking the y-axis
#
# Every ranking plot takes the same two options:
#
# * `selection` — `None` (default) gives best / median / worst. A list of
#   1-based ranks, e.g. `[1, 2, 3]`, shows the three best regions. A list of
#   region names shows exactly those, in the order given, with each panel
#   reporting where that region falls in the ranking.
# * `share_y_per_row` — `False` by default. Set `True` to give every panel
#   in a row the same y-limits, making the columns directly comparable.

# %%
lol = plot_psd_curves(
    psd_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    nperseg=256, selection=[1, 2, 3, 4], share_y_per_row=True,
    save_path='/home/schwind/emuvaluate/plots/psd_curves_top4.png',
)
lol = plot_timeseries_regional(
    ts_error_data, sim_mm, emu_mm,
    baseline_emulations={'Pattern Scaling': ps_mm},
    years=years, selection=['SOO', 'MED', 'WAF'], share_y_per_row=True,
    save_path='/home/schwind/emuvaluate/plots/error_timeseries_named_regions.png',
)

# %%

# %% [markdown]
# ## Error matrix — several models side by side
#
# Every figure above evaluates one emulated model. `plot_error_matrix_regional`
# takes an `ErrorData` *per model* and lays them all out as one heatmap:
# regions across the columns, one row per (model, indicator, experiment),
# error value as colour. "Experiment" is what the `ErrorData` already carries
# — the Emulator plus every baseline it was built with.
#
# The dict keys are free-form labels, so they can name a model, a scenario, or
# both. Every entry has to cover the same regions and use the same metric.
# Colours are normalised per indicator by default, so a temperature error in K
# and a precipitation error in mm/day never share a colour scale.

# %%
# One entry per model. With a single model loaded this notebook shows one
# block; loop over models to fill the matrix out:
#
#     error_data_by_model = {
#         model: build_error_data_regional(
#             sim[model], emu[model], region_list, metric="mae",
#             baseline_emulations={'Simulations vs Simulations': sim_only[model]},
#         )
#         for model in ('ACCESS-ESM1-5', 'MPI-ESM1-2-LR', 'MIROC6')
#     }
error_data_by_model = {MODEL: error_data_vs_simsim}

lol = plot_error_matrix_regional(
    error_data_by_model,
    save_path='/home/schwind/emuvaluate/plots/error_matrix.png',
)

# vmin="auto" spends the whole colour range on the differences between rows,
# which is what you want once every row is a similar shade. A handful of named
# regions plus show_values turns it into a readable table.
lol = plot_error_matrix_regional(
    error_data_by_model, vmin="auto",
    save_path='/home/schwind/emuvaluate/plots/error_matrix_autoscale.png',
)
lol = plot_error_matrix_regional(
    error_data_by_model,
    regions=['SOO', 'MED', 'WAF', 'EAS', 'NEU', 'ARO'], show_values=True,
    save_path='/home/schwind/emuvaluate/plots/error_matrix_selected_regions.png',
)

# %%
