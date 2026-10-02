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
# # SCALES paper figures

# %% [markdown]
# Imports from the emuvaluate package: https://github.com/iiasa/emuvaluate 
#
# This notebook only works if the notebook is run with a kernel that has emuvaluate installed. See README.md for installation instructions. You additionally need to install ipykernel (`pip install ipykernel`) and install the environment with emuvaluate in it as a kernel (`python -m ipykernel install --user --name=name_of_kernal`). 

# %%
import numpy as np
import matplotlib.pyplot as plt
import joblib

from emuvaluate.data_preparation import load_scenarios
from emuvaluate.baseline_methods import fit_regional_regressions_monthly, predict_pattern_scaling
from emuvaluate.transforms import yearly_average
from emuvaluate.preprocessing import preprocess_regional, select_members, scale_indicators, ensemble_mean_regional
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
# # Configs
# - `DATA_PATH_EMULATIONS`: path to the SCALES emulations on your local machine
# - `DATA_PATH_SIMULATIONS`: path to the simulations on your local machine
#
# The data needed to reproduce the SCALES evaluation figures in the study are archived on Zenodo: https://doi.org/10.5281/zenodo.23034786
#
# Set `DATA_PATH_EMULATIONS` to the path of "" and `DATA_PATH_SIMULATIONS` is the path to simulation_aggregates (that you get when decompressing "simulation_aggregates.zip").

# %%
DATA_PATH_EMULATIONS = '/path/to/emulated_data'
DATA_PATH_SIMULATIONS = '/path/to/cmip6-ng-inc-oceans'

# %% [markdown]
# ## Data loading & pattern-scaling baseline for SSP2-4.5
#
# Loads the simulation ensemble and the emulator's predictions for SSP2-4.5, and fits the Pattern
# Scaling baseline used throughout the figures below.

# %%
MODEL = 'ACCESS-ESM1-5'

if MODEL == 'MPI-ESM1-2-LR':
    members = 30
elif MODEL == 'IPSL-CM6A-LR':
    members = 11
else:
    members = 40

SCENARIO = 'ssp245'
scenario_data_ssp245 = load_scenarios(
    model=MODEL,
    indicators=['tas', 'pr'],
    scenarios=[SCENARIO],
    model_path=f'{DATA_PATH_SIMULATIONS}/{MODEL}',
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

gmt_data_ssp245 = np.stack([_[0, 120:months].transpose() for _ in scenario_data_ssp245], axis=0)
scenario_data_ssp245_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data_ssp245], axis=0)[0:min(members,ensembles),:,:]
scenario_data_ssp245_pr = np.stack([_[59:, 120:months].transpose() for _ in scenario_data_ssp245], axis=0)[0:min(members,ensembles),:,:]

scenario_data_ssp585 = load_scenarios(
    model=MODEL,
    indicators=['tas', 'pr'],
    scenarios=['ssp585'],
    model_path=f'{DATA_PATH_SIMULATIONS}/{MODEL}',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp585',
)

gmt_data_ssp585 = np.stack([_[0, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)
scenario_data_ssp585_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)
scenario_data_ssp585_pr = np.stack([_[59:, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)

#SCALES-SSM
#sims = joblib.load('/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/ACCESS/ssp245/ssp245_ens40_pr_tas_monthly_correlated_rank4_11Aug26.pkl')#
#y_pred_ensemble_ssp245_tas = sims['tas']
#y_pred_ensemble_ssp245_pr = sims['pr']

#SCALES-DiT
sims = np.array(list(np.load(f'{DATA_PATH_EMULATIONS}/{SCENARIO}_{MODEL}_emulated.npy', allow_pickle=True)))
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

# %% [markdown]
# ## Data loading & pattern-scaling baseline for SSP1-2.6
#
# Loads the simulation ensemble and the emulator's predictions for SSP1-2.6, and fits the Pattern
# Scaling baseline used throughout the figures below.

# %%
MODEL = 'ACCESS-ESM1-5'

if MODEL == 'MPI-ESM1-2-LR':
    members = 30
elif MODEL == 'IPSL-CM6A-LR':
    members = 11
else:
    members = 40

SCENARIO = 'ssp126'
scenario_data_ssp126 = load_scenarios(
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

gmt_data_ssp126 = np.stack([_[0, 120:months].transpose() for _ in scenario_data_ssp126], axis=0)
scenario_data_ssp126_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data_ssp126], axis=0)[0:min(members,ensembles),:,:]
scenario_data_ssp126_pr = np.stack([_[59:, 120:months].transpose() for _ in scenario_data_ssp126], axis=0)[0:min(members,ensembles),:,:]

scenario_data_ssp585 = load_scenarios(
    model=MODEL,
    indicators=['tas', 'pr'],
    scenarios=['ssp585'],
    model_path=f'{DATA_PATH_SIMULATIONS}/{MODEL}',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp585',
)

gmt_data_ssp585 = np.stack([_[0, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)
scenario_data_ssp585_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)
scenario_data_ssp585_pr = np.stack([_[59:, 120:months].transpose() for _ in scenario_data_ssp585], axis=0)

# SCALES SSM
#sims = joblib.load('/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/ACCESS/ssp126/ssp126_ens40_pr_tas_monthly_correlated_rank4_11Aug26.pkl')#
#y_pred_ensemble_ssp126_tas = sims['tas']
#y_pred_ensemble_ssp126_pr = sims['pr']

# SCALES DiT 
sims = np.array(list(np.load(f'{DATA_PATH_EMULATIONS}/{SCENARIO}_{MODEL}_emulated.npy', allow_pickle=True)))
y_pred_ensemble_ssp126_tas =sims.transpose(0,2,1)[0:min(members,ensembles),120:,1:59]
y_pred_ensemble_ssp126_pr = sims.transpose(0,2,1)[0:min(members,ensembles),120:,59:]

if SCENARIO == 'ssp126':
    y_pred_ensemble_ssp126_tas = np.concatenate([y_pred_ensemble_ssp126_tas, y_pred_ensemble_ssp126_tas[-1:]], axis=0)
    y_pred_ensemble_ssp126_pr = np.concatenate([y_pred_ensemble_ssp126_pr, y_pred_ensemble_ssp126_pr[-1:]], axis=0)
    scenario_data_ssp126_tas = np.concatenate([scenario_data_ssp126_tas, scenario_data_ssp126_tas[-1:]], axis=0)
    scenario_data_ssp126_pr = np.concatenate([scenario_data_ssp126_pr, scenario_data_ssp126_pr[-1:]], axis=0)
elif MODEL == 'MIROC6':
    scenario_data_ssp126_tas = np.concatenate([scenario_data_ssp126_tas, scenario_data_ssp126_tas[-7:]], axis=0)
    scenario_data_ssp126_pr = np.concatenate([scenario_data_ssp126_pr, scenario_data_ssp126_pr[-7:]], axis=0)
print(yearly_average(y_pred_ensemble_ssp126_tas).shape)
print(yearly_average(y_pred_ensemble_ssp126_pr).shape)
print(yearly_average(scenario_data_ssp126_tas).shape)
print(yearly_average(scenario_data_ssp126_pr).shape)

fit_tas_os = fit_regional_regressions_monthly(
    global_series=yearly_average(gmt_data_ssp585[0, :]),
    regional_series=yearly_average(scenario_data_ssp585_tas[0, :]),
    monthly=False,
    train_ramp_down=False,
)
fit_pr_os = fit_regional_regressions_monthly(
    global_series=yearly_average(gmt_data_ssp585[0, :]),
    regional_series=yearly_average(scenario_data_ssp585_pr[0, :]),
    monthly=False,
    train_ramp_down=False,
)

ps_emulation_tas_os = predict_pattern_scaling(
    fit=fit_tas_os, global_series=yearly_average(gmt_data_ssp126[0, :]), monthly=False, n_members=members,
)
ps_emulation_pr_os = predict_pattern_scaling(
    fit=fit_pr_os, global_series=yearly_average(gmt_data_ssp126[0, :]), monthly=False, n_members=members,
)
print(ps_emulation_pr_os.shape)
print(ps_emulation_tas_os.shape)

region_list = sorted(['ARO', 'ARP', 'ARS', 'BOB', 'CAF', 'CAR', 'CAU', 'CNA', 'EAN', 'EAO', 'EAS', 'EAU', 'ECA',
                       'EEU', 'EIO', 'ENA', 'EPO', 'ESAF', 'ESB', 'GIC', 'MDG', 'MED', 'NAO', 'NAU', 'NCA',
                       'NEAF', 'NEN', 'NES', 'NEU', 'NPO', 'NSA', 'NWN', 'NWS', 'NZ', 'RAR', 'RFE', 'SAH',
                       'SAM', 'SAO', 'SAS', 'SAU', 'SCA', 'SEA', 'SEAF', 'SES', 'SIO', 'SOO', 'SPO', 'SSA',
                       'SWS', 'TIB', 'WAF', 'WAN', 'WCA', 'WCE', 'WNA', 'WSAF', 'WSB'])

# Raw ensembles, keyed by indicator — this dict shape is what every
# preprocessing / metric / plotting call below takes.
raw_sim_os = {'tas': scenario_data_ssp126_tas, 'pr': scenario_data_ssp126_pr}
raw_emu_os = {'tas': y_pred_ensemble_ssp126_tas, 'pr': y_pred_ensemble_ssp126_pr}
raw_ps_os = {'tas': ps_emulation_tas_os, 'pr': ps_emulation_pr_os}

# %% [markdown]
# ## Error maps — regional annual means (NMAE)
#
# Calculates the NMAE per IPCC region between the remaining ensemble members of the original simulation (member 21 onward) and:
#
# 1. the first 20 ensemble members of the simulation
# 2. the 20 emulated ensemble members, forced by the GMT values of the same first 20 simulation ensemble members
#
# All data are first aggregated to annual means. Both comparisons use members that are independent of the reference members. This compares natural variability and trend combined. The better the emulator, the closer the emulation–simulation NMAE is to the simulation–simulation NMAE.

# %%
# 1. Preprocess
sim_annual = preprocess_regional(raw_sim, yearly_average=True)
emu_annual = preprocess_regional(raw_emu, yearly_average=True)
ps_annual = preprocess_regional(raw_ps, smooth=True)

sim = select_members(sim_annual, slice(20, None))
emu = select_members(emu_annual, slice(None, 20))
ps = select_members(ps_annual, slice(None, 20))
sim_only = select_members(sim_annual, slice(None, 20))

error_data_vs_simsim = build_error_data_regional(
    sim, emu, region_list, metric="nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only},
)

_ = plot_map_regional(
    error_data_vs_simsim,
    save_path='../plots/error_maps_ssp245.png',
)

# %% [markdown]
# ## Error maps — regional ensemble means (NMAE), emulator vs. pattern scaling
#
# Calculates the NMAE per IPCC region between the ensemble mean of all simulation ensemble members (annual means) and:
#
# 1. the ensemble mean of the emulated ensemble members from member 21 onward
# 2. the ensemble mean of the pattern-scaling ensemble members from member 21 onward
#
# The simulation, the emulator and pattern scaling all share the same underlying GMT data, so differences come from how each method maps that GMT to regional temperature and precipitation. The pattern-scaling output is already annual, so it is not aggregated further or smoothed. Precipitation is converted from kg m⁻² s⁻¹ to mm day⁻¹.
#
# Averaging over the ensemble removes most of the natural variability, so this mainly compares the forced response (the trend). Using all simulation members gives the best estimate of that forced response. The map shows the difference in NMAE (emulator − pattern scaling). Negative values (blue) mean the emulator is closer to the simulated forced response than pattern scaling.

# %%
sim_annual_os = preprocess_regional(raw_sim_os, yearly_average=True)
emu_annual_os = preprocess_regional(raw_emu_os, yearly_average=True)
#sim_annual_os = {key: value[:,-433:] for (key,value) in sim_annual_os.items()}
# Pattern Scaling already comes out annual from predict_pattern_scaling(monthly=False),
# so this is just a copy — no temporal aggregation, no smoothing.
ps_annual_os = preprocess_regional(raw_ps_os)
#ps_annual_os = {key: value[:,-433:] for (key,value) in ps_annual_os.items()}
sim_os = select_members(sim_annual_os, slice(0, None))
emu_os = select_members(emu_annual_os, slice(0, None))
ps_os = select_members(ps_annual_os, slice(0, None))


# 1. Preprocess — ensemble means instead of per-member LOWESS smoothing
mean_sim_os = scale_indicators(ensemble_mean_regional(sim_os), {'pr': 86400})
mean_emu_os = scale_indicators(ensemble_mean_regional(emu_os), {'pr': 86400})
mean_ps_os = scale_indicators(ensemble_mean_regional(ps_os), {'pr': 86400})

# 2. Build error data
mean_vs_ps_os = build_error_data_regional(
    mean_sim_os, mean_emu_os, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': mean_ps_os},
)
mean_vs_ps_ranked_by_ps_os = build_error_data_regional(
    mean_sim_os, mean_emu_os, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': mean_ps_os},
    ranking_strategy='Pattern Scaling',
)
# 3. Plot
years = np.array([1860 + i for i in range(sim_os['tas'].shape[1])])
_ = plot_map_regional(mean_vs_ps_os, show_difference=True, save_path='../plots/error_maps_ssp126.png')

# %% [markdown]
# ## Error timeseries ranked by NMAE
#
# Compares the annual means of all simulation ensemble members with all emulated ensemble members and all pattern-scaling ensemble members, per IPCC region. The emulations are forced by the regional temperature and precipitation values of the corresponding simulation members, and all three share the same underlying GMT data. Precipitation is converted from kg m⁻² s⁻¹ to mm day⁻¹. Regions are ranked by the emulator's NMAE, and the plot shows the best, median and worst regions of the error distribution, with pattern scaling as a baseline.
#
# This illustrates how well the emulator reproduces the trend and natural variability over time, compared with pattern scaling. The more closely the emulator ensemble resembles the simulation ensemble, the better the emulator. A good match even in the region with the highest error is especially convincing.

# %%
years = np.array([1860 + i for i in range(sim['tas'].shape[1])])
sim = select_members(sim_annual, slice(0 , None))
emu = select_members(emu_annual, slice(0, None))
ps = select_members(ps_annual, slice(0, None))

sim_mm = scale_indicators(sim, {'pr': 86400})
emu_mm = scale_indicators(emu, {'pr': 86400})
ps_mm = scale_indicators(ps, {'pr': 86400})


ts_error_data = build_error_data_regional(
    sim_mm, emu_mm, region_list, metric="nmae",
    baseline_emulations={'Pattern Scaling': ps_mm},
)


_ = plot_timeseries_regional(
    ts_error_data, sim_mm, emu_mm,
    baseline_emulations={'Pattern Scaling': ps_mm},
    years=years, save_path='../plots/error_timeseries_ssp245.png',
)

# %% [markdown]
# ## Natural variability — ranked Q-Q plots (QQ-NMAE)
#
# Isolates the natural variability in the simulations and emulations by subtracting the ensemble mean from each. Then, for each IPCC region, it calculates the distance between the quantiles of the natural variability distributions of all 40 simulation ensemble members and all 40 emulated ensemble members. Each emulated member is forced by GMT of the corresponding simulation member. Precipitation is converted from kg m⁻² s⁻¹ to mm day⁻¹. Finally, it plots Q-Q plots of the best, median and worst regions of the error distribution.
#
# The Q-Q plots illustrate how well the emulator reproduces the natural variability. The more closely the emulator ensemble resembles the simulation ensemble, the better the emulator. A good match even in the region with the highest error is especially convincing.

# %%
# 1. Preprocess
sim_full_det = preprocess_regional(raw_sim, remove_ensemble_mean=True)
emu_full_det = preprocess_regional(raw_emu, remove_ensemble_mean=True)

sim_det = scale_indicators(select_members(sim_full_det, slice(None, 40)), {'pr': 86400})
emu_det = scale_indicators(select_members(emu_full_det, slice(None, 40)), {'pr': 86400})

# 2. Build error data
qq_error_data = build_error_data_regional(sim_det, emu_det, region_list, metric="qq_nmae")

# 3. Plot
_ = plot_qq_scatter(
    qq_error_data, sim_det, emu_det,
    save_path='../plots/error_variability.png',
)

# %%
_ = plot_map_regional(qq_error_data, save_path='../plots/error_variability_map.png')

# %% [markdown]
# ## Spatial correlations of natural variability between regions
#
# Uses the natural variability of the simulations and emulations (ensemble mean subtracted, as above). For temperature and precipitation separately, it calculates the correlation matrix between all pairs of IPCC regions and compares the first 20 simulation ensemble members with:
#
# 1. the emulated ensemble members 21–40, forced by the regional temperature and precipitation values of simulation members 21–40
# 2. the simulation ensemble members 21–40 (independent), as the simulation–simulation baseline
#
# This shows whether the emulator captures how natural variability co-varies across regions, such as teleconnections and coherent large-scale patterns. The more closely the emulated correlation matrix resembles the simulated one, the better the emulator. A perfect emulator would reproduce the simulation–simulation comparison, since two independent sets of simulation members also differ because of sampling noise.

# %%
tas_correlation_data = build_correlation_data(
    sim=sim_full_det['tas'][20:, :, :], pred=emu_full_det['tas'][:20, :, :],
    baseline_emulations={'Simulation vs Simulation': sim_full_det['tas'][:20, :, :]},
)
pr_correlation_data = build_correlation_data(
    sim=sim_full_det['pr'][20:, :, :], pred=emu_full_det['pr'][:20, :, :],
    baseline_emulations={'Simulation vs Simulation': sim_full_det['pr'][:20, :, :]},
)

_ = plot_correlation_comparison(tas_correlation_data, region_names=region_list, save_path='../plots/spatial_correlations_tas.png')
_ = plot_correlation_comparison(pr_correlation_data, region_names=region_list, save_path='../plots/spatial_correlations_pr.png')

# %% [markdown]
# ## Inter-variable correlations of natural variability
#
# Uses the natural variability of all 40 simulation and all 40 emulated ensemble members (ensemble mean subtracted, as above). Each emulated member is forced by the GMT values of the corresponding simulation member. For each IPCC region, it calculates the correlation between temperature and precipitation variability, for both the simulations and the emulations. The bar plot shows the absolute simulated and emulated correlation side by side for every region.
#
# The more closely the emulated correlations match the simulated ones, the better the emulator.

# %%
# 2. Build error data
intervar_error_data = build_error_data_intervariable_correlation_values(
    sim_det, emu_det,
    # baseline_emulations={'Simulations vs Simulations': sim_only_det},
    unit_labels=region_list,
    diff_baseline=None,
)

# 3. Plot
_ = bar_plot_regional(
    intervar_error_data, show_difference=False,
    figsize_per_panel=(16, 4),
    save_path='../plots/intervariable_correlation_bar_plot.png',
)

# %% [markdown]
# ## Temporal autocorrelation of natural variability (temporal-correlation NMAE)
#
# Uses the natural variability of the simulations and emulations (ensemble mean subtracted, as above). For each IPCC region, it calculates the autocorrelation function up to a lag of 200 months. It then calculates the NMAE between the autocorrelation functions of the first 20 simulation ensemble members and:
#
# 1. the emulated ensemble members 21–40, forced by the regional temperature and precipitation values of simulation members 21–40
# 2. the simulation ensemble members 21–40 (independent), as the simulation–simulation baseline
#
# The map shows this error per region. The curves show the simulated, emulated and simulation-baseline autocorrelation functions for the best, median and worst regions of the error distribution.
#
# The better the emulator, the closer the emulation–simulation error is to the simulation–simulation error. A good match even in the region with the highest error is especially convincing.

# %%
# 2. Build error data
sim_det = scale_indicators(select_members(sim_full_det, slice(None, 20)), {'pr': 86400})
emu_det = scale_indicators(select_members(emu_full_det, slice(20, None)), {'pr': 86400})
sim_only_det = scale_indicators(select_members(sim_full_det, slice(20, None)), {'pr': 86400})

temporal_corr_error_data = build_error_data_regional(
    sim_det, emu_det, region_list, metric="temporal_corr_nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=200,
)

# 3. Plot
_ = plot_map_regional(
    temporal_corr_error_data,
    save_path='../plots/temporal_correlation_nmae_map_long.png',
)
_ = plot_temporal_correlation_curves(
    temporal_corr_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=200,
    save_path='../plots/temporal_correlation_curves_long.png',
)

# %% [markdown]
# ## Temporal autocorrelation of natural variability — short lags
#
# Same as above, but only up to a lag of 20 months, focusing on short-term autocorrelation.

# %%
# 2. Build error data
temporal_corr_error_data = build_error_data_regional(
    sim_det, emu_det, region_list, metric="temporal_corr_nmae",
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=20,
)

# 3. Plot
_ = plot_map_regional(
    temporal_corr_error_data,
    save_path='../plots/temporal_correlation_nmae_map_short.png',
)
_ = plot_temporal_correlation_curves(
    temporal_corr_error_data, sim_det, emu_det,
    baseline_emulations={'Simulations vs Simulations': sim_only_det},
    n_lags=20,
    save_path='../plots/temporal_correlation_curves_short.png',
)

# %% [markdown]
# ## CRPS maps 
#
# Calculates the continuous ranked probability score (CRPS) per IPCC region on monthly data. For each region and timestep, the ensemble is treated as a probability distribution and scored against the first 20 simulation ensemble members. The CRPS is then averaged over all timesteps. This is done for:
#
# 1. the emulated ensemble members 21–40, forced by the regional temperature and precipitation values of simulation members 21–40
# 2. the simulation ensemble members 21–40 (independent), as the simulation–simulation baseline
#
# The map shows the difference between the emulation–simulation CRPS and the simulation–simulation CRPS.
#
# The CRPS tests the trend and natural variability together, as a full distribution. A perfect emulator would match the simulation–simulation CRPS, so the difference would be near zero.

# %%
# CRPS
# 1. Preprocess — monthly, no temporal aggregation. CRPS carries physical
#    units, so pr is converted to mm/day here (the annual NMAE cell doesn't
#    need this because NMAE is unitless).
sim_mon    = scale_indicators(select_members(raw_sim, slice(None, 20)), {'pr': 86400})
emu_mon    = scale_indicators(select_members(raw_emu, slice(20, None)), {'pr': 86400})
simsim_mon = scale_indicators(select_members(raw_sim, slice(20, None)), {'pr': 86400})

# 2. Build error data — one CRPS per region, from the ensemble members at each
#    (region, month) cell, averaged over the months.
crps_monthly = build_error_data_regional(
    sim_mon, emu_mon, region_list, metric="crps",
    baseline_emulations={'Simulations vs Simulations': simsim_mon},
)

# 3. Plot
_ = plot_map_regional(
    crps_monthly,
    units={'tas': 'K', 'pr': 'mm day⁻¹'},
    show_difference = True,
    save_path='../plots/crps_maps_monthly_ssp245.png',
)

# %% [markdown]
# ## Error matrices across ESMs
#
# Repeats the regional evaluation for all five ESMs (ACCESS-ESM1-5, MPI-ESM1-2-LR, MIROC6, CanESM5, IPSL-CM6A-LR) under SSP2-4.5, using these metrics:
#
# - NMAE of the annual timeseries, with the simulation–simulation baseline
# - QQ-NMAE of the natural variability
# - temporal-autocorrelation NMAE, up to lags of 200 months
# - the TAS–PR correlation values themselves
#
# To save space, the results are shown as region × model matrices of error values instead of maps, with one figure per metric and the colour scale normalised per indicator.
#
# How many ensemble members a metric uses depends on whether it carries a simulation–simulation baseline.
#
# The three metrics without a baseline — QQ-NMAE, the autocorrelation NMAE and the TAS–PR correlation values — compare the full simulated ensemble against the full emulated one, since there is no baseline a half-split would have to be matched to. That is 40 members per side for ACCESS-ESM1-5 and CanESM5, 33 for MIROC6, 30 for MPI-ESM1-2-LR, and 11 for IPSL-CM6A-LR. None of these metrics pairs members, so the unequal counts for MIROC6 are not a problem.
#
# The annual-timeseries NMAE does carry a baseline, so the simulated ensemble is split into two equal, disjoint halves and the emulated ensemble is cut to the same size, so that the emulator and the baseline are estimated from equally many members. The simulation reference is the lower half, the simulation–simulation baseline is the upper half, and the emulations are those forced by the regional temperature and precipitation values of the upper-half members. That gives 20/20 members for ACCESS-ESM1-5 and CanESM5, 16/16 for MIROC6, 15/15 for MPI-ESM1-2-LR and 5/5 for IPSL-CM6A-LR, whose 11th member is used only by the
# full-ensemble metrics.
#
# Preprocessing — the annual averaging and the removal of the ensemble mean — is applied to the full ensemble before any split, so both halves share a single ensemble-mean reference.

# %%
# ── Error matrices across 5 models ───────────────────────────────────────────
import gc
import numpy as np
from emuvaluate.data_preparation import load_scenarios
from emuvaluate.preprocessing import preprocess_regional, select_members, scale_indicators
from emuvaluate.metrics import (
    build_error_data_regional,
    build_error_data_intervariable_correlation_values,
)
from emuvaluate.plots import plot_error_matrix_regional

MODELS = ['ACCESS-ESM1-5', 'MPI-ESM1-2-LR', 'MIROC6', 'CanESM5', 'IPSL-CM6A-LR']
SCENARIO = 'ssp245'
PLOT_DIR = '../plots'
MEMBERS = {'MPI-ESM1-2-LR': 30, 'IPSL-CM6A-LR': 11}   # everything else: 40
PR = {'pr': 86400}

if SCENARIO in ('ssp126', 'ssp534-over'):
    months, ensembles = 5416, 39
else:
    months, ensembles = 3012, 40

region_list = sorted(['ARO', 'ARP', 'ARS', 'BOB', 'CAF', 'CAR', 'CAU', 'CNA', 'EAN', 'EAO', 'EAS',
                      'EAU', 'ECA', 'EEU', 'EIO', 'ENA', 'EPO', 'ESAF', 'ESB', 'GIC', 'MDG', 'MED',
                      'NAO', 'NAU', 'NCA', 'NEAF', 'NEN', 'NES', 'NEU', 'NPO', 'NSA', 'NWN', 'NWS',
                      'NZ', 'RAR', 'RFE', 'SAH', 'SAM', 'SAO', 'SAS', 'SAU', 'SCA', 'SEA', 'SEAF',
                      'SES', 'SIO', 'SOO', 'SPO', 'SSA', 'SWS', 'TIB', 'WAF', 'WAN', 'WCA', 'WCE',
                      'WNA', 'WSAF', 'WSB'])

scores = {}          # scores[metric_label][model] -> ErrorData
members_used = {}    # bookkeeping, so the caption can state the real counts

for MODEL in MODELS:
    print(f'--- {MODEL} ---', flush=True)
    n_keep = min(MEMBERS.get(MODEL, 40), ensembles)

    scenario_data = load_scenarios(
        model=MODEL, indicators=['tas', 'pr'], scenarios=[SCENARIO],
        model_path=f'{DATA_PATH_SIMULATIONS}/{MODEL}',
        monthly_flag=True, use_smoothing=False, train_pattern_scaling_name=SCENARIO,
    )
    sim_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data], axis=0)[:n_keep]
    sim_pr  = np.stack([_[59:,  120:months].transpose() for _ in scenario_data], axis=0)[:n_keep]
    del scenario_data

    sims = np.array(list(np.load(
        f'{DATA_PATH_EMULATIONS}/'
        f'{SCENARIO}_{MODEL}_emulated.npy', allow_pickle=True)))
    emu_tas = sims.transpose(0, 2, 1)[:n_keep, 120:, 1:59]
    emu_pr  = sims.transpose(0, 2, 1)[:n_keep, 120:, 59:]
    del sims

    raw_sim = {'tas': sim_tas, 'pr': sim_pr}
    raw_emu = {'tas': emu_tas, 'pr': emu_pr}
    del sim_tas, sim_pr, emu_tas, emu_pr

    # No member padding. Both ensembles are cut to the same size (MIROC6 has
    # only 33 simulated members, so its 40 emulations are cut to 33 as well),
    # and the cut happens before preprocessing so the ensemble mean is taken
    # over exactly the members that are used.
    n = min(raw_sim['tas'].shape[0], raw_emu['tas'].shape[0])
    raw_sim = select_members(raw_sim, slice(0, n))
    raw_emu = select_members(raw_emu, slice(0, n))
    half = n // 2
    lower, upper, full = slice(0, half), slice(half, 2 * half), slice(None)
    members_used[MODEL] = dict(n=n, half=half)
    print(f'   full-ensemble metrics: {n} vs {n}  |  '
          f'baselined metric: {half} vs {half}', flush=True)

    # Preprocess on the FULL ensemble — remove_ensemble_mean has to see every
    # member so the two halves below share one ensemble-mean reference.
    ann_sim = preprocess_regional(raw_sim, yearly_average=True)
    ann_emu = preprocess_regional(raw_emu, yearly_average=True)
    det_sim = preprocess_regional(raw_sim, remove_ensemble_mean=True)
    det_emu = preprocess_regional(raw_emu, remove_ensemble_mean=True)
    del raw_sim, raw_emu

    def pick(d, members):
        return scale_indicators(select_members(d, members), PR)

    # (a) No simulation-vs-simulation baseline -> nothing to match, use every member.
    sim_det_all, emu_det_all = pick(det_sim, full), pick(det_emu, full)

    # (b) With a baseline -> the simulation ensemble is split in two and the
    #     emulator is cut to the same size, so all three rows carry the same
    #     sampling uncertainty. Reference = lower half, baseline = upper half,
    #     matching the single-model notebook cells.
    sim_ann, emu_ann, simsim_ann = pick(ann_sim, lower), pick(ann_emu, upper), pick(ann_sim, upper)
    del ann_sim, ann_emu, det_sim, det_emu

    per_model = {
        # ── full ensemble ────────────────────────────────────────────────────
        'Temporal-Corr NMAE (200 lag)': build_error_data_regional(
            sim_det_all, emu_det_all, region_list, metric='temporal_corr_nmae', n_lags=200),
        'QQ-NMAE': build_error_data_regional(
            sim_det_all, emu_det_all, region_list, metric='qq_nmae'),
        # The |correlation| itself for each data source side by side, so you
        # can see how strong the coupling actually is, not just the error.
        'TAS-PR correlation': build_error_data_intervariable_correlation_values(
            sim_det_all, emu_det_all,
            unit_labels=region_list, diff_baseline=None),

        # ── half ensembles, matched against a baseline ───────────────────────
        'NMAE (annual timeseries)': build_error_data_regional(
            sim_ann, emu_ann, region_list, metric='nmae',
            baseline_emulations={'Simulations vs Simulations': simsim_ann}),
    }

    # The colourbar bracket defaults to the indicator's name, which would read
    # "[temperature]" for a correlation — relabel so it reads "[correlation]".
    per_model['TAS-PR correlation'].indicator_labels['tas-pr'] = 'Correlation (TAS-PR)'

    for metric_label, ed in per_model.items():
        scores.setdefault(metric_label, {})[MODEL] = ed

    del sim_det_all, emu_det_all, sim_ann, emu_ann, simsim_ann, per_model
    gc.collect()

print('done:', list(scores), flush=True)
print('members used:', members_used, flush=True)

# ── one figure per metric ────────────────────────────────────────────────────
figs = {}
for metric_label, by_model in scores.items():
    slug = (metric_label.lower().replace(' ', '_').replace('-', '_')
            .replace('(', '').replace(')', ''))
    figs[metric_label] = plot_error_matrix_regional(
        by_model,
        normalise='indicator',
        suptitle=f'Per-Region {metric_label}',
        save_path=f'{PLOT_DIR}/error_matrix_{slug}.png',
    )

# %% [markdown]
# ## Spatial correlations across ESMs
#
# Repeats the spatial correlation analysis for all five ESMs under SSP2-4.5, using the same member split as the error matrices above. For temperature and precipitation separately, the inter-regional correlation matrix of the natural variability is calculated for:
#
# - the upper-half simulation members (reference)
# - the lower-half emulations
# - the lower-half simulation members (simulation–simulation baseline)
#
# The full correlation matrices are produced for each model. For the paper, they are summarised by the MAE and RMSE between the emulated and simulated correlation matrices, alongside the same scores for the simulation–simulation comparison, and these values are reported in a table. The closer the emulation–simulation scores are to the simulation–simulation scores, the better the emulator reproduces the spatial structure of natural variability.

# %%
# ── Spatial correlation comparison plots across 5 models ────────────────────
import gc
import numpy as np
from emuvaluate.data_preparation import load_scenarios
from emuvaluate.preprocessing import preprocess_regional, select_members, scale_indicators
from emuvaluate.metrics import build_correlation_data
from emuvaluate.plots import plot_correlation_comparison

MODELS = ['ACCESS-ESM1-5', 'MPI-ESM1-2-LR', 'MIROC6', 'CanESM5', 'IPSL-CM6A-LR']
SCENARIO = 'ssp245'
PLOT_DIR = '../plots'
MEMBERS = {'MPI-ESM1-2-LR': 30, 'IPSL-CM6A-LR': 11}   # everything else: 40

if SCENARIO == 'ssp126' or SCENARIO == 'ssp534-over':
    months, ensembles = 5416, 39
else:
    months, ensembles = 3012, 40

region_list = sorted(['ARO', 'ARP', 'ARS', 'BOB', 'CAF', 'CAR', 'CAU', 'CNA', 'EAN', 'EAO', 'EAS',
                      'EAU', 'ECA', 'EEU', 'EIO', 'ENA', 'EPO', 'ESAF', 'ESB', 'GIC', 'MDG', 'MED',
                      'NAO', 'NAU', 'NCA', 'NEAF', 'NEN', 'NES', 'NEU', 'NPO', 'NSA', 'NWN', 'NWS',
                      'NZ', 'RAR', 'RFE', 'SAH', 'SAM', 'SAO', 'SAS', 'SAU', 'SCA', 'SEA', 'SEAF',
                      'SES', 'SIO', 'SOO', 'SPO', 'SSA', 'SWS', 'TIB', 'WAF', 'WAN', 'WCA', 'WCE',
                      'WNA', 'WSAF', 'WSB'])

corr_figs = {}   # corr_figs[model][indicator] -> figure

for MODEL in MODELS:
    print(f'--- {MODEL} ---', flush=True)
    n_keep = min(MEMBERS.get(MODEL, 40), ensembles)
    scenario_data = load_scenarios(
        model=MODEL, indicators=['tas', 'pr'], scenarios=[SCENARIO],
        model_path=f'{DATA_PATH_SIMULATIONS}/{MODEL}',
        monthly_flag=True, use_smoothing=False, train_pattern_scaling_name=SCENARIO,
    )
    sim_tas = np.stack([_[1:59, 120:months].transpose() for _ in scenario_data], axis=0)[:n_keep]
    sim_pr = np.stack([_[59:, 120:months].transpose() for _ in scenario_data], axis=0)[:n_keep]
    del scenario_data

    sims = np.array(list(np.load(
        f'{DATA_PATH_EMULATIONS}/'
        f'{SCENARIO}_{MODEL}_emulated.npy', allow_pickle=True)))
    emu_tas = sims.transpose(0, 2, 1)[:n_keep, 120:, 1:59]
    emu_pr = sims.transpose(0, 2, 1)[:n_keep, 120:, 59:]
    del sims

    if SCENARIO == 'ssp126':
        emu_tas = np.concatenate([emu_tas, emu_tas[-1:]], axis=0)
        emu_pr = np.concatenate([emu_pr, emu_pr[-1:]], axis=0)
        sim_tas = np.concatenate([sim_tas, sim_tas[-1:]], axis=0)
        sim_pr = np.concatenate([sim_pr, sim_pr[-1:]], axis=0)
    elif MODEL == 'MIROC6':
        sim_tas = np.concatenate([sim_tas, sim_tas[-7:]], axis=0)
        sim_pr = np.concatenate([sim_pr, sim_pr[-7:]], axis=0)

    raw_sim = {'tas': sim_tas, 'pr': sim_pr}
    raw_emu = {'tas': emu_tas, 'pr': emu_pr}
    del sim_tas, sim_pr, emu_tas, emu_pr

    # Two equal disjoint halves — sim pairs with baseline member-for-member.
    # 40 members -> [20:40]/[0:20]; IPSL (11) -> [5:10]/[0:5].
    half = raw_sim['tas'].shape[0] // 2
    upper, lower = slice(half, 2 * half), slice(0, half)

    det_sim = preprocess_regional(raw_sim, remove_ensemble_mean=True)
    det_emu = preprocess_regional(raw_emu, remove_ensemble_mean=True)
    sim_det = scale_indicators(select_members(det_sim, upper), {'pr': 86400})
    emu_det = scale_indicators(select_members(det_emu, lower), {'pr': 86400})
    simsim_det = scale_indicators(select_members(det_sim, lower), {'pr': 86400})
    del det_sim, det_emu, raw_sim, raw_emu

    corr_figs[MODEL] = {}
    for var in ('tas', 'pr'):
        correlation_data = build_correlation_data(
            sim=sim_det[var], pred=emu_det[var],
            baseline_emulations={'Simulation vs Simulation': simsim_det[var]},
        )
        corr_figs[MODEL][var] = plot_correlation_comparison(
            correlation_data, region_names=region_list,
            save_path=f'{PLOT_DIR}/spatial_correlations_{var}_{MODEL}.png',
        )

    del sim_det, emu_det, simsim_det
    gc.collect()

print('done:', list(corr_figs), flush=True)

# %%
