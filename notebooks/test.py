# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pickle
import emuvaluate
from emuvaluate.plots import plot_metric_extremes, plot_random_timeseries, plot_spatial_correlations, plot_qq_extremes, plot_error_metrics_bar, plot_gmt_phases, plot_region_ensemble_extremes, plot_gmt_vs_regional_linearity_extremes
from emuvaluate.data_preparation import load_scenarios
from emuvaluate.baseline_methods import fit_regional_regressions_monthly, predict_pattern_scaling
from emuvaluate.transforms import yearly_average, detrend_gaussian, select_month, split_at_indices, find_phase_split_points, weighted_linear_smoothing


# %% [markdown]
# Load the data: The data is 40 ACCESS-ESM1.5 simulations for ssp245 and 40 SCALES emulations of the ssp245 sceanrio - SCALES was trained with all of CMIP-6 + some extra overshoot scenarios excluding ssp245. 

# %%
# --- Data loading (unchanged) ---
MODEL = 'ACCESS-ESM1-5'
scenario_data = load_scenarios(
    model=MODEL,
    indicators=['tas'],
    scenarios=['ssp245'],
    model_path=f'/Users/hoegner/Projects/data/CMIP6/{MODEL}/cmip6-ng-inc-oceans',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp245',
)
gmt_data = np.stack([lol[0,-2507:-7].transpose() for lol in scenario_data], axis=0)
scenario_data = np.stack([lol[1:,-2507:-7].transpose() for lol in scenario_data], axis=0)

pickle_filename = "/Users/hoegner/GitHub/scales_causal/outputs/ssp245_ensemble40_tas_monthly_3000m_21Apr26-monthly_variability.pkl"
y_pred_ensemble = pickle.load(open(pickle_filename, "rb"))

#y_pred_ensemble = scenario_data[20:, :, :]
#scenario_data   = scenario_data[0:20, :, :]

print(y_pred_ensemble.shape)
print(scenario_data.shape)
print(gmt_data.shape)
phases = ['stable', 'ramp-up']
split_points = find_phase_split_points(gmt_data[0,:], phases, window_size = gmt_data.shape[1]//len(phases))

plot = plot_gmt_phases(
    gmt =gmt_data[0,:],
    split_points = split_points,
    phases = phases,
    #tau: float = 20,
    #smooth: bool = True,
    #title: str | None = None,
    #save_path: str | None = None,
    #dpi: int = 300,
)


# %%
'''
import matplotlib.pyplot as plt
examples = {'ssp585': ['stable','ramp-up'],
           'ssp370':['stable','ramp-up'],
           'flat10cdrincspinoff':['stable', 'ramp-up','ramp-down'],
           'flat10zecincspinoff': ['stable', 'ramp-up', 'stable'],
           'ssp126': ['stable', 'ramp-up', 'stable'],
           'ssp534-over': ['stable', 'ramp-up','ramp-down', 'stable'],
           'abrupt-4xco2': ['stable', 'ramp-up','stable']}

for scenario, split in examples.items():
    print(scenario)
    MODEL = 'ACCESS-ESM1-5'
    scenario_data_ssp534 = load_scenarios(
        model=MODEL,
        indicators=['tas'],
        scenarios=[scenario],
        model_path=f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}',
        monthly_flag=True,
        use_smoothing=False,
        train_pattern_scaling_name=scenario,
    )
    gmt_data_ssp534 = np.stack([lol[0,:].transpose() for lol in scenario_data_ssp534], axis=0)
    
    print(gmt_data_ssp534.shape)
    #split = ['stable', 'ramp-up','ramp-down', 'stable']
    split_points = find_phase_split_points(gmt_data_ssp534[0,:], split, 50, window_size =gmt_data_ssp534.shape[1]//len(split))
    
    plot = plot_gmt_phases(
        gmt =gmt_data_ssp534[0,:],
        split_points = split_points,#split_points,#split_points,
        phases = split,
        tau = 50,
        #smooth: bool = True,
        #title: str | None = None,
        #save_path: str | None = None,
        #dpi: int = 300,
    )
    plt.show()
    '''

# %%
# Fit pattern scaling
print(gmt_data[0,:].shape)
print(scenario_data[0,:].shape)
fit = fit_regional_regressions_monthly(
    global_series=yearly_average(gmt_data[0,:]),
    regional_series=yearly_average(scenario_data[0,:]),
    monthly=False,
    train_ramp_down=False,
)

# Predict
ps_emulation = predict_pattern_scaling(
    fit=fit,
    global_series=yearly_average(gmt_data[0,:]),
    monthly=False,
    n_members=40,
)

lol = plot_error_metrics_bar(
    scenario_data = weighted_linear_smoothing(yearly_average(scenario_data)),
    y_pred_ensemble = weighted_linear_smoothing(yearly_average(y_pred_ensemble)),
    baseline_emulations = {'Pattern Scaling':weighted_linear_smoothing(ps_emulation)},
)

# %%
lol = plot_region_ensemble_extremes(
    scenario_data = weighted_linear_smoothing(yearly_average(scenario_data)),
    y_pred_ensemble = weighted_linear_smoothing(yearly_average(y_pred_ensemble)),
    baseline_emulations = {'Pattern Scaling':weighted_linear_smoothing(ps_emulation)},
)

# %%
lol = plot_gmt_vs_regional_linearity_extremes(
    gmt = weighted_linear_smoothing(yearly_average(gmt_data[0,:])),
    regional_data={'Simulation': weighted_linear_smoothing(yearly_average(scenario_data[0,:])),
                  'Emulation': weighted_linear_smoothing(yearly_average(y_pred_ensemble[0,:])),
                  'Pattern Scaling':weighted_linear_smoothing(ps_emulation[0,:])})


# %% [markdown]
# Plot some random emulations and simulations corresponding to the same ensemble member and region: 

# %%
plot_random_timeseries(y_pred_ensemble=select_month(y_pred_ensemble[:,:750,:],6),#yearly_average(ps_emulation),
    scenario_data=select_month(scenario_data[:,:750,:],6))



# %% [markdown]
# Plot some emulation/simulation pairs according to their performance, given an error metric
# (1) An example: Absolute difference between the overall standard deviations of the
#     two ensembles. ('sigma')
#
# Also possible would be:
# - Mean Continuous Ranked Probability Score (CRPS) over all timesteps ('cprs')
# - Wasserstein distance between the mean power spectral densities ('psd')
# - Absolute difference between the means ('mean')
# - MAE between mean Autocorrelation Function Curves ('mae_acf')
#
# The data can here be preprocessed, e.g., by removing seasonality, trend, or computing annual averages, or looking at a specific month. 

# %% [markdown]
# How this looks for annually averaged data and a perfect emulator:

# %%
scores = plot_metric_extremes(
    y_pred_ensemble=scenario_data[0:19,:,:],#
    scenario_data=scenario_data[20:,:,:],
    n_examples=5,
    yearly_average=True,
    metric='sigma',
    #detrend=True,
    #detrend_tau=5,
    #deseasonalise=True,
)  

# %% [markdown]
# Current version of SCALES:

# %%
scores = plot_metric_extremes(
    y_pred_ensemble=y_pred_ensemble[0:19,:,:],#
    scenario_data=scenario_data[20:,:,:],
    n_examples=5,
    yearly_average=True,
    metric='sigma',
    #detrend=True,
    #detrend_tau=5,
    #deseasonalise=True,
)  

# %% [markdown]
# How this would look for detrended and deseasonalised data and a perfect emulator

# %%
scores = plot_metric_extremes(
    y_pred_ensemble=scenario_data[0:19,:,:],#
    scenario_data=scenario_data[20:,:,:],
    n_examples=5,
    yearly_average=False,
    metric='sigma',
    detrend=True,
    detrend_tau=5,
    deseasonalise=True,
)  

# %% [markdown]
# How this looks for the current testing data from SCALES

# %%
scores = plot_metric_extremes(
    y_pred_ensemble=y_pred_ensemble[0:19,:,:],
    scenario_data=scenario_data[20:,:,:],
    n_examples=5,
    yearly_average=False,
    metric='sigma',
    detrend=True,
    detrend_tau=5,
    deseasonalise=True,
)

# %% [markdown]
# Plot spatial correlation patterns within emulations and simulations

# %%
results = plot_spatial_correlations(
    scenario_data[0:19,:750,:],#y_pred_ensemble,
    scenario_data[20:,:750,:],
    yearly_average=False,
    month_selection = 10,
    #detrend=True,
    #detrend_tau=20,
    #region_names=my_region_names,
    #save_path="spatial_corr.png",
)

# Access raw matrices
sim_corr = results["sim_corr"]
diff     = results["diff"]
print(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")

# %%
results = plot_spatial_correlations(
    y_pred_ensemble[0:19,:750,:],
    scenario_data[0:19,:750,:],
    yearly_average=False,
    month_selection = 10,
    #yearly_average=False,
    #detrend=True,
    #detrend_tau=20,
    #region_names=my_region_names,
    #save_path="spatial_corr.png",
)

# Access raw matrices
sim_corr = results["sim_corr"]
diff     = results["diff"]
print(f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")

# %% [markdown]
# And within 2 different sets of simulation, to check for sample uncertainty.

# %% [markdown]
# Q-Q plots as they should look for a perfect emulator: 

# %%
results = plot_qq_extremes(
    scenario_data[0:19,:,:],#y_pred_ensemble,
    scenario_data[20:,:,:],
    n_examples=5,
    metric="tail_mae",       # focus on tail performance
    yearly_average=True,
    detrend=True,
    detrend_tau=5,
    #save_path="qq_extremes.png",
)

# %% [markdown]
# Q-Q plots as they actually look: 

# %%
results = plot_qq_extremes(
    y_pred_ensemble[0:19,:,:],#y_pred_ensemble,
    scenario_data[20:,:,:],
    n_examples=5,
    metric="tail_mae",       # focus on tail performance
    yearly_average=True,
    detrend=True,
    detrend_tau=5,
    #save_path="qq_extremes.png",
)

# %%
