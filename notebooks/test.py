# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: climate-data-processing
#     language: python
#     name: climate-data-processing
# ---

# %%
import numpy as np
import pickle
import emuvaluate
from emuvaluate.plots import plot_metric_extremes, plot_random_timeseries
from emuvaluate.data_preparation import load_scenarios

# %%
# --- Data loading (unchanged) ---
MODEL = 'ACCESS-ESM1-5'
scenario_data = load_scenarios(
    model=MODEL,
    indicators=['tas'],
    scenarios=['ssp245'],
    model_path=f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp245',
)
scenario_data = np.stack([lol[1:,-2507:-7].transpose() for lol in scenario_data], axis=0)

pickle_filename = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/ACCESS/ssp245/ssp245_ensemble40_monthly_2500m.pkl"
y_pred_ensemble = pickle.load(open(pickle_filename, "rb"))

#y_pred_ensemble = scenario_data[20:, :, :]
#scenario_data   = scenario_data[0:20, :, :]

print(y_pred_ensemble.shape)
print(scenario_data.shape)




# %%
plot_random_timeseries(y_pred_ensemble=y_pred_ensemble,
    scenario_data=scenario_data)

# %%
# --- Evaluate ---
crps_scores = plot_metric_extremes(
    y_pred_ensemble=y_pred_ensemble,
    scenario_data=scenario_data,
    n_examples=5,
    yearly_average=False,
    metric='mean',
    detrend=True,
    detrend_tau=5,
    deseasonalise=True,
)

# %%
