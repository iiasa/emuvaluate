# emuvaluate — Climate emulator evaluation and validation pipeline

Evaluation toolkit for climate emulator ensembles. Provides preprocessing, scoring metrics, and diagnostic plots to compare emulated regional temperature outputs against CMIP6 simulations.

## Status

- prototype: the project is just starting up and the code is all prototype

---

## Installation

We do all our environment management using [uv](https://docs.astral.sh/uv/).
To get started, you will need to make sure that uv is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/),
we found that using uv's standalone installer was best on a Mac).

To create the virtual environment, run

```sh
uv sync
uv run pre-commit install
```

These steps are also captured in the `Makefile` so if you want a single
command, you can instead simply run `make virtual-enviroment`.

Having installed your virtual environment, you can now run commands in your
virtual environment using

```sh
uv run <command>
```

For example, to run Python within the virtual environment, run

```sh
uv run python
```

As another example, to run a notebook server, run

```sh
uv run jupyter lab
```

Alternatively, to install directly with pip in editable mode:

```bash
git clone git@github.com:iiasa/emuvaluate.git
cd emuvaluate
pip install -e .
```

---

## Package structure

```
emuvaluate/
├── pyproject.toml
├── notebooks/
└── src/
    └── emuvaluate/
        ├── metrics.py           # scoring functions
        ├── transforms.py        # preprocessing and smoothing functions
        ├── plots.py             # diagnostic plots
        ├── baseline_methods.py  # pattern scaling and other baselines
        └── data_preparation.py  # data loading pipeline
```

---

## Usage

### Data loading

```python
from emuvaluate.data_preparation import load_scenarios

scenario_data = load_scenarios(
    model='ACCESS-ESM1-5',
    indicators=['tas'],
    scenarios=['ssp245'],
    model_path='/path/to/model/data',
    monthly_flag=True,
    use_smoothing=False,
    train_pattern_scaling_name='ssp245',
)
```

### Full diagnostic plot

```python
from emuvaluate.plots import plot_metric_extremes

scores = plot_metric_extremes(
    y_pred_ensemble=y_pred,   # (n_members, T, n_regions)
    scenario_data=scenario,   # (n_members, T, n_regions)
    metric='crps',
    n_examples=5,
    detrend=True,
    detrend_tau=5,
    deseasonalise=True,
)
```

### QQ diagnostic plot

```python
from emuvaluate.plots import plot_qq_extremes

results = plot_qq_extremes(
    y_pred_ensemble=y_pred,
    scenario_data=scenario,
    metric='tail_mae',
    n_examples=5,
    yearly_average=True,
    detrend=True,
    detrend_tau=40,
    save_path='qq_extremes.png',
)
```

### Spatial correlation diagnostic

```python
from emuvaluate.plots import plot_spatial_correlations

results = plot_spatial_correlations(
    y_pred_ensemble=y_pred,
    scenario_data=scenario,
    yearly_average=True,
    detrend=True,
    detrend_tau=40,
    region_names=my_region_names,
    save_path='spatial_corr.png',
)
# results contains: 'sim_corr', 'emu_corr', 'diff', 'mae', 'rmse'
```

### Error metric bar chart with baselines

```python
from emuvaluate.plots import plot_error_metrics_bar
from emuvaluate.baseline_methods import fit_regional_regressions_monthly, predict_pattern_scaling

# Fit pattern scaling on training data
fit = fit_regional_regressions_monthly(
    global_series=gmt_train,
    regional_series=regional_train,
    monthly=True,
    train_ramp_down=True,
)

# Predict on test GMT
ps_emulation = predict_pattern_scaling(
    fit=fit,
    global_series=gmt_test,
    monthly=True,
    n_members=40,
)

# Plot emulator vs baseline
results = plot_error_metrics_bar(
    scenario_data=simulator_data,
    y_pred_ensemble=emulator_data,
    baseline_emulations={'Pattern Scaling': ps_emulation},
    yearly_average=True,
    metrics=['mae', 'rmse', 'max_ae'],
    region_names=my_region_names,
    save_path='error_metrics.png',
)
```

### Ensemble timeseries comparison with baselines

```python
from emuvaluate.plots import plot_region_ensemble_extremes

results = plot_region_ensemble_extremes(
    scenario_data=simulator_data,
    y_pred_ensemble=emulator_data,
    baseline_emulations={'Pattern Scaling': ps_emulation},
    n_examples=5,
    metric='mae',
    ranking_mode='difference',   # rank by how much emulator improves on baseline
    yearly_average=True,
    region_names=my_region_names,
    save_path='ensemble_extremes.png',
)
```

### GMT vs regional scatter plots

```python
from emuvaluate.plots import plot_gmt_vs_regional, plot_gmt_vs_regional_linearity_extremes
from emuvaluate.transforms import weighted_linear_smoothing, yearly_average

gmt_smooth       = weighted_linear_smoothing(yearly_average(gmt_data[0, :]))
scenario_smooth  = weighted_linear_smoothing(yearly_average(scenario_data))
emulation_smooth = weighted_linear_smoothing(yearly_average(y_pred_ensemble))

# Plot specific or random regions and samples
plot_gmt_vs_regional(
    gmt=gmt_smooth,
    regional_data={
        'Simulation':      scenario_smooth,
        'Emulation':       emulation_smooth,
        'Pattern Scaling': weighted_linear_smoothing(yearly_average(ps_emulation)),
    },
    n_random=5,
    region_names=my_region_names,
)

# Plot most and least linear regions
results = plot_gmt_vs_regional_linearity_extremes(
    gmt=gmt_smooth,
    regional_data={
        'Simulation':      scenario_smooth,
        'Emulation':       emulation_smooth,
        'Pattern Scaling': weighted_linear_smoothing(yearly_average(ps_emulation)),
    },
    n_examples=5,
    n_random_samples=5,
    region_names=my_region_names,
    save_path='linearity.png',
)
```

### GMT phase detection and splitting

```python
from emuvaluate.transforms import find_phase_split_points, split_at_indices
from emuvaluate.plots import plot_gmt_phases

# Find split points that best match the requested phase sequence
split_points = find_phase_split_points(
    gmt=gmt,
    phases=['stable', 'ramp-up', 'ramp-down', 'stable'],
    tau=20,
)

# Visualise the detected phases
plot_gmt_phases(
    gmt=gmt,
    split_points=split_points,
    phases=['stable', 'ramp-up', 'ramp-down', 'stable'],
    save_path='gmt_phases.png',
)

# Split data along T at those points
segments = split_at_indices(scenario_data, split_points)
# segments[0] covers the first stable phase, segments[1] the ramp-up, etc.
```

### Smoothing

```python
from emuvaluate.transforms import weighted_linear_smoothing

# 1-D GMT timeseries
gmt_smooth = weighted_linear_smoothing(gmt, tau=20)

# 2-D array (T, n_regions)
regional_smooth = weighted_linear_smoothing(regional_data, tau=20)

# Full ensemble array (n_members, T, n_regions)
data_smooth = weighted_linear_smoothing(scenario_data, tau=20)

# Monthly data — smooth each calendar month independently
data_smooth_monthly = weighted_linear_smoothing(scenario_data, tau=20, monthly=True)
```

### Using components individually

```python
from emuvaluate.transforms import preprocess, detrend_gaussian, deseasonalise
from emuvaluate.metrics import compute_metric_all_regions, crps_score
from emuvaluate.plots import rank_regions

# Preprocess
data_clean = deseasonalise(detrend_gaussian(data, tau=20))

# Score all regions
scores = compute_metric_all_regions(obs, pred, metric='crps')

# Rank
best, worst = rank_regions(scores, n_examples=5)
```

---

## API reference

### `emuvaluate.metrics`

#### Ensemble scoring metrics
Accept arrays of shape `(n_members, T)` and return a single float (lower = better).

| Function | Description |
|---|---|
| `crps_score(obs, pred)` | Mean Continuous Ranked Probability Score over all timesteps. |
| `mean_score(obs, pred)` | Absolute difference between ensemble grand means. |
| `sigma_score(obs, pred)` | Absolute difference between ensemble standard deviations. |
| `psd_score(obs, pred)` | Wasserstein distance between mean power spectral densities (Welch method). |
| `compute_metric_all_regions(obs, pred, metric)` | Applies a named metric to every region. Returns `dict[region_idx → score]`. |

Available metric names: `'crps'`, `'mean'`, `'sigma'`, `'psd'`.

#### Error metrics
Accept arrays of shape `(n_members, T)` and return a single float (lower = better).

| Function | Description |
|---|---|
| `mae_score(obs, pred)` | Mean Absolute Error between ensemble means. |
| `rmse_score(obs, pred)` | Root Mean Squared Error between ensemble means. |
| `max_ae_score(obs, pred)` | Maximum Absolute Error between ensemble means. |
| `mse_score(obs, pred)` | Mean Squared Error between ensemble means. |
| `nmae_score(obs, pred)` | Normalised MAE — divided by the range of the observed ensemble mean. |
| `nmse_score(obs, pred)` | Normalised MSE — divided by the variance of the observed ensemble mean. |

Available error metric names: `'mae'`, `'rmse'`, `'max_ae'`, `'mse'`, `'nmae'`, `'nmse'`.

#### QQ metrics
Compare quantile distributions between simulations and emulations.

| Function | Description |
|---|---|
| `qq_mae(obs_q, pred_q)` | Mean Absolute Error between quantiles. |
| `qq_nmae(obs_q, pred_q)` | Normalised MAE — divided by the range of observed quantiles. |
| `qq_max_ae(obs_q, pred_q)` | Maximum Absolute Error between quantiles. |
| `qq_mse(obs_q, pred_q)` | Mean Squared Error between quantiles. |
| `qq_nmse(obs_q, pred_q)` | Normalised MSE — divided by variance of observed quantiles. |
| `qq_ks(obs_q, pred_q)` | Kolmogorov-Smirnov statistic. |
| `qq_tail_mae(obs_q, pred_q, tail_frac=0.1)` | MAE restricted to the upper and lower tails of the quantile grid. |
| `compute_qq_scores_all_regions(obs, pred, metric, quantiles)` | Applies a named QQ metric to every region. Returns `(scores, obs_qq, pred_qq)`. |

Available QQ metric names: `'mae'`, `'nmae'`, `'max_ae'`, `'mse'`, `'nmse'`, `'ks'`, `'tail_mae'`.

#### Spatial correlation metrics

| Function | Description |
|---|---|
| `spatial_correlation_matrix(data)` | Computes mean spatial correlation matrix `(n_regions, n_regions)` across ensemble members. |
| `spatial_correlation_scores(obs, pred)` | Returns dict with `sim_corr`, `emu_corr`, `diff`, `mae`, `rmse`. |

#### Linearity metric

| Function | Description |
|---|---|
| `linearity_score(obs, gmt)` | R² of a linear regression of regional values on GMT. Higher = more linear. Accepts `obs` of shape `(n_members, T)` and `gmt` of shape `(n_members, T)` or `(T,)`. |

---

### `emuvaluate.transforms`

#### Aggregation and selection

| Function | Accepted shapes | Description |
|---|---|---|
| `yearly_average(data)` | `(T,)`, `(T, n_regions)`, `(n_members, T, n_regions)` | Collapses 12 consecutive monthly timesteps into annual means. Trailing incomplete years are dropped. |
| `select_month(data, month)` | `(n_members, T, n_regions)` | Retains only timesteps for a given calendar month (1–12). |
| `deseasonalise(data, period=12)` | `(n_members, T, n_regions)` | Subtracts the mean seasonal cycle per member and region. |
| `detrend_gaussian(data, tau=20)` | `(n_members, T, n_regions)` | Removes a Gaussian-smoothed trend per member and region. |
| `preprocess(data, ...)` | `(n_members, T, n_regions)` | Convenience wrapper: aggregation → deseasonalise → detrend. |

`preprocess` keyword arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `apply_yearly_average` | bool | False | Collapse to annual means |
| `month_selection` | int or None | None | Keep one calendar month only |
| `apply_deseasonalise` | bool | False | Remove seasonal cycle |
| `apply_detrend` | bool | False | Remove Gaussian trend |
| `detrend_tau` | float | 20 | Smoothing sigma for detrending |

`apply_yearly_average` and `month_selection` are mutually exclusive.

#### Smoothing

| Function | Accepted shapes | Description |
|---|---|---|
| `weighted_linear_smoothing(data, tau=20, monthly=False)` | `(T,)`, `(T, n_regions)`, `(n_members, T, n_regions)` | Local weighted regression smoothing along T. If `monthly=True`, each calendar month is smoothed independently across years (T must be divisible by 12). |

#### Phase detection and splitting

| Function | Description |
|---|---|
| `split_at_indices(data, split_points)` | Splits `(n_members, T, n_regions)` along T at the given indices. Returns `len(split_points) + 1` arrays. |
| `find_phase_split_points(gmt, phases, ...)` | Finds split points that best divide a GMT timeseries into a requested sequence of phases (`'ramp-up'`, `'ramp-down'`, `'stable'`). GMT is smoothed before detection. Always returns `len(phases) - 1` split points. |

`find_phase_split_points` keyword arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `tau` | float | 20 | Smoothing bandwidth |
| `smooth` | bool | True | Smooth GMT before phase detection |
| `stable_window` | int | 30 | Timesteps over which to assess stability |
| `stable_threshold` | float | 0.1 | Max GMT change over `stable_window` to count as stable |
| `min_phase_length` | int | 10 | Minimum timesteps for a valid phase segment |
| `window_size` | int or None | None | Rolling window size for ramp detection (defaults to T // 3) |

---

### `emuvaluate.plots`

| Function | Description |
|---|---|
| `plot_metric_extremes(...)` | Scores all regions by a chosen metric and plots the best and worst examples side by side. |
| `plot_qq_extremes(...)` | Scores all regions by a QQ metric and plots QQ scatter and quantile curves for best and worst regions. |
| `plot_spatial_correlations(...)` | Plots spatial correlation matrices for simulations and emulations, plus their difference. |
| `plot_error_metrics_bar(...)` | Per-region error metrics as grouped bar charts, comparing emulator against one or more baselines. |
| `plot_region_ensemble_extremes(...)` | Plots best and worst regions with all methods overlaid (simulation, emulator, baselines), ranked by a chosen error metric. |
| `plot_gmt_vs_regional(...)` | Scatter plot of regional values vs GMT for multiple data sources and a chosen set of regions and samples. |
| `plot_gmt_vs_regional_linearity_extremes(...)` | Ranks regions by GMT–regional linearity (R²) and plots the most and least linear using `plot_gmt_vs_regional`. |
| `plot_gmt_phases(...)` | Plots a GMT timeseries with phase splits highlighted as shaded bands, with raw and smoothed GMT overlaid. |
| `plot_random_timeseries(...)` | Plots randomly sampled (ensemble, region) pairs: true vs predicted. |
| `rank_regions(metric_scores, n_examples=5)` | Splits a scores dict into the `n_examples` best and worst region pairs. |

`plot_metric_extremes` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `y_pred_ensemble` | ndarray | — | Emulated ensemble `(n_members, T, n_regions)` |
| `scenario_data` | ndarray | — | Ground-truth ensemble `(n_members, T, n_regions)` |
| `n_examples` | int | 5 | Number of best/worst regions to show |
| `metric` | str | `'crps'` | One of `'crps'`, `'mean'`, `'sigma'`, `'psd'` |
| `yearly_average` | bool | False | Aggregate to annual before scoring |
| `month_selection` | int or None | None | Restrict to one calendar month |
| `detrend` | bool | False | Remove Gaussian trend before scoring |
| `detrend_tau` | float | 20 | Smoothing sigma for detrending |
| `deseasonalise` | bool | False | Remove seasonal cycle before scoring |
| `save_path` | str or None | None | Save figure to this path |

`plot_error_metrics_bar` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `scenario_data` | ndarray | — | Ground-truth ensemble `(n_members, T, n_regions)` |
| `y_pred_ensemble` | ndarray | — | Emulated ensemble `(n_members, T, n_regions)` |
| `baseline_emulations` | dict or None | None | Dict of `name → (n_members, T, n_regions)` |
| `metrics` | list | `['mae','rmse','max_ae']` | Error metrics to plot |
| `yearly_average` | bool | False | Aggregate to annual before scoring |
| `month_selection` | int or None | None | Restrict to one calendar month |
| `detrend` | bool | False | Remove Gaussian trend |
| `detrend_tau` | float | 20 | Smoothing sigma for detrending |
| `deseasonalise` | bool | False | Remove seasonal cycle |
| `region_names` | list or None | None | Region label strings |
| `save_path` | str or None | None | Save figure to this path |

`plot_region_ensemble_extremes` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `scenario_data` | ndarray | — | Ground-truth ensemble `(n_members, T, n_regions)` |
| `y_pred_ensemble` | ndarray | — | Emulated ensemble `(n_members, T, n_regions)` |
| `baseline_emulations` | dict or None | None | Dict of `name → (n_members, T, n_regions)` |
| `n_examples` | int | 5 | Number of best/worst regions to show |
| `metric` | str | `'mae'` | Error metric key from `ERROR_METRIC_REGISTRY` |
| `ranking_mode` | str | `'emulator'` | One of `'emulator'`, `'baseline'`, `'difference'` |
| `yearly_average` | bool | False | Aggregate to annual before scoring |
| `month_selection` | int or None | None | Restrict to one calendar month |
| `detrend` | bool | False | Remove Gaussian trend |
| `detrend_tau` | float | 20 | Smoothing sigma for detrending |
| `deseasonalise` | bool | False | Remove seasonal cycle |
| `region_names` | list or None | None | Region label strings |
| `save_path` | str or None | None | Save figure to this path |

`plot_gmt_vs_regional` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `gmt` | ndarray | — | GMT timeseries `(T,)` or `(n_samples, T)` |
| `regional_data` | dict | — | Dict of `name → (n_samples, T, n_regions)` or `(T, n_regions)` |
| `region_indices` | list or None | None | Region indices to plot; if None uses `n_random` |
| `sample_indices` | list or None | None | Sample indices to plot; if None uses `n_random` |
| `n_random` | int or None | None | Number of random regions/samples to draw if indices not given |
| `region_names` | list or None | None | Region label strings |
| `save_path` | str or None | None | Save figure to this path |

`plot_gmt_vs_regional_linearity_extremes` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `gmt` | ndarray | — | GMT timeseries `(T,)` or `(n_samples, T)` |
| `regional_data` | dict | — | Dict of `name → (n_samples, T, n_regions)` or `(T, n_regions)`; linearity ranked on first entry |
| `n_examples` | int | 5 | Number of most/least linear regions to show |
| `sample_indices` | list or None | None | Fixed sample indices; if None draws `n_random_samples` |
| `n_random_samples` | int | 5 | Number of random samples if `sample_indices` not given |
| `region_names` | list or None | None | Region label strings |
| `save_path` | str or None | None | Base path — `_most_linear` / `_least_linear` appended before extension |

`plot_gmt_phases` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `gmt` | ndarray | — | GMT timeseries `(T,)` |
| `split_points` | list | — | Split indices along T |
| `phases` | list or None | None | Phase label strings, length `len(split_points) + 1` |
| `tau` | float | 20 | Smoothing bandwidth for overlay |
| `smooth` | bool | True | Whether to overlay the smoothed GMT |
| `title` | str or None | None | Optional figure title |
| `save_path` | str or None | None | Save figure to this path |

---

### `emuvaluate.baseline_methods`

| Function | Description |
|---|---|
| `fit_regional_regressions(global_series, regional_series, ...)` | Fits per-region linear regressions against GMT with optional ramp-down correction. |
| `fit_regional_regressions_monthly(global_series, regional_series, ...)` | Fits regressions stratified by calendar month. Returns a list of 12 dicts (monthly) or a single dict (annual). |
| `predict_pattern_scaling(fit, global_series, ...)` | Applies fitted coefficients to a GMT series. Returns `(n_members, T, n_regions)`. |

`fit_regional_regressions_monthly` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `global_series` | ndarray | — | GMT timeseries `(T,)` |
| `regional_series` | ndarray | — | Regional indicators `(T, n_regions)` |
| `train_ramp_down` | bool | False | Fit a separate correction for the ramp-down phase |
| `monthly` | bool | True | Stratify by calendar month; set False for annual data |

`predict_pattern_scaling` arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `fit` | list or dict | — | Output of `fit_regional_regressions_monthly` |
| `global_series` | ndarray | — | GMT timeseries to emulate `(T,)` |
| `monthly` | bool | True | Must match how `fit` was produced |
| `n_members` | int | 1 | Number of output ensemble members (deterministic — all identical) |

---

### `emuvaluate.data_preparation`

| Function | Description |
|---|---|
| `load_scenarios(model, indicators, scenarios, model_path, ...)` | Full pipeline: finds files, loads CSVs, computes anomalies relative to baseline, optionally applies pattern scaling. Returns a list of arrays, one per scenario. |
| `prepare_scenario_data(...)` | Single-indicator version of `load_scenarios`. |
| `process_scenarios(...)` | Loads and aligns one baseline/scenario CSV pair and returns `(gmt_df, regional_df)`. |

---

## Development

Install and run instructions are the same as the above (this is a simple
repository, without tests etc. so there are no development-only dependencies).

### Contributing

This is a very thin repository. There aren't any strict guidelines for
contributing, partly because we don't know what we're trying to achieve (we're
just exploring). If you would like to contribute, it is best to raise an issue
to discuss what you want to do (without a discussion, we can't guarantee that
any contribution can actually be used).

### Repository structure

The repository is very basic. It imposes no structure on you so you can layout
your Python files, notebooks etc. in any way you wish. We do have a basic
`Makefile` which captures key commands in one place (for more thoughts on why
this makes sense, see
[general principles: automation](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/general-principles/automation.md)).
For an introduction to `make`, see
[this introduction from Software Carpentry](https://swcarpentry.github.io/make-novice/).
Having said this, if you're not interested in `make`, you can just copy the
commands out of the `Makefile` by hand and you will be 90% as happy for a
simple repository like this.

### Tools

In this repository, we use the following tools:

- git for version-control (for more on version control, see
  [general principles: version control](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/theory/version-control.md))
    - for these purposes, git is a great version-control system so we don't
      complicate things any further. For an introduction to Git, see
      [this introduction from Software Carpentry](http://swcarpentry.github.io/git-novice/).
- [uv](https://docs.astral.sh/uv/) for environment management
  (for more on environment management, see
  [general principles: environment management](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/theory/environment-management.md))
    - there are lots of environment management systems.
      uv works well in our experience.
    - we track the `uv.lock` file so that the environment
      is completely reproducible on other machines or by other people
      (e.g. if you want a colleague to take a look at what you've done)
- [pre-commit](https://pre-commit.com/) with some very basic settings to get some
  easy wins in terms of maintenance, specifically:
    - code formatting with [ruff](https://docs.astral.sh/ruff/formatter/)
    - basic file checks (removing unneeded whitespace, not committing large
      files etc.)
    - (for more thoughts on the usefulness of pre-commit, see
      [general principles: automation](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/general-principles/automation.md))
    - track your notebooks using
      [jupytext](https://jupytext.readthedocs.io/en/latest/index.html)
      (for more thoughts on the usefulness of Jupytext, see
      [tips and tricks: Jupytext](https://gitlab.com/znicholls/mullet-rse/-/blob/main/book/tips-and-tricks/managing-notebooks-jupytext.md))
        - this avoids nasty merge conflicts and incomprehensible diffs

---

## Authors

Annika Högner, Verena Kain, Tessa Möller, Zebedee Nicholls, Niklas Schwind, Marco Zecchetto — IIASA

---

## Original template

This project was generated from this template:
[basic python repository](https://gitlab.com/openscm/copier-basic-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.