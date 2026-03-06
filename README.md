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
        ├── metrics.py          # scoring functions
        ├── transforms.py       # preprocessing functions
        ├── plots.py            # diagnostic plots
        └── data_preparation.py # data loading pipeline
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

| Function | Description |
|---|---|
| `crps_score(obs, pred)` | Mean Continuous Ranked Probability Score over all timesteps. Lower is better. |
| `mean_score(obs, pred)` | Absolute difference between ensemble grand means. |
| `sigma_score(obs, pred)` | Absolute difference between ensemble standard deviations. |
| `psd_score(obs, pred)` | Wasserstein distance between mean power spectral densities (Welch method). |
| `compute_metric_all_regions(obs, pred, metric)` | Applies a named metric to every region. Returns `dict[region_idx → score]`. |

All score functions accept arrays of shape `(n_members, T)` and return a single float.
Available metric names: `'crps'`, `'mean'`, `'sigma'`, `'psd'`.

---

### `emuvaluate.transforms`

All functions accept arrays of shape `(n_members, T, n_regions)` and return a transformed copy.

| Function | Description |
|---|---|
| `yearly_average(data)` | Collapses 12 consecutive monthly timesteps into annual means. |
| `select_month(data, month)` | Retains only timesteps for a given calendar month (1–12). |
| `deseasonalise(data, period=12)` | Subtracts the mean seasonal cycle per member and region. |
| `detrend_gaussian(data, tau=20)` | Removes a Gaussian-smoothed trend (sigma=`tau` timesteps) per member and region. |
| `preprocess(data, ...)` | Convenience wrapper applying all steps in order: aggregation → deseasonalise → detrend. |

`preprocess` keyword arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `apply_yearly_average` | bool | False | Collapse to annual means |
| `month_selection` | int or None | None | Keep one calendar month only |
| `apply_deseasonalise` | bool | False | Remove seasonal cycle |
| `apply_detrend` | bool | False | Remove Gaussian trend |
| `detrend_tau` | float | 20 | Smoothing sigma for detrending |

`apply_yearly_average` and `month_selection` are mutually exclusive.

---

### `emuvaluate.plots`

| Function | Description |
|---|---|
| `plot_metric_extremes(y_pred_ensemble, scenario_data, ...)` | Preprocesses data, scores all regions, and plots the best and worst examples side by side. Returns `dict[region_idx → score]`. |
| `rank_regions(metric_scores, n_examples=5)` | Splits a scores dict into the `n_examples` best and worst region pairs. Useful if you already have scores from a previous run. |

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
| `save_path` | str or None | None | Save figure to this path at 300 dpi |

---

### `emuvaluate.data_preparation`

| Function | Description |
|---|---|
| `load_scenarios(model, indicators, scenarios, model_path, ...)` | Full pipeline: finds files, loads CSVs, computes anomalies relative to baseline, optionally applies pattern scaling. Returns a list of arrays, one per scenario. |
| `prepare_scenario_data(...)` | Single-indicator version of `load_scenarios`. |
| `process_scenarios(...)` | Loads and aligns one baseline/scenario CSV pair and returns `(gmt_df, regional_df)`. |
| `fit_regional_regressions(global_series, regional_series, ...)` | Fits per-region linear regressions against GMT, with optional ramp-down correction. |
| `predict_regional_temperatures(global_test_series, slopes, intercepts, ...)` | Applies fitted regression parameters to a new GMT series to predict regional temperatures. |

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
