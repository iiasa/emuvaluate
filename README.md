# emuvaluate

Evaluate climate emulators against the simulations they are trying to emulate.

You give it a simulation ensemble and an emulator ensemble — plus, optionally,
one or more *baseline* emulators to compare against — and it tells you where
and how the emulator differs, as per-region or per-gridpoint error values and
as publication-ready figures.

The package works at two spatial scales, with the same API on both sides:

| flavour      | unit           | data shape                                          |
| ------------ | -------------- | --------------------------------------------------- |
| **regional** | an AR6 region  | `(n_members, T, n_regions)` numpy arrays             |
| **gridded**  | a gridpoint    | `xr.DataArray` with dims `(ensemble, time, lat, lon)`|

Function names follow that split: `preprocess_regional` / `preprocess_gridded`,
`build_error_data_regional` / `build_error_data_gridded`, `plot_map_regional` /
`plot_map_gridded`. Functions that work on both take plain arrays and are
named without a suffix (`plot_qq_scatter`, `plot_psd_curves`,
`plot_temporal_correlation_curves`).

---

## Installation

```bash
pip install -e .
```

Needs Python ≥ 3.9. The heavier dependencies are `cartopy` and `regionmask`
(map drawing), `statsmodels` (autocorrelation), `scipy` (Welch spectra,
Wasserstein distance) and `xarray` (gridded data).

`regionmask` downloads the IPCC AR6 reference-region shapefile on first use,
and `cartopy` downloads Natural Earth coastlines — so the first map you draw
needs network access.

If you want to use the notebooks run: 

```bash
jupytext --sync notebooks/*.py
```

---

## The three steps

Every figure in this package is produced the same way, and the boundaries
between the steps are strict: **plots never compute scores, and metrics never
preprocess data.**

```
preprocess  ─────►  build error data  ─────►  plot
preprocessing.py     metrics.py               plots.py
```

```python
from emuvaluate.preprocessing import preprocess_regional, select_members
from emuvaluate.metrics import build_error_data_regional
from emuvaluate.plots import plot_map_regional, plot_timeseries_regional

# 1. preprocess — dict in, dict out, one entry per indicator
sim = preprocess_regional({"tas": tas_raw, "pr": pr_raw}, yearly_average=True)
emu = preprocess_regional({"tas": tas_emu_raw, "pr": pr_emu_raw}, yearly_average=True)

# 2. score every region, for every indicator and comparison, with one metric
error_data = build_error_data_regional(
    sim, emu, region_names,
    metric="nmae",
    baseline_emulations={"Pattern Scaling": ps},
)

# 3. plot — the figure functions only lay out what step 2 computed
plot_map_regional(error_data, show_difference=True)
plot_timeseries_regional(error_data, sim, emu, {"Pattern Scaling": ps}, years)
```

`error_data` is an `ErrorData` object: per-unit scores for every comparison and
indicator, a shared colour-scale maximum, an optional Emulator-minus-baseline
difference, and a full ranking of units from best to worst. Every plotting
function reads from it — which is why swapping `metric="nmae"` for
`metric="crps"` or `metric="psd_log_nmae"` changes what a map shows without
touching the plotting call.

---

## Package structure

```
src/emuvaluate/
├── data_preparation.py   load raw CMIP6 scenario files into ensemble arrays
├── baseline_methods.py   the Pattern Scaling baseline emulator to compare against
├── transforms.py         stateless array operations the rest builds on
├── preprocessing.py      raw ensembles ──► scoring-ready arrays
├── metrics.py            arrays ──► error values, packaged as ErrorData
└── plots.py              ErrorData ──► figures
```

### `data_preparation.py` — loading

Reads CMIP6-ng scenario files off disk and turns them into the
`(n_members, T, n_regions)` arrays everything else consumes.

| function                            | what it does                                                     |
| ----------------------------------- | ---------------------------------------------------------------- |
| `load_scenarios(...)`               | the one entry point: model + indicators + scenarios ──► arrays    |
| `process_scenarios(...)`            | per-scenario file handling, baseline subtraction, smoothing       |
| `process_gmt_and_regions_into_array` | GMT + regional dataframes ──► the stacked numpy array            |
| `filter_climate_files`, `get_all_files_`, `get_baseline_filename`, `parse_filename` | file discovery and matching |

### `baseline_methods.py` — the reference emulator

Pattern Scaling: regress each region/gridpoint on GMT, then predict.
It's what "is the emulator actually better than the obvious thing?" is
measured against.

| function                                | scale    |
| --------------------------------------- | -------- |
| `fit_regional_regressions_monthly`      | regional |
| `predict_pattern_scaling`               | regional |
| `fit_gridpoint_regressions_monthly`     | gridded  |
| `predict_pattern_scaling_gridded`       | gridded  |
| `compute_gmt_area_weighted`             | both — area-weighted GMT from a gridded `tas` field |

### `transforms.py` — stateless array operations

`yearly_average`, `yearly_average_gridded`, `select_month`,
`weighted_linear_smoothing` (locally-weighted-regression "forced response"
smoothing). Used by `preprocessing.py`; occasionally useful directly.

### `preprocessing.py` — getting data ready to score

Two entry points, one per flavour, applying the same optional steps in the
same order so regional and gridded results stay comparable:

1. temporal aggregation (`yearly_average` **or** `month_selection`)
2. `remove_ensemble_mean` — subtract the member-axis mean
3. `smooth` — locally-weighted-regression smoothing, per member

| function                          | notes                                                        |
| --------------------------------- | ------------------------------------------------------------ |
| `preprocess_regional(data, ...)`  | array or `{indicator: array}` in, same shape out              |
| `preprocess_gridded(data, ...)`   | `xr.DataArray` or dict in, `GriddedArray` out; validates every entry against a reference grid via `ref_lat`/`ref_lon` |
| `select_members(data, members)`   | slice the ensemble axis of anything — array, `GriddedArray`, or dict of either |
| `scale_indicators(data, factors)` | multiply named indicators by a constant, e.g. `{"pr": 86400}` |
| `GriddedArray`                    | flattened `(ensemble, time, n_grid)` values + the lat/lon metadata needed to fold per-gridpoint scores back into a map |

`remove_ensemble_mean` removes whatever trend and seasonal cycle are common
across ensemble members, leaving internal variability around the forced
response. Apply it **before** splitting into subsets, so that a main ensemble
and a "Simulations vs Simulations" baseline drawn from the same pool share one
ensemble-mean reference:

```python
full = preprocess_regional(raw, remove_ensemble_mean=True)
sim = select_members(full, slice(20, None))
baseline = select_members(full, slice(0, 20))
```

### `metrics.py` — every number

Four metric families. Pass any of these names as `metric=`; call
`available_metrics()` for the live list.

**Scalar** — one score straight from a unit's `(n_members, T)` pair:

| name | meaning |
| ---- | ------- |
| `mae`, `mse`, `rmse`, `max_ae` | absolute / squared error |
| `nmae`, `nmse`        | the same, normalised by the observed range / variance |
| `mean_bias`, `sigma`  | difference in grand mean / standard deviation |
| `crps`                | Continuous Ranked Probability Score, generalised to a multi-member "truth" — every simulation member is treated as an independent verification draw |

**Quantile curves** — the per-unit quantity is a quantile curve, the score is
the distance between the simulated and the emulated one:

| name | meaning |
| ---- | ------- |
| `qq_mae`, `qq_nmae` | mean (normalised) absolute error between quantile curves |
| `qq_ks`             | max difference — the Kolmogorov–Smirnov statistic |
| `qq_tail_mae`       | restricted to the distribution tails |

**Autocorrelation curves** — `temporal_corr_nmae`, the NMAE between the
simulated and emulated lag-autocorrelation curves.

**Power spectral density** — per-member Welch spectra, scored on the
member-mean curve:

| name | meaning |
| ---- | ------- |
| `psd_log_nmae`    | NMAE between the log10 spectra — weights every frequency band comparably. Usually the one you want. |
| `psd_nmae`        | NMAE between the raw spectra — dominated by whichever frequencies carry the most power |
| `psd_wasserstein` | earth-mover's distance — penalises power sitting at the *wrong* frequency |

Builders, all of which return an `ErrorData`:

| function | produces |
| -------- | -------- |
| `build_error_data_regional(sim, emulator, region_names, metric, ...)` | per-region scores |
| `build_error_data_gridded(sim, emulator, metric, ...)`                | per-gridpoint scores |
| `build_error_data_intervariable_correlation(...)`                     | per-unit error in the correlation *between two indicators* |
| `build_error_data_intervariable_correlation_values(...)`              | the correlation magnitude itself, per data source, rather than the error |
| `build_crps_timeseries_data_gridded(...)`                             | CRPS(t), averaged over gridpoints — a plain dict, not `ErrorData` |
| `build_correlation_data(sim, pred, ...)`                              | across-region correlation matrices (regional only — a gridded one would be 10,000² ) |

Key builder options:

- **`baseline_emulations={name: {indicator: array}}`** — every baseline is
  scored the same way as the emulator and gets its own row in the maps and its
  own line in the timeseries.
- **`ranking_strategy`** — `"emulator"` (default) or a baseline name. Decides
  whose error orders the units, i.e. which ones the ranking plots call "best"
  and "worst".
- **`diff_baseline`** — which baseline `show_difference=True` subtracts.
- **`n_lags` / `window`** (autocorrelation), **`quantiles`** (QQ),
  **`psd_fs` / `psd_nperseg`** (PSD) — pass the same values to the matching
  plotting function so the drawn curves match the scores.

### `plots.py` — every figure

Nothing here computes a score; each function is handed an `ErrorData` and
decides how to lay it out.

**Maps** — one column per indicator, one row per comparison:

| function | notes |
| -------- | ----- |
| `plot_map_regional(error_data, ...)` | AR6 choropleth. `show_difference=True` returns a second figure with the Emulator-minus-baseline map. |
| `plot_map_gridded(error_data, ...)`  | the data's native lat/lon grid. Also renders the CRPS map — build with `metric="crps"`; there is no separate CRPS-map function. |
| `bar_plot_regional(error_data, ...)` | the same data as a grouped bar chart, all comparisons side by side per region |
| `plot_error_matrix_regional(...)`    | **many models at once** — regions across the columns, one row per (model, indicator, experiment), error as colour. See below. |

**Ranking grids** — one row per indicator, one column per selected unit:

| function | what each panel shows |
| -------- | --------------------- |
| `plot_timeseries_regional` / `plot_timeseries_gridded` | Simulation and Emulator ensembles (members + median + 90 % CI) plus a dashed median per baseline |
| `plot_qq_scatter`                    | simulated vs emulated quantiles, with a sim-vs-sim bootstrap band |
| `plot_temporal_correlation_curves`   | autocorrelation against lag |
| `plot_psd_curves`                    | spectral power against period (or frequency), with an inter-member percentile band |

**Aggregates** — the unit axis is already collapsed:

| function | notes |
| -------- | ----- |
| `plot_correlation_comparison` | across-region correlation matrices, sim / emu / difference (regional only) |
| `plot_crps_timeseries_gridded` | spatially-averaged CRPS over time, one line per comparison |

---

## Comparing several models: the error matrix

Every other function evaluates one emulated model at a time.
`plot_error_matrix_regional` takes an `ErrorData` per model and lays them all
out as a single heatmap — regions across the columns, one row per
(model, indicator, experiment), error value as colour.

```python
from emuvaluate.plots import plot_error_matrix_regional

error_data = {
    model: build_error_data_regional(
        sim[model], emu[model], region_list, metric="mae",
        baseline_emulations={
            "Simulations vs Simulations": sim_only[model],
            "Pattern Scaling": ps[model],
        },
    )
    for model in ("ACCESS-ESM1-5", "MPI-ESM1-2-LR", "MIROC6")
}

plot_error_matrix_regional(error_data)
```
The dict keys are free-form labels, so they can name anything that varies
between runs — a model, a scenario, or both (`"ACCESS-ESM1-5 ssp245"`).
"Experiment" is just what an `ErrorData` already carries: the Emulator plus
every baseline you built it with, each getting its own row.

Every entry must cover the same regions and use the same metric, or the columns
won't line up and the colours won't mean the same thing — both are checked and
raise. A combination one model doesn't have (a baseline only some runs include)
is drawn grey rather than dropped, so the grid stays rectangular and the gap is
visible.

Useful options:

| option | effect |
| ------ | ------ |
| `row_order` | any permutation of `("model", "indicator", "comparison")`, outermost first. `("indicator", "comparison", "model")` groups all models of one experiment together instead. |
| `models`, `indicators`, `comparisons`, `regions` | restrict and/or reorder each axis |
| `normalise` | `"indicator"` (default) — one scale and colourbar per indicator, so K and mm/day never share a colour. `"global"` for one scale; `"row"` to scale each row to its own max and compare error *shapes*. |
| `vmin`, `vmax` | scalar, `{indicator: value}`, or `"auto"`. `vmin` defaults to `0` like the maps; set `vmin="auto"` when every row looks the same flat colour and you want the range spent on the differences between rows. |
| `sort_regions` | `"mean"` / `"max"` to put the worst regions first, or a callable. Default keeps the region order stable across figures. |
| `show_values` | write the number in each cell — readable up to about a dozen columns |

---

## Working with any number of indicators

Everything is indicator-generic. The `{indicator: array}` dict you pass in
decides how many columns a map has and how many rows a ranking grid has, in
the order you give them. The default is two indicators named `tas` and `pr`,
which are labelled and given units automatically.

```python
# one indicator
build_error_data_regional({"tas": sim}, {"tas": emu}, regions)

# three, one of them custom-named
build_error_data_regional(
    {"tas": a, "pr": b, "hurs": c},
    {"tas": d, "pr": e, "hurs": f},
    regions,
    indicator_labels={"hurs": "Relative humidity (HURS)"},
    indicator_units={"hurs": "%"},
)

# reorder or subset without rebuilding the dicts
build_error_data_regional(sim, emu, regions, indicators=["pr", "tas"])
```

Any key that isn't `tas` or `pr` is labelled by its uppercased name and gets no
unit string unless you supply one.

---

## Choosing which units the ranking plots show

Every ranking plot takes `selection`, in three forms:

```python
# default — best, median and worst unit by error_data.ranking
plot_timeseries_regional(ed, sim, emu, baselines, years)

# 1-based ranks: the four best regions
plot_psd_curves(ed, sim, emu, nperseg=256, selection=[1, 2, 3, 4])

# named regions, in the order given
plot_timeseries_regional(ed, sim, emu, baselines, years,
                         selection=["SOO", "MED", "WAF"])

# gridded: (lat, lon) pairs, matched to the nearest gridpoint
plot_timeseries_gridded(ed, sim, emu, baselines, years,
                        selection=[(0.0, 0.0), (50.0, 10.0)])
```

With an explicit selection each panel title also reports where that unit falls
in the ranking, so you can still see whether you picked a good one.

## Locking the y-axis across a row

`share_y_per_row=False` is the default — each panel scales to its own data.
Set it to `True` to give every panel in a row the same y-limits, which makes
the columns directly comparable at a glance:

```python
plot_timeseries_regional(ed, sim, emu, baselines, years, share_y_per_row=True)
```

Available on `plot_timeseries_regional`, `plot_timeseries_gridded`,
`plot_qq_scatter`, `plot_temporal_correlation_curves` and `plot_psd_curves`.
`plot_crps_timeseries_gridded` has the equivalent `share_y` (it has no columns
to share across, only rows).

---

## Notebooks

| notebook | what it produces |
| -------- | ---------------- |
| `notebooks/paper-MESH.py` | the gridded evaluations of MESH |
| `notebooks/paper-SCALES.py` | the regional evaluations of SCALES |

They are jupytext-paired (`ipynb,py:percent`), so the `.py` files are the
source of truth and `jupytext --sync notebooks/*.py` regenerates the `.ipynb`.

