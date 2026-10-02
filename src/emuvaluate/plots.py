"""
plots.py
--------
Every figure this package produces. Pure plotting: nothing here preprocesses
an ensemble or computes a score. Each function is handed an `ErrorData` (or
one of the two plain dicts) already built by `metrics.py`, and only decides
how to lay it out.

The figures come in four shapes:

  maps            `plot_map_regional`, `plot_map_gridded`
                  one column per indicator, one row per comparison.

  bars            `bar_plot_regional`
                  the map's data as a grouped bar chart — all comparisons
                  side by side per region.

  matrix          `plot_error_matrix_regional`
                  many models at once: regions across the columns, one row
                  per (model, indicator, comparison), error value as colour.

  ranking grids   `plot_timeseries_regional`, `plot_timeseries_gridded`,
                  `plot_qq_scatter`, `plot_temporal_correlation_curves`,
                  `plot_psd_curves`
                  one row per indicator, one column per selected unit. By
                  default the columns are the best / median / worst unit by
                  `error_data.ranking`; pass `selection=` to choose units
                  yourself. All of them accept `share_y_per_row`.

  aggregates      `plot_correlation_comparison` (regional only),
                  `plot_crps_timeseries_gridded`
                  figures where the unit axis has already been collapsed.

Indicators
----------
Everything is indicator-generic. `error_data.indicators` drives how many
columns a map has and how many rows a ranking grid has, in the order the
`ErrorData` was built with. Two indicators named "tas" and "pr" is the
default, but one indicator, or five, works the same way — see
`metrics.build_error_data_regional`'s `indicators` / `indicator_labels` /
`indicator_units` parameters for naming them.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import regionmask
import xarray as xr

from .metrics import (
    ErrorData,
    RankingResult,
    compute_psd_curves,
    compute_temporal_correlation_curve,
    DEFAULT_QQ_QUANTILES,
    DEFAULT_TEMPORAL_CORR_N_LAGS,
)

__all__ = [
    "plot_map_regional",
    "plot_map_gridded",
    "bar_plot_regional",
    "plot_error_matrix_regional",
    "plot_timeseries_regional",
    "plot_timeseries_gridded",
    "plot_qq_scatter",
    "plot_temporal_correlation_curves",
    "plot_psd_curves",
    "plot_correlation_comparison",
    "plot_crps_timeseries_gridded",
    "resolve_selection",
    "plot_spatial_correlation_curves_gridded",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────

# Colourblind-safe palette (Paul Tol "bright"), shared by every function below.
_PAPER_C = {
    "sim":      "#4477AA",   # blue
    "emulator": "#EE7733",   # orange
    "sim_band": "#BBBBBB",   # light grey, sim-vs-sim envelope
    "diag":     "#000000",   # black diagonal / zero line
    "baseline": ["#009988", "#AA3377", "#CCBB44", "#66CCEE"],
}
_PAPER_ALPHA_MEMBERS = 0.08
_PAPER_ALPHA_CI      = 0.20
_PAPER_ALPHA_BAND    = 0.35
_PAPER_LW_MEDIAN     = 1.6
_PAPER_LW_MEMBER     = 0.4
_PAPER_LW_BASELINE   = 1.4
_PAPER_LW_MAIN       = 1.5
_PAPER_LW_DIAG       = 0.8


def _baseline_color(i: int) -> str:
    return _PAPER_C["baseline"][i % len(_PAPER_C["baseline"])]


def _style_axes(ax, fontsize_ax: int, grid_axis: str | None = "y"):
    """The shared spine/tick/grid treatment used by every line panel."""
    ax.tick_params(labelsize=fontsize_ax, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6)
    if grid_axis:
        ax.grid(axis=grid_axis, linewidth=0.35, alpha=0.45, linestyle=":")


def _indicator_unit(error_data: ErrorData, indicator: str,
                    override: dict[str, str] | None) -> str:
    if override and indicator in override:
        return override[indicator]
    return error_data.indicator_units.get(indicator, "")


def _colorbar_unit(error_data: ErrorData, indicator: str,
                   override: dict[str, str] | None) -> str:
    """Bracketed text on a map colourbar: an explicit unit, else the indicator name."""
    if override and indicator in override:
        return override[indicator]
    return error_data.indicator_labels.get(indicator, indicator).split()[0].lower()


def _add_bottom_colorbars(fig, axes, im_handles, col_labels, cbar_label_fontsize):
    """One horizontal colorbar per column, placed below the bottom row."""
    fig.canvas.draw()
    fig_height = fig.get_size_inches()[1]
    cbar_height_inches, cbar_gap_inches = 0.15, 0.15
    cbar_height_fig = cbar_height_inches / fig_height
    for col_idx, label in enumerate(col_labels):
        ax_ref = axes[-1, col_idx] if axes.ndim == 2 else axes[col_idx]
        pos = ax_ref.get_position()
        gap_fig = cbar_gap_inches / fig_height
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - gap_fig - cbar_height_fig, pos.width, cbar_height_fig])
        cb = fig.colorbar(im_handles[col_idx], cax=cbar_ax, orientation="horizontal")
        cb.set_label(label, fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_label_fontsize - 1)
        cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(top=1 - 0.3 / fig_height)


# ─────────────────────────────────────────────────────────────────────────────
# Unit selection for the ranking grids
# ─────────────────────────────────────────────────────────────────────────────

def resolve_selection(
    error_data: ErrorData,
    ranking: RankingResult,
    selection=None,
) -> list[tuple[int, str]]:
    """
    Turn a `selection` argument into the ordered list of units a ranking
    grid should show, one per column.

    Parameters
    ----------
    error_data : the `ErrorData` being plotted (supplies unit labels)
    ranking    : `error_data.ranking[indicator]` for the row being drawn —
                 selection is resolved per indicator, because rank 1 for one
                 indicator is generally a different unit than rank 1 for
                 another.
    selection  : one of
        None (default)
            The classic three columns: best, median and worst unit by
            `ranking`.
        list of int
            1-based ranks, so ``[1, 2, 3]`` shows the three best units and
            ``[1, 5, 50]`` the 1st, 5th and 50th best. Ranks run 1 (best) to
            ``ranking.n_units`` (worst).
        list of str (regional) / list of str or (lat, lon) pairs (gridded)
            Explicit units, e.g. ``["SOO", "MED", "WAF"]`` or
            ``[(45.0, 10.0), "(45.00°, 10.00°)"]``. Shown in the order given,
            no ranking applied — the column header still reports each unit's
            rank so you can see where it falls.

    Returns
    -------
    list of ``(unit_index, column_header, header_names_the_ranking)``
        The third element is True when the header describes a position in
        the ranking ("Best", "Rank 3") rather than a specific unit, so the
        caller knows whether appending "(by <ranking source>)" makes sense.
    """
    if selection is None:
        return [
            (ranking.best, "Best", True),
            (ranking.median, "Median", True),
            (ranking.worst, "Worst", True),
        ]
    if isinstance(selection, (str, int, np.integer)):
        selection = [selection]
    if len(selection) == 0:
        raise ValueError("selection must contain at least one unit")

    out: list[tuple[int, str, bool]] = []
    for item in selection:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            rank = int(item)
            out.append((ranking.unit_at_rank(rank), f"Rank {rank}", True))
        else:
            idx = error_data.index_of_unit(item)
            out.append((idx, error_data.label_for(idx), False))
    return out


def _apply_share_y(row_axes: list) -> None:
    """Give every axis in a row the same y-limits (the union of theirs)."""
    lims = [ax.get_ylim() for ax in row_axes]
    lo = min(l[0] for l in lims)
    hi = max(l[1] for l in lims)
    for ax in row_axes:
        ax.set_ylim(lo, hi)


def _ranking_grid(
    error_data: ErrorData,
    *,
    draw_cell,
    selection,
    share_y_per_row: bool,
    row_ylabel,
    xlabel: str,
    legend_handles: list,
    panel_width: float,
    panel_height: float,
    fontsize_title: int,
    fontsize_ax: int,
    fontsize_legend: int,
    hspace: float,
    wspace: float,
    show_ranking_source: bool,
    suptitle: str | None,
    save_path: str | None,
    dpi: int,
    per_cell_setup=None,
) -> plt.Figure:
    """
    The shared layout behind every best/median/worst-style figure: one row
    per indicator, one column per selected unit.

    `draw_cell(ax, indicator, unit_idx)` draws a single panel; everything
    else — column headers, per-panel titles with the unit label and score,
    axis labels, the shared legend, `share_y_per_row` and `selection` — is
    handled here so all the ranking plots behave identically.
    """
    indicators = error_data.indicators
    n_rows = len(indicators)
    unit_noun = "gridpoint" if error_data.is_gridded else "region"
    per_row_selection = [
        resolve_selection(error_data, error_data.ranking[ind], selection)
        for ind in indicators
    ]
    n_cols = len(per_row_selection[0])

    fig = plt.figure(figsize=(panel_width * n_cols, panel_height * n_rows))
    gs = mgridspec.GridSpec(
        n_rows, n_cols, figure=fig, hspace=hspace, wspace=wspace,
        left=0.08, right=0.99, top=0.93 if suptitle else 0.96, bottom=0.13,
    )

    for row_idx, indicator in enumerate(indicators):
        ranking = error_data.ranking[indicator]
        short = error_data.short_label(indicator)
        row_axes = []
        for col_idx, (uidx, header, header_is_rank) in enumerate(per_row_selection[row_idx]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)

            draw_cell(ax, indicator, uidx)

            _style_axes(ax, fontsize_ax)
            if per_cell_setup is not None:
                per_cell_setup(ax)

            # With an explicit `selection` the column header no longer says
            # where the unit falls in the ranking, so put its rank in the
            # panel title instead.
            score_txt = f"{error_data.metric_label}={ranking.scores[uidx]:.3f}"
            if not header_is_rank:
                score_txt = f"rank {ranking.rank_of(uidx)} \N{MIDDLE DOT} {score_txt}"
            title = f"{short} \N{MIDDLE DOT} {error_data.label_for(uidx)}  ({score_txt})"
            if row_idx == 0:
                head = header
                if header_is_rank:
                    head = f"{header} {unit_noun}"
                    if show_ranking_source:
                        head += f" (by {ranking.ranking_label})"
                title = f"{head}\n{title}"
            ax.set_title(title, fontsize=fontsize_title, pad=3)

            if col_idx == 0:
                ax.set_ylabel(row_ylabel(indicator), fontsize=fontsize_ax)
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize_ax)

        if share_y_per_row:
            _apply_share_y(row_axes)

    if legend_handles:
        fig.legend(
            handles=legend_handles, ncol=len(legend_handles), loc="lower center",
            bbox_to_anchor=(0.5, -0.01), fontsize=fontsize_legend, frameon=False,
            handlelength=1.6, columnspacing=0.8, handletextpad=0.4,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize_title + 1, y=1.0)
    plt.subplots_adjust(bottom=0.13)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Map plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_map_core(
    error_data: ErrorData,
    draw,
    *,
    show_difference: bool,
    cmap_emulator: str,
    cmap_baseline: str,
    cmap_difference: str,
    projection,
    figsize_per_panel: tuple[float, float],
    cbar_label_fontsize: int,
    units: dict[str, str] | None,
    default_suptitle: str,
    suptitle: str | None,
    save_path: str | None,
    dpi: int,
):
    """Shared body of `plot_map_regional` / `plot_map_gridded`.

    `draw(ax, values, cmap, norm)` renders one panel's per-unit values.
    """
    indicators = error_data.indicators
    comparisons = error_data.comparisons
    n_rows, n_cols = len(comparisons), len(indicators)
    fw, fh = figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fw, fh), squeeze=False,
        subplot_kw={"projection": projection}, gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    )

    row_cmaps = [cmap_emulator] + [cmap_baseline] * (n_rows - 1)
    norms = {i: mcolors.Normalize(vmin=0, vmax=error_data.vmax[i]) for i in indicators}
    im_handles = {}
    for row_idx, label in enumerate(comparisons):
        for col_idx, indicator in enumerate(indicators):
            ax = axes[row_idx, col_idx]
            im = draw(ax, error_data.scores[label][indicator], row_cmaps[row_idx], norms[indicator])
            im_handles[col_idx] = im
            if col_idx == 0:
                ax.text(-0.02, 0.5, label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=9, fontweight="bold", rotation=90)
            if row_idx == 0:
                ax.set_title(error_data.indicator_labels[indicator], fontsize=10, pad=6)

    col_labels = [
        f"{error_data.metric_label}  [{_colorbar_unit(error_data, i, units)}]"
        for i in indicators
    ]
    _add_bottom_colorbars(fig, axes, im_handles, col_labels, cbar_label_fontsize)
    fig.suptitle(suptitle or default_suptitle, fontsize=12, y=1.01)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if not show_difference:
        return fig
    if error_data.diff is None:
        raise ValueError(
            "show_difference=True requires error_data.diff — build error_data "
            "with at least one baseline in baseline_emulations."
        )
    diff_label = f"Emulator \N{MINUS SIGN} {error_data.diff_baseline}"
    diff_norms = {
        i: mcolors.Normalize(vmin=-error_data.diff_vmax[i], vmax=error_data.diff_vmax[i])
        for i in indicators
    }
    diff_fig, diff_axes = plt.subplots(
        1, n_cols, figsize=(fw, figsize_per_panel[1]), squeeze=False,
        subplot_kw={"projection": projection}, gridspec_kw={"hspace": 0.12, "wspace": 0.04},
    )
    diff_axes = diff_axes[0]
    diff_im_handles = {}
    for col_idx, indicator in enumerate(indicators):
        ax = diff_axes[col_idx]
        im = draw(ax, error_data.diff[indicator], cmap_difference, diff_norms[indicator])
        diff_im_handles[col_idx] = im
        if col_idx == 0:
            ax.text(-0.02, 0.5, diff_label, transform=ax.transAxes, ha="right", va="center",
                    fontsize=9, fontweight="bold", rotation=90)
        ax.set_title(error_data.indicator_labels[indicator], fontsize=10, pad=6)
    diff_col_labels = [
        f"\N{GREEK CAPITAL LETTER DELTA} {error_data.metric_label}  "
        f"[{_colorbar_unit(error_data, i, units)}]"
        for i in indicators
    ]
    _add_bottom_colorbars(diff_fig, diff_axes, diff_im_handles, diff_col_labels, cbar_label_fontsize)
    if save_path:
        base, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
        diff_fig.savefig(f"{base}_difference.{ext}", dpi=dpi, bbox_inches="tight")
    return fig, diff_fig


def plot_map_regional(
    error_data: ErrorData,
    *,
    show_difference: bool = False,
    cmap_emulator: str = "YlOrRd",
    cmap_baseline: str = "YlOrRd",
    cmap_difference: str = "RdBu_r",
    projection=None,
    figsize_per_panel: tuple[float, float] = (7, 3.2),
    cbar_label_fontsize: int = 9,
    units: dict[str, str] | None = None,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure | tuple[plt.Figure, plt.Figure]:
    """
    Per-AR6-region scores plotted as a choropleth map: one column per
    indicator, one row per comparison (Emulator, then each baseline).

    Works with any `ErrorData`, whatever metric family it was built with —
    the scores are already computed; this function only lays them out.

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` or
        `metrics.build_error_data_intervariable_correlation*`. Its
        `indicators` list decides how many columns are drawn and their
        titles come from its `indicator_labels`.
    show_difference : if True, additionally build and return a second figure
        with the (Emulator − `error_data.diff_baseline`) difference maps.
        Requires `error_data.diff`, i.e. at least one baseline was supplied
        when building `error_data`.
    cmap_emulator, cmap_baseline, cmap_difference : colormaps
    projection : Cartopy CRS (default Robinson)
    figsize_per_panel, cbar_label_fontsize : figure geometry
    units : optional ``{"tas": "K", ...}`` to show physical units in the
        colorbar label instead of the indicator name (useful e.g. for "crps").
    suptitle, save_path, dpi : as usual

    Returns
    -------
    matplotlib Figure, or (Figure, Figure) if show_difference=True
    """
    ar6 = regionmask.defined_regions.ar6.all
    expected = sorted(ar6.abbrevs)
    region_names = error_data.unit_labels
    if region_names is None or list(region_names) != expected:
        raise ValueError(
            "error_data.unit_labels must equal sorted(ar6.abbrevs) to plot a "
            f"regional map.\nGot:      {region_names}\nExpected: {expected}"
        )
    abbrev_to_idx = {a: i for i, a in enumerate(region_names)}
    abbrev_to_number = {a: ar6[a].number for a in expected}

    ds = xr.Dataset(coords={
        "lon": np.linspace(-179.5, 179.5, 720),
        "lat": np.linspace(-89.5, 89.5, 360),
    })
    mask = ar6.mask(ds)

    def draw(ax, scores, cmap, norm):
        region_values = xr.full_like(mask, np.nan, dtype=float)
        for abbr in expected:
            region_values = region_values.where(
                mask != abbrev_to_number[abbr], float(scores[abbrev_to_idx[abbr]])
            )
        im = ax.pcolormesh(
            ds.lon, ds.lat, region_values,
            transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto",
        )
        ar6.plot(
            ax=ax, add_label=False, add_coastlines=True, add_ocean=False, add_land=False,
            line_kws={"color": "0.3", "linewidth": 0.4},
        )
        ax.set_global()
        return im

    return _plot_map_core(
        error_data, draw,
        show_difference=show_difference, cmap_emulator=cmap_emulator,
        cmap_baseline=cmap_baseline, cmap_difference=cmap_difference,
        projection=ccrs.Robinson() if projection is None else projection,
        figsize_per_panel=figsize_per_panel, cbar_label_fontsize=cbar_label_fontsize,
        units=units, default_suptitle=f"Per-Region {error_data.metric_label}",
        suptitle=suptitle, save_path=save_path, dpi=dpi,
    )


def plot_map_gridded(
    error_data: ErrorData,
    *,
    show_difference: bool = False,
    cmap_emulator: str = "YlOrRd",
    cmap_baseline: str = "YlOrRd",
    cmap_difference: str = "RdBu_r",
    projection=None,
    figsize_per_panel: tuple[float, float] = (7, 3.2),
    cbar_label_fontsize: int = 9,
    units: dict[str, str] | None = None,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure | tuple[plt.Figure, plt.Figure]:
    """
    Gridded analogue of `plot_map_regional`: per-gridpoint scores drawn on
    the data's native (lat, lon) grid — no regionmask / AR6 choropleth. Also
    renders the CRPS map (build `error_data` with ``metric="crps"``).

    Parameters
    ----------
    error_data : from `metrics.build_error_data_gridded` (or
        `build_error_data_intervariable_correlation*` with `lat`/`lon` set)
    (all other parameters: see `plot_map_regional`)

    Returns
    -------
    matplotlib Figure, or (Figure, Figure) if show_difference=True
    """
    if error_data.lat is None:
        raise ValueError(
            "error_data.lat/lon must be set to plot a gridded map "
            "(build it with build_error_data_gridded)."
        )
    ref_lat, ref_lon = error_data.lat, error_data.lon
    n_lat, n_lon = error_data.n_lat, error_data.n_lon

    def draw(ax, scores, cmap, norm):
        grid_vals = np.asarray(scores).reshape(n_lat, n_lon)
        data_cyclic, lon_cyclic = add_cyclic_point(grid_vals, coord=ref_lon)
        im = ax.pcolormesh(
            lon_cyclic, ref_lat, data_cyclic,
            transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto",
        )
        ax.coastlines(color="0.3", linewidth=0.4)
        ax.set_global()
        return im

    return _plot_map_core(
        error_data, draw,
        show_difference=show_difference, cmap_emulator=cmap_emulator,
        cmap_baseline=cmap_baseline, cmap_difference=cmap_difference,
        projection=ccrs.Robinson() if projection is None else projection,
        figsize_per_panel=figsize_per_panel, cbar_label_fontsize=cbar_label_fontsize,
        units=units, default_suptitle=f"Per-Gridpoint {error_data.metric_label}",
        suptitle=suptitle, save_path=save_path, dpi=dpi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart (regional only — one bar per AR6 region)
# ─────────────────────────────────────────────────────────────────────────────

def bar_plot_regional(
    error_data: ErrorData,
    *,
    show_difference: bool = False,
    cmap_emulator: str = "YlOrRd",
    cmap_baseline: str = "YlOrRd",
    cmap_difference: str = "RdBu_r",
    figsize_per_panel: tuple[float, float] = (7, 3.2),
    label_fontsize: int = 9,
    units: dict[str, str] | None = None,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Bar-chart equivalent of `plot_map_regional`: same `ErrorData`, one panel
    per indicator, but instead of one row per comparison, *all* comparisons
    (plus the Emulator-minus-baseline difference, if requested) are drawn as
    a grouped bar chart — for each AR6 region the bars sit in a tight
    cluster, and clusters are spaced along the x-axis.

    Panels whose scores are identical across indicators are collapsed to a
    single panel, so the intervariable-correlation `ErrorData` (which has
    one score per *pair*, not per indicator) renders as one chart rather
    than two identical ones.

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` or
        `build_error_data_intervariable_correlation*`
    show_difference : add one extra bar per region group with
        (Emulator − `error_data.diff_baseline`). Requires `error_data.diff`.
    cmap_emulator, cmap_baseline, cmap_difference : used to pick one flat
        colour per bar *series* (not per bar value — the bar height already
        encodes the value; colour only distinguishes the series).
    figsize_per_panel : (width, height) of a single indicator panel. With 58
        AR6 regions the default width is usually too narrow for readable
        labels — consider e.g. ``(16, 4)``.
    label_fontsize : sizes axis, tick and legend text.
    units : optional ``{"tas": "K", ...}`` shown in the y-axis label.
    suptitle, save_path, dpi : as usual

    Returns
    -------
    matplotlib Figure (a single figure — every comparison and, if requested,
    the difference are all bars within it)
    """
    ar6 = regionmask.defined_regions.ar6.all
    expected = sorted(ar6.abbrevs)
    region_names = error_data.unit_labels
    if region_names is None or list(region_names) != expected:
        raise ValueError(
            "error_data.unit_labels must equal sorted(ar6.abbrevs) to plot a "
            f"regional bar chart.\nGot:      {region_names}\nExpected: {expected}"
        )
    abbrev_to_idx = {a: i for i, a in enumerate(region_names)}
    x = np.arange(len(expected))

    if show_difference and error_data.diff is None:
        raise ValueError(
            "show_difference=True requires error_data.diff — build error_data "
            "with at least one baseline in baseline_emulations."
        )

    comparisons = list(error_data.comparisons)
    n_baselines = len(comparisons) - 1

    series_labels = list(comparisons)
    series_colors = [plt.get_cmap(cmap_emulator)(0.95)]
    if n_baselines == 1:
        series_colors.append(plt.get_cmap(cmap_baseline)(0.35))
    elif n_baselines > 1:
        for frac in np.linspace(0.3, 0.6, n_baselines):
            series_colors.append(plt.get_cmap(cmap_baseline)(frac))
    diff_label = None
    if show_difference:
        diff_label = f"Emulator \N{MINUS SIGN} {error_data.diff_baseline}"
        series_labels.append(diff_label)
        series_colors.append(plt.get_cmap(cmap_difference)(0.15))

    n_series = len(series_labels)
    bar_width = 0.75 / n_series

    # Collapse indicators that carry identical scores (e.g. an
    # intervariable-correlation ErrorData) into a single panel.
    indicators = list(error_data.indicators)
    kept: list[str] = []
    for ind in indicators:
        dup = any(
            all(np.allclose(error_data.scores[c][ind], error_data.scores[c][k])
                for c in comparisons)
            for k in kept
        )
        if not dup:
            kept.append(ind)
    show_titles = len(kept) > 1

    n_cols = len(kept)
    fw, fh = figsize_per_panel[0] * n_cols, figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_cols, figsize=(fw, fh), squeeze=False,
                             gridspec_kw={"wspace": 0.18})
    axes = axes[0]

    tick_fontsize = max(label_fontsize - 4, 4)
    legend_handles = []

    for col_idx, indicator in enumerate(kept):
        ax = axes[col_idx]
        for k, label in enumerate(series_labels):
            source = error_data.diff if label == diff_label else error_data.scores[label]
            vals = np.array([source[indicator][abbrev_to_idx[a]] for a in expected], dtype=float)
            offset = (k - n_series / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, width=bar_width * 0.95,
                          color=series_colors[k], alpha=0.85, label=label)
            if col_idx == 0:
                legend_handles.append(bars)

        if show_difference:
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.6, len(expected) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(expected, rotation=90, fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5, linestyle="--")
        if show_titles:
            ax.set_title(error_data.indicator_labels[indicator], fontsize=10, pad=6)
        unit_str = _indicator_unit(error_data, indicator, units)
        ax.set_ylabel(
            f"{error_data.metric_label}  [{unit_str}]" if unit_str else error_data.metric_label,
            fontsize=label_fontsize,
        )

    fig.legend(legend_handles, series_labels, loc="upper center", ncol=n_series,
               frameon=False, fontsize=label_fontsize, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(suptitle or f"Per-Region {error_data.metric_label}", fontsize=12, y=1.14)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Error matrix — many models at once, regions as columns
# ─────────────────────────────────────────────────────────────────────────────

_MATRIX_ROW_KEYS = ("model", "indicator", "comparison")


def _resolve_matrix_inputs(error_data, models, indicators, comparisons):
    """Normalise the input to a {model: ErrorData} dict and resolve the three row axes."""
    if isinstance(error_data, ErrorData):
        error_data = {"": error_data}
    if not isinstance(error_data, dict) or not error_data:
        raise TypeError(
            "error_data must be an ErrorData or a non-empty {model_name: ErrorData} dict"
        )
    for name, ed in error_data.items():
        if not isinstance(ed, ErrorData):
            raise TypeError(f"error_data['{name}'] is {type(ed).__name__}, not an ErrorData")
        if ed.unit_labels is None:
            raise ValueError(
                f"error_data['{name}'] has no unit_labels — plot_error_matrix_regional "
                "needs regional ErrorData (a gridded one would have thousands of columns)."
            )

    model_names = list(error_data) if models is None else list(models)
    unknown = [m for m in model_names if m not in error_data]
    if unknown:
        raise ValueError(f"models names unknown key(s) {unknown}; available: {list(error_data)}")

    # Every ErrorData must describe the same regions, or the columns don't line up.
    ref_name = model_names[0]
    ref_units = list(error_data[ref_name].unit_labels)
    for name in model_names[1:]:
        if list(error_data[name].unit_labels) != ref_units:
            raise ValueError(
                f"error_data['{name}'] has different unit_labels than error_data['{ref_name}'] — "
                "every ErrorData must cover the same regions, in the same order."
            )

    metrics = {error_data[m].metric_label for m in model_names}
    if len(metrics) > 1:
        raise ValueError(
            f"every ErrorData must use the same metric; got {sorted(metrics)}. "
            "Build one matrix per metric."
        )

    def _union(attr):
        seen = []
        for m in model_names:
            for v in getattr(error_data[m], attr):
                if v not in seen:
                    seen.append(v)
        return seen

    ind_names = _union("indicators") if indicators is None else list(indicators)
    cmp_names = _union("comparisons") if comparisons is None else list(comparisons)
    return error_data, model_names, ind_names, cmp_names, ref_units


def plot_error_matrix_regional(
    error_data,
    *,
    row_order: tuple[str, str, str] = ("model", "indicator", "comparison"),
    models: list[str] | None = None,
    indicators: list[str] | None = None,
    comparisons: list[str] | None = None,
    regions: list[str] | None = None,
    sort_regions=None,
    normalise: str = "indicator",
    cmap: str = "YlOrRd",
    vmin=0.0,
    vmax=None,
    missing_color: str = "0.9",
    show_values: bool = False,
    value_fmt: str = "{:.2f}",
    group_separators: bool = True,
    figsize: tuple[float, float] | None = None,
    fontsize_tick: int = 6,
    fontsize_label: int = 9,
    cbar_label_fontsize: int = 9,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Every model, indicator and experiment in one matrix: regions across the
    columns, one row per (model, indicator, comparison) combination, and the
    error value as colour.

    This is the many-models counterpart to `plot_map_regional`. A map shows
    one model at a time and spends its space on geography; this shows an
    arbitrary number of models, indicators and comparisons side by side and
    spends its space on the comparison itself, at the cost of the spatial
    layout. Reach for it when the question is "which model/experiment is
    worst, and in which regions?" rather than "where on Earth is the error?".

    "Comparison" is what an `ErrorData` already carries — Emulator plus every
    baseline you built it with, so `Simulations vs Simulations` and
    `Pattern Scaling` become their own rows automatically.

    Parameters
    ----------
    error_data : ``{model_name: ErrorData}``, or a single `ErrorData`
        One entry per model. Every entry must cover the same regions in the
        same order and use the same metric — otherwise the columns don't line
        up or the colours don't mean the same thing, and this raises. The
        keys are free-form labels, so they can name anything that varies
        between the runs: a model, a scenario, or both
        (``"ACCESS-ESM1-5 ssp245"``). A bare `ErrorData` is accepted for the
        single-model case and gets an unlabelled model axis.

        Build the dict by looping over whatever you are comparing::

            error_data = {
                model: build_error_data_regional(
                    sim[model], emu[model], region_list, metric="nmae",
                    baseline_emulations={"Simulations vs Simulations": sim_only[model]},
                )
                for model in ("ACCESS-ESM1-5", "MPI-ESM1-2-LR", "MIROC6")
            }
            plot_error_matrix_regional(error_data)

    row_order : the three row axes, outermost first. Any permutation of
        ``("model", "indicator", "comparison")``. Rows are the full product
        in that order, so ``("indicator", "comparison", "model")`` groups all
        models of one experiment together instead of all experiments of one
        model.
    models, indicators, comparisons : restrict and/or reorder each axis.
        Default is everything, in first-seen order across the input. A
        combination one model doesn't have (a baseline only some runs
        include) is drawn in `missing_color` rather than dropped, so the grid
        stays rectangular and the gap is visible.
    regions : restrict and/or reorder the columns, e.g. ``["SOO", "MED", "WAF"]``.
        Default is every region, in `error_data`'s own order.
    sort_regions : ``None`` (default, keep `regions` order), ``"mean"`` /
        ``"max"`` to order columns by the mean / max error across all rows
        (worst first), or a callable ``(n_rows, n_cols) array -> column
        order``. Sorting makes the worst regions cluster together but moves a
        region's column between figures, so cross-referencing two figures
        needs the default.
    normalise : how colours map to values.
        ``"indicator"`` (default)
            One shared scale per indicator, across every model and
            comparison, with one colourbar each. The right choice whenever
            indicators carry different units — an MAE of 2 K and an MAE of
            2 mm/day should not be the same colour.
        ``"global"``
            A single scale and colourbar for the whole matrix. Only
            meaningful when every indicator shares units, or for a unitless
            metric such as NMAE.
        ``"row"``
            Each row scaled to its own maximum, so every row spans the full
            colour range. Shows the *shape* of each error pattern across
            regions; absolute magnitudes become unreadable.
    cmap : colormap for the error values.
    vmin, vmax : the ends of the colour scale. Each takes a scalar,
        ``{indicator: value}`` when ``normalise="indicator"``, or the string
        ``"auto"`` for the data's own minimum / maximum. `vmin` defaults to
        ``0`` — the same zero-anchored scale the maps use, which keeps
        "how big is this error" readable. Set ``vmin="auto"`` when every row
        has a similar error and the whole matrix comes out one flat colour:
        it spends the full colour range on the differences *between* rows,
        at the cost of no longer showing magnitude against zero. Passing
        explicit numbers holds the scale fixed across several figures.
        Both are ignored when ``normalise="row"``.
    missing_color : colour for cells with no value (a missing combination, or
        a NaN score).
    show_values : write the number in each cell. Readable up to roughly a
        dozen columns; leave off for all 58 regions.
    value_fmt : format string used by `show_values`.
    group_separators : draw a line between blocks of the outermost row axis,
        and a lighter one between blocks of the middle axis.
    figsize : defaults to a size derived from the row and column counts.
    fontsize_tick, fontsize_label, cbar_label_fontsize : text sizes.
    suptitle, save_path, dpi : as usual.

    Returns
    -------
    matplotlib Figure
    """
    if tuple(sorted(row_order)) != tuple(sorted(_MATRIX_ROW_KEYS)):
        raise ValueError(
            f"row_order must be a permutation of {_MATRIX_ROW_KEYS}, got {row_order}"
        )
    if normalise not in ("indicator", "global", "row"):
        raise ValueError("normalise must be 'indicator', 'global' or 'row'")

    data, model_names, ind_names, cmp_names, all_regions = _resolve_matrix_inputs(
        error_data, models, indicators, comparisons
    )

    # ── columns ─────────────────────────────────────────────────────────────
    col_names = list(all_regions) if regions is None else list(regions)
    unknown = [r for r in col_names if r not in all_regions]
    if unknown:
        raise ValueError(f"regions names unknown region(s) {unknown}")
    col_idx = [all_regions.index(r) for r in col_names]

    # ── rows: the full product of the three axes, outermost first ───────────
    axis_values = {"model": model_names, "indicator": ind_names, "comparison": cmp_names}
    rows = [{}]
    for key in row_order:
        rows = [{**r, key: v} for r in rows for v in axis_values[key]]

    n_rows, n_cols = len(rows), len(col_names)
    values = np.full((n_rows, n_cols), np.nan)
    for r, row in enumerate(rows):
        ed = data[row["model"]]
        scores = ed.scores.get(row["comparison"], {}).get(row["indicator"])
        if scores is not None:
            values[r] = np.asarray(scores, dtype=float)[col_idx]

    if np.all(np.isnan(values)):
        raise ValueError(
            "no scores matched — check that `indicators` and `comparisons` name "
            "things the ErrorData objects actually contain."
        )

    # ── optional column reordering ──────────────────────────────────────────
    if sort_regions is not None:
        if callable(sort_regions):
            order = np.asarray(sort_regions(values), dtype=int)
        elif sort_regions in ("mean", "max"):
            agg = np.nanmean if sort_regions == "mean" else np.nanmax
            with np.errstate(invalid="ignore"):
                stat = agg(np.where(np.isnan(values), np.nan, values), axis=0)
            order = np.argsort(np.where(np.isnan(stat), -np.inf, stat))[::-1]
        else:
            raise ValueError("sort_regions must be None, 'mean', 'max', or a callable")
        values = values[:, order]
        col_names = [col_names[i] for i in order]

    # ── colour normalisation ────────────────────────────────────────────────
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(missing_color)

    def _limit(spec, key, auto_value, default):
        """Resolve one end of a colour scale from a scalar / dict / 'auto'."""
        if isinstance(spec, dict):
            spec = spec.get(key, default)
        if spec is None:
            spec = default
        if isinstance(spec, str):
            if spec != "auto":
                raise ValueError(f"vmin/vmax strings must be 'auto', got {spec!r}")
            return auto_value
        return float(spec)

    def _norm_for(key, subset) -> mcolors.Normalize:
        with np.errstate(invalid="ignore"):
            lo_auto = float(np.nanmin(subset)) if subset.size else 0.0
            hi_auto = float(np.nanmax(subset)) if subset.size else 1.0
        if not np.isfinite(lo_auto):
            lo_auto = 0.0
        if not np.isfinite(hi_auto):
            hi_auto = 1.0
        lo = _limit(vmin, key, lo_auto, 0.0)
        hi = _limit(vmax, key, hi_auto, hi_auto)
        if hi <= lo:  # degenerate (all cells equal, or a bad override)
            hi = lo + max(abs(lo) * 1e-6, 1e-9)
        return mcolors.Normalize(lo, hi)

    row_indicators = [row["indicator"] for row in rows]
    if normalise == "row":
        with np.errstate(invalid="ignore"):
            row_max = np.nanmax(np.abs(values), axis=1)
        row_max = np.where(~np.isfinite(row_max) | (row_max == 0), 1.0, row_max)
        scaled = values / row_max[:, None]
        cbar_specs = [("fraction of row maximum", mcolors.Normalize(0, 1))]
    else:
        if normalise == "global":
            norms = {None: _norm_for(None, values)}
            key_of_row = [None] * n_rows
            cbar_specs = [(_matrix_metric_label(data, model_names), norms[None])]
        else:
            norms = {}
            for ind in ind_names:
                mask = np.array([ri == ind for ri in row_indicators])
                norms[ind] = _norm_for(ind, values[mask] if mask.any() else values[:0])
            key_of_row = row_indicators
            ref_ed = data[model_names[0]]
            cbar_specs = [
                (f"{_matrix_metric_label(data, model_names)} "
                 f"[{_colorbar_unit(ref_ed, ind, None)}]", norms[ind])
                for ind in ind_names if ind in norms
            ]
        scaled = np.empty_like(values)
        for r in range(n_rows):
            scaled[r] = norms[key_of_row[r]](values[r])

    rgba = cmap_obj(np.ma.masked_invalid(scaled))

    # ── figure ──────────────────────────────────────────────────────────────
    row_labels = [
        " \N{MIDDLE DOT} ".join(
            _matrix_row_part(data, row, key) for key in row_order
            if _matrix_row_part(data, row, key)
        )
        for row in rows
    ]
    label_width_in = 0.06 * fontsize_tick * max((len(s) for s in row_labels), default=1)
    if figsize is None:
        figsize = (
            max(6.0, label_width_in + 0.19 * n_cols),
            max(2.2, 0.9 + 0.24 * n_rows),
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_names, rotation=90, fontsize=fontsize_tick)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=fontsize_tick)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.6")

    if group_separators and n_rows > 1:
        outer_len = len(axis_values[row_order[0]])
        middle_len = len(axis_values[row_order[1]])
        block = n_rows // max(outer_len, 1)
        sub = block // max(middle_len, 1)
        for r in range(1, n_rows):
            if block and r % block == 0:
                ax.axhline(r - 0.5, color="0.1", linewidth=1.2, zorder=5)
            elif sub and r % sub == 0:
                ax.axhline(r - 0.5, color="0.45", linewidth=0.7, zorder=4)

    if show_values:
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isnan(values[r, c]):
                    continue
                lum = 0.299 * rgba[r, c, 0] + 0.587 * rgba[r, c, 1] + 0.114 * rgba[r, c, 2]
                ax.text(c, r, value_fmt.format(values[r, c]), ha="center", va="center",
                        fontsize=max(fontsize_tick - 1, 3),
                        color="white" if lum < 0.5 else "0.1")

    ax.set_xlabel("Region", fontsize=fontsize_label, labelpad=4)

    # ── colourbars, one per scale, in a row underneath ──────────────────────
    fig.canvas.draw()
    pos = ax.get_position()
    fig_h = fig.get_size_inches()[1]
    cbar_h = 0.13 / fig_h
    gap = 0.75 / fig_h
    n_cb = len(cbar_specs)
    span = pos.width / n_cb
    for i, (label, norm) in enumerate(cbar_specs):
        cax = fig.add_axes([pos.x0 + i * span, pos.y0 - gap - cbar_h, span * 0.82, cbar_h])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=cax, orientation="horizontal"
        )
        cb.set_label(label, fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_label_fontsize - 1)
        cb.outline.set_linewidth(0.5)

    fig.suptitle(
        suptitle or f"Per-Region {_matrix_metric_label(data, model_names)}",
        fontsize=12, y=pos.y1 + 0.6 / fig_h,
    )
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _matrix_metric_label(data: dict, model_names: list[str]) -> str:
    return data[model_names[0]].metric_label


def _matrix_row_part(data: dict, row: dict, key: str) -> str:
    """The text for one level of a matrix row label."""
    if key == "model":
        return row["model"]
    if key == "comparison":
        return row["comparison"]
    ed = data[row["model"]]
    return ed.short_label(row["indicator"])


# ─────────────────────────────────────────────────────────────────────────────
# Timeseries ranking grid
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """arr : (n_members, T) -> median, p5, p95, each (T,)."""
    return np.median(arr, axis=0), np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0)


def _plot_ensemble_band(ax, years, arr, color, label, zorder=2):
    """Draw individual members (faint) + median + 90% CI shading."""
    med, p5, p95 = _ensemble_stats(arr)
    for member in arr:
        ax.plot(years, member, color=color, lw=_PAPER_LW_MEMBER,
                alpha=_PAPER_ALPHA_MEMBERS, zorder=zorder)
    ax.fill_between(years, p5, p95, color=color, alpha=_PAPER_ALPHA_CI,
                    zorder=zorder + 1, linewidth=0)
    ax.plot(years, med, color=color, lw=_PAPER_LW_MEDIAN, label=label, zorder=zorder + 2)


def _timeseries_legend(baselines) -> list:
    handles = [
        mlines.Line2D([], [], color=_PAPER_C["sim"], lw=_PAPER_LW_MEDIAN,
                      label="Simulation (median + 90 % CI)"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MEDIAN,
                      label="Emulator (median + 90 % CI)"),
    ]
    for b_idx, bname in enumerate(baselines):
        handles.append(mlines.Line2D([], [], color=_baseline_color(b_idx),
                                     lw=_PAPER_LW_BASELINE, linestyle="--", label=bname))
    handles += [
        mpatches.Patch(color=_PAPER_C["sim"], alpha=_PAPER_ALPHA_CI + 0.15, label="Simulation 90 % CI"),
        mpatches.Patch(color=_PAPER_C["emulator"], alpha=_PAPER_ALPHA_CI + 0.15, label="Emulator 90 % CI"),
    ]
    return handles


def _plot_timeseries_core(
    error_data, sim, emulator, baseline_emulations, years,
    selection, share_y_per_row, indicator_units, panel_width, panel_height,
    fontsize_title, fontsize_ax, fontsize_legend, suptitle, save_path, dpi,
) -> plt.Figure:
    n_time = np.asarray(sim[error_data.indicators[0]]).shape[1]
    years = np.arange(n_time) if years is None else np.asarray(years)
    baselines = baseline_emulations or {}

    def draw_cell(ax, indicator, uidx):
        _plot_ensemble_band(ax, years, np.asarray(sim[indicator])[:, :, uidx],
                            color=_PAPER_C["sim"], label="Simulation", zorder=2)
        _plot_ensemble_band(ax, years, np.asarray(emulator[indicator])[:, :, uidx],
                            color=_PAPER_C["emulator"], label="Emulator", zorder=3)
        for b_idx, (bname, barrs) in enumerate(baselines.items()):
            b_data = np.asarray(barrs[indicator])[:, :, uidx]
            ax.plot(years, np.median(b_data, axis=0), color=_baseline_color(b_idx),
                    lw=_PAPER_LW_BASELINE, linestyle="--", label=bname, zorder=4)

    def row_ylabel(indicator):
        unit = _indicator_unit(error_data, indicator, indicator_units)
        short = error_data.short_label(indicator)
        return f"{short} [{unit}]" if unit else short

    return _ranking_grid(
        error_data, draw_cell=draw_cell, selection=selection,
        share_y_per_row=share_y_per_row,
        row_ylabel=row_ylabel, xlabel="Year",
        legend_handles=_timeseries_legend(baselines),
        panel_width=panel_width, panel_height=panel_height,
        fontsize_title=fontsize_title, fontsize_ax=fontsize_ax,
        fontsize_legend=fontsize_legend, hspace=0.42, wspace=0.32,
        show_ranking_source=True, suptitle=suptitle, save_path=save_path, dpi=dpi,
        per_cell_setup=lambda ax: ax.xaxis.set_major_locator(
            plt.MaxNLocator(nbins=4, integer=True, prune="both")
        ),
    )


def plot_timeseries_regional(
    error_data: ErrorData,
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    years: np.ndarray | None = None,
    *,
    selection=None,
    share_y_per_row: bool = False,
    indicator_units: dict[str, str] | None = None,
    panel_width: float = 3.4,
    panel_height: float = 2.2,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Timeseries of selected AR6 regions — by default the best, median and
    worst region per `error_data.ranking` — one row per indicator.

    Each panel shows the Simulation and Emulator ensembles (faint members +
    median + 90 % CI) plus one dashed median line per baseline.

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` — supplies the
        ranking, the per-unit scores shown in each panel title, the metric
        label, and the indicator list/labels/units.
    sim, emulator : ``{indicator: (n_members, T, n_regions)}`` — the same
        preprocessed arrays `error_data` was built from.
    baseline_emulations : ``{name: {indicator: array}}`` — same baselines and
        preprocessing as passed to `build_error_data_regional`.
    years : x-axis values, length T. Defaults to 0..T-1.
    selection : which regions to show, one per column. ``None`` (default)
        gives best / median / worst. A list of 1-based ranks, e.g.
        ``[1, 2, 3]``, shows the three best. A list of region abbreviations,
        e.g. ``["SOO", "MED"]``, shows exactly those. See `resolve_selection`.
    share_y_per_row : if True, every panel in a row (i.e. every region shown
        for one indicator) gets the same y-limits, so columns are directly
        comparable. Default False — each panel scales to its own data.
    indicator_units : override the unit strings in the y-axis labels.
    (remaining parameters control figure geometry/cosmetics)

    Returns
    -------
    matplotlib Figure
    """
    return _plot_timeseries_core(
        error_data, sim, emulator, baseline_emulations, years,
        selection, share_y_per_row, indicator_units, panel_width, panel_height,
        fontsize_title, fontsize_ax, fontsize_legend, suptitle, save_path, dpi,
    )


def plot_timeseries_gridded(
    error_data: ErrorData,
    sim: dict,
    emulator: dict,
    baseline_emulations: dict | None = None,
    years: np.ndarray | None = None,
    *,
    selection=None,
    share_y_per_row: bool = False,
    indicator_units: dict[str, str] | None = None,
    panel_width: float = 3.4,
    panel_height: float = 2.2,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Gridded analogue of `plot_timeseries_regional`: timeseries of selected
    gridpoints, one row per indicator.

    Parameters
    ----------
    error_data : from `metrics.build_error_data_gridded`
    sim, emulator : ``{indicator: preprocessing.GriddedArray}`` — the same
        preprocessed arrays `error_data` was built from.
    baseline_emulations : ``{name: {indicator: GriddedArray}}``
    selection : ``None`` for best / median / worst, a list of 1-based ranks,
        or a list of gridpoints given as ``(lat, lon)`` pairs or
        ``"(45.00°, 10.00°)"``-style labels. See `resolve_selection`.
    (remaining parameters: see `plot_timeseries_regional`)

    Returns
    -------
    matplotlib Figure
    """
    def _values(d):
        return {k: v.values for k, v in d.items()}

    baselines_values = None
    if baseline_emulations:
        baselines_values = {n: _values(a) for n, a in baseline_emulations.items()}
    return _plot_timeseries_core(
        error_data, _values(sim), _values(emulator), baselines_values, years,
        selection, share_y_per_row, indicator_units, panel_width, panel_height,
        fontsize_title, fontsize_ax, fontsize_legend, suptitle, save_path, dpi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QQ scatter — one function for both regional and gridded data
# ─────────────────────────────────────────────────────────────────────────────

def _sim_vs_sim_band(sim: np.ndarray, j: int, quantiles: np.ndarray,
                     n_boot: int, rng: np.random.Generator):
    """
    Draw `n_boot` random half-splits of the simulation ensemble at unit j,
    compute QQ curves of split-A quantiles vs split-B quantiles, and return
    (sim_quantiles, band_low_p5, band_high_p95). x-axis reference = quantiles
    of the full sim ensemble.
    """
    n = sim.shape[0]
    half = max(n // 2, 1)
    curves = []
    for _ in range(n_boot):
        idx_a = rng.choice(n, size=half, replace=False)
        idx_b = np.setdiff1d(np.arange(n), idx_a)
        if len(idx_b) == 0:
            idx_b = idx_a
        qa = np.quantile(sim[idx_a, :, j].ravel(), quantiles)
        qb = np.quantile(sim[idx_b, :, j].ravel(), quantiles)
        curves.append(qb - qa)
    curves = np.array(curves)
    q_ref = np.quantile(sim[:, :, j].ravel(), quantiles)
    return q_ref, q_ref + np.percentile(curves, 5, axis=0), q_ref + np.percentile(curves, 95, axis=0)


def plot_qq_scatter(
    error_data: ErrorData,
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    selection=None,
    share_y_per_row: bool = False,
    quantiles: np.ndarray = DEFAULT_QQ_QUANTILES,
    n_boot: int = 200,
    random_seed: int = 0,
    indicator_units: dict[str, str] | None = None,
    equal_aspect: bool = True,
    panel_width: float = 3.4,
    panel_height: float = 2.4,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    QQ scatter of selected units, one row per indicator. One function for
    both regional and gridded data — pass plain
    ``{indicator: (n_members, T, n_units)}`` arrays either way (for gridded,
    that's ``mesh_array.values``).

    Each panel:
      x-axis    : simulation quantiles (full ensemble, flattened)
      y-axis    : emulator / baseline quantiles
      grey band : 5th–95th percentile of `n_boot` sim-vs-sim QQ residuals
      diagonal  : perfect-emulation reference

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` /
        `build_error_data_gridded`, built with a quantile-curve metric
        ("qq_mae", "qq_nmae", "qq_ks", "qq_tail_mae") so the ranking reflects
        QQ error rather than a raw-timeseries error.
    sim, emulator : ``{indicator: (n_members, T, n_units)}`` — the same
        preprocessed arrays `error_data` was built from, usually with the
        ensemble mean removed (`preprocess_*(..., remove_ensemble_mean=True)`).
    baseline_emulations : ``{name: {indicator: array}}``
    selection : which units to show — see `resolve_selection`.
    share_y_per_row : give every panel in a row the same y-limits (default
        False). Note this only equalises the *y* axis; with
        ``equal_aspect=True`` the x range follows, keeping panels square.
    quantiles : quantile levels plotted (match what `error_data` was built with)
    n_boot, random_seed : sim-vs-sim bootstrap settings for the grey band
    equal_aspect : keep a 1:1 data aspect so the diagonal is at 45°
    (remaining parameters control figure geometry/cosmetics)

    Returns
    -------
    matplotlib Figure
    """
    rng = np.random.default_rng(random_seed)
    baselines = baseline_emulations or {}

    def draw_cell(ax, indicator, uidx):
        sim_arr = np.asarray(sim[indicator])
        q_ref, band_lo, band_hi = _sim_vs_sim_band(sim_arr, uidx, quantiles, n_boot, rng)
        ax.fill_between(q_ref, band_lo, band_hi, color=_PAPER_C["sim_band"],
                        alpha=_PAPER_ALPHA_BAND, linewidth=0, zorder=1)
        ax.plot(q_ref, q_ref, color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG,
                linestyle="--", zorder=2)
        q_emu = np.quantile(np.asarray(emulator[indicator])[:, :, uidx].ravel(), quantiles)
        ax.plot(q_ref, q_emu, color=_PAPER_C["emulator"], lw=_PAPER_LW_MAIN, zorder=4)
        for b_idx, (bname, barrs) in enumerate(baselines.items()):
            q_b = np.quantile(np.asarray(barrs[indicator])[:, :, uidx].ravel(), quantiles)
            ax.plot(q_ref, q_b, color=_baseline_color(b_idx), lw=_PAPER_LW_MAIN,
                    linestyle="--", zorder=3)

    def row_ylabel(indicator):
        unit = _indicator_unit(error_data, indicator, indicator_units)
        return f"Emulated [{unit}]" if unit else "Emulated"

    def setup(ax):
        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))

    legend_handles = [
        mpatches.Patch(color=_PAPER_C["sim_band"], alpha=_PAPER_ALPHA_BAND + 0.2,
                       label="Sim-Sim 90 % band"),
        mlines.Line2D([], [], color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG,
                      linestyle="--", label="Perfect (diagonal)"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MAIN, label="Emulator"),
    ]
    for b_idx, bname in enumerate(baselines):
        legend_handles.append(mlines.Line2D([], [], color=_baseline_color(b_idx),
                                            lw=_PAPER_LW_MAIN, linestyle="--", label=bname))

    return _ranking_grid(
        error_data, draw_cell=draw_cell, selection=selection,
        share_y_per_row=share_y_per_row,
        row_ylabel=row_ylabel, xlabel="Simulated",
        legend_handles=legend_handles,
        panel_width=panel_width, panel_height=panel_height,
        fontsize_title=fontsize_title, fontsize_ax=fontsize_ax,
        fontsize_legend=fontsize_legend, hspace=0.48, wspace=0.35,
        show_ranking_source=False, suptitle=suptitle, save_path=save_path, dpi=dpi,
        per_cell_setup=setup,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation curves — one function for both regional and gridded data
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_correlation_curves(
    error_data: ErrorData,
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    n_lags: int = DEFAULT_TEMPORAL_CORR_N_LAGS,
    window: int | None = None,
    selection=None,
    share_y_per_row: bool = False,
    panel_width: float = 3.4,
    panel_height: float = 2.2,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Temporal-autocorrelation curves — lag on the x-axis, autocorrelation on
    the y-axis — for selected units, one row per indicator. One function for
    both regional and gridded data: pass plain
    ``{indicator: (n_members, T, n_units)}`` arrays either way (for gridded,
    that's ``mesh_array.values``).

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` /
        `build_error_data_gridded`, built with ``metric="temporal_corr_nmae"``
        so the ranking reflects the distance between simulated and emulated
        autocorrelation curves.
    sim, emulator : ``{indicator: (n_members, T, n_units)}`` — the same
        preprocessed arrays `error_data` was built from.
    baseline_emulations : ``{name: {indicator: array}}`` — drawn as a dashed
        curve each.
    n_lags, window : must match what `error_data` was built with — the number
        of lags plotted and the trailing-window length used by
        `metrics.compute_temporal_correlation_curve`.
    selection : which units to show — see `resolve_selection`.
    share_y_per_row : give every panel in a row the same y-limits (default
        False).
    (remaining parameters control figure geometry/cosmetics)

    Returns
    -------
    matplotlib Figure
    """
    lags = np.arange(n_lags + 1)
    baselines = baseline_emulations or {}

    def draw_cell(ax, indicator, uidx):
        ax.axhline(0, color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG, linestyle=":", zorder=1)
        sim_curve = compute_temporal_correlation_curve(
            np.asarray(sim[indicator])[:, :, uidx], n_lags, window)
        emu_curve = compute_temporal_correlation_curve(
            np.asarray(emulator[indicator])[:, :, uidx], n_lags, window)
        ax.plot(lags, sim_curve, color=_PAPER_C["sim"], lw=_PAPER_LW_MEDIAN, zorder=3)
        ax.plot(lags, emu_curve, color=_PAPER_C["emulator"], lw=_PAPER_LW_MEDIAN, zorder=4)
        for b_idx, (bname, barrs) in enumerate(baselines.items()):
            b_curve = compute_temporal_correlation_curve(
                np.asarray(barrs[indicator])[:, :, uidx], n_lags, window)
            ax.plot(lags, b_curve, color=_baseline_color(b_idx),
                    lw=_PAPER_LW_BASELINE, linestyle="--", zorder=2)

    legend_handles = [
        mlines.Line2D([], [], color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG,
                      linestyle=":", label="Zero correlation"),
        mlines.Line2D([], [], color=_PAPER_C["sim"], lw=_PAPER_LW_MEDIAN, label="Simulation"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MEDIAN, label="Emulator"),
    ]
    for b_idx, bname in enumerate(baselines):
        legend_handles.append(mlines.Line2D([], [], color=_baseline_color(b_idx),
                                            lw=_PAPER_LW_BASELINE, linestyle="--", label=bname))

    return _ranking_grid(
        error_data, draw_cell=draw_cell, selection=selection,
        share_y_per_row=share_y_per_row,
        row_ylabel=lambda i: f"{error_data.short_label(i)} autocorrelation",
        xlabel="Lag (timesteps)", legend_handles=legend_handles,
        panel_width=panel_width, panel_height=panel_height,
        fontsize_title=fontsize_title, fontsize_ax=fontsize_ax,
        fontsize_legend=fontsize_legend, hspace=0.42, wspace=0.32,
        show_ranking_source=True, suptitle=suptitle, save_path=save_path, dpi=dpi,
        per_cell_setup=lambda ax: ax.xaxis.set_major_locator(
            plt.MaxNLocator(nbins=4, integer=True, prune="both")
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Power spectral density curves — one function for both regional and gridded
# ─────────────────────────────────────────────────────────────────────────────

def plot_psd_curves(
    error_data: ErrorData,
    sim: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    baseline_emulations: dict[str, dict[str, np.ndarray]] | None = None,
    *,
    fs: float = 1.0,
    nperseg: int | None = None,
    x_axis: str = "period",
    log_x: bool = True,
    log_y: bool = True,
    band: tuple[float, float] = (10, 90),
    reference_periods: dict[str, float] | None = None,
    selection=None,
    share_y_per_row: bool = False,
    panel_width: float = 3.4,
    panel_height: float = 2.2,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Power spectral density curves for selected units, one row per indicator
    — the curve counterpart of `plot_temporal_correlation_curves`, and the
    companion figure to a ``metric="psd_log_nmae"`` (or "psd_nmae" /
    "psd_wasserstein") map.

    Each panel shows the member-median PSD for the Simulation and the
    Emulator with a shaded inter-member percentile band, so you can see
    whether an emulator/simulation gap is larger than the spread across
    ensemble members, plus one dashed median curve per baseline.

    Parameters
    ----------
    error_data : from `metrics.build_error_data_regional` /
        `build_error_data_gridded`, built with a PSD metric so the ranking
        reflects the distance between simulated and emulated spectra.
    sim, emulator : ``{indicator: (n_members, T, n_units)}`` — the same
        preprocessed arrays `error_data` was built from (for gridded, pass
        ``mesh_array.values``). Remove the ensemble mean first
        (`preprocess_*(..., remove_ensemble_mean=True)`) if you want a pure
        internal-variability spectrum.
    baseline_emulations : ``{name: {indicator: array}}``
    fs, nperseg : sampling frequency and Welch segment length — pass the same
        values used to build `error_data` (`psd_fs` / `psd_nperseg`) so the
        curves match the scores. With monthly data, ``fs=1`` gives
        cycles/month; ``fs=12`` gives cycles/year.
    x_axis : "period" (default, ``1/f``, the usual way to read a climate
        spectrum) or "frequency".
    log_x, log_y : log-scale the axes (both default True — a PSD normally
        spans several orders of magnitude).
    band : inter-member percentiles shaded around each median curve, as
        ``(low, high)``. Default (10, 90).
    reference_periods : optional vertical guide lines, e.g.
        ``{"Annual": 12, "ENSO ~3 yr": 36}`` — keys are labels, values are
        periods **in timesteps** (they are converted using `fs`, so they land
        correctly whichever `x_axis` is chosen). Pass ``{}`` for none;
        default draws annual and semi-annual lines when the data looks
        monthly (``fs`` of 1 or 12).
    selection : which units to show — see `resolve_selection`.
    share_y_per_row : give every panel in a row the same y-limits (default
        False).
    (remaining parameters control figure geometry/cosmetics)

    Returns
    -------
    matplotlib Figure
    """
    if x_axis not in ("period", "frequency"):
        raise ValueError("x_axis must be 'period' or 'frequency'")
    baselines = baseline_emulations or {}
    lo_pct, hi_pct = band

    if reference_periods is None:
        reference_periods = {"Annual": 12, "Semi-annual": 6} if fs in (1.0, 1, 12, 12.0) else {}

    def _x_of(freqs):
        keep = freqs > 0
        return keep, (1.0 / freqs[keep] if x_axis == "period" else freqs[keep])

    def draw_cell(ax, indicator, uidx):
        freqs, psd_sim = compute_psd_curves(np.asarray(sim[indicator])[:, :, uidx], fs, nperseg)
        keep, x = _x_of(freqs)
        for arr, color in ((psd_sim, _PAPER_C["sim"]),
                           (compute_psd_curves(np.asarray(emulator[indicator])[:, :, uidx],
                                               fs, nperseg)[1], _PAPER_C["emulator"])):
            vals = arr[:, keep]
            ax.fill_between(x, np.percentile(vals, lo_pct, axis=0),
                            np.percentile(vals, hi_pct, axis=0),
                            color=color, alpha=_PAPER_ALPHA_CI, linewidth=0, zorder=2)
            ax.plot(x, np.median(vals, axis=0), color=color, lw=_PAPER_LW_MEDIAN, zorder=4)
        for b_idx, (bname, barrs) in enumerate(baselines.items()):
            _, psd_b = compute_psd_curves(np.asarray(barrs[indicator])[:, :, uidx], fs, nperseg)
            ax.plot(x, np.median(psd_b[:, keep], axis=0), color=_baseline_color(b_idx),
                    lw=_PAPER_LW_BASELINE, linestyle="--", zorder=3)
        for label, period_steps in reference_periods.items():
            xv = period_steps / fs if x_axis == "period" else fs / period_steps
            ax.axvline(xv, color="0.55", linestyle=":", linewidth=0.8, zorder=1)

    def setup(ax):
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    freq_unit = "cycles / timestep" if fs == 1.0 else "cycles / time unit"
    xlabel = "Period (timesteps)" if x_axis == "period" else f"Frequency ({freq_unit})"

    legend_handles = [
        mlines.Line2D([], [], color=_PAPER_C["sim"], lw=_PAPER_LW_MEDIAN,
                      label=f"Simulation (median + {lo_pct:g}\N{EN DASH}{hi_pct:g} %)"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MEDIAN,
                      label=f"Emulator (median + {lo_pct:g}\N{EN DASH}{hi_pct:g} %)"),
    ]
    for b_idx, bname in enumerate(baselines):
        legend_handles.append(mlines.Line2D([], [], color=_baseline_color(b_idx),
                                            lw=_PAPER_LW_BASELINE, linestyle="--", label=bname))
    if reference_periods:
        legend_handles.append(mlines.Line2D([], [], color="0.55", linestyle=":", linewidth=0.8,
                                            label=" / ".join(reference_periods)))

    return _ranking_grid(
        error_data, draw_cell=draw_cell, selection=selection,
        share_y_per_row=share_y_per_row,
        row_ylabel=lambda i: f"{error_data.short_label(i)} PSD",
        xlabel=xlabel, legend_handles=legend_handles,
        panel_width=panel_width, panel_height=panel_height,
        fontsize_title=fontsize_title, fontsize_ax=fontsize_ax,
        fontsize_legend=fontsize_legend, hspace=0.42, wspace=0.32,
        show_ranking_source=True, suptitle=suptitle, save_path=save_path, dpi=dpi,
        per_cell_setup=setup,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spatial correlation matrices (regional only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_comparison(
    correlation_data: dict[str, dict],
    region_names: list[str] | None = None,
    figsize_per_panel: float = 4.5,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Spatial correlation matrices for the Emulator and every baseline, side by
    side with their difference from the simulation. One row per comparison.

    Parameters
    ----------
    correlation_data : from `metrics.build_correlation_data` —
        ``{comparison: {"sim_corr", "emu_corr", "diff", "mae", "rmse"}}``.
        One indicator at a time, since the correlation is *across regions*.
    region_names : labels for the matrix axes (length n_regions)
    figsize_per_panel, suptitle, save_path, dpi : as usual

    Returns
    -------
    matplotlib Figure
    """
    comparisons = list(correlation_data)
    m = correlation_data[comparisons[0]]["sim_corr"].shape[0]
    labels = region_names if region_names is not None else [str(i) for i in range(m)]

    n_rows, n_cols = len(comparisons), 3
    fw = figsize_per_panel * n_cols + 1.5
    fh = figsize_per_panel * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh), squeeze=False,
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})

    col_titles = ["Simulation correlations", "Emulation correlations", "Difference (emu \N{MINUS SIGN} sim)"]
    cbar_labels = ["Correlation (sim)", "Correlation (emu)", "Difference"]
    im_handles = {}
    diff_abs_global = max(max(np.abs(correlation_data[c]["diff"]).max() for c in comparisons), 1e-9)
    panels_template = [
        ("sim_corr", "RdBu_r", -1, 1),
        ("emu_corr", "RdBu_r", -1, 1),
        ("diff", "PiYG", -diff_abs_global, diff_abs_global),
    ]

    for row_idx, label in enumerate(comparisons):
        res = correlation_data[label]
        for col_idx, (key, cmap, vmin, vmax) in enumerate(panels_template):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(res[key], cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest")
            im_handles[(row_idx, col_idx)] = im
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis="both", which="both", length=2, pad=2)
            if row_idx == n_rows - 1:
                ax.set_xticks(range(m))
                ax.set_xticklabels(labels, rotation=90, fontsize=5, ha="center")
            else:
                ax.set_xticks([])
            if col_idx == 0:
                ax.set_yticks(range(m))
                ax.set_yticklabels(labels, fontsize=5, va="center")
            else:
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, pad=6)
        axes[row_idx, 0].text(
            -0.18, 0.5, f"{label}\nMAE {res['mae']:.3f} | RMSE {res['rmse']:.3f}",
            transform=axes[row_idx, 0].transAxes, ha="center", va="center",
            fontsize=7, fontweight="bold", rotation=90,
        )

    fig.canvas.draw()
    fig_height = fig.get_size_inches()[1]
    cbar_height_inches, cbar_gap_inches = 0.12, 0.6
    cbar_height_fig = cbar_height_inches / fig_height
    for col_idx in range(n_cols):
        pos = axes[-1, col_idx].get_position()
        gap_fig = cbar_gap_inches / fig_height
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - gap_fig - cbar_height_fig, pos.width, cbar_height_fig])
        cb = fig.colorbar(im_handles[(n_rows - 1, col_idx)], cax=cbar_ax, orientation="horizontal")
        cb.set_label(cbar_labels[col_idx], fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)

    fig.subplots_adjust(top=0.93)
    fig.suptitle(suptitle or "Spatial correlations", fontsize=12, y=0.96)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CRPS timeseries (gridded — the CRPS *map* is plot_map_gridded(metric="crps"))
# ─────────────────────────────────────────────────────────────────────────────

def plot_crps_timeseries_gridded(
    crps_data: dict,
    years: np.ndarray | None = None,
    *,
    share_y: bool = False,
    indicator_units: dict[str, str] | None = None,
    panel_width: float = 6.0,
    panel_height: float = 2.4,
    fontsize_title: int = 9,
    fontsize_ax: int = 8,
    fontsize_legend: int = 8,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Spatially-averaged CRPS at every timestep, one line per comparison and
    one row per indicator. Unlike `plot_timeseries_gridded` there is no
    best/median/worst split — spatial averaging already collapsed the
    gridpoint axis.

    Parameters
    ----------
    crps_data : from `metrics.build_crps_timeseries_data_gridded` — a dict
        with "indicators", "indicator_labels", "indicator_units" and
        ``series[comparison][indicator] -> (T,)``.
    years : x-axis values, length T. Defaults to 0..T-1.
    share_y : give every row the same y-limits (default False).
    indicator_units : override the unit strings in the y-axis labels.
    (remaining parameters control figure geometry/cosmetics)

    Returns
    -------
    matplotlib Figure
    """
    indicators = crps_data["indicators"]
    labels = crps_data["indicator_labels"]
    units = dict(crps_data["indicator_units"])
    units.update(indicator_units or {})
    series = crps_data["series"]

    comparisons = list(series)
    n_time = len(series[comparisons[0]][indicators[0]])
    years = np.arange(n_time) if years is None else np.asarray(years)

    n_rows = len(indicators)
    fig, axes = plt.subplots(n_rows, 1, figsize=(panel_width, panel_height * n_rows),
                             squeeze=False, gridspec_kw={"hspace": 0.35})
    axes = axes[:, 0]

    line_colors = {"Emulator": _PAPER_C["emulator"]}
    baseline_names = [c for c in comparisons if c != "Emulator"]
    for b_idx, bname in enumerate(baseline_names):
        line_colors[bname] = _baseline_color(b_idx)

    for row_idx, indicator in enumerate(indicators):
        ax = axes[row_idx]
        ax.axhline(0, color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG, linestyle="--", zorder=1)
        for label in comparisons:
            ax.plot(years, series[label][indicator], color=line_colors[label],
                    lw=_PAPER_LW_MAIN, label=label, zorder=3)
        _style_axes(ax, fontsize_ax)
        unit = units.get(indicator, "")
        ax.set_ylabel(f"CRPS  [{unit}]" if unit else "CRPS", fontsize=fontsize_ax)
        ax.set_title(f"{labels[indicator]} — spatial-mean CRPS over time",
                     fontsize=fontsize_title, pad=4)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Year", fontsize=fontsize_ax)

    if share_y:
        _apply_share_y(list(axes))

    legend_handles = [
        mlines.Line2D([], [], color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG,
                      linestyle="--", label="Perfect (CRPS=0)"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MAIN, label="Emulator"),
    ]
    for bname in baseline_names:
        legend_handles.append(mlines.Line2D([], [], color=line_colors[bname],
                                            lw=_PAPER_LW_MAIN, label=bname))
    fig.legend(handles=legend_handles, ncol=len(legend_handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.03), fontsize=fontsize_legend, frameon=False,
               handlelength=1.6, columnspacing=0.8, handletextpad=0.4)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize_title + 1, y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig

def plot_spatial_correlation_curves_gridded(
    error_data: ErrorData,
    curve_data: dict,
    *,
    show_spread: bool = True,
    x_unit: str = "km",
    selection=None,
    share_y_per_row: bool = False,
    panel_width: float = 3.4,
    panel_height: float = 2.2,
    fontsize_title: int = 7,
    fontsize_ax: int = 6,
    fontsize_legend: int = 6,
    suptitle: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Correlation-vs-distance curves for selected gridpoints — great-circle
    distance on the x-axis, mean correlation with the gridpoints at that
    distance on the y-axis — one row per indicator.
 
    The companion figure to a ``metric="spatial_corr_nmae"`` map: the map says
    *where* the emulator gets the spatial structure wrong, this says *how*. An
    emulator that draws gridpoints too independently gives a curve that decays
    to zero faster than the simulation's; one that over-smooths stays high too
    far out.
 
    Parameters
    ----------
    error_data : from `metrics.build_error_data_spatial_correlation_gridded` —
        supplies the ranking, the per-gridpoint scores in the panel titles, and
        the indicator list.
    curve_data : the matching dict from
        `metrics.build_spatial_correlation_curve_data_gridded` — the curves
        themselves. Both come from the same call chain, so they are already
        consistent; nothing is recomputed here.
    show_spread : shade ±1 standard deviation of the individual pairwise
        correlations inside each distance bin, around each curve. This is the
        scatter across gridpoint *pairs*, not across ensemble members — it
        shows whether a gap between two curves is larger than the variation
        among the pairs that went into them.
    x_unit : "km" (default) or "1000 km", purely a display scaling.
    selection : which gridpoints to show — ``None`` for best / median / worst,
        a list of 1-based ranks, or a list of ``(lat, lon)`` pairs. See
        `resolve_selection`.
    share_y_per_row : give every panel in a row the same y-limits (default
        False). Often worth setting True here, since correlation is already on
        a common scale.
    (remaining parameters control figure geometry/cosmetics)
 
    Returns
    -------
    matplotlib Figure
    """
    if x_unit not in ("km", "1000 km"):
        raise ValueError("x_unit must be 'km' or '1000 km'")
    scale = 1.0 if x_unit == "km" else 1e-3
    x = np.asarray(curve_data["bin_centers"]) * scale
 
    curves = curve_data["curves"]
    spread = curve_data["spread"]
    baselines = [c for c in error_data.comparisons if c != "Emulator"]
 
    def _line(ax, source, uidx, indicator, color, dashed):
        y = curves[source][indicator][uidx]
        if show_spread and source in spread:
            s = spread[source][indicator][uidx]
            ax.fill_between(x, y - s, y + s, color=color, alpha=_PAPER_ALPHA_CI,
                            linewidth=0, zorder=2)
        ax.plot(x, y, color=color, lw=_PAPER_LW_BASELINE if dashed else _PAPER_LW_MEDIAN,
                linestyle="--" if dashed else "-", zorder=4 if not dashed else 3)
 
    def draw_cell(ax, indicator, uidx):
        ax.axhline(0, color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG, linestyle=":", zorder=1)
        _line(ax, "Simulations", uidx, indicator, _PAPER_C["sim"], dashed=False)
        _line(ax, "Emulator", uidx, indicator, _PAPER_C["emulator"], dashed=False)
        for b_idx, bname in enumerate(baselines):
            if bname in curves:
                _line(ax, bname, uidx, indicator, _baseline_color(b_idx), dashed=True)
 
    legend_handles = [
        mlines.Line2D([], [], color=_PAPER_C["diag"], lw=_PAPER_LW_DIAG,
                      linestyle=":", label="Zero correlation"),
        mlines.Line2D([], [], color=_PAPER_C["sim"], lw=_PAPER_LW_MEDIAN, label="Simulation"),
        mlines.Line2D([], [], color=_PAPER_C["emulator"], lw=_PAPER_LW_MEDIAN, label="Emulator"),
    ]
    for b_idx, bname in enumerate(baselines):
        legend_handles.append(mlines.Line2D([], [], color=_baseline_color(b_idx),
                                            lw=_PAPER_LW_BASELINE, linestyle="--", label=bname))
    if show_spread:
        legend_handles.append(mpatches.Patch(color=_PAPER_C["sim"],
                                             alpha=_PAPER_ALPHA_CI + 0.15,
                                             label="\N{PLUS-MINUS SIGN}1 sd across pairs in bin"))
 
    return _ranking_grid(
        error_data, draw_cell=draw_cell, selection=selection,
        share_y_per_row=share_y_per_row,
        row_ylabel=lambda i: f"{error_data.short_label(i)} correlation",
        xlabel=f"Great-circle distance ({x_unit})",
        legend_handles=legend_handles,
        panel_width=panel_width, panel_height=panel_height,
        fontsize_title=fontsize_title, fontsize_ax=fontsize_ax,
        fontsize_legend=fontsize_legend, hspace=0.42, wspace=0.32,
        show_ranking_source=True, suptitle=suptitle, save_path=save_path, dpi=dpi,
        per_cell_setup=lambda ax: ax.xaxis.set_major_locator(
            plt.MaxNLocator(nbins=4, prune="both")
        ),
    )
