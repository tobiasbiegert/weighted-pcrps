"""
Script to create the figures for paper, appendix and supplement.
"""

import logging
import os
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from isodisreg import idr
import xarray as xr
import cartopy.crs as ccrs
from weatherbench2.metrics import _spatial_average
from weatherbench2.visualization import set_wb2_style


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def get_environment_path(name: str) -> Path:
    """Return a required path from an environment variable."""
    value = os.environ.get(name)

    if value is None:
        raise RuntimeError(
            f"{name} is not set. Load local_paths.env before running this script."
        )

    return Path(value)


RESULTS_DIR = get_environment_path("PCRPS_RESULTS_DIR")
THRESHOLDS_DIR = get_environment_path("PCRPS_THRESHOLDS_DIR")
PLOTS_DIR = Path("plots")


# ---------------------------------------------------------------------
# Variable names, labels, and score selection
# ---------------------------------------------------------------------


VARIABLE_INFO = {
    "mean_sea_level_pressure": {
        "title": "MSLP",
        "unit": r"$\left[ \text{Pa} \right]$",
    },
    "2m_temperature": {
        "title": "T2M",
        "unit": r"$\left[ \text{K} \right]$",
    },
    "10m_wind_speed": {
        "title": "WS10",
        "unit": r"$\left[ \text{m/s} \right]$",
    },
    "total_precipitation_24hr": {
        "title": "TP24hr",
        "unit": r"$\left[ \text{m} \right]$",
    },
}

SCORE_SUFFIXES = {
    "pcrps": "_pcrps",
    "qw_pcrps": "_qw_pcrps",
    "tw_pcrps": "_tw_pcrps",
}

variables = list(VARIABLE_INFO.keys())

lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
]

lead_time_days = lead_times/np.timedelta64(1, 'D')

plot_kwargs = dict(marker='|', markersize=6, markeredgewidth=1.5)

def _match_variable(var: str) -> str:
    for key in VARIABLE_INFO:
        if key in var:
            return key
    raise ValueError(f"No known variable found in: {var!r}")

def get_title(var: str) -> str:
    key = _match_variable(var)
    return VARIABLE_INFO[key]["title"]


def get_unit(var: str) -> str:
    key = _match_variable(var)
    return VARIABLE_INFO[key]["unit"]

def get_score_var_name(var: str, score: str) -> str:
    if var not in VARIABLE_INFO:
        raise ValueError(f"Unknown base variable: {var!r}")
    if score not in SCORE_SUFFIXES:
        raise ValueError(f"Unknown score: {score!r}")
    return f"{var}{SCORE_SUFFIXES[score]}"


def get_score_data(ds: xr.Dataset, var: str, score: str) -> xr.DataArray:
    return ds[get_score_var_name(var, score)]

def available_base_variables(ds, score="pcrps"):
    suffix = SCORE_SUFFIXES[score]
    return [var for var in VARIABLE_INFO if f"{var}{suffix}" in ds.data_vars]

def get_upper_skill_score_data(forecast_ds, reference_ds, var, score):
    forecast_score = get_score_data(forecast_ds, var, score)
    reference_score = get_score_data(reference_ds, var, score)

    forecast_score = forecast_score.clip(min=0)
    reference_score = reference_score.clip(min=0)

    return 1 - forecast_score / reference_score

def get_lower_skill_score_data(forecast_ds, reference_ds, var, score):
    forecast_score = (
        get_score_data(forecast_ds, var, "pcrps")
        - get_score_data(forecast_ds, var, score)
    )

    reference_score = (
        get_score_data(reference_ds, var, "pcrps")
        - get_score_data(reference_ds, var, score)
    )

    forecast_score = forecast_score.clip(min=0)
    reference_score = reference_score.clip(min=0)

    return 1 - forecast_score / reference_score

def is_lower_half(t):
    if t == "monthly_min":
        return True
    if t == "monthly_max":
        return False
    return int(t[1:]) < 50


def get_tw_pcrps_data(ds, var, threshold):
    """Select twPCRPS for the tail associated with the threshold."""
    tw = get_score_data(ds, var, "tw_pcrps").sel(threshold=threshold)

    if threshold != "monthly_min" and is_lower_half(threshold):
        return get_score_data(ds, var, "pcrps") - tw

    return tw


def get_map_score_data(ds, var, score, threshold=None, quantile=None):
    """Select the score field used in a map."""
    if score == "pcrps":
        return get_score_data(ds, var, "pcrps")

    if score == "tw_pcrps":
        if threshold is None:
            raise ValueError("For score='tw_pcrps', provide threshold.")

        return get_tw_pcrps_data(ds, var, threshold)

    if score == "qw_pcrps":
        if quantile is None:
            raise ValueError("For score='qw_pcrps', provide quantile.")

        qw = get_score_data(ds, var, "qw_pcrps").sel(quantile=quantile)

        if quantile < 0.5:
            return get_score_data(ds, var, "pcrps") - qw

        return qw

    raise ValueError("score must be 'pcrps', 'tw_pcrps', or 'qw_pcrps'.")


def get_map_skill_score_data(forecast_ds, reference_ds, var, score, threshold=None, quantile=None):
    """Calculate skill from the selected forecast and reference scores."""
    forecast_score = get_map_score_data(forecast_ds, var, score, threshold=threshold, quantile=quantile).clip(min=0)
    reference_score = get_map_score_data(reference_ds, var, score, threshold=threshold, quantile=quantile).clip(min=0)

    return 1 - forecast_score / reference_score.where(reference_score > 0)


def make_skill_cmap(vmin, vmax=1, n_colors=256, bad_color="lightgray"):
    if vmin < 0:
        frac_neg = -vmin / (vmax - vmin)
        n_neg = int(np.round(frac_neg * n_colors))
        n_neg = np.clip(n_neg, 1, n_colors - 1)

        neg_cmap = plt.cm.OrRd(np.linspace(0, 1, n_neg))
        pos_cmap = plt.cm.viridis(np.linspace(0, 1, n_colors - n_neg))
        combined_cmap = np.vstack([neg_cmap, pos_cmap])

        cmap_obj = mcolors.ListedColormap(combined_cmap)
    else:
        cmap_obj = plt.get_cmap("viridis").copy()

    cmap_obj.set_bad(bad_color)
    return cmap_obj

# ---------------------------------------------------------------------
# Map figures
# ---------------------------------------------------------------------

def plot_skill_score_maps(
    forecast_ds,
    reference_ds,
    score="tw_pcrps",
    threshold=None,
    quantile=None,
    variables=None,
    lead_times=lead_times,
    figsize_per_panel=(5.8, 3.3),
    title_fontsize=28,
    rowlabel_fontsize=22,
    cbar_label_fontsize=22,
    cbar_tick_fontsize=18,
    vmin=None,
    vmax=1,
    wspace=0.0,
    hspace=0.08,
    cbar_position=(0.22, 0.07, 0.56, 0.022),
    add_zero_contour=False,
    mask_below_vmin=False,
    save_path=None,
    save_dpi=300,
):
    if variables is None:
        variables = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]

    skill_by_var = {
        var: get_map_skill_score_data(
            forecast_ds,
            reference_ds,
            var,
            score,
            threshold=threshold,
            quantile=quantile,
        )
        for var in variables
    }

    if mask_below_vmin and vmin is not None:
        skill_by_var = {var: data.where(data >= vmin) for var, data in skill_by_var.items()}

    finite_values = []

    for var in variables:
        values = skill_by_var[var].sel(prediction_timedelta=lead_times).values
        values = values[np.isfinite(values)]

        if values.size > 0:
            finite_values.append(values)

    if len(finite_values) == 0:
        raise ValueError("All selected skill-score values are non-finite.")

    finite_values = np.concatenate(finite_values)
    data_min = finite_values.min()
    data_max = finite_values.max()

    if vmin is None:
        vmin = min(data_min, 0)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    missing_color = "#b0b0b0"

    cmap_obj = make_skill_cmap(vmin=vmin, vmax=vmax, bad_color=(0, 0, 0, 0))

    if data_min < vmin and data_max > vmax:
        extend = "both"
    elif data_min < vmin:
        extend = "min"
    elif data_max > vmax:
        extend = "max"
    else:
        extend = "neither"

    n_rows = len(lead_times)
    n_cols = len(variables)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        subplot_kw={"projection": ccrs.Robinson()},
        squeeze=False,
    )

    for i, lt in enumerate(lead_times):
        lt_days = int(lt / np.timedelta64(1, "D"))

        for j, var in enumerate(variables):
            ax = axes[i, j]
            ax.set_facecolor(missing_color)
            ax.coastlines(linewidth=0.7)

            data = skill_by_var[var].sel(prediction_timedelta=lt)
            data = data.where(np.isfinite(data))

            data = data.transpose("latitude", "longitude")

            lon_grid, lat_grid = np.meshgrid(data.longitude.values, data.latitude.values)

            mesh = ax.pcolormesh(
                lon_grid,
                lat_grid,
                data.values,
                cmap=cmap_obj,
                norm=norm,
                transform=ccrs.PlateCarree(),
                linewidth=0.5,
                rasterized=True,
            )

            if add_zero_contour:
                lon_grid, lat_grid = np.meshgrid(data.longitude.values, data.latitude.values)

                ax.contour(
                    lon_grid,
                    lat_grid,
                    data.values,
                    levels=[0.0],
                    colors="lightgray",
                    linewidths=0.6,
                    transform=ccrs.PlateCarree(),
                )

            if i == 0:
                ax.set_title(get_title(var), fontsize=title_fontsize)

        axes[i, 0].annotate(
            f"Lead Time: {lt_days} day" if lt_days == 1 else f"Lead Time: {lt_days} days",
            fontsize=rowlabel_fontsize,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )

    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.96,
        bottom=0.11,
        wspace=wspace,
        hspace=hspace,
    )

    cax = fig.add_axes(cbar_position)
    cax.grid(False)

    cbar = plt.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    score_label = {"pcrps": "PCRPS", "tw_pcrps": "twPCRPS", "qw_pcrps": "qwPCRPS"}[score]

    if add_zero_contour:
        cbar.set_label(f"{score_label} Skill", fontsize=cbar_label_fontsize)
    else:
        cbar.set_label(f"{score_label}-S", fontsize=cbar_label_fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")

    plt.close(fig)

def plot_best_model_maps(
    model_datasets,
    score="pcrps",
    threshold=None,
    quantile=None,
    variables=None,
    lead_times=lead_times,
    figsize_per_panel=(5.8, 3.3),
    title_fontsize=28,
    rowlabel_fontsize=22,
    legend_fontsize=28,
    wspace=0.0,
    hspace=0.08,
    legend_y=0.05,
    bad_color="#b0b0b0",
    print_tie_summary=True,
    save_path=None,
    save_dpi=300,
):
    if variables is None:
        variables = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"]

    model_order = list(model_datasets.keys())

    model_colors = {
        "HRES": "tab:blue",
        "PW-ERA5": "tab:orange",
        "PW-IFS": "tab:orange",
        "GC-ERA5": "tab:green",
        "GC-IFS": "tab:green",
        "FuXi": "tab:pink",
    }

    color_list = [model_colors.get(m, f"C{i}") for i, m in enumerate(model_order)]

    tie_idx = len(model_order)
    cmap = ListedColormap(color_list + [bad_color])
    norm = BoundaryNorm(np.arange(-0.5, len(model_order) + 1.5, 1), cmap.N)

    first_ds = next(iter(model_datasets.values()))
    lon = first_ds["longitude"].values
    lat = first_ds["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    n_rows = len(lead_times)
    n_cols = len(variables)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        subplot_kw={"projection": ccrs.Robinson()},
        squeeze=False,
    )

    win_counts = np.zeros(len(model_order), dtype=int)
    total_count = 0
    tie_count = 0

    for i, lt in enumerate(lead_times):
        lt_days = int(lt / np.timedelta64(1, "D"))

        for j, var in enumerate(variables):
            ax = axes[i, j]
            ax.coastlines(linewidth=0.7)

            arrays = []
            shape_ref = None

            for model_name in model_order:
                ds = model_datasets[model_name]
                score_var_name = get_score_var_name(var, score)

                if score_var_name not in ds.data_vars:
                    arrays.append(None)
                    continue

                da = get_map_score_data(ds, var, score, threshold=threshold, quantile=quantile)

                if (score == "tw_pcrps" and is_lower_half(threshold)) or (score == "qw_pcrps" and quantile < 0.5):
                    da = da.clip(min=0)

                da = da.sel(prediction_timedelta=lt, longitude=lon, latitude=lat).transpose("latitude", "longitude")
                arr = da.values

                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"Non-finite values found for model={model_name}, "
                        f"variable={var}, score={score}, lead_time={lt_days} days."
                    )

                if np.any(arr < 0):
                    raise ValueError(
                        f"Negative score values found for model={model_name}, "
                        f"variable={var}, score={score}, lead_time={lt_days} days."
                    )

                if shape_ref is None:
                    shape_ref = arr.shape
                elif arr.shape != shape_ref:
                    raise ValueError(
                        f"Shape mismatch for model={model_name}, variable={var}, "
                        f"score={score}, lead_time={lt_days} days. "
                        f"Expected {shape_ref}, got {arr.shape}."
                    )

                arrays.append(arr)

            if shape_ref is None:
                raise ValueError(f"No model has variable={var!r} for score={score!r}.")

            stacked = np.stack(
                [arr if arr is not None else np.full(shape_ref, np.inf) for arr in arrays],
                axis=0,
            )

            min_score = stacked.min(axis=0)
            is_winner = stacked == min_score[None, ...]
            n_winners = is_winner.sum(axis=0)

            unique = n_winners == 1
            ties = n_winners > 1

            winner_idx = np.empty(shape_ref, dtype=float)
            winner_idx[unique] = np.argmax(is_winner[:, unique], axis=0)
            winner_idx[ties] = tie_idx

            wins = np.bincount(winner_idx[unique].astype(int), minlength=len(model_order))
            win_counts += wins
            total_count += winner_idx.size
            tie_count += ties.sum()

            ax.pcolormesh(
                lon_grid,
                lat_grid,
                winner_idx,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

            if i == 0:
                ax.set_title(get_title(var), fontsize=title_fontsize)

        axes[i, 0].annotate(
            f"Lead Time: {lt_days} day" if lt_days == 1 else f"Lead Time: {lt_days} days",
            fontsize=rowlabel_fontsize,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )

    win_pcts = 100 * win_counts / total_count

    legend_patches = [
        Patch(facecolor=color_list[k], label=f"{model_name} ({win_pcts[k]:.1f}%)")
        for k, model_name in enumerate(model_order)
    ]

    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.96,
        bottom=0.11,
        wspace=wspace,
        hspace=hspace,
    )

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(model_order),
        fontsize=legend_fontsize,
    )

    if print_tie_summary:
        print(f"Ties: {tie_count} of {total_count} comparisons ({100 * tie_count / total_count:.1f}%).")

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")

    plt.close(fig)

def plot_significant_best_model_maps(
    pairwise_p_values,
    score="pcrps",
    threshold=None,
    quantile=None,
    variables=None,
    lead_times=lead_times,
    alpha=0.05,
    model_order=None,
    figsize_per_panel=(5.8, 3.3),
    title_fontsize=28,
    rowlabel_fontsize=22,
    legend_fontsize=26,
    wspace=0.0,
    hspace=0.08,
    legend_y=0.05,
    no_sig_color="#b0b0b0",
    include_ai_better=False,
    save_path=None,
    save_dpi=300,
):
    if variables is None:
        variables = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"]

    if model_order is None:
        model_order = []
        for a, b in pairwise_p_values:
            if a not in model_order:
                model_order.append(a)
            if b not in model_order:
                model_order.append(b)

    model_colors = {
        "HRES": "tab:blue",
        "PW-ERA5": "tab:orange",
        "PW-IFS": "tab:orange",
        "GC-ERA5": "tab:green",
        "GC-IFS": "tab:green",
        "FuXi": "tab:pink",
    }

    color_list = [model_colors.get(m, f"C{i}") for i, m in enumerate(model_order)]

    # Category 0: no significantly best model.
    # Categories 1..n_models: individual models.
    # Optional last category: all available AI models beat HRES, but no individual AI model is significantly best.
    ai_better_idx = len(model_order) + 1

    if include_ai_better:
        cmap = ListedColormap([no_sig_color] + color_list + ["tab:brown"])
        norm = BoundaryNorm(np.arange(-0.5, len(model_order) + 2.5, 1), cmap.N)
        count_by_category = np.zeros(len(model_order) + 2, dtype=int)
    else:
        cmap = ListedColormap([no_sig_color] + color_list)
        norm = BoundaryNorm(np.arange(-0.5, len(model_order) + 1.5, 1), cmap.N)
        count_by_category = np.zeros(len(model_order) + 1, dtype=int)

    first_ds = next(iter(pairwise_p_values.values()))
    lon = first_ds["longitude"].values
    lat = first_ds["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    n_rows = len(lead_times)
    n_cols = len(variables)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        subplot_kw={"projection": ccrs.Robinson()},
        squeeze=False,
    )

    total_count = 0

    def _select_p_field(p_ds, var):
        p = get_score_data(p_ds, var, score)

        if score == "tw_pcrps":
            if threshold is None:
                raise ValueError("For score='tw_pcrps', please provide threshold=...")
            p = p.sel(threshold=threshold)

        elif score == "qw_pcrps":
            if quantile is None:
                raise ValueError("For score='qw_pcrps', please provide quantile=...")
            p = p.sel(quantile=quantile)

        elif score != "pcrps":
            raise ValueError("score must be 'pcrps', 'qw_pcrps', or 'tw_pcrps'.")

        return p

    def _has_p_field(p_ds, var):
        return get_score_var_name(var, score) in p_ds.data_vars

    def _model_available(model_name, var):
        for other in model_order:
            if other == model_name:
                continue

            if (model_name, other) in pairwise_p_values:
                if _has_p_field(pairwise_p_values[(model_name, other)], var):
                    return True

            if (other, model_name) in pairwise_p_values:
                if _has_p_field(pairwise_p_values[(other, model_name)], var):
                    return True

        return False

    def _get_p_values(a, b, var, lt):
        lt_days = int(lt / np.timedelta64(1, "D"))

        if (a, b) in pairwise_p_values and _has_p_field(pairwise_p_values[(a, b)], var):
            p = _select_p_field(pairwise_p_values[(a, b)], var).sel(prediction_timedelta=lt, longitude=lon, latitude=lat).transpose("latitude", "longitude")
            arr = p.values

        elif (b, a) in pairwise_p_values and _has_p_field(pairwise_p_values[(b, a)], var):
            p = _select_p_field(pairwise_p_values[(b, a)], var).sel(prediction_timedelta=lt, longitude=lon, latitude=lat).transpose("latitude", "longitude")
            arr = 1.0 - p.values

        else:
            raise ValueError(f"Missing pairwise p-values for {a} vs {b}, variable={var}, score={score}, lead_time={lt_days} days.")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite p-values found for {a} vs {b}, variable={var}, score={score}, lead_time={lt_days} days.")

        if np.any((arr < 0) | (arr > 1)):
            raise ValueError(f"p-values outside [0, 1] found for {a} vs {b}, variable={var}, score={score}, lead_time={lt_days} days.")

        return arr

    for i, lt in enumerate(lead_times):
        lt_days = int(lt / np.timedelta64(1, "D"))

        for j, var in enumerate(variables):
            ax = axes[i, j]
            ax.coastlines(linewidth=0.7)

            available_models = [model_name for model_name in model_order if _model_available(model_name, var)]

            if len(available_models) < 2:
                raise ValueError(f"Need at least two available models for variable={var!r}, but found {available_models}.")

            shape_ref = None
            sig_best = []

            for model_name in model_order:
                if model_name not in available_models:
                    sig_best.append(None)
                    continue

                beats_all = None

                for other in available_models:
                    if other == model_name:
                        continue

                    p_values = _get_p_values(model_name, other, var, lt)
                    model_beats_other = p_values < alpha

                    if shape_ref is None:
                        shape_ref = p_values.shape
                    elif p_values.shape != shape_ref:
                        raise ValueError(f"Shape mismatch for pair {model_name} vs {other}, variable={var}, score={score}, lead_time={lt_days} days. Expected {shape_ref}, got {p_values.shape}.")

                    if beats_all is None:
                        beats_all = model_beats_other
                    else:
                        beats_all = beats_all & model_beats_other

                sig_best.append(beats_all)

            sig_best = np.stack([x if x is not None else np.zeros(shape_ref, dtype=bool) for x in sig_best], axis=0)
            n_sig_best = sig_best.sum(axis=0)

            if np.any(n_sig_best > 1):
                raise ValueError(f"More than one significantly best model found for variable={var}, score={score}, lead_time={lt_days} days.")

            combo = np.zeros(shape_ref, dtype=int)
            unique = n_sig_best == 1
            combo[unique] = np.argmax(sig_best[:, unique], axis=0) + 1

            if include_ai_better:
                if "HRES" not in available_models:
                    raise ValueError("include_ai_better=True requires 'HRES' to be available.")

                ai_models = [model_name for model_name in available_models if model_name != "HRES"]
                ai_better_than_hres = np.ones(shape_ref, dtype=bool)

                for model_name in ai_models:
                    p_values = _get_p_values(model_name, "HRES", var, lt)
                    ai_better_than_hres = ai_better_than_hres & (p_values < alpha)

                no_individual_ai_best = np.ones(shape_ref, dtype=bool)

                for model_name in ai_models:
                    model_idx = model_order.index(model_name)
                    no_individual_ai_best = no_individual_ai_best & ~sig_best[model_idx]

                combo = np.where(ai_better_than_hres & no_individual_ai_best, ai_better_idx, combo)

            counts = np.bincount(combo.ravel(), minlength=len(count_by_category))
            count_by_category += counts
            total_count += combo.size

            ax.pcolormesh(
                lon_grid,
                lat_grid,
                combo,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

            if i == 0:
                ax.set_title(get_title(var), fontsize=title_fontsize)

        axes[i, 0].annotate(
            f"Lead Time: {lt_days} day" if lt_days == 1 else f"Lead Time: {lt_days} days",
            fontsize=rowlabel_fontsize,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
        )

    pcts = 100 * count_by_category / total_count
    legend_patches = [Patch(facecolor=color_list[k], label=f"{model_name} ({pcts[k + 1]:.1f}%)") for k, model_name in enumerate(model_order)]

    if include_ai_better:
        legend_patches.append(Patch(facecolor="tab:brown", label=f"AIWP better than NWP ({pcts[ai_better_idx]:.1f}%)"))

    legend_patches.append(Patch(facecolor=no_sig_color, label=f"No significant best ({pcts[0]:.1f}%)"))

    fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.11, wspace=wspace, hspace=hspace)
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=len(legend_patches), fontsize=legend_fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")

    plt.close(fig)

# ---------------------------------------------------------------------
# Threshold-weighted scores
# ---------------------------------------------------------------------

def opposite_threshold(t):
    if t == "monthly_min":
        return "monthly_max"
    if t == "monthly_max":
        return "monthly_min"
    i = int(t[1:])
    return f"q{100 - i:03d}"

def format_threshold_label(t):
    if t == "monthly_max":
        return "monthly record high"
    if t == "monthly_min":
        return "monthly record low"
    if t == "q100":
        return "record high"
    if t == "q000":
        return "record low"

    q = int(t[1:])

    if 10 <= q % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(q % 10, "th")

    return f"{q}{suffix} percentile"

def plot_tw_pcrps_panels(
    dataset_map,
    threshold,
    ground_truth,
    figsize_per_panel=(6, 4.5),
    title_fontsize=22,
    label_fontsize=17,
    tick_fontsize=16,
    legend_fontsize=20,
    save_path=None,
    save_dpi=300,
):
    if ground_truth not in {"era5", "ifs_analysis"}:
        raise ValueError("ground_truth must be 'era5' or 'ifs_analysis'.")

    threshold_opposite = opposite_threshold(threshold)

    if ground_truth == "era5":
        panels = [
            ("mean_sea_level_pressure", threshold),
            ("2m_temperature", threshold),
            ("10m_wind_speed", threshold),
            ("mean_sea_level_pressure", threshold_opposite),
            ("2m_temperature", threshold_opposite),
            ("total_precipitation_24hr", threshold),
        ]

    else:  # ifs_analysis
        panels = [
            ("mean_sea_level_pressure", threshold),
            ("2m_temperature", threshold),
            ("10m_wind_speed", threshold),
            ("mean_sea_level_pressure", threshold_opposite),
            ("2m_temperature", threshold_opposite),
        ]

    n_panels = len(panels)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        constrained_layout=True,
    )

    flat_axes = axes.ravel()

    for ax, (var, thr) in zip(flat_axes, panels):
        title = f"{get_title(var)} – {format_threshold_label(thr)}"

        for label, (color, ds) in dataset_map.items():
            if var == "total_precipitation_24hr" and label in {"PW-ERA5", "PW-IFS"}:
                continue

            pcrps = get_score_data(ds, var, "pcrps")
            tw = get_score_data(ds, var, "tw_pcrps").sel(threshold=thr)

            if thr != "monthly_min" and is_lower_half(thr):
                y = pcrps.values - tw.values
            else:
                y = tw.values

            line_kwargs = dict(
                color=color,
                label=label,
                **plot_kwargs,
            )

            if label == "ERA5 Climatology":
                line_kwargs["linestyle"] = "--"

            ax.plot(
                lead_time_days,
                y,
                **line_kwargs,
            )

        ax.set_xlabel("Lead Time [d]", fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xticks(lead_time_days)
        ax.set_ylim(0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
        ax.set_ylabel(f"Latitude-weighted twPCRPS {get_unit(var)}", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    first_ax = flat_axes[0]
    handles, labels = first_ax.get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.07),
        fontsize=legend_fontsize,
    )

    for h in legend.legend_handles:
        h.set_marker("")

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')

    plt.close(fig)

def plot_tw_pcrps_skill_by_threshold(
    forecast_map,
    reference_ds,
    ground_truth="era5",
    skill_reference="pcrps0",
    tail="upper",
    title_fontsize=24,
    label_fontsize=18,
    tick_fontsize=14,
    legend_fontsize=22,
    ylim_map=None,
    save_path=None,
    save_dpi=300,
):
    """
    Plot twPCRPS skill score curves over threshold quantiles for several lead times.

    Parameters
    ----------
    ground_truth : {"era5", "ifs_analysis"}
        Ground truth dataset.
    skill_reference : {"pcrps0", "climatology"}
        Denominator used in the skill score:
        - "pcrps0": 1 - forecast / pcrps0
        - "climatology": 1 - forecast / climatology
        The climatology reference is available only for ERA5.
    tail : {"upper", "lower", "both"}
        Which skill curves to plot.
        - "upper": plot only upper-tail skill (solid)
        - "lower": plot only lower-tail skill (dashed), only for MSLP and T2M
        - "both": plot upper for all variables and lower only for MSLP and T2M
    """
    if ground_truth not in {"era5", "ifs_analysis"}:
        raise ValueError("ground_truth must be 'era5' or 'ifs_analysis'.")
    if skill_reference not in {"pcrps0", "climatology"}:
        raise ValueError("skill_reference must be 'pcrps0' or 'climatology'.")
    if tail not in {"upper", "lower", "both"}:
        raise ValueError("tail must be 'upper', 'lower', or 'both'.")
    if ground_truth == "ifs_analysis" and skill_reference == "climatology":
        raise ValueError(
            "The climatology reference is available only for ground_truth='era5'."
        )

    lower_tail_vars = {"mean_sea_level_pressure", "2m_temperature"}

    variables = available_base_variables(reference_ds, score="tw_pcrps")

    if tail == "lower":
        variables = [var for var in variables if var in lower_tail_vars]

    th_all = reference_ds["threshold"].values.astype(str)
    quantile_thresholds = np.array(
        [th for th in th_all if th.startswith("q")]
    )
    t = np.array(
        [int(s[1:]) / 100 for s in quantile_thresholds],
        dtype=float,
    )

    lead_times = reference_ds["prediction_timedelta"].values

    if ylim_map is None:
        if skill_reference == "pcrps0":
            ylim_map = {
                1: (0.6, 1),
                3: (0.3, 1),
                5: (0.2, 1),
                7: (0, 1),
                10: (0, 1),
            }
        else:
            ylim_map = {
                1: (0.5, 1),
                3: (0.3, 1),
                5: (0.1, 1),
                7: (0.05, 0.7),
                10: (-0.1, 0.4),
            }

    fig, axes = plt.subplots(
        len(lead_times),
        len(variables),
        figsize=(5 * len(variables), 4 * len(lead_times)),
        sharex=True,
        sharey=False,
    )

    if len(lead_times) == 1 and len(variables) == 1:
        axes = np.array([[axes]])
    elif len(lead_times) == 1:
        axes = np.array([axes])
    elif len(variables) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, lead_time in enumerate(lead_times):
        lead_days = int(lead_time / np.timedelta64(1, "D"))

        for j, var in enumerate(variables):
            ax = axes[i, j]

            for label, (color, ds) in forecast_map.items():
                if (
                    var == "total_precipitation_24hr"
                    and label in {"PW-ERA5", "PW-IFS"}
                ):
                    continue

                if tail in {"upper", "both"}:
                    upper_skill = get_upper_skill_score_data(
                        ds, reference_ds, var, "tw_pcrps"
                    )

                    ax.plot(
                        t,
                        upper_skill.sel(
                            prediction_timedelta=lead_time,
                            threshold=quantile_thresholds,
                        ).values,
                        label=label,
                        color=color,
                        linestyle="-",
                    )

                if tail in {"lower", "both"} and var in lower_tail_vars:
                    lower_skill = get_lower_skill_score_data(
                        ds, reference_ds, var, "tw_pcrps"
                    )

                    lower_label = label if tail == "lower" else None

                    ax.plot(
                        t,
                        lower_skill.sel(
                            prediction_timedelta=lead_time,
                            threshold=quantile_thresholds,
                        ).values,
                        label=lower_label,
                        color=color,
                        linestyle="--",
                    )

            if skill_reference == "climatology":
                ax.axhline(0, color="black", linestyle="--")

            if i == 0:
                ax.set_title(get_title(var), fontsize=title_fontsize)

            if i == len(lead_times) - 1:
                ax.set_xlabel(
                    "Threshold quantile", fontsize=label_fontsize
                )

            if j == 0:
                if tail == "upper":
                    ylabel = "latitude-weighted upper twPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted upper twPCRPS skill"
                elif tail == "lower":
                    ylabel = "latitude-weighted lower twPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted lower twPCRPS skill"
                else:
                    ylabel = "latitude-weighted twPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted twPCRPS skill"

                ax.set_ylabel(
                    f"{lead_days} day\n{ylabel}",
                    fontsize=label_fontsize,
                )

            if lead_days in ylim_map:
                ax.set_ylim(*ylim_map[lead_days])

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=tick_fontsize,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.52, -0.035),
        fontsize=legend_fontsize,
    )

    for h in legend.legend_handles:
        if hasattr(h, "set_marker"):
            h.set_marker("")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    plt.close(fig)

# ---------------------------------------------------------------------
# Deterministic and ensemble score comparisons
# ---------------------------------------------------------------------

def compute_correlations(ens_stacked, det_stacked, lead_times):
    """Calculate correlations from aligned, stacked score fields."""
    x = det_stacked.values
    y = ens_stacked.values

    pearson_all = pearsonr(x, y)[0]
    spearman_all = spearmanr(x, y)[0]

    pearson_by_lt = {}
    spearman_by_lt = {}

    point_lead_times = det_stacked["prediction_timedelta"].values

    for lt in lead_times:
        lt_days = int(lt / np.timedelta64(1, "D"))
        selected = point_lead_times == lt

        pearson_by_lt[lt_days] = pearsonr(x[selected], y[selected])[0]
        spearman_by_lt[lt_days] = spearmanr(x[selected], y[selected])[0]

    return pearson_all, spearman_all, pearson_by_lt, spearman_by_lt


def plot_tw_scatter_grid(
    det_ds,
    ens_crps_ds,
    ens_tw_ds,
    thresholds,
    ground_truth="era5",
    comparison="ifs",
    title_fontsize=26,
    label_fontsize=24,
    tick_fontsize=20,
    legend_fontsize=24,
    include_spearman=False,
    scientific=True,
    save_path=None,
    save_dpi=300,
):
    if ground_truth not in {"era5", "ifs_analysis"}:
        raise ValueError("ground_truth must be 'era5' or 'ifs_analysis'.")
    if comparison not in {"ifs", "gencast"}:
        raise ValueError("comparison must be 'ifs' or 'gencast'.")
    if len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty list.")

    lower_flags = [is_lower_half(th) for th in thresholds]
    if len(set(lower_flags)) != 1:
        raise ValueError("All thresholds in one call must refer to the same tail.")

    lower_tail = lower_flags[0]

    if comparison == "ifs":
        x_label = "IFS HRES twPCRPS"
        y_label = "IFS ENS twCRPS"
    else:
        x_label = "GraphCast twPCRPS"
        y_label = "GenCast twCRPS"

    variables = ["mean_sea_level_pressure", "2m_temperature"]

    if not lower_tail:
        variables.append("10m_wind_speed")
        if ground_truth == "era5":
            variables.append("total_precipitation_24hr")

    lead_times = det_ds["prediction_timedelta"].values
    lead_time_days = [int(lt / np.timedelta64(1, "D")) for lt in lead_times]

    n_rows = len(thresholds)
    n_cols = len(variables)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 6.8 * n_rows),
        squeeze=False,
    )

    standard_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    lead_to_color = dict(zip(lead_time_days, standard_colors))

    for i, threshold in enumerate(thresholds):
        for j, var in enumerate(variables):
            ax = axes[i, j]

            det_score = get_tw_pcrps_data(det_ds, var, threshold)
            ens_tw = ens_tw_ds[var].sel(threshold=threshold)

            if lower_tail:
                ens_score = ens_crps_ds[var] - ens_tw
            else:
                ens_score = ens_tw

            det_stacked = det_score.stack(points=("prediction_timedelta", "latitude", "longitude"))
            ens_stacked = ens_score.stack(points=("prediction_timedelta", "latitude", "longitude"))
            ens_aligned, det_aligned = xr.align(ens_stacked, det_stacked, join="inner")

            pearson_all, spearman_all, pearson_by_lt, spearman_by_lt = compute_correlations(
                ens_aligned, det_aligned, lead_times
            )

            x = det_aligned.values
            y = ens_aligned.values
            point_lead_days = det_aligned["prediction_timedelta"].dt.days.values
            point_colors = np.array([lead_to_color[int(lt)] for lt in point_lead_days])

            ax.scatter(
                x[::-1],
                y[::-1],
                c=point_colors[::-1],
                s=4,
                alpha=0.7,
                rasterized=True,
            )

            xy_max = max(np.nanmax(x), np.nanmax(y))
            ax.plot([0, xy_max], [0, xy_max], color="k", linestyle="--", linewidth=1.2)

            ax.set_xlabel(f"{x_label} {get_unit(var)}", fontsize=label_fontsize)
            ax.set_ylabel(f"{y_label} {get_unit(var)}", fontsize=label_fontsize)
            ax.set_title(
                f"{get_title(var)} – {format_threshold_label(threshold)}",
                fontsize=title_fontsize,
            )
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

            if scientific:
                ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
                ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
                ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

            corr_handles = [
                Line2D(
                    [], [],
                    color=lead_to_color[lt],
                    marker="o",
                    linestyle="None",
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                )
                for lt in lead_time_days
            ]

            if include_spearman:
                spearman_labels = [f"{spearman_by_lt[lt]:.3f}" for lt in lead_time_days]

                leg_spearman = ax.legend(
                    corr_handles,
                    spearman_labels,
                    title=f"$\\rho$ = {spearman_all:.3f}",
                    loc="upper left",
                    fontsize=legend_fontsize,
                    title_fontsize=legend_fontsize,
                    framealpha=0.5,
                    borderpad=0.2,
                    borderaxespad=0.2,
                    handletextpad=0.4,
                )
                leg_spearman.get_title().set_bbox(
                    dict(
                        mutation_aspect=0.37,
                        fill=False,
                        edgecolor="black",
                        boxstyle="round,pad=0.45,rounding_size=0.3",
                    )
                )
                ax.add_artist(leg_spearman)

            pearson_labels = [f"{pearson_by_lt[lt]:.3f}" for lt in lead_time_days]

            leg_pearson = ax.legend(
                corr_handles,
                pearson_labels,
                title=f"$r$ = {pearson_all:.3f}",
                loc="lower right",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                framealpha=0.5,
                borderpad=0.2,
                borderaxespad=0.2,
                handletextpad=0.4,
            )
            leg_pearson.get_title().set_bbox(
                dict(
                    mutation_aspect=0.33,
                    fill=False,
                    edgecolor="black",
                    boxstyle="round,pad=0.55,rounding_size=0.3",
                )
            )

    lead_handles = [
        Line2D(
            [], [],
            color=lead_to_color[lt],
            marker="o",
            linestyle="None",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        for lt in lead_time_days
    ]

    text_handle = Line2D([], [], color="none", marker="None", linestyle="None")

    fig.legend(
        [text_handle] + lead_handles,
        ["Lead Time [d]"] + [str(lt) for lt in lead_time_days],
        loc="lower center",
        ncol=len(lead_time_days) + 1,
        bbox_to_anchor=(0.515, 0.005),
        fontsize=26,
        framealpha=0.5,
        handletextpad=0.3,
        columnspacing=1,
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    fig.subplots_adjust(hspace=0.25)

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    plt.close(fig)

# ---------------------------------------------------------------------
# Quantile-weighted scores
# ---------------------------------------------------------------------

def format_quantile_label(q, tail):
    p = int(round(100 * q))

    if tail == "upper":
        return f"upper {100 - p}%"

    else:  # lower
        return f"lower {p}%"


def plot_qw_pcrps_panels(
    dataset_map,
    quantile,
    ground_truth,
    include_climatology=False,
    climatology_ds=None,
    figsize_per_panel=(6, 4.5),
    title_fontsize=22,
    label_fontsize=17,
    tick_fontsize=16,
    legend_fontsize=20,
    save_path=None,
    save_dpi=300,
):
    if ground_truth not in {"era5", "ifs_analysis"}:
        raise ValueError("ground_truth must be 'era5' or 'ifs_analysis'.")

    q_upper = quantile
    q_lower = 1 - quantile

    if ground_truth == "era5":
        panels = [
            ("mean_sea_level_pressure", "upper", q_upper),
            ("2m_temperature", "upper", q_upper),
            ("10m_wind_speed", "upper", q_upper),
            ("mean_sea_level_pressure", "lower", q_lower),
            ("2m_temperature", "lower", q_lower),
            ("total_precipitation_24hr", "upper", q_upper),
        ]

    else:  # ifs_analysis
        panels = [
            ("mean_sea_level_pressure", "upper", q_upper),
            ("2m_temperature", "upper", q_upper),
            ("10m_wind_speed", "upper", q_upper),
            ("mean_sea_level_pressure", "lower", q_lower),
            ("2m_temperature", "lower", q_lower),
        ]

    if include_climatology:
        dataset_map = {"ERA5 Climatology": ("black", climatology_ds), **dataset_map}

    n_panels = len(panels)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]),
        constrained_layout=True,
    )

    flat_axes = axes.ravel()

    for ax, (var, tail, q_panel) in zip(flat_axes, panels):
        title = f"{get_title(var)} – {format_quantile_label(q_panel, tail)}"

        for label, (color, ds) in dataset_map.items():
            if var == "total_precipitation_24hr" and label in {"PW-ERA5", "PW-IFS"}:
                continue

            pcrps = get_score_data(ds, var, "pcrps")
            qw = get_score_data(ds, var, "qw_pcrps").sel(quantile=q_panel)

            if tail == "lower":
                y = pcrps.values - qw.values
            else:
                y = qw.values

            line_kwargs = dict(
                color=color,
                label=label,
                **plot_kwargs,
            )

            if label == "ERA5 Climatology":
                line_kwargs["linestyle"] = "--"

            ax.plot(
                lead_time_days,
                y,
                **line_kwargs,
            )

        ax.set_xlabel("Lead Time [d]", fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xticks(lead_time_days)
        ax.set_ylim(0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
        ax.set_ylabel(f"Latitude-weighted qwPCRPS {get_unit(var)}", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    first_ax = flat_axes[0]
    handles, labels = first_ax.get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.07),
        fontsize=legend_fontsize,
    )

    for h in legend.legend_handles:
        h.set_marker("")

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')

    plt.close(fig)

def plot_qw_pcrps_skill_by_quantile(
    forecast_map,
    reference_ds,
    ground_truth="era5",
    skill_reference="pcrps0",
    tail="upper",
    title_fontsize=24,
    label_fontsize=18,
    tick_fontsize=14,
    legend_fontsize=22,
    ylim_map=None,
    save_path=None,
    save_dpi=300,
):
    """
    Plot qwPCRPS skill score curves over quantiles for several lead times.

    Parameters
    ----------
    ground_truth : {"era5", "ifs_analysis"}
        Ground truth dataset.
    skill_reference : {"pcrps0", "climatology"}
        Denominator used in the skill score:
        - "pcrps0": 1 - forecast / pcrps0
        - "climatology": 1 - forecast / climatology
    tail : {"upper", "lower", "both"}
        Which skill curves to plot.
        - "upper": plot only upper-tail skill (solid)
        - "lower": plot only lower-tail skill (dashed), only for MSLP and T2M
        - "both": plot upper for all variables and lower only for MSLP and T2M
    """
    if ground_truth not in {"era5", "ifs_analysis"}:
        raise ValueError("ground_truth must be 'era5' or 'ifs_analysis'.")
    if skill_reference not in {"pcrps0", "climatology"}:
        raise ValueError("skill_reference must be 'pcrps0' or 'climatology'.")
    if tail not in {"upper", "lower", "both"}:
        raise ValueError("tail must be 'upper', 'lower', or 'both'.")

    lower_tail_vars = {"mean_sea_level_pressure", "2m_temperature"}

    variables = available_base_variables(reference_ds, score="qw_pcrps")

    if tail == "lower":
        variables = [var for var in variables if var in lower_tail_vars]

    q = reference_ds["quantile"].values
    lead_times = reference_ds["prediction_timedelta"].values

    if ylim_map is None:
        if skill_reference == "pcrps0":
            ylim_map = {
                1: (0.7, 1),
                3: (0.5, 1),
                5: (0.3, 1),
                7: (0.1, 1),
                10: (0, 1),
            }
        else:
            ylim_map = {
                1: (0.65, 1),
                3: (0.40, 0.9),
                5: (0.20, 0.70),
                7: (0.0, 0.50),
                10: (-0.3, 0.25),
            }

    fig, axes = plt.subplots(
        len(lead_times),
        len(variables),
        figsize=(5 * len(variables), 4 * len(lead_times)),
        sharex=True,
        sharey=False,
    )

    if len(lead_times) == 1 and len(variables) == 1:
        axes = np.array([[axes]])
    elif len(lead_times) == 1:
        axes = np.array([axes])
    elif len(variables) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, lead_time in enumerate(lead_times):
        lead_days = int(lead_time / np.timedelta64(1, "D"))

        for j, var in enumerate(variables):
            ax = axes[i, j]

            for label, (color, ds) in forecast_map.items():
                if var == "total_precipitation_24hr" and label in {"PW-ERA5", "PW-IFS"}:
                    continue

                if tail in {"upper", "both"}:
                    upper_skill = get_upper_skill_score_data(ds, reference_ds, var, "qw_pcrps")

                    ax.plot(
                        q,
                        upper_skill.sel(prediction_timedelta=lead_time).values,
                        label=label,
                        color=color,
                        linestyle="-",
                    )

                if tail in {"lower", "both"} and var in lower_tail_vars:
                    lower_skill = get_lower_skill_score_data(ds, reference_ds, var, "qw_pcrps")

                    lower_label = label if tail == "lower" else None

                    ax.plot(
                        q,
                        lower_skill.sel(prediction_timedelta=lead_time).values,
                        label=lower_label,
                        color=color,
                        linestyle="--",
                    )

            if skill_reference == "climatology":
                ax.axhline(0, color="black", linestyle="--")

            if i == 0:
                ax.set_title(get_title(var), fontsize=title_fontsize)

            if i == len(lead_times) - 1:
                ax.set_xlabel("Quantile", fontsize=label_fontsize)

            if j == 0:
                if tail == "upper":
                    ylabel = "latitude-weighted upper qwPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted upper qwPCRPS skill"
                elif tail == "lower":
                    ylabel = "latitude-weighted lower qwPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted lower qwPCRPS skill"
                else:
                    ylabel = "latitude-weighted qwPCRPS-S"
                    if skill_reference == "climatology":
                        ylabel = "latitude-weighted qwPCRPS skill"

                ax.set_ylabel(
                    f"{lead_days} day\n{ylabel}",
                    fontsize=label_fontsize,
                )

            if lead_days in ylim_map:
                ax.set_ylim(*ylim_map[lead_days])

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.52, -0.035),
        fontsize=legend_fontsize,
    )

    for h in legend.legend_handles:
        if hasattr(h, "set_marker"):
            h.set_marker("")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')

    plt.close(fig)


# ---------------------------------------------------------------------
# Score illustration and threshold maps
# ---------------------------------------------------------------------

def plot_weighted_crps(points, cdf, y=298.2, threshold=298.0, quantile=0.75, save_path=None, save_dpi=300):
    """Illustrate the CRPS and its upper-tail weighted versions for a discrete CDF."""
    X = np.hstack([points[0], points])
    F = np.hstack([0, cdf])

    # Insert the observation and threshold without changing the step CDF.
    for value in np.unique([y, threshold]):
        i = np.searchsorted(X, value, side="right")
        height = F[i - 1] if i > 0 else 0.0
        X = np.insert(X, i, value)
        F = np.insert(F, i, height)

    left_height = F**2
    right_bottom = 1 - (1 - F)**2
    q_left = quantile**2
    q_right = 1 - (1 - quantile)**2

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

    for ax, title in zip(axes, ["CRPS", "qwCRPS", "twCRPS"]):
        ax.step(X, F, where="post", label=r"$\hat{F}(x)$")
        ax.axvline(y, color="red", label=r"$y$")
        ax.set_title(title, fontsize=28, pad=7)

    # CRPS: shade the squared distance on either side of the observation.
    ax = axes[0]
    ax.fill_between(X, 0, left_height, step="post", where=(X <= y), alpha=0.5, color="tab:blue", label="CRPS")
    ax.fill_between(X, right_bottom, 1, step="post", where=(X >= y), alpha=0.5, color="tab:blue")

    # qwCRPS: restrict the quantile integration to levels above quantile.
    ax = axes[1]
    ax.axhline(quantile, color="grey", linestyle="--", label=rf"$\tau={quantile}$")
    ax.axhline(q_left, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.axhline(q_right, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.fill_between(X, q_left, left_height, step="post", where=(X <= y) & (F > quantile), alpha=0.5, color="tab:blue", label="qwCRPS")
    ax.fill_between(X, np.maximum(right_bottom, q_right), 1, step="post", where=(X >= y), alpha=0.5, color="tab:blue")
    ax.text(0.03, q_left + 0.003, r"$\tau^2$", transform=ax.get_yaxis_transform(), va="bottom", fontsize=18)
    ax.text(0.03, q_right + 0.003, r"$1-(1-\tau)^2$", transform=ax.get_yaxis_transform(), va="bottom", fontsize=18)

    # twCRPS: retain only the region above the outcome threshold.
    ax = axes[2]
    ax.axvline(threshold, color="grey", linestyle="--", label=r"$t$")
    ax.fill_between(X, 0, left_height, step="post", where=(X <= y) & (X >= threshold), alpha=0.5, color="tab:blue", label="twCRPS")
    ax.fill_between(X, right_bottom, 1, step="post", where=(X >= y) & (X >= threshold), alpha=0.5, color="tab:blue")

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlabel("x", fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=20)

    axes[0].set_ylabel("CDF", fontsize=22)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")

    plt.close(fig)


def plot_record_threshold_maps(threshold_ds, save_path=None, save_dpi=150):
    """Plot historical record highs and lows for the selected weather variables."""
    panels = [
        ("mean_sea_level_pressure", "MSLP - max record", "q100", "viridis"),
        ("2m_temperature", "T2M - max record", "q100", "viridis"),
        ("10m_wind_speed", "WS10 - max record", "q100", "viridis"),
        ("mean_sea_level_pressure", "MSLP - min record", "q000", "viridis_r"),
        ("2m_temperature", "T2M - min record", "q000", "viridis_r"),
        ("total_precipitation_24hr", "TP24hr - max record", "q100", "viridis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), subplot_kw={"projection": ccrs.Robinson()}, constrained_layout=True)

    for ax, (var, title, threshold, cmap) in zip(axes.ravel(), panels):
        ax.coastlines()
        data = threshold_ds[var].sel(threshold=threshold).transpose("latitude", "longitude")

        vmax = None
        if var == "10m_wind_speed" and threshold == "q100":
            values = data.values.ravel()
            values = values[np.isfinite(values)]
            vmax = np.partition(values, -2)[-2]

        mesh = ax.pcolormesh(data["longitude"].values, data["latitude"].values, data.values, cmap=cmap, vmax=vmax, transform=ccrs.PlateCarree(), linewidth=0.5)
        ax.set_title(title, fontsize=24)

        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.75, pad=0.05)
        cbar.ax.tick_params(labelsize=12)

    for ax in axes[-1]:
        ax.set_xlabel("Longitude")

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")

    plt.close(fig)


def plot_monthly_record_exceedance_maps(max_counts, min_counts, save_path=None, save_dpi=150):
    """Plot the number of observations beyond the monthly record thresholds."""
    panels = [
        ("mean_sea_level_pressure", "MSLP - max exceedances", "max"),
        ("2m_temperature", "T2M - max exceedances", "max"),
        ("10m_wind_speed", "WS10 - max exceedances", "max"),
        ("mean_sea_level_pressure", "MSLP - min non-exceedances", "min"),
        ("2m_temperature", "T2M - min non-exceedances", "min"),
        ("total_precipitation_24hr", "TP24hr - max exceedances", "max"),
    ]

    global_max = max((max_counts if record == "max" else min_counts)[var].max().item() for var, _, record in panels)
    bounds = [1, 6, 11, 21, max(22, int(global_max) + 1)]
    labels = ["1–5", "6-10", "11–20", ">20"]

    cmap_disc = plt.get_cmap("YlOrBr_r", 4)
    norm = BoundaryNorm(bounds, cmap_disc.N)
    grey_cmap = ListedColormap(["lightgrey"])

    fig, axes = plt.subplots(2, 3, figsize=(20, 9), subplot_kw={"projection": ccrs.Robinson()}, constrained_layout=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.03, hspace=0.10)

    for ax, (var, title, record) in zip(axes.ravel(), panels):
        counts = max_counts if record == "max" else min_counts
        data = counts[var].transpose("latitude", "longitude")
        lon = data["longitude"].values
        lat = data["latitude"].values

        zeros = np.where(data.values == 0, 1.0, np.nan)
        nonzeros = np.where(data.values == 0, np.nan, data.values)

        ax.pcolormesh(lon, lat, zeros, cmap=grey_cmap, transform=ccrs.PlateCarree(), shading="auto", zorder=1)
        ax.pcolormesh(lon, lat, nonzeros, cmap=cmap_disc, norm=norm, transform=ccrs.PlateCarree(), shading="auto", zorder=2)

        coast = ax.coastlines(zorder=10)
        coast.set_path_effects([pe.Stroke(linewidth=1.25, foreground="white"), pe.Normal()])
        ax.set_title(title, fontsize=24)

    leg_ax = fig.add_axes([0.08, 0.03, 0.84, 0.06])
    leg_ax.axis("off")
    handles = [Patch(facecolor="lightgrey", edgecolor="k", label="0")]
    handles += [Patch(facecolor=cmap_disc(i), edgecolor="k", label=label) for i, label in enumerate(labels)]
    leg_ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=True, fontsize=22)

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Score illustration
    logging.info("Creating weighted_crps.pdf")
    illustration_variable = "2m_temperature"
    illustration_lead_time = np.timedelta64(3, "D")
    illustration_forecast_path = "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"
    illustration_observation_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

    with ProgressBar():
        with xr.open_zarr(illustration_forecast_path, storage_options={"token": "anon"}, decode_timedelta=True) as ds:
            illustration_forecast = ds[illustration_variable].sel(time=slice("2020-01-01", "2020-12-31"), prediction_timedelta=illustration_lead_time).sel(latitude=49.00937, longitude=8.40444, method="nearest").load()

        with xr.open_zarr(illustration_observation_path, storage_options={"token": "anon"}, decode_timedelta=True) as ds:
            illustration_obs = ds[illustration_variable].sel(time=illustration_forecast["time"] + illustration_lead_time, latitude=illustration_forecast["latitude"], longitude=illustration_forecast["longitude"]).load()

    illustration_df = illustration_forecast.to_dataframe()[[illustration_variable]]
    illustration_fit = idr(illustration_obs.values, illustration_df)
    illustration_prediction = illustration_fit.predict(illustration_df, digits=8).predictions[279]
    illustration_points = np.asarray(illustration_prediction.points)
    illustration_cdf = np.asarray(illustration_prediction.ecdf)

    plot_weighted_crps(
        illustration_points,
        illustration_cdf,
        y=298.2,
        threshold=298.0,
        quantile=0.75,
        save_path=PLOTS_DIR / "weighted_crps.pdf",
    )

    # Historical record thresholds
    logging.info("Creating q100_thresholds_maps.png")
    with ProgressBar():
        era5_record_thresholds = xr.open_zarr(f"{THRESHOLDS_DIR}/era5_quantiles_1979_2019.zarr").sel(threshold=["q000", "q100"]).load()

    plot_record_threshold_maps(era5_record_thresholds, save_path=PLOTS_DIR / "q100_thresholds_maps.png")

    # Monthly record exceedances
    logging.info("Creating monthly_record_exceedances_maps.png")
    with ProgressBar():
        era5_record_max_month = xr.open_zarr(f"{THRESHOLDS_DIR}/era5_record_max_month_1979_2019.zarr").load()
        era5_record_min_month = xr.open_zarr(f"{THRESHOLDS_DIR}/era5_record_min_month_1979_2019.zarr").load()

    record_variables = ["2m_temperature", "mean_sea_level_pressure", "10m_wind_speed", "total_precipitation_24hr"]
    era5_record_obs_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

    era5_record_obs = xr.open_zarr(era5_record_obs_path, storage_options={"token": "anon"}, decode_timedelta=True).sel(time=slice("2020-01-02", "2020-12-26"))[record_variables].chunk({"time": -1})
    era5_record_obs = era5_record_obs.sel(time=era5_record_obs["time"].dt.hour.isin([0, 12]))

    era5_max_record_counts = (era5_record_obs.groupby("time.month") > era5_record_max_month).sum(dim="time")
    era5_min_record_counts = (era5_record_obs[["2m_temperature", "mean_sea_level_pressure"]].groupby("time.month") < era5_record_min_month).sum(dim="time")

    with ProgressBar():
        era5_max_record_counts, era5_min_record_counts = dask.compute(era5_max_record_counts, era5_min_record_counts)

    plot_monthly_record_exceedance_maps(era5_max_record_counts, era5_min_record_counts, save_path=PLOTS_DIR / "monthly_record_exceedances_maps.png")

    # Apply WeatherBench2 styling to the score comparison figures.
    set_wb2_style()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Score datasets and spatial averages
    era5_pcrps0 = xr.open_zarr(f"{RESULTS_DIR}/pcrps0/era5_pcrps0.zarr", decode_timedelta=True)
    ifs_analysis_pcrps0 = xr.open_zarr(f"{RESULTS_DIR}/pcrps0/ifs_analysis_pcrps0.zarr", decode_timedelta=True)

    hres_vs_era5_pcrps = xr.open_zarr(f"{RESULTS_DIR}/hres_240x121_vs_era5.zarr", decode_timedelta=True)[list(era5_pcrps0.data_vars)]
    pangu_vs_era5_pcrps = xr.open_zarr(f"{RESULTS_DIR}/pangu_240x121_vs_era5.zarr", decode_timedelta=True)[list(ifs_analysis_pcrps0.data_vars)]
    graphcast_vs_era5_pcrps = xr.open_zarr(f"{RESULTS_DIR}/graphcast_240x121_vs_era5.zarr", decode_timedelta=True)[list(era5_pcrps0.data_vars)]
    fuxi_vs_era5_pcrps = xr.open_zarr(f"{RESULTS_DIR}/fuxi_240x121_vs_era5.zarr", decode_timedelta=True)[list(era5_pcrps0.data_vars)]
    era5_climatology_vs_era5_pcrps = xr.open_zarr(f"{RESULTS_DIR}/era5_climatology_240x121_vs_era5.zarr", decode_timedelta=True)[list(era5_pcrps0.data_vars)]

    hres_vs_ifs_analysis_pcrps = xr.open_zarr(f"{RESULTS_DIR}/hres_240x121_vs_ifs_analysis.zarr", decode_timedelta=True)[list(ifs_analysis_pcrps0.data_vars)]
    pangu_operational_vs_ifs_analysis_pcrps = xr.open_zarr(f"{RESULTS_DIR}/pangu_operational_240x121_vs_ifs_analysis.zarr", decode_timedelta=True)[list(ifs_analysis_pcrps0.data_vars)]
    graphcast_operational_vs_ifs_analysis_pcrps = xr.open_zarr(f"{RESULTS_DIR}/graphcast_operational_240x121_vs_ifs_analysis.zarr", decode_timedelta=True)[list(ifs_analysis_pcrps0.data_vars)]

    logging.info('Computing latitude-weighted spatial averages.')
    with ProgressBar():
        era5_pcrps0_spatial_avg = _spatial_average(era5_pcrps0, region=None, skipna=False).load()
        ifs_analysis_pcrps0_spatial_avg = _spatial_average(ifs_analysis_pcrps0, region=None, skipna=False).load()

        hres_vs_era5_pcrps_spatial_avg = _spatial_average(hres_vs_era5_pcrps, region=None, skipna=False).load()
        pangu_vs_era5_pcrps_spatial_avg = _spatial_average(pangu_vs_era5_pcrps, region=None, skipna=False).load()
        graphcast_vs_era5_pcrps_spatial_avg = _spatial_average(graphcast_vs_era5_pcrps, region=None, skipna=False).load()
        fuxi_vs_era5_pcrps_spatial_avg = _spatial_average(fuxi_vs_era5_pcrps, region=None, skipna=False).load()
        era5_climatology_vs_era5_pcrps_spatial_avg = _spatial_average(era5_climatology_vs_era5_pcrps, region=None, skipna=False).load()

        hres_vs_ifs_analysis_pcrps_spatial_avg = _spatial_average(hres_vs_ifs_analysis_pcrps, region=None, skipna=False).load()
        pangu_operational_vs_ifs_analysis_pcrps_spatial_avg = _spatial_average(pangu_operational_vs_ifs_analysis_pcrps, region=None, skipna=False).load()
        graphcast_operational_vs_ifs_analysis_pcrps_spatial_avg = _spatial_average(graphcast_operational_vs_ifs_analysis_pcrps, region=None, skipna=False).load()

    # Load data for the maps.
    reduced_thresholds = ['q099', 'monthly_max']
    reduced_quantiles = [0.99]

    logging.info('Loading map scores and pairwise p-values.')
    with ProgressBar():
        era5_pcrps0_maps = era5_pcrps0.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()
        hres_vs_era5_pcrps_maps = hres_vs_era5_pcrps.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()
        pangu_vs_era5_pcrps_maps = pangu_vs_era5_pcrps.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()
        graphcast_vs_era5_pcrps_maps = graphcast_vs_era5_pcrps.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()
        fuxi_vs_era5_pcrps_maps = fuxi_vs_era5_pcrps.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()
        era5_climatology_vs_era5_pcrps_maps = era5_climatology_vs_era5_pcrps.sel(quantile=reduced_quantiles, threshold=reduced_thresholds).load()

        fuxi_graphcast_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/fuxi_graphcast_vs_era5_p.zarr", decode_timedelta=True).load()
        fuxi_pangu_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/fuxi_pangu_vs_era5_p.zarr", decode_timedelta=True).load()
        fuxi_hres_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/fuxi_hres_vs_era5_p.zarr", decode_timedelta=True).load()
        graphcast_pangu_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/graphcast_pangu_vs_era5_p.zarr", decode_timedelta=True).load()
        graphcast_hres_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/graphcast_hres_vs_era5_p.zarr", decode_timedelta=True).load()
        pangu_hres_vs_era5_p = xr.open_zarr(f"{RESULTS_DIR}/tests/pangu_hres_vs_era5_p.zarr", decode_timedelta=True).load()

    pairwise_p_values_era5 = {
        ('FuXi', 'GC-ERA5'): fuxi_graphcast_vs_era5_p,
        ('FuXi', 'PW-ERA5'): fuxi_pangu_vs_era5_p,
        ('FuXi', 'HRES'): fuxi_hres_vs_era5_p,
        ('GC-ERA5', 'PW-ERA5'): graphcast_pangu_vs_era5_p,
        ('GC-ERA5', 'HRES'): graphcast_hres_vs_era5_p,
        ('PW-ERA5', 'HRES'): pangu_hres_vs_era5_p,
    }

    # Scatterplot datasets
    scatter_thresholds = ['q000', 'q001', 'q099', 'q100', 'monthly_min', 'monthly_max']
    scatter_variables = [
        '10m_wind_speed_pcrps', '10m_wind_speed_tw_pcrps',
        '2m_temperature_pcrps', '2m_temperature_tw_pcrps',
        'mean_sea_level_pressure_pcrps', 'mean_sea_level_pressure_tw_pcrps',
        'total_precipitation_24hr_pcrps', 'total_precipitation_24hr_tw_pcrps',
    ]

    ensemble_dir = f"{RESULTS_DIR}/ensemble"
    logging.info('Loading deterministic and ensemble scatterplot scores.')
    with ProgressBar():
        ifs_ens_vs_era5_crps = xr.open_zarr(f"{ensemble_dir}/ifs_ens_vs_era5_crps.zarr", decode_timedelta=True).load()
        gencast_vs_era5_crps = xr.open_zarr(f"{ensemble_dir}/gencast_vs_era5_crps.zarr", decode_timedelta=True).load()
        ifs_ens_vs_era5_tw_crps = xr.open_zarr(f"{ensemble_dir}/ifs_ens_vs_era5_tw_crps.zarr", decode_timedelta=True).sel(threshold=scatter_thresholds).load()
        gencast_vs_era5_tw_crps = xr.open_zarr(f"{ensemble_dir}/gencast_vs_era5_tw_crps.zarr", decode_timedelta=True).sel(threshold=scatter_thresholds).load()
        hres_vs_era5_scatter = hres_vs_era5_pcrps[scatter_variables].sel(threshold=scatter_thresholds).load()
        graphcast_vs_era5_scatter = graphcast_vs_era5_pcrps[scatter_variables].sel(threshold=scatter_thresholds).load()

    era5_forecast_map = {
        "HRES": ("tab:blue", hres_vs_era5_pcrps_spatial_avg),
        "PW-ERA5": ("tab:orange", pangu_vs_era5_pcrps_spatial_avg),
        "GC-ERA5": ("tab:green", graphcast_vs_era5_pcrps_spatial_avg),
        "FuXi": ("tab:pink", fuxi_vs_era5_pcrps_spatial_avg),
    }
    ifs_forecast_map = {
        "HRES": ("tab:blue", hres_vs_ifs_analysis_pcrps_spatial_avg),
        "PW-IFS": ("tab:orange", pangu_operational_vs_ifs_analysis_pcrps_spatial_avg),
        "GC-IFS": ("tab:green", graphcast_operational_vs_ifs_analysis_pcrps_spatial_avg),
    }

    # Main paper
    logging.info("Saving tw_pcrpss_vs_era5.pdf")
    plot_tw_pcrps_skill_by_threshold(
        forecast_map=era5_forecast_map,
        reference_ds=era5_pcrps0_spatial_avg,
        ground_truth="era5",
        skill_reference="pcrps0",
        tail='both',
        save_path=PLOTS_DIR / "tw_pcrpss_vs_era5.pdf",
    )

    logging.info("Saving tw_pcrps_monthly_records_vs_era5.pdf")
    plot_tw_pcrps_panels(
        dataset_map=era5_forecast_map,
        threshold="monthly_max",
        ground_truth="era5",
        save_path=PLOTS_DIR / "tw_pcrps_monthly_records_vs_era5.pdf",
    )

    logging.info("Saving maps_fuxi_tw_pcrpss_q099_vs_era5.png")
    plot_skill_score_maps(
        forecast_ds=fuxi_vs_era5_pcrps_maps,
        reference_ds=era5_pcrps0_maps,
        score="tw_pcrps",
        threshold="q099",
        variables=["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"],
        vmin=0,
        mask_below_vmin=True,
        save_path=PLOTS_DIR / "maps_fuxi_tw_pcrpss_q099_vs_era5.png",
    )

    logging.info("Saving maps_best_model_tw_pcrps_q099_vs_era5.png")
    plot_best_model_maps(
        model_datasets={
            "HRES": hres_vs_era5_pcrps_maps,
            "PW-ERA5": pangu_vs_era5_pcrps_maps,
            "GC-ERA5": graphcast_vs_era5_pcrps_maps,
            "FuXi": fuxi_vs_era5_pcrps_maps,
        },
        score="tw_pcrps",
        threshold="q099",
        save_path=PLOTS_DIR / "maps_best_model_tw_pcrps_q099_vs_era5.png",
    )

    logging.info("Saving scatterplot_aiwp_tw_crps_vs_era5.png")
    plot_tw_scatter_grid(
        det_ds=graphcast_vs_era5_scatter,
        ens_crps_ds=gencast_vs_era5_crps,
        ens_tw_ds=gencast_vs_era5_tw_crps,
        thresholds=["q099", "monthly_max", "q100"],
        ground_truth="era5",
        comparison="gencast",
        save_path=PLOTS_DIR / "scatterplot_aiwp_tw_crps_vs_era5.png",
    )

    # Appendix
    logging.info("Saving maps_best_model_tw_pcrps_monthly_max_vs_era5.png")
    plot_best_model_maps(
        model_datasets={
            "HRES": hres_vs_era5_pcrps_maps,
            "PW-ERA5": pangu_vs_era5_pcrps_maps,
            "GC-ERA5": graphcast_vs_era5_pcrps_maps,
            "FuXi": fuxi_vs_era5_pcrps_maps,
        },
        score="tw_pcrps",
        threshold="monthly_max",
        save_path=PLOTS_DIR / "maps_best_model_tw_pcrps_monthly_max_vs_era5.png",
    )

    logging.info("Saving maps_fuxi_tw_pcrps_skill_q099_vs_era5.png")
    plot_skill_score_maps(
        forecast_ds=fuxi_vs_era5_pcrps_maps,
        reference_ds=era5_climatology_vs_era5_pcrps_maps,
        score="tw_pcrps",
        threshold="q099",
        variables=["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"],
        add_zero_contour=True,
        vmin=-1,
        save_path=PLOTS_DIR / "maps_fuxi_tw_pcrps_skill_q099_vs_era5.png",
    )

    logging.info("Saving maps_sig_best_model_ai_tw_pcrps_monthly_max_vs_era5.png")
    plot_significant_best_model_maps(
        pairwise_p_values=pairwise_p_values_era5,
        model_order=["HRES", "PW-ERA5", "GC-ERA5", "FuXi"],
        score="tw_pcrps",
        threshold="monthly_max",
        include_ai_better=True,
        legend_fontsize=21,
        save_path=PLOTS_DIR / "maps_sig_best_model_ai_tw_pcrps_monthly_max_vs_era5.png",
    )

    logging.info("Saving tw_pcrps_q100_vs_era5.pdf")
    plot_tw_pcrps_panels(
        dataset_map=era5_forecast_map,
        threshold="q100",
        ground_truth="era5",
        save_path=PLOTS_DIR / "tw_pcrps_q100_vs_era5.pdf",
    )

    logging.info("Saving tw_pcrps_q099_vs_era5.pdf")
    plot_tw_pcrps_panels(
        dataset_map=era5_forecast_map,
        threshold="q099",
        ground_truth="era5",
        save_path=PLOTS_DIR / "tw_pcrps_q099_vs_era5.pdf",
    )

    logging.info("Saving tw_pcrps_monthly_records_operational.pdf")
    plot_tw_pcrps_panels(
        dataset_map=ifs_forecast_map,
        threshold="monthly_max",
        ground_truth="ifs_analysis",
        save_path=PLOTS_DIR / "tw_pcrps_monthly_records_operational.pdf",
    )

    logging.info("Saving scatterplot_nwp_tw_crps_vs_era5.png")
    plot_tw_scatter_grid(
        det_ds=hres_vs_era5_scatter,
        ens_crps_ds=ifs_ens_vs_era5_crps,
        ens_tw_ds=ifs_ens_vs_era5_tw_crps,
        thresholds=["q099", "monthly_max", "q100"],
        ground_truth="era5",
        comparison="ifs",
        save_path=PLOTS_DIR / "scatterplot_nwp_tw_crps_vs_era5.png",
    )

    # Supplement
    logging.info("Saving qw_pcrpss_vs_era5.pdf")
    plot_qw_pcrps_skill_by_quantile(
        forecast_map=era5_forecast_map,
        reference_ds=era5_pcrps0_spatial_avg,
        ground_truth="era5",
        skill_reference="pcrps0",
        tail="both",
        save_path=PLOTS_DIR / "qw_pcrpss_vs_era5.pdf",
    )

    logging.info("Saving qw_pcrps_99_vs_era5.pdf")
    plot_qw_pcrps_panels(
        dataset_map=era5_forecast_map,
        quantile=0.99,
        ground_truth="era5",
        include_climatology=False,
        save_path=PLOTS_DIR / "qw_pcrps_99_vs_era5.pdf",
    )

    logging.info("Saving maps_fuxi_qw_pcrpss_99_vs_era5.png")
    plot_skill_score_maps(
        forecast_ds=fuxi_vs_era5_pcrps_maps,
        reference_ds=era5_pcrps0_maps,
        score="qw_pcrps",
        quantile=0.99,
        variables=["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"],
        save_path=PLOTS_DIR / "maps_fuxi_qw_pcrpss_99_vs_era5.png",
    )

    logging.info("Saving maps_fuxi_qw_pcrps_skill_99_vs_era5.png")
    plot_skill_score_maps(
        forecast_ds=fuxi_vs_era5_pcrps_maps,
        reference_ds=era5_climatology_vs_era5_pcrps_maps,
        score="qw_pcrps",
        quantile=0.99,
        variables=["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed", "total_precipitation_24hr"],
        add_zero_contour=True,
        vmin=-1,
        save_path=PLOTS_DIR / "maps_fuxi_qw_pcrps_skill_99_vs_era5.png",
    )

    logging.info("Saving maps_best_model_qw_pcrps_99_vs_era5.png")
    plot_best_model_maps(
        model_datasets={
            "HRES": hres_vs_era5_pcrps_maps,
            "PW-ERA5": pangu_vs_era5_pcrps_maps,
            "GC-ERA5": graphcast_vs_era5_pcrps_maps,
            "FuXi": fuxi_vs_era5_pcrps_maps,
        },
        score="qw_pcrps",
        quantile=0.99,
        save_path=PLOTS_DIR / "maps_best_model_qw_pcrps_99_vs_era5.png",
    )

    logging.info("Saving maps_sig_best_model_ai_qw_pcrps_monthly_max_vs_era5.png")
    plot_significant_best_model_maps(
        pairwise_p_values=pairwise_p_values_era5,
        model_order=["HRES", "PW-ERA5", "GC-ERA5", "FuXi"],
        score="qw_pcrps",
        quantile=0.99,
        include_ai_better=True,
        legend_fontsize=21,
        save_path=PLOTS_DIR / "maps_sig_best_model_ai_qw_pcrps_monthly_max_vs_era5.png",
    )

    logging.info('Finished. Figures saved to %s', PLOTS_DIR)


if __name__ == '__main__':
    main()
