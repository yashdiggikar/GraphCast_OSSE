#!/usr/bin/env python3
"""
World-map plots of GraphCast spatial bias (forecast - ERA5 truth) at 500 hPa,
averaged over lead days 40-60, control + 3 global-insertion OSSE experiments.

Reads spatial_bias/spatial_bias_day40_60.nc (from compute_spatial_bias.py).
Coastlines are drawn by contouring land_sea_mask_0.25deg.nc at the 0.5 level
(same native 0.25deg grid as the bias fields -- no cartopy needed).

Outputs (in spatial_bias/):
    spatial_bias_grid_500hPa.png                     -- 4x4 overview (rows=var, cols=dataset)
    fig_bias_<dataset>_<var>_500hPa.png  x16          -- individual full-size maps
"""
from pathlib import Path

import numpy as np
import xarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/umbc/rs/pi_halem/users/yashd1/Graphcast-full")
BIAS_NC = BASE / "spatial_bias" / "spatial_bias_day40_60.nc"
LAND_MASK_NC = BASE / "land_sea_mask_0.25deg.nc"
OUT_DIR = BASE / "spatial_bias"

DATASETS = ["control", "T-Ins", "UV-Ins", "Z-Ins"]
DATASET_LABEL = {"control": "Control", "T-Ins": "T-Insertion",
                  "UV-Ins": "UV-Insertion", "Z-Ins": "Z-Insertion"}
VARS = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential"]
VAR_LABEL = {"temperature": "T500", "u_component_of_wind": "U500",
             "v_component_of_wind": "V500", "geopotential": "Z500"}
VAR_UNITS = {"temperature": "K", "u_component_of_wind": "m/s",
             "v_component_of_wind": "m/s", "geopotential": "m$^2$/s$^2$"}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
})


def coastline_overlay(ax, land, lat, lon):
    ax.contour(lon, lat, land, levels=[0.5], colors="#333333", linewidths=0.5, zorder=5)


def make_map(ax, field, lat, lon, land, vmax, cmap, title, units, show_cbar=True):
    im = ax.pcolormesh(lon, lat, field, vmin=-vmax, vmax=vmax, cmap=cmap,
                        shading="nearest", rasterized=True)
    coastline_overlay(ax, land, lat, lon)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_aspect(1.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.026, pad=0.02)
        cb.set_label(units, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return im


def main():
    ds = xarray.open_dataset(BIAS_NC)
    lat = ds.coords["lat"].values
    lon = ds.coords["lon"].values
    level = ds.attrs.get("level_hpa", 500)
    d0, d1 = ds.attrs.get("lead_day_min", 40), ds.attrs.get("lead_day_max", 60)

    land_ds = xarray.open_dataset(LAND_MASK_NC)
    land = land_ds["land"].values

    # ─── individual maps ─────────────────────────────────────────────────
    for var in VARS:
        fields = {dsname: ds[f"{dsname}__{var}"].values for dsname in DATASETS}
        vmax = max(np.nanpercentile(np.abs(f), 99.5) for f in fields.values())
        cmap = "RdBu_r" if var != "geopotential" else "PuOr_r"

        for dsname in DATASETS:
            fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
            make_map(ax, fields[dsname], lat, lon, land, vmax, cmap,
                     f"GraphCast {DATASET_LABEL[dsname]} — {VAR_LABEL[var]} Bias "
                     f"(Forecast − ERA5), Days {d0:g}–{d1:g}",
                     VAR_UNITS[var])
            ax.set_xlabel("Longitude", fontsize=9)
            ax.set_ylabel("Latitude", fontsize=9)
            out_png = OUT_DIR / f"fig_bias_{dsname}_{VAR_LABEL[var]}.png"
            fig.savefig(out_png, facecolor="white")
            plt.close(fig)
            print(f"Saved: {out_png}")

    # ─── 4x4 overview grid (rows=var, cols=dataset) ─────────────────────
    fig, axes = plt.subplots(len(VARS), len(DATASETS),
                              figsize=(4.4 * len(DATASETS), 2.7 * len(VARS)),
                              constrained_layout=True)

    for ri, var in enumerate(VARS):
        fields = {dsname: ds[f"{dsname}__{var}"].values for dsname in DATASETS}
        vmax = max(np.nanpercentile(np.abs(f), 99.5) for f in fields.values())
        cmap = "RdBu_r" if var != "geopotential" else "PuOr_r"
        for ci, dsname in enumerate(DATASETS):
            ax = axes[ri, ci]
            title = f"{DATASET_LABEL[dsname]}" if ri == 0 else ""
            im = make_map(ax, fields[dsname], lat, lon, land, vmax, cmap, title,
                          VAR_UNITS[var], show_cbar=(ci == len(DATASETS) - 1))
            if ci == 0:
                ax.set_ylabel(f"{VAR_LABEL[var]}\nLatitude", fontsize=9, fontweight="bold")
            if ri == len(VARS) - 1:
                ax.set_xlabel("Longitude", fontsize=9)

    fig.suptitle(f"GraphCast Spatial Bias (Forecast − ERA5) at {level:g} hPa, "
                 f"Days {d0:g}–{d1:g} Mean",
                 fontsize=15, fontweight="bold")
    out_grid = OUT_DIR / "spatial_bias_grid_500hPa.png"
    fig.savefig(out_grid, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_grid}")


if __name__ == "__main__":
    main()
