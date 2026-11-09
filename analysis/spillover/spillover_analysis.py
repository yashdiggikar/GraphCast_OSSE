#!/usr/bin/env python3
"""
Spillover-distance decay analysis for GC-DETER-2023's land_temp_ins /
ocean_temp_ins truth-insertion experiments (GraphCast full, 0.25deg, 2023-02-01 IC).

Ports the methodology from /umbc/rs/pi_halem/users/yashd1/spillover_analysis.py
("Spillover-distance decay", AMS paper Appendix A.1) onto this study's own
land/ocean-masked temperature-insertion experiments. Same core method,
adapted to a single model (GraphCast full) and this study's own grid/paths:

WHAT THIS SHOWS
----------------
For land_temp_ins and ocean_temp_ins, quantifies how the forecast-skill
benefit of inserting ERA5 T truth into one geographic region (land or ocean)
"spills over" into the complementary, NOT-inserted region, as a function of
great-circle distance from the insertion boundary.

HOW (identical to the reference method)
-----------------------------------------
1. Land mask: this study's own IC file's `land_sea_mask` field
   (dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc),
   thresholded > 0.5 -- the EXACT mask used when the land_temp_ins/
   ocean_temp_ins experiments were actually run (--insert_mask_var land
   --land_threshold 0.5), not an independently-derived mask. Native
   721x1440 grid, no crop needed (GraphCast full includes both poles).
2. Great-circle distance field: every grid cell's (lat,lon) -> unit-sphere
   Cartesian coords; a cKDTree over the SOURCE (inserted) class's cells;
   every cell queried for nearest-source chord distance, converted to
   great-circle km via d = 2*R_Earth*arcsin(chord/2).
3. Evaluation restricted to the NOT-inserted class only (ocean cells for
   land_temp_ins, land cells for ocean_temp_ins).
4. Same 15 distance bins as the reference (km): 0,100,200,350,500,700,900,
   1200,1500,1900,2400,3000,4000,6000,10000,20100.
5. T500 RMSE per bin: cosine-latitude-weighted squared error vs. ERA5 truth,
   accumulated jointly over space AND the last-20-day mature window
   (days 41-60 of the 60-day rollout -- this study's own insertion starts
   day 30.25/step 121, matching the reference's "last validated 20-day
   window" convention), RMSE = sqrt(sum(weighted SE) / sum(weight)) per bin.
   Computed independently for control and for the experiment, both at their
   native 6-hourly cadence against the matching ERA5 truth timestep.
6. % improvement per bin = 100 * (RMSE_control - RMSE_experiment) / RMSE_control.

Outputs (this dir):
  spillover_decay_T_LAND.csv / fig_spillover_decay_T_LAND.png    (land-inserted -> measured over ocean)
  spillover_decay_T_OCEAN.csv / fig_spillover_decay_T_OCEAN.png  (ocean-inserted -> measured over land)
  spillover_decay_methodology.txt
"""
import warnings; warnings.filterwarnings("ignore")

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

BASE = Path("/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023")
OUT_DIR = BASE / "spillover_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EARTH_R_KM = 6371.0
LEVEL_HPA = 500.0
MATURE_LO_DAY = 41.0
MATURE_HI_DAY = 60.0

CHUNK_DIRS = {
    "control": BASE / "control" / "full_predictions" / "rollout_60day_2023-02-01",
    "land":    BASE / "land_temp_ins" / "full_predictions" / "land_temp_ins_2023-02-01",
    "ocean":   BASE / "ocean_temp_ins" / "full_predictions" / "ocean_temp_ins_2023-02-01",
}
TRUTH_NC = BASE / "truth" / "truth_leadtime_2023-02-01IC.nc"
IC_SAMPLE_NC = BASE / "dataset_samples" / "source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc"

# ─── DISTANCE BINS (km), identical to the reference implementation ──────────
DIST_BIN_EDGES = np.array([0, 100, 200, 350, 500, 700, 900, 1200, 1500,
                            1900, 2400, 3000, 4000, 6000, 10000, 20100], dtype=float)
N_BINS = len(DIST_BIN_EDGES) - 1
BIN_CENTERS = 0.5 * (DIST_BIN_EDGES[:-1] + DIST_BIN_EDGES[1:])
BIN_CENTERS[-1] = DIST_BIN_EDGES[-2] + 1500.0


def latlon_to_xyz(lat_deg, lon_deg):
    latr = np.deg2rad(lat_deg)
    lonr = np.deg2rad(lon_deg)
    return np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)


def dist_to_nearest_source(lat1d, lon1d, source_mask):
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    x, y, z = latlon_to_xyz(lat2d, lon2d)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    src_idx = np.where(source_mask.ravel())[0]
    tree = cKDTree(pts[src_idx])
    chord, _ = tree.query(pts, k=1)
    chord = np.clip(chord, 0.0, 2.0)
    theta = 2.0 * np.arcsin(chord / 2.0)
    return (theta * EARTH_R_KM).reshape(source_mask.shape)


def load_grid_and_land_mask():
    ds = xr.open_dataset(IC_SAMPLE_NC, decode_timedelta=True)
    lat = ds.coords["lat"].values.astype(np.float64)
    lon = ds.coords["lon"].values.astype(np.float64)
    lsm = ds["land_sea_mask"].values
    if lsm.ndim > 2:
        lsm = lsm.reshape(lsm.shape[-2], lsm.shape[-1])
    land_mask = lsm > 0.5
    ds.close()
    return lat, lon, land_mask


def load_truth_t500():
    """Returns (lead_days 1D array, T500 (time,lat,lon) array) for the whole
    60-day truth file, level=500hPa only (avoids loading all 37 levels)."""
    ds = xr.open_dataset(TRUTH_NC, decode_timedelta=True)
    lead_ns = ds.coords["time"].values.astype("int64")
    lead_days = lead_ns / 1e9 / 3600.0 / 24.0
    t500 = ds["temperature"].sel(level=LEVEL_HPA).load().values  # (time, lat, lon)
    if t500.ndim == 4:  # possible leading batch dim
        t500 = t500[:, 0]
    ds.close()
    return lead_days, t500


def iter_chunk_t500_mature(chunk_dir):
    """Yield (lead_days, T500 2D array) for every step in [MATURE_LO_DAY, MATURE_HI_DAY].
    Opens each chunk lazily and pulls only the temperature/level=500 slice --
    never loads the other 10 saved variables or the other 12 levels."""
    chunk_paths = sorted(glob.glob(str(chunk_dir / "chunk_*.nc")))
    for cp in chunk_paths:
        ds = xr.open_dataset(cp, decode_timedelta=True)
        lead_ns = ds.coords["time"].values.astype("int64")
        lead_days = lead_ns / 1e9 / 3600.0 / 24.0
        if lead_days.max() < MATURE_LO_DAY or lead_days.min() > MATURE_HI_DAY:
            ds.close()
            continue
        T = ds["temperature"].sel(level=LEVEL_HPA)
        if "batch" in T.dims:
            T = T.isel(batch=0)
        t500 = T.load().values  # (time, lat, lon) -- only this slice touches disk
        ds.close()
        for i, ld in enumerate(lead_days):
            if MATURE_LO_DAY <= ld <= MATURE_HI_DAY:
                yield ld, t500[i]


def accumulate_bins(chunk_dir, truth_lead_days, truth_t500, weight2d, eval_mask, dist_field):
    flat_mask = eval_mask.ravel()
    bin_idx_all = np.clip(np.digitize(dist_field.ravel(), DIST_BIN_EDGES) - 1, 0, N_BINS - 1)
    bin_idx_eval = bin_idx_all[flat_mask]
    w_eval = weight2d.ravel()[flat_mask]
    sumsq = np.zeros(N_BINS)
    sumw = np.zeros(N_BINS)
    for ld, field in iter_chunk_t500_mature(chunk_dir):
        ti = int(np.argmin(np.abs(truth_lead_days - ld)))
        truth = truth_t500[ti]
        fc = field.ravel()[flat_mask]
        tr = truth.ravel()[flat_mask]
        sq = w_eval * (fc - tr) ** 2
        sumsq += np.bincount(bin_idx_eval, weights=sq, minlength=N_BINS)
        sumw += np.bincount(bin_idx_eval, weights=w_eval, minlength=N_BINS)
    return sumsq, sumw


def bins_to_rmse(sumsq, sumw):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sumsq / sumw)


plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 13, "axes.titlesize": 15,
    "axes.labelsize": 13.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.dpi": 300, "savefig.dpi": 400, "savefig.bbox": "tight",
})


def plot_decay(bin_centers, ctrl, exp, exp_type, out_path):
    fig, ax = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_imp = 100.0 * (ctrl - exp) / ctrl
    valid = np.isfinite(pct_imp)
    ax.plot(bin_centers[valid], pct_imp[valid], marker="o", markersize=6.5, linewidth=2.2,
            color="#4C72B0", label="GraphCast (GC-DETER-2023)",
            markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color="black", lw=1.0, ls="-", alpha=0.6)
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel(f"Great-circle distance from nearest inserted "
                   f"({'land' if exp_type == 'land' else 'ocean'}) cell (km)", fontweight="bold")
    ax.set_ylabel("T500 RMSE Improvement vs. Control (%)", fontweight="bold")
    region_word = "Ocean" if exp_type == "land" else "Land"
    src_word = "Land" if exp_type == "land" else "Ocean"
    ax.set_title(f"Spillover of Temperature-Insertion Skill from {src_word} into {region_word}\n"
                 f"GC-DETER-2023, T500, Last-20-Day Average (Days 41-60) vs. Control", fontweight="bold")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.45, color="#999999")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#555555")
    ax.legend(loc="best", frameon=True, framealpha=0.92, edgecolor="#bbbbbb")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


METHODOLOGY_TEXT = """\
SPILLOVER-DISTANCE DECAY -- GC-DETER-2023 (GraphCast full, 2023-02-01 IC)
==========================================================================

Ports the method from /umbc/rs/pi_halem/users/yashd1/spillover_analysis.py
(AMS paper Appendix A.1) onto this study's own land_temp_ins/ocean_temp_ins
experiments. See that script/the paper text for the full method rationale;
summary of this port's specifics:

1. Land mask: this study's own IC file's `land_sea_mask` (721x1440, no
   crop), thresholded > 0.5 -- the SAME mask/threshold actually used when
   land_temp_ins/ocean_temp_ins were run (--insert_mask_var land
   --land_threshold 0.5).
2. Great-circle distance field via cKDTree on unit-sphere coordinates,
   identical construction to the reference.
3. Evaluation restricted to the NOT-inserted class (ocean for land_temp_ins,
   land for ocean_temp_ins).
4. Same 15 great-circle-distance bins (km) as the reference: 0,100,200,350,
   500,700,900,1200,1500,1900,2400,3000,4000,6000,10000,20100.
5. T500 (500hPa temperature) RMSE per bin: cosine-latitude-weighted squared
   error vs. ERA5 truth (truth/truth_leadtime_2023-02-01IC.nc), pooled
   jointly over space and the last-20-day mature window (days 41-60 of the
   60-day rollout -- this study's insertion starts day 30.25/step 121),
   RMSE = sqrt(sum(weighted SE)/sum(weight)) per bin. Computed independently
   for control (control/full_predictions/) and each experiment
   (land_temp_ins/, ocean_temp_ins/full_predictions/), both at native
   6-hourly cadence, matched to the nearest ERA5 truth timestep.
6. % improvement = 100 * (RMSE_control - RMSE_experiment) / RMSE_control.

Only GraphCast (this study) is included -- there is no equivalent GraphCast
run in the original 3-model (Aurora/FCNv2/Prithvi) AMS-paper spillover
analysis (GraphCast was excluded from Appendix A.1 there, "no matching
run"). This is therefore GraphCast's first spillover-decay result.
"""


def main():
    print("Loading grid/land mask + ERA5 truth (T500 only)...", flush=True)
    lat, lon, land_mask = load_grid_and_land_mask()
    weight2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    truth_lead_days, truth_t500 = load_truth_t500()
    print(f"  grid: {lat.size}x{lon.size}, land fraction {land_mask.mean()*100:.1f}%", flush=True)

    for exp_type in ["land", "ocean"]:
        print(f"-- T_{exp_type.upper()} --", flush=True)
        eval_mask = ~land_mask if exp_type == "land" else land_mask
        source_mask = land_mask if exp_type == "land" else ~land_mask
        dist_field = dist_to_nearest_source(lat, lon, source_mask)

        sumsq_c, sumw_c = accumulate_bins(CHUNK_DIRS["control"], truth_lead_days, truth_t500,
                                           weight2d, eval_mask, dist_field)
        print("  control done", flush=True)
        sumsq_e, sumw_e = accumulate_bins(CHUNK_DIRS[exp_type], truth_lead_days, truth_t500,
                                           weight2d, eval_mask, dist_field)
        print(f"  {exp_type} experiment done", flush=True)

        ctrl_rmse = bins_to_rmse(sumsq_c, sumw_c)
        exp_rmse = bins_to_rmse(sumsq_e, sumw_e)

        rows = []
        for i in range(N_BINS):
            rows.append({
                "model": "GraphCast (GC-DETER-2023)",
                "bin_low_km": DIST_BIN_EDGES[i], "bin_high_km": DIST_BIN_EDGES[i + 1],
                "bin_center_km": BIN_CENTERS[i],
                "control_rmse": ctrl_rmse[i], "experiment_rmse": exp_rmse[i],
                "pct_improvement": 100 * (ctrl_rmse[i] - exp_rmse[i]) / ctrl_rmse[i] if ctrl_rmse[i] else np.nan,
            })
        pd.DataFrame(rows).to_csv(OUT_DIR / f"spillover_decay_T_{exp_type.upper()}.csv", index=False)
        plot_decay(BIN_CENTERS, ctrl_rmse, exp_rmse, exp_type,
                   OUT_DIR / f"fig_spillover_decay_T_{exp_type.upper()}.png")

    (OUT_DIR / "spillover_decay_methodology.txt").write_text(METHODOLOGY_TEXT)
    print(f"  Saved: {OUT_DIR / 'spillover_decay_methodology.txt'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
