"""Full global spatial bias maps (mean(pred - truth) over the mature/post-
injection phase, lead_days >= 30) for a GraphCast full-model 60-day
deterministic rollout vs ERA5 truth, at 3 pressure levels (300/500/1000 hPa)
for T/U/V/Z. Saved as .npy arrays, one file per variable/level, at
GraphCast-full's own native 0.25deg grid (721x1440) -- no downscaling,
unlike the older GraphCast-Small reference study.

Same mature-phase convention (bias = time-mean forecast-minus-truth over the
post-injection half of the run) as the reference bias study
(Organized_results/GraphCast/bias/scripts/section_gc_bias_analysis.py,
Organized_results/Prithvi/bias/scripts/section_5_8_prithvi_bias_analysis.py):
here lead_days >= 30 matches this study's --inject_from_step 120 (0-based)
truth-insertion start used throughout compute_rmse.py / the regime analyses.

Same streaming chunk/ERA5-loading approach as compute_rmse.py.

Usage:
    python compute_bias_maps.py \
        --pred_dir control/full_predictions/rollout_60day_2023-02-01 \
        --era5_dir era5_feb2023 \
        --sample_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
        --out_dir bias_maps/control \
        --mature_start_day 30.0
"""
import argparse
import glob
import os

import numpy as np
import xarray

VAR_MAP = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
}
LEVELS = [300, 500, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_dir", required=True)
    p.add_argument("--sample_nc", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mature_start_day", type=float, default=30.0,
                    help="Lead time (days) the mature/post-injection window starts at.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ic_ds = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()
    print("IC reference (last input) datetime:", ic_last_datetime, flush=True)

    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "era5_pressure_2023_*.grib")))
    print("ERA5 truth files:", era5_files, flush=True)
    truth_by_month = {}
    for f in era5_files:
        ds = xarray.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
        truth_by_month[f] = ds

    def get_truth_slice(var_era5, valid_times):
        for f, ds in truth_by_month.items():
            tmin, tmax = ds.coords["time"].min().values, ds.coords["time"].max().values
            if all((vt >= tmin) and (vt <= tmax) for vt in valid_times):
                return ds[var_era5].sel(time=valid_times, isobaricInhPa=LEVELS).load()
        slices = []
        for vt in valid_times:
            for f, ds in truth_by_month.items():
                tmin, tmax = ds.coords["time"].min().values, ds.coords["time"].max().values
                if tmin <= vt <= tmax:
                    slices.append(ds[var_era5].sel(time=vt, isobaricInhPa=LEVELS).load())
                    break
        return xarray.concat(slices, dim="time")

    sum_diff = {}   # (var, lvl) -> (lat, lon) float64 accumulator
    count = {}      # (var, lvl) -> int
    lat_out = lon_out = None

    chunk_paths = sorted(glob.glob(os.path.join(args.pred_dir, "chunk_*.nc")))
    print(f"Found {len(chunk_paths)} prediction chunks", flush=True)

    n_mature_total = 0
    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values
        valid_times = ic_last_datetime + lead_times
        lead_days = lead_times / np.timedelta64(1, "D")
        mature_mask = lead_days >= args.mature_start_day

        if not np.any(mature_mask):
            pred.close()
            continue

        if lat_out is None:
            lat_out = pred.coords["lat"].values.astype(np.float64)
            lon_out = pred.coords["lon"].values.astype(np.float64)
            for var_gc in VAR_MAP:
                for lvl in LEVELS:
                    sum_diff[(var_gc, lvl)] = np.zeros((len(lat_out), len(lon_out)), dtype=np.float64)
                    count[(var_gc, lvl)] = 0

        mature_idx = np.nonzero(mature_mask)[0]

        for var_gc, var_era5 in VAR_MAP.items():
            truth = get_truth_slice(var_era5, valid_times[mature_idx])
            truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon",
                                                        "isobaricInhPa": "level"})
            truth = truth.assign_coords(lat=pred.coords["lat"].values,
                                          lon=pred.coords["lon"].values)

            pred_var = pred[var_gc]
            if "batch" in pred_var.dims:
                pred_var = pred_var.squeeze("batch", drop=True)
            pred_var = pred_var.sel(level=LEVELS).isel(time=mature_idx)

            diff = pred_var.values - truth.values   # (n_mature_here, level, lat, lon)
            for li, lvl in enumerate(LEVELS):
                sum_diff[(var_gc, lvl)] += diff[:, li, :, :].sum(axis=0)
                count[(var_gc, lvl)] += diff.shape[0]

        n_mature_total += len(mature_idx)
        print(f"chunk {ci+1}/{len(chunk_paths)}: {len(mature_idx)} mature steps this chunk "
              f"({n_mature_total} total so far)", flush=True)
        pred.close()

    np.save(os.path.join(args.out_dir, "lat.npy"), lat_out)
    np.save(os.path.join(args.out_dir, "lon.npy"), lon_out)

    for (var_gc, lvl), s in sum_diff.items():
        n = count[(var_gc, lvl)]
        bias = (s / n) if n > 0 else np.full_like(s, np.nan)
        fname = f"bias_{var_gc}_{lvl}hpa.npy"
        np.save(os.path.join(args.out_dir, fname), bias)
        print(f"Saved {fname}: mean={np.nanmean(bias):.4f}  "
              f"MAB={np.nanmean(np.abs(bias)):.4f}  (n={n} mature steps)", flush=True)

    print(f"Done. {n_mature_total} mature (lead_days>={args.mature_start_day}) "
          f"timesteps used. Output: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
