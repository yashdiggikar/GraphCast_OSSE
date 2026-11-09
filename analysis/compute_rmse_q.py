"""Global (cosine-latitude weighted) RMSE of specific humidity (q) for a
GraphCast full-res 60-day GC-DETER-2023 rollout vs ERA5 truth.

Q analysis did not exist for any experiment in this study before this
script -- unlike temperature/u/v/geopotential, ERA5 `q` truth was only
downloaded at 4 pressure levels (1000/850/500/300 hPa, not the standard
13-level set), so this is a standalone script rather than an extra
VAR_MAP entry in compute_rmse.py (whose LEVELS list assumes the full
13-level truth is available for every variable).

Same streaming/cosine-weighting approach as compute_rmse.py.

Usage:
    python compute_rmse_q.py \
        --pred_dir control/full_predictions/rollout_60day_2023-02-01 \
        --era5_q_file era5_feb2023/era5_q_2023_febapr.grib \
        --sample_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
        --out_csv control/results/csv/rmse_q_2023-02-01.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray

LEVELS = [300, 500, 850, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_q_file", required=True)
    p.add_argument("--sample_nc", required=True,
                    help="Original sample file used as IC, to recover the reference datetime")
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    ic_ds = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()
    print("IC reference (last input) datetime:", ic_last_datetime, flush=True)

    print("ERA5 q truth file:", args.era5_q_file, flush=True)
    truth_ds = xarray.open_dataset(args.era5_q_file, engine="cfgrib", backend_kwargs={"indexpath": ""})

    cos_w_2d = None
    total_w = None

    def get_truth_slice(valid_times):
        return truth_ds["q"].sel(time=valid_times, isobaricInhPa=LEVELS).load()

    rows = []
    chunk_paths = sorted(glob.glob(os.path.join(args.pred_dir, "chunk_*.nc")))
    print(f"Found {len(chunk_paths)} prediction chunks", flush=True)

    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values
        valid_times = (ic_last_datetime + lead_times)

        if cos_w_2d is None:
            pred_lat = pred.coords["lat"].values.astype(np.float64)
            cos_w_1d = np.cos(np.deg2rad(pred_lat))
            cos_w_2d = cos_w_1d[:, None] * np.ones((1, pred.sizes["lon"]))
            total_w = cos_w_2d.sum()

        truth = get_truth_slice(valid_times)
        truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon",
                                                    "isobaricInhPa": "level"})
        truth = truth.assign_coords(lat=pred.coords["lat"].values,
                                      lon=pred.coords["lon"].values)

        pred_var = pred["specific_humidity"]
        if "batch" in pred_var.dims:
            pred_var = pred_var.squeeze("batch", drop=True)
        pred_var = pred_var.sel(level=LEVELS)

        for ti, (lt, vt) in enumerate(zip(lead_times, valid_times)):
            pred_t = pred_var.isel(time=ti)
            truth_t = truth.isel(time=ti)
            diff2 = (pred_t.values - truth_t.values) ** 2
            rmse_per_level = np.sqrt(
                (diff2 * cos_w_2d[None, :, :]).sum(axis=(1, 2)) / total_w
            )
            lead_hours = lt / np.timedelta64(1, "h")
            for lvl, r in zip(LEVELS, rmse_per_level):
                rows.append({
                    "variable": "specific_humidity",
                    "level_hpa": lvl,
                    "lead_hours": float(lead_hours),
                    "lead_days": float(lead_hours) / 24.0,
                    "valid_time": str(vt),
                    "rmse": float(r),
                })

        elapsed_steps = (ci + 1) * pred.sizes["time"]
        print(f"chunk {ci+1}/{len(chunk_paths)} done (through step "
              f"{elapsed_steps}, {elapsed_steps*6/24:.1f}d) -- {len(rows)} rows so far",
              flush=True)
        pred.close()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
