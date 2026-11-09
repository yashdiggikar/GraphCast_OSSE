"""Global (cosine-latitude weighted) RMSE of the 60-day GraphCast rollout vs ERA5 truth.

Formula matches the Aurora study convention (Aurora_Pred_Files/nrmse/compute_aurora_nrmse.py):
    rmse(t, level) = sqrt( sum((pred - truth)^2 * cos(lat)) / sum(cos(lat)) )

Streams through the 60 prediction chunks (each already on disk from the rollout) and,
for each chunk's valid times, pulls the matching ERA5 truth slice from the monthly
GRIB files -- never loading a full month of ERA5 into memory at once.

Usage:
    python compute_rmse.py \
        --pred_dir out/rollout_60day_2022-01-01 \
        --era5_dir /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 \
        --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --out_csv out/rmse_60day_2022-01-01.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray

VAR_MAP = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
}
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_dir", required=True)
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

    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "era5_pressure_2023_*.grib")))
    print("ERA5 truth files:", era5_files, flush=True)
    truth_by_month = {}
    for f in era5_files:
        ds = xarray.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
        truth_by_month[f] = ds

    lat = ic_ds_lat = None
    # cos-lat weights computed once we know the prediction grid's lat.
    cos_w_2d = None
    total_w = None

    def get_truth_slice(var_era5, valid_times):
        """Returns a (time, level, lat, lon) DataArray for the given absolute times."""
        for f, ds in truth_by_month.items():
            tmin, tmax = ds.coords["time"].min().values, ds.coords["time"].max().values
            if all((vt >= tmin) and (vt <= tmax) for vt in valid_times):
                return ds[var_era5].sel(time=valid_times, isobaricInhPa=LEVELS).load()
        # Times may straddle a month boundary -- fetch each separately and concat.
        slices = []
        for vt in valid_times:
            for f, ds in truth_by_month.items():
                tmin, tmax = ds.coords["time"].min().values, ds.coords["time"].max().values
                if tmin <= vt <= tmax:
                    slices.append(ds[var_era5].sel(time=vt, isobaricInhPa=LEVELS).load())
                    break
        return xarray.concat(slices, dim="time")

    rows = []
    chunk_paths = sorted(glob.glob(os.path.join(args.pred_dir, "chunk_*.nc")))
    print(f"Found {len(chunk_paths)} prediction chunks", flush=True)

    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values  # timedelta64, global lead times
        valid_times = (ic_last_datetime + lead_times)

        if cos_w_2d is None:
            pred_lat = pred.coords["lat"].values.astype(np.float64)
            cos_w_1d = np.cos(np.deg2rad(pred_lat))
            cos_w_2d = cos_w_1d[:, None] * np.ones((1, pred.sizes["lon"]))
            total_w = cos_w_2d.sum()

        for var_gc, var_era5 in VAR_MAP.items():
            truth = get_truth_slice(var_era5, valid_times)  # (time, level, lat, lon), lat descending
            truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon",
                                                        "isobaricInhPa": "level"})
            # Align lon/lat float precision by snapping truth coords onto the prediction's.
            truth = truth.assign_coords(lat=pred.coords["lat"].values,
                                          lon=pred.coords["lon"].values)

            pred_var = pred[var_gc]
            if "batch" in pred_var.dims:
                pred_var = pred_var.squeeze("batch", drop=True)

            for ti, (lt, vt) in enumerate(zip(lead_times, valid_times)):
                pred_t = pred_var.isel(time=ti)  # (level, lat, lon)
                truth_t = truth.isel(time=ti)    # (level, lat, lon)
                diff2 = (pred_t.values - truth_t.values) ** 2  # (level, lat, lon)
                rmse_per_level = np.sqrt(
                    (diff2 * cos_w_2d[None, :, :]).sum(axis=(1, 2)) / total_w
                )
                lead_hours = lt / np.timedelta64(1, "h")
                for lvl, r in zip(LEVELS, rmse_per_level):
                    rows.append({
                        "variable": var_gc,
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
