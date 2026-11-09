"""Latitude-SLICE (4 narrow +-5deg NH bands centred on 2N/14N/40N/60N) cosine-latitude
weighted RMSE of a GraphCast full-model 60-day rollout vs ERA5 truth, restricted to
3 pressure levels (300/500/1000 hPa).

Same slice definition and cosine-weighting convention as the existing cross-model
latitude-slice study (Organized_results/*/lat_bands/scripts/section_*_latslice_analysis.py):
each slice is a +-5deg band centred on the reference latitude, Northern Hemisphere only.

Streams through the 60 prediction chunks (each already on disk from the rollout) and,
for each chunk's valid times, pulls the matching ERA5 truth slice from the monthly
GRIB files -- never loading a full month of ERA5 into memory at once. Same approach as
compute_rmse.py, just reduced over narrow latitude bands instead of the whole globe,
and restricted to 3 levels instead of all 13 saved levels.

Usage:
    python compute_latslice_rmse.py \
        --pred_dir out/rollout_60day_2022-01-01 \
        --era5_dir /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 \
        --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --out_csv latslice_analysis/control/rmse_latslice_control.csv
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
LEVELS = [300, 500, 1000]
HALF_WIDTH = 5.0
SLICES = {
    "2N": (2.0 - HALF_WIDTH, 2.0 + HALF_WIDTH),
    "14N": (14.0 - HALF_WIDTH, 14.0 + HALF_WIDTH),
    "40N": (40.0 - HALF_WIDTH, 40.0 + HALF_WIDTH),
    "60N": (60.0 - HALF_WIDTH, 60.0 + HALF_WIDTH),
}


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

    slice_weights = None   # {slice_name: (nlat, nlon) weight array}
    slice_wsum = None      # {slice_name: scalar}

    def get_truth_slice(var_era5, valid_times):
        """Returns a (time, level, lat, lon) DataArray for the given absolute times."""
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

    rows = []
    chunk_paths = sorted(glob.glob(os.path.join(args.pred_dir, "chunk_*.nc")))
    print(f"Found {len(chunk_paths)} prediction chunks", flush=True)

    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values  # timedelta64, global lead times
        valid_times = (ic_last_datetime + lead_times)

        if slice_weights is None:
            pred_lat = pred.coords["lat"].values.astype(np.float64)
            nlon = pred.sizes["lon"]
            cos_w_1d = np.cos(np.deg2rad(pred_lat))
            cos_w_2d = cos_w_1d[:, None] * np.ones((1, nlon))
            slice_weights = {}
            slice_wsum = {}
            for sn, (lo, hi) in SLICES.items():
                lat_mask = (pred_lat >= lo) & (pred_lat < hi)
                w = cos_w_2d * lat_mask[:, None]
                slice_weights[sn] = w
                slice_wsum[sn] = float(w.sum())
                print(f"  slice {sn}: {lo:.1f} to {hi:.1f} deg, "
                      f"{int(lat_mask.sum())} lat rows, weight sum {slice_wsum[sn]:.4f}",
                      flush=True)

        for var_gc, var_era5 in VAR_MAP.items():
            truth = get_truth_slice(var_era5, valid_times)  # (time, level, lat, lon), lat descending
            truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon",
                                                        "isobaricInhPa": "level"})
            truth = truth.assign_coords(lat=pred.coords["lat"].values,
                                          lon=pred.coords["lon"].values)

            pred_var = pred[var_gc]
            if "batch" in pred_var.dims:
                pred_var = pred_var.squeeze("batch", drop=True)
            pred_var = pred_var.sel(level=LEVELS)

            for ti, (lt, vt) in enumerate(zip(lead_times, valid_times)):
                pred_t = pred_var.isel(time=ti)   # (level, lat, lon)
                truth_t = truth.isel(time=ti)     # (level, lat, lon)
                diff2 = (pred_t.values - truth_t.values) ** 2  # (level, lat, lon)
                lead_hours = lt / np.timedelta64(1, "h")

                for sn, w in slice_weights.items():
                    wsum = slice_wsum[sn]
                    rmse_per_level = np.sqrt(
                        (diff2 * w[None, :, :]).sum(axis=(1, 2)) / wsum
                    )
                    for lvl, r in zip(LEVELS, rmse_per_level):
                        rows.append({
                            "variable": var_gc,
                            "level_hpa": lvl,
                            "lat_slice": sn,
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
