"""Cloud-regime-stratified (4 ERA5 TCC-based regimes: Clear/Cirrus/Mixed/Opaque)
cosine-latitude weighted RMSE of specific humidity (q) for a GraphCast
full-model GC-DETER-2023 rollout vs ERA5 truth.

Q variant of compute_cloud_rmse.py -- same regime classification/weighting
logic, but restricted to Q's own 4-level truth (300/500/850/1000 hPa, the
only levels the era5_q_2023_febapr.grib file has -- see compute_rmse_q.py)
and a single ERA5 truth file instead of a directory of monthly files.

Usage:
    python compute_cloud_rmse_q.py \
        --pred_dir control/full_predictions/rollout_60day_2023-02-01 \
        --era5_q_file era5_feb2023/era5_q_2023_febapr.grib \
        --sample_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
        --tcc_nc /umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc \
        --out_csv regime_analysis_results/cloud/csv/rmse_cloud_q_control.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray

LEVELS = [300, 500, 850, 1000]
REGIME_NAMES = ["Clear", "Cirrus", "Mixed", "Opaque"]
REGIME_BOUNDS = [(0.0, 0.10), (0.10, 0.40), (0.40, 0.70), (0.70, 1.01)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_q_file", required=True)
    p.add_argument("--sample_nc", required=True)
    p.add_argument("--tcc_nc", required=True)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def classify_regime(tcc_2d):
    reg = np.zeros(tcc_2d.shape, dtype=np.int8)
    for r, (lo, hi) in enumerate(REGIME_BOUNDS):
        reg[(tcc_2d >= lo) & (tcc_2d < hi)] = r
    return reg


def main():
    args = parse_args()

    ic_ds = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()
    print("IC reference (last input) datetime:", ic_last_datetime, flush=True)

    print("Loading ERA5 TCC file (full):", args.tcc_nc, flush=True)
    tcc_ds = xarray.open_dataset(args.tcc_nc, engine="netcdf4")
    tcc_data = tcc_ds["tcc"].values
    tcc_times = tcc_ds["valid_time"].values
    tcc_lat_desc = tcc_ds["latitude"].values
    tcc_lon = tcc_ds["longitude"].values
    tcc_ds.close()
    lat_sort_idx = np.argsort(tcc_lat_desc)
    tcc_lat_asc = tcc_lat_desc[lat_sort_idx]
    tcc_data = tcc_data[:, lat_sort_idx, :]

    def nearest_tcc_index(vt):
        return int(np.argmin(np.abs(tcc_times - vt)))

    print("ERA5 q truth file:", args.era5_q_file, flush=True)
    truth_ds = xarray.open_dataset(args.era5_q_file, engine="cfgrib", backend_kwargs={"indexpath": ""})

    def get_truth_slice(valid_times):
        return truth_ds["q"].sel(time=valid_times, isobaricInhPa=LEVELS).load()

    cos_w_2d = None
    rows = []
    chunk_paths = sorted(glob.glob(os.path.join(args.pred_dir, "chunk_*.nc")))
    print(f"Found {len(chunk_paths)} prediction chunks", flush=True)

    for ci, chunk_path in enumerate(chunk_paths):
        pred = xarray.open_dataset(chunk_path, decode_timedelta=True).load()
        lead_times = pred.coords["time"].values
        valid_times = (ic_last_datetime + lead_times)

        if cos_w_2d is None:
            pred_lat = pred.coords["lat"].values.astype(np.float64)
            nlon = pred.sizes["lon"]
            assert np.allclose(pred_lat, tcc_lat_asc), "prediction/TCC grid lat mismatch"
            assert np.allclose(pred.coords["lon"].values.astype(np.float64), tcc_lon), \
                "prediction/TCC grid lon mismatch"
            cos_w_1d = np.cos(np.deg2rad(pred_lat))
            cos_w_2d = cos_w_1d[:, None] * np.ones((1, nlon))

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
            lead_hours = lt / np.timedelta64(1, "h")

            tcc_idx = nearest_tcc_index(vt)
            regime_map = classify_regime(tcc_data[tcc_idx])

            for r, name in enumerate(REGIME_NAMES):
                rmask = (regime_map == r)
                w = cos_w_2d * rmask
                wsum = float(w.sum())
                if wsum < 1e-8:
                    rmse_per_level = np.full(len(LEVELS), np.nan)
                else:
                    rmse_per_level = np.sqrt(
                        (diff2 * w[None, :, :]).sum(axis=(1, 2)) / wsum
                    )
                for lvl, r_val in zip(LEVELS, rmse_per_level):
                    rows.append({
                        "variable": "specific_humidity",
                        "level_hpa": lvl,
                        "cloud_regime": name,
                        "lead_hours": float(lead_hours),
                        "lead_days": float(lead_hours) / 24.0,
                        "valid_time": str(vt),
                        "rmse": float(r_val),
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
