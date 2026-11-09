"""Cloud-regime-stratified (4 ERA5 TCC-based regimes: Clear/Cirrus/Mixed/Opaque)
cosine-latitude weighted RMSE of a GraphCast full-model 60-day rollout vs
ERA5 truth, restricted to 3 pressure levels (300/500/1000 hPa), for all 4
tracked variables (temperature, u/v wind, geopotential).

Same regime definition as the reference cross-model study
(Organized_results/common/tcc/fig5_75_76_temp_ins_cloud_regime.py::
classify_regime): Clear TCC<10%, Cirrus 10-40%, Mixed 40-70%, Opaque >=70%.

UNLIKE terrain (compute_terrain_rmse.py, a static mask reused every step),
cloud regime is TIME-VARYING -- the regime map is reclassified at every
6-hourly step from the matching ERA5 total-cloud-cover snapshot. Truth
source: /umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/
era5_total_cloud_cover_2022_jan_feb_6hourly.nc (despite the filename, this
file actually covers 2022-01-01 through 2022-03-31, confirmed directly --
comfortably spans this project's full 60-day Jan-Mar 2022 rollout window).

Same streaming chunk/ERA5-pressure-loading approach as compute_rmse.py /
compute_latslice_rmse.py; the TCC file is loaded once in full (721x1440x360
float32, ~1.5GB) since it's needed at essentially every step.

Usage:
    python compute_cloud_rmse.py \
        --pred_dir control/out/rollout_60day_2022-01-01 \
        --era5_dir /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 \
        --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --tcc_nc /umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2022_jan_feb_6hourly.nc \
        --out_csv cloud_analysis/control/rmse_cloud_control.csv
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
REGIME_NAMES = ["Clear", "Cirrus", "Mixed", "Opaque"]
REGIME_BOUNDS = [(0.0, 0.10), (0.10, 0.40), (0.40, 0.70), (0.70, 1.01)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_dir", required=True)
    p.add_argument("--sample_nc", required=True)
    p.add_argument("--tcc_nc", required=True)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def classify_regime(tcc_2d):
    """tcc_2d: (lat, lon) fractional [0,1]. Returns int8 class map 0-3."""
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
    tcc_data = tcc_ds["tcc"].values  # (time, lat_desc, lon), descending lat
    tcc_times = tcc_ds["valid_time"].values  # datetime64[ns]
    tcc_lat_desc = tcc_ds["latitude"].values
    tcc_lon = tcc_ds["longitude"].values
    tcc_ds.close()
    print(f"  TCC shape: {tcc_data.shape}, time range {tcc_times.min()} to {tcc_times.max()}",
          flush=True)
    # Sort to ascending lat once, matching the prediction grid's convention.
    lat_sort_idx = np.argsort(tcc_lat_desc)
    tcc_lat_asc = tcc_lat_desc[lat_sort_idx]
    tcc_data = tcc_data[:, lat_sort_idx, :]

    def nearest_tcc_index(vt):
        return int(np.argmin(np.abs(tcc_times - vt)))

    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "era5_pressure_2023_*.grib")))
    print("ERA5 truth files:", era5_files, flush=True)
    truth_by_month = {}
    for f in era5_files:
        ds = xarray.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
        truth_by_month[f] = ds

    cos_w_2d = None

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

        for var_gc, var_era5 in VAR_MAP.items():
            truth = get_truth_slice(var_era5, valid_times)
            truth = truth.sortby("latitude").rename({"latitude": "lat", "longitude": "lon",
                                                        "isobaricInhPa": "level"})
            truth = truth.assign_coords(lat=pred.coords["lat"].values,
                                          lon=pred.coords["lon"].values)

            pred_var = pred[var_gc]
            if "batch" in pred_var.dims:
                pred_var = pred_var.squeeze("batch", drop=True)
            pred_var = pred_var.sel(level=LEVELS)

            for ti, (lt, vt) in enumerate(zip(lead_times, valid_times)):
                pred_t = pred_var.isel(time=ti)
                truth_t = truth.isel(time=ti)
                diff2 = (pred_t.values - truth_t.values) ** 2  # (level, lat, lon)
                lead_hours = lt / np.timedelta64(1, "h")

                tcc_idx = nearest_tcc_index(vt)
                regime_map = classify_regime(tcc_data[tcc_idx])  # (lat, lon)

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
                            "variable": var_gc,
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
