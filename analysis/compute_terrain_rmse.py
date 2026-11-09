"""Terrain-class-stratified (5 classes: Land Low/Mod/High, Ocean, Polar)
cosine-latitude weighted RMSE of a GraphCast full-model 60-day rollout vs
ERA5 truth, restricted to 3 pressure levels (300/500/1000 hPa), for all 4
tracked variables (temperature, u/v wind, geopotential).

Same terrain classification convention as the reference cross-model study
(Organized_results/common/terrain/fig_terrain_temp_ins.py::classify_terrain),
built from GraphCast's own static fields -- see build_terrain_mask.py.
Terrain is a STATIC mask, reused for every step (unlike cloud regime, which
is time-varying -- see compute_cloud_rmse.py).

Same streaming chunk/ERA5-loading approach as compute_rmse.py /
compute_latslice_rmse.py.

Usage:
    python compute_terrain_rmse.py \
        --pred_dir control/out/rollout_60day_2022-01-01 \
        --era5_dir /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 \
        --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --terrain_mask_nc terrain_mask_0.25deg.nc \
        --out_csv terrain_analysis/control/rmse_terrain_control.csv
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
TERR_NAMES = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--era5_dir", required=True)
    p.add_argument("--sample_nc", required=True)
    p.add_argument("--terrain_mask_nc", required=True)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    ic_ds = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()
    print("IC reference (last input) datetime:", ic_last_datetime, flush=True)

    terrain_ds = xarray.load_dataset(args.terrain_mask_nc)
    terrain_class = terrain_ds["terrain_class"].values  # (lat, lon), int8 0-4
    terrain_lat = terrain_ds["lat"].values
    terrain_lon = terrain_ds["lon"].values

    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "era5_pressure_2023_*.grib")))
    print("ERA5 truth files:", era5_files, flush=True)
    truth_by_month = {}
    for f in era5_files:
        ds = xarray.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
        truth_by_month[f] = ds

    class_weights = None   # {class_idx: (nlat, nlon) weight array}
    class_wsum = None

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

        if class_weights is None:
            pred_lat = pred.coords["lat"].values.astype(np.float64)
            nlon = pred.sizes["lon"]
            assert np.allclose(pred_lat, terrain_lat), "prediction/terrain grid lat mismatch"
            assert np.allclose(pred.coords["lon"].values.astype(np.float64), terrain_lon), \
                "prediction/terrain grid lon mismatch"
            cos_w_1d = np.cos(np.deg2rad(pred_lat))
            cos_w_2d = cos_w_1d[:, None] * np.ones((1, nlon))
            class_weights = {}
            class_wsum = {}
            for c, name in enumerate(TERR_NAMES):
                cmask = (terrain_class == c)
                w = cos_w_2d * cmask
                class_weights[c] = w
                class_wsum[c] = float(w.sum())
                print(f"  class {name}: {int(cmask.sum())} grid cells, weight sum {class_wsum[c]:.4f}",
                      flush=True)

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

                for c, name in enumerate(TERR_NAMES):
                    w = class_weights[c]
                    wsum = class_wsum[c]
                    rmse_per_level = np.sqrt(
                        (diff2 * w[None, :, :]).sum(axis=(1, 2)) / wsum
                    )
                    for lvl, r in zip(LEVELS, rmse_per_level):
                        rows.append({
                            "variable": var_gc,
                            "level_hpa": lvl,
                            "terrain_class": name,
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
