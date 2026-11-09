"""Builds a lead-time-indexed, all-37-level ERA5 truth Dataset for the
GC-DETER-2023 study (2023-02-01 IC, 240-step/60-day, deterministic settings),
from this study's own era5_feb2023/era5_pressure_2023_febapr.grib (3-month
Feb-Apr 2023 bulk download -- covers the full 60-day lead-time window).

Mirrors ../prepare_truth.py (the original 2022-01-01/240-step version) and
../one_month_osse_2023/prepare_truth_mar2023.py's SHARED_GRIB_IDX pattern
(same index path build_ic_feb2023.py uses, so whichever of the two scripts
runs first builds the cfgrib index and the other reuses it -- worth doing
here since the grib is ~3x one_month_osse_2023's).

Usage:
    python prepare_truth_feb2023.py \
        --era5_37level_grib era5_feb2023/era5_pressure_2023_febapr.grib \
        --sample_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
        --num_steps 240 \
        --out_nc truth/truth_leadtime_2023-02-01IC.nc
"""
import argparse

import numpy as np
import xarray

VAR_MAP = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}

SHARED_GRIB_IDX = (
    "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/"
    "era5_feb2023/era5_pressure_2023_febapr.grib.gcidx"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--era5_37level_grib", required=True)
    p.add_argument("--sample_nc", required=True,
                    help="Same sample file used as the rollout IC, to recover ic_last_datetime")
    p.add_argument("--num_steps", type=int, default=240)
    p.add_argument("--out_nc", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    ic_ds = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    ic_last_datetime = ic_ds.coords["datetime"].isel(time=1).values
    ic_ds.close()
    print("IC reference (last input) datetime:", ic_last_datetime, flush=True)

    lead_times = np.arange(1, args.num_steps + 1) * np.timedelta64(6, "h")
    valid_times = ic_last_datetime + lead_times
    print(f"Need {args.num_steps} valid times: {valid_times[0]} .. {valid_times[-1]}", flush=True)

    print("Opening ERA5 37-level GRIB (cfgrib index build/reuse via shared "
          f"persistent index {SHARED_GRIB_IDX})...", flush=True)
    ds = xarray.open_dataset(
        args.era5_37level_grib,
        engine="cfgrib",
        backend_kwargs={"indexpath": SHARED_GRIB_IDX},
        chunks={"time": 4},
    )
    print("Opened. dims:", dict(ds.dims), flush=True)

    print("Selecting valid times + all levels...", flush=True)
    sub = ds.sel(time=valid_times)

    # cfgrib: latitude descending (90->-90), longitude 0->359.75. GraphCast: lat
    # ascending, same lon convention. Sort lat only.
    sub = sub.sortby("latitude")

    sub = sub.rename({
        "latitude": "lat",
        "longitude": "lon",
        "isobaricInhPa": "level",
    })

    data_vars = {}
    for era5_name, gc_name in VAR_MAP.items():
        if era5_name not in sub.data_vars:
            raise KeyError(f"Expected variable '{era5_name}' not found in GRIB. "
                            f"Available: {list(sub.data_vars)}")
        data_vars[gc_name] = sub[era5_name].astype("float32")

    truth = xarray.Dataset(data_vars)

    # Relabel the time dim from absolute valid-time to lead-time (matches
    # targets_template.time in the rollout scripts), keep absolute time for reference.
    truth = truth.assign_coords(valid_time=("time", valid_times))
    truth = truth.assign_coords(time=lead_times)

    truth = truth.transpose("time", "level", "lat", "lon")

    print("Truth dataset:", truth, flush=True)
    print(f"Writing to {args.out_nc} (this materializes the full selection -- "
          f"expect tens of GB)...", flush=True)
    truth.to_netcdf(args.out_nc)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
