"""Builds a lead-time-indexed, all-37-level ERA5 truth Dataset for GraphCast OSSE
truth-insertion runs, from the raw ERA5 GRIB file downloaded in era5_37level_2022/.

This is a one-time (per IC) CPU-bound prep step, kept separate from the GPU
insertion run itself so the (slow, one-time) cfgrib indexing of the ~103GB source
GRIB file doesn't eat into GPU walltime. Output is a plain NetCDF that any OSSE
variant script (Z-only, T-only, UV, combined, ...) can open cheaply and slice
whichever variable(s) it actually inserts -- shared across all variants, since the
truth data itself doesn't depend on which variable(s) a given run chooses to insert.

Output variable naming/dims matches what `graphcast/rollout.py`'s insertion code
expects: data_vars named after GraphCast's own variable names (geopotential,
temperature, u_component_of_wind, v_component_of_wind), dims (time, level, lat, lon)
[+ batch added at insertion time if needed], with `time` holding LEAD TIMES
(timedelta64), exactly matching `targets_template.time` in the rollout scripts --
NOT absolute datetimes.

Usage:
    python prepare_truth.py \
        --era5_37level_grib era5_37level_2022/ce7b7214a173e25f4fd170f3e7722a2b.grib \
        --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --num_steps 240 \
        --out_nc era5_37level_2022/truth_leadtime_2022-01-01IC.nc
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

    print("Opening ERA5 37-level GRIB (lazy, cfgrib index build -- one-time, can take a while "
          "on a 100GB+ file over network storage)...", flush=True)
    ds = xarray.open_dataset(
        args.era5_37level_grib,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
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
