#!/usr/bin/env python3
"""
Propagation-map |error| array extraction for GC-DETER-2023's land_temp_ins
experiment -- ports /umbc/rs/pi_halem/users/yashd1/CC_aurora/extract_propagation_arrays.py
onto this study's own T-insertion, land-masked run (GraphCast full, 0.25deg,
2023-02-01 IC).

Loads only the 3 day-snapshot timesteps needed (First Injection / Day45 /
Day60) from land_temp_ins's raw prediction chunks plus the matching ERA5
T500 truth snapshots, computes |forecast - truth| per gridpoint (unsigned,
no averaging -- NOT RMSE, NOT signed bias), and saves as .npy + .json so
propagation maps can be plotted without re-touching the raw chunk files.

Output (this dir):
  propagation_error_land_temp.npy   float32 (3, 721, 1440)
  propagation_meta_land_temp.json
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import xarray as xr

BASE = Path("/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023")
CHUNK_DIR = BASE / "land_temp_ins" / "full_predictions" / "land_temp_ins_2023-02-01"
TRUTH_NC = BASE / "truth" / "truth_leadtime_2023-02-01IC.nc"
OUT_DIR = BASE / "land_temp_ins"
PLOT_LEVEL = 500.0
STEPS_PER_CHUNK = 4

# 0-indexed global step -> (chunk index, position within chunk). This study's
# insertion starts at global_step=120 (0-indexed; --inject_from_step 120),
# i.e. the FIRST step that actually receives masked truth. Day45/Day60 are
# the same 0-indexed step numbers used everywhere else in this study's
# 6-hourly, 240-step convention (step 179 = lead 1080h = day45; step 239 =
# lead 1440h = day60, the final step).
DAY_CONFIGS = [
    ("First Injection", 120),
    ("Day 45", 179),
    ("Day 60", 239),
]


def log(msg):
    print(msg, flush=True)


def read_chunk_slice(step0based):
    chunk_i = step0based // STEPS_PER_CHUNK
    pos = step0based % STEPS_PER_CHUNK
    cp = CHUNK_DIR / f"chunk_{chunk_i:04d}.nc"
    ds = xr.open_dataset(cp, decode_timedelta=True)
    T = ds["temperature"].sel(level=PLOT_LEVEL)
    if "batch" in T.dims:
        T = T.isel(batch=0)
    field = T.isel(time=pos).load().values.astype(np.float32)  # (lat, lon)
    lead_ns = int(ds.coords["time"].values[pos].astype("int64"))
    ds.close()
    return field, lead_ns


def main():
    log("Loading land_temp_ins forecast T500 snapshots ...")
    exp_slices = []
    lead_ns_list = []
    for label, step in DAY_CONFIGS:
        field, lead_ns = read_chunk_slice(step)
        exp_slices.append(field)
        lead_ns_list.append(lead_ns)
        log(f"  {label} (step {step}): lead={lead_ns/1e9/3600.0:.2f}h, field shape {field.shape}")
    exp_slices = np.stack(exp_slices)

    log("Loading matching ERA5 T500 truth snapshots ...")
    truth_ds = xr.open_dataset(TRUTH_NC, decode_timedelta=True)
    truth_lead_ns = truth_ds.coords["time"].values.astype("int64")
    truth_t500 = truth_ds["temperature"].sel(level=PLOT_LEVEL)
    truth_slices = []
    for lead_ns in lead_ns_list:
        ti = int(np.argmin(np.abs(truth_lead_ns - lead_ns)))
        truth_slices.append(truth_t500.isel(time=ti).load().values.astype(np.float32))
    truth_ds.close()
    truth_slices = np.stack(truth_slices)

    errors = np.abs(exp_slices - truth_slices)
    vmax = max(float(np.nanpercentile(errors, 97)), 0.1)
    vmax = round(vmax, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "propagation_error_land_temp.npy", errors)
    meta = {
        "exp_name": "land_temp",
        "ins_type": "temp",
        "var_label": "T @ 500 hPa (K)",
        "vmax": vmax,
        "day_labels": [d for d, _ in DAY_CONFIGS],
        "shape": list(errors.shape),
    }
    with open(OUT_DIR / "propagation_meta_land_temp.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved -> {OUT_DIR}/propagation_error_land_temp.npy  (shape={errors.shape}, vmax={vmax})")


if __name__ == "__main__":
    main()
