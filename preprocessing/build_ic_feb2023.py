"""Assembles a 2023-02-01 IC file (3 timesteps: 00:00, 06:00, 12:00Z) matching
the schema of ../dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc,
for use as --input_nc to control/run_graphcast_longrollout.py and
osse_*/run_osse_insertion.py in this GC-DETER-2023/ study (deterministic
re-run settings, 2023-02-01 IC, matched to the prithvi-feb23 study's IC).

Mirrors ../one_month_osse_2023/build_ic_mar2023.py, one month earlier, with
one difference: --pressure_grib here is the FULL 3-month (Feb-Apr 2023) bulk
download from download_era5_37level_feb2023.py (not a single-month file),
since GC-DETER-2023 needs a 240-step (60-day) truth-insertion window, not a
120-step one -- see prepare_truth_feb2023.py, which reads the same bulk grib
for the truth build. build_ic only ever slices 3 specific IC timesteps out
of it, so the extra 2 months in the file cost nothing here.

Only the first 2 timesteps (00:00, 06:00) are real GraphCast inputs; the 3rd
(12:00) is used by data_utils.extract_inputs_targets_forcings only as a
shape/dtype template -- see one_month_osse/build_ic_mar2022.py's docstring
for the full explanation (identical reasoning applies here).

total_precipitation_6hr accumulation window (same forecast-stream behavior,
different date -- re-verify if reused for yet another date):
    acc_6h(2023-02-01 00:00) = sum(tp(base=2023-01-31T18:00, step=1h..6h))
    acc_6h(2023-02-01 06:00) = sum(tp(base=2023-01-31T18:00, step=7h..12h))

Usage:
    python build_ic_feb2023.py \
        --pressure_grib era5_feb2023/era5_pressure_2023_febapr.grib \
        --extras_grib era5_feb2023/extras_feb2023.grib \
        --surface_grib_jan era5_feb2023/surface_jan31_2023.grib \
        --surface_grib_feb era5_feb2023/surface_feb01_2023.grib \
        --sample_nc ../dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --out_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
"""
import argparse

import numpy as np
import pandas as pd
import xarray

IC_TIMES = pd.to_datetime([
    "2023-02-01T00:00:00",
    "2023-02-01T06:00:00",
    "2023-02-01T12:00:00",
])
TP_BASE = pd.Timestamp("2023-01-31T18:00:00")

# Shared with prepare_truth_feb2023.py and each variant's compute_rmse.py --
# keeping this identical everywhere lets cfgrib build the persistent index
# for this study's own grib exactly ONCE and have every later script reuse
# it (the grib is ~3x the size of one_month_osse_2023's since it spans 3
# months, so this indexing step matters more here).
SHARED_GRIB_IDX = (
    "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/"
    "era5_feb2023/era5_pressure_2023_febapr.grib.gcidx"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pressure_grib", required=True,
                    help="T/Z/U/V, 37 levels -- this study's own era5_pressure_2023_febapr.grib")
    p.add_argument("--extras_grib", required=True,
                    help="q/w, 37 levels, 2023-02-01 00z+06z only")
    p.add_argument("--surface_grib_jan", required=True,
                    help="2t/msl/10u/10v/tp, hourly 2023-01-31")
    p.add_argument("--surface_grib_feb", required=True,
                    help="2t/msl/10u/10v/tp, hourly 2023-02-01")
    p.add_argument("--sample_nc", required=True,
                    help="DeepMind 2022-01-01 IC sample, for static fields + exact schema")
    p.add_argument("--out_nc", required=True)
    return p.parse_args()


def _drop_extra_coords(ds, keep):
    extra = [c for c in ds.coords if c not in keep]
    return ds.drop_vars(extra)


def open_pressure(path, rename, times=IC_TIMES, idx_path=None):
    print(f"Opening (cfgrib, indexed) {path} ...", flush=True)
    backend_kwargs = {"indexpath": idx_path} if idx_path else {}
    ds = xarray.open_dataset(
        path, engine="cfgrib", backend_kwargs=backend_kwargs,
        chunks={"time": 4},
    )
    ds = ds.sel(time=times).load()
    ds = ds.sortby("latitude").rename(
        {"latitude": "lat", "longitude": "lon", "isobaricInhPa": "level", **rename}
    )
    ds = _drop_extra_coords(ds, keep={"time", "level", "lat", "lon"})
    return ds.transpose("time", "level", "lat", "lon")


def open_surface(paths, rename):
    print(f"Opening (cfgrib) {paths} for instantaneous surface vars ...", flush=True)
    parts = []
    for path in paths:
        ds = xarray.open_dataset(
            path, engine="cfgrib", backend_kwargs={"indexpath": f"{path}.idx"},
        )
        parts.append(ds)
    ds = xarray.concat(parts, dim="time", coords="minimal", compat="override")
    ds = ds.sel(time=IC_TIMES)
    ds = ds.sortby("latitude").rename({"latitude": "lat", "longitude": "lon", **rename})
    ds = _drop_extra_coords(ds, keep={"time", "lat", "lon"})
    return ds.transpose("time", "lat", "lon")


def open_tp_6hr(paths):
    print(f"Opening (cfgrib) {paths} for accumulated tp ...", flush=True)
    parts = []
    for path in paths:
        ds = xarray.open_dataset(
            path, engine="cfgrib",
            backend_kwargs={"indexpath": f"{path}.idx", "filter_by_keys": {"shortName": "tp"}},
        )
        parts.append(ds)
    # The base=2023-01-31T18:00 forecast run's 12 hourly steps are split
    # across the two source files (steps 1-5 in the Jan31 file, steps 6-12
    # in the Feb01 file) -- merge (union on matching time+step coords), not
    # concat, reassembles the full 12-step block. See
    # one_month_osse/build_ic_mar2022.py's docstring for the full
    # explanation of why concat fails here.
    ds = xarray.merge(parts, compat="no_conflicts")
    tp = ds["tp"].sel(time=TP_BASE)
    hourly_steps = [pd.Timedelta(hours=h) for h in range(1, 13)]
    tp = tp.sel(step=hourly_steps)
    acc_00 = tp.isel(step=slice(0, 6)).sum("step")   # steps 1h..6h -> valid 19:00..00:00
    acc_06 = tp.isel(step=slice(6, 12)).sum("step")  # steps 7h..12h -> valid 01:00..06:00
    out = xarray.concat([acc_00, acc_06, acc_06], dim=pd.Index(IC_TIMES, name="time"),
                         coords="minimal", compat="override")
    out = out.sortby("latitude").rename({"latitude": "lat", "longitude": "lon"})
    out = _drop_extra_coords(out, keep={"time", "lat", "lon"})
    return out.transpose("time", "lat", "lon").astype("float32")


def main():
    args = parse_args()

    sample = xarray.open_dataset(args.sample_nc, decode_timedelta=True)
    lat_ref = sample.coords["lat"]
    lon_ref = sample.coords["lon"]
    level_ref = sample.coords["level"]
    static = sample[["geopotential_at_surface", "land_sea_mask"]].load()
    sample.close()

    press = open_pressure(args.pressure_grib, rename={
        "t": "temperature", "z": "geopotential",
        "u": "u_component_of_wind", "v": "v_component_of_wind",
    }, idx_path=SHARED_GRIB_IDX)

    extras = open_pressure(args.extras_grib, rename={
        "q": "specific_humidity", "w": "vertical_velocity",
    }, times=IC_TIMES[:2], idx_path=None)
    # extras_grib only has real data for [00z, 06z]; duplicate 06z for the
    # 12z template slot (values unused downstream -- see module docstring).
    extras = xarray.concat(
        [extras.isel(time=0), extras.isel(time=1), extras.isel(time=1)],
        dim=pd.Index(IC_TIMES, name="time"),
        coords="minimal", compat="override",
    )

    surf = open_surface([args.surface_grib_jan, args.surface_grib_feb], rename={
        "t2m": "2m_temperature", "msl": "mean_sea_level_pressure",
        "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
    })

    tp = open_tp_6hr([args.surface_grib_jan, args.surface_grib_feb])

    dyn = xarray.merge([press, extras, surf], compat="override")
    dyn["total_precipitation_6hr"] = tp

    assert np.allclose(dyn.coords["lat"].values, lat_ref.values)
    assert np.allclose(sorted(dyn.coords["lon"].values), sorted(lon_ref.values))
    dyn = dyn.reindex(lon=lon_ref.values)
    dyn = dyn.sel(level=level_ref.values)
    dyn.coords["lat"] = lat_ref
    dyn.coords["lon"] = lon_ref
    dyn.coords["level"] = level_ref.astype("int32")

    datetimes = IC_TIMES.values
    rel_time = (datetimes - datetimes[0]).astype("timedelta64[ns]")
    dyn = dyn.assign_coords(time=rel_time)
    dyn = dyn.expand_dims("batch")  # bare size-1 dim, no coordinate values
    dyn = dyn.assign_coords(datetime=(("batch", "time"), datetimes[None, :]))

    for var in dyn.data_vars:
        dyn[var] = dyn[var].astype("float32")

    ds = xarray.merge([dyn, static])

    print("Assembled IC dataset:", ds, flush=True)
    print("Writing to", args.out_nc, flush=True)
    ds.to_netcdf(args.out_nc)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
