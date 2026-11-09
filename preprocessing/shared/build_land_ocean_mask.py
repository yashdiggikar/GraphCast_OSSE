"""Build a small, reusable land/ocean boolean mask at GraphCast's native 0.25deg
grid (721x1440), from the checkpoint's own land_sea_mask static input field
(fractional ERA5 land-sea-mask, 0=ocean..1=land). Thresholded at >0.5 = land,
matching the convention used elsewhere in this cross-model study
(fcn_inference/inf.py: land_bool = lsm > 0.5).

Using GraphCast's OWN land_sea_mask (rather than sourcing a separate LSM file,
as the other 3 models' pipelines do) keeps the mask exactly self-consistent
with the static field the model itself was trained on and already conditions
its predictions on.

Output: land_sea_mask_0.25deg.nc (lat, lon), variable 'land' (bool-as-int8,
1=land, 0=ocean), shared across all land/ocean insertion experiments in
extended_exp/.
"""
import xarray as xr
import numpy as np

SAMPLE_NC = "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
OUT_NC = "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/land_sea_mask_0.25deg.nc"

ds = xr.open_dataset(SAMPLE_NC, decode_timedelta=True)
lsm_frac = ds["land_sea_mask"]  # (lat, lon), float, 0..1

land_bool = (lsm_frac > 0.5)
land_da = xr.DataArray(
    land_bool.values.astype(np.int8),
    dims=("lat", "lon"),
    coords={"lat": lsm_frac.coords["lat"].values, "lon": lsm_frac.coords["lon"].values},
    name="land",
    attrs={
        "description": "1 = land, 0 = ocean; thresholded at land_sea_mask > 0.5",
        "source": "GraphCast checkpoint's own land_sea_mask static input field",
    },
)
out = land_da.to_dataset()
out.to_netcdf(OUT_NC)

n_land = int(land_bool.sum())
n_total = land_bool.size
print(f"Saved {OUT_NC}")
print(f"Land points: {n_land}/{n_total} ({100*n_land/n_total:.1f}%)")
print(f"Ocean points: {n_total-n_land}/{n_total} ({100*(n_total-n_land)/n_total:.1f}%)")
