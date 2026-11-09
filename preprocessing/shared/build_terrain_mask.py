"""Build a static 5-class terrain mask at GraphCast's native 0.25deg grid
(721x1440), matching the classification convention used for the other 3
models in this cross-model study (common/terrain/fig_terrain_temp_ins.py::
classify_terrain()):

    Land Low   land, elevation <300 m,  non-polar
    Land Mod   land, 300 <= elevation < 1500 m, non-polar
    Land High  land, elevation >= 1500 m, non-polar
    Ocean      not land, non-polar
    Polar      |lat| > 60 deg (land or ocean pooled together)

Elevation and land fraction both come directly from GraphCast's OWN static
input fields (geopotential_at_surface / g, and land_sea_mask), the same
self-consistency choice already made for land_sea_mask_0.25deg.nc in
build_land_ocean_mask.py -- rather than sourcing the reference study's
MERRA-2 terrain file and regridding it.

Output: terrain_mask_0.25deg.nc (lat, lon), variable 'terrain_class'
(int8, 0=Land Low, 1=Land Mod, 2=Land High, 3=Ocean, 4=Polar).
"""
import xarray as xr
import numpy as np

SAMPLE_NC = "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
OUT_NC = "/umbc/rs/pi_halem/users/yashd1/Graphcast-full/terrain_mask_0.25deg.nc"

TERR_NAMES = ["Land Low", "Land Mod", "Land High", "Ocean", "Polar"]

ds = xr.open_dataset(SAMPLE_NC, decode_timedelta=True)
elev = (ds["geopotential_at_surface"] / 9.80665).values  # (lat, lon), meters
land_bool = (ds["land_sea_mask"] > 0.5).values             # (lat, lon)
lat_2d = np.broadcast_to(ds.coords["lat"].values[:, None], elev.shape)

polar = np.abs(lat_2d) > 60.0
terrain_class = np.full(elev.shape, 3, dtype=np.int8)  # default: Ocean
terrain_class[(~polar) & land_bool & (elev < 300)] = 0
terrain_class[(~polar) & land_bool & (elev >= 300) & (elev < 1500)] = 1
terrain_class[(~polar) & land_bool & (elev >= 1500)] = 2
terrain_class[polar] = 4

out = xr.DataArray(
    terrain_class, dims=("lat", "lon"),
    coords={"lat": ds.coords["lat"].values, "lon": ds.coords["lon"].values},
    name="terrain_class",
    attrs={
        "description": "0=Land Low(<300m) 1=Land Mod(300-1500m) 2=Land High(>=1500m) 3=Ocean 4=Polar(|lat|>60)",
        "source": "GraphCast checkpoint's own geopotential_at_surface (elevation) and land_sea_mask static fields",
    },
).to_dataset()
out.to_netcdf(OUT_NC)

print(f"Saved {OUT_NC}")
total = terrain_class.size
for c, name in enumerate(TERR_NAMES):
    n = int((terrain_class == c).sum())
    print(f"  {name:10s}: {n:7d}/{total} ({100*n/total:.1f}%)")
