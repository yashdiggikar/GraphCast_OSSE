"""IC-only surface/single-level variables for the GC-DETER-2023 2023-02-01
control run. Mirrors one_month_osse_2023/download_era5_ic_surface_mar2023.py,
one month earlier -- same day-boundary two-request split (day before IC ->
IC day) for the total_precipitation_6hr accumulation window.

Downloads to the current working directory -- run from era5_feb2023/.
Produces surface_jan31_2023.grib and surface_feb01_2023.grib.
"""
import cdsapi

dataset = "reanalysis-era5-single-levels"
HOURLY = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
]
VARIABLES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation"
]

client = cdsapi.Client()

# 2023-01-31 (full day) -- supplies tp forecast steps 1h-6h (base 18:00 the
# day before, 2023-01-31T18:00 minus 24h = 2023-01-30T18:00... actually base
# is 2023-01-31T18:00, covering valid times 19:00..23:59 that day plus the
# instantaneous fields at every hour of 2023-01-31.
request_jan = {
    "product_type": ["reanalysis"],
    "variable": VARIABLES,
    "year": ["2023"],
    "month": ["01"],
    "day": ["31"],
    "time": HOURLY,
    "data_format": "grib",
    "download_format": "unarchived"
}
client.retrieve(dataset, request_jan).download("surface_jan31_2023.grib")

# 2023-02-01 (full day) -- supplies tp forecast steps 7h-12h (same base,
# 2023-01-31T18:00) covering valid times 00:00..06:00, plus the
# instantaneous fields at every hour.
request_feb = {
    "product_type": ["reanalysis"],
    "variable": VARIABLES,
    "year": ["2023"],
    "month": ["02"],
    "day": ["01"],
    "time": HOURLY,
    "data_format": "grib",
    "download_format": "unarchived"
}
client.retrieve(dataset, request_feb).download("surface_feb01_2023.grib")
