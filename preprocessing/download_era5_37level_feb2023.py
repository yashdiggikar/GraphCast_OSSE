"""ERA5 truth for the GC-DETER-2023 study: T/Z/U/V, all 37 pressure levels,
6-hourly, Feb-Apr 2023 -- covers the full 240-step (60-day) rollout window
from a 2023-02-01 IC (2023-02-01 through 2023-04-02), matching the original
Jan-Mar-2022 download's 3-month buffer convention
(../download_era5_37level.py).

One combined CDS request (all 3 months in one retrieve call) -> one grib
file, same as the original 2022 recipe, so it can serve BOTH the truth-
insertion build (prepare_truth_feb2023.py) AND RMSE verification
(each variant's compute_rmse.py, glob-adjusted to match this filename)
without a second duplicate download.

Downloads to the current working directory -- run from era5_feb2023/ (see
slurm/run_dl_era5_37level_feb2023.sbatch). Credentials via venvs/cds, same
as every other ERA5 download in this project.
"""
import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": ["2023"],
    "month": ["02", "03", "04"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "pressure_level": [
        "1", "2", "3",
        "5", "7", "10",
        "20", "30", "50",
        "70", "100", "125",
        "150", "175", "200",
        "225", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000"
    ],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("era5_pressure_2023_febapr.grib")
