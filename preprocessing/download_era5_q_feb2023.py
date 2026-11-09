"""ERA5 specific-humidity truth for the GC-DETER-2023 study: q, at 4 pressure
levels (300/500/850/1000 hPa), 6-hourly, Feb-Apr 2023 -- covers the full
240-step (60-day) rollout window from a 2023-02-01 IC (2023-02-01 through
2023-04-02), same day/month/time coverage as the existing T/Z/U/V truth
download (download_era5_37level_feb2023.py).

Unlike download_era5_ic_extras_feb2023.py (q at ALL 37 levels but only the
2 IC timesteps), this is verification truth: full 6-hourly cadence across
the whole 3-month window, restricted to the 4 levels this study reports on
(300/500/850/1000 hPa) to keep the download small.

Downloads to the current working directory -- run from era5_feb2023/ (see
slurm/run_dl_era5_q_feb2023.sbatch). Credentials via venvs/cds, same as
every other ERA5 download in this project.
"""
import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "specific_humidity"
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
        "300", "500", "850", "1000"
    ],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("era5_q_2023_febapr.grib")
