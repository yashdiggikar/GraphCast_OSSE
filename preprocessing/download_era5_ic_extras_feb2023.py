"""IC-only pressure-level extras for the GC-DETER-2023 2023-02-01 control run.

Mirrors one_month_osse_2023/download_era5_ic_extras_mar2023.py, one month
earlier. q/w at 37 levels for the two real IC input timesteps (2023-02-01
00:00 and 06:00) -- T/Z/U/V for the same times come from the bulk
download_era5_37level_feb2023.py download instead (no need to duplicate).

Downloads to the current working directory -- run from era5_feb2023/.
"""
import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "specific_humidity",
        "vertical_velocity"
    ],
    "year": ["2023"],
    "month": ["02"],
    "day": ["01"],
    "time": [
        "00:00", "06:00"
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
client.retrieve(dataset, request).download("extras_feb2023.grib")
