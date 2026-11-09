#!/bin/bash
#SBATCH --job-name=gc_build_ic_feb2023
#SBATCH --cluster=chip-cpu
#SBATCH --partition=2018
#SBATCH --account=pi_halem
#SBATCH --qos=long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/build_ic_feb2023_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/build_ic_feb2023_%j.err

set -e
cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023

source /umbc/rs/pi_halem/users/yashd1/Graphcast-full/venv_graphcast/bin/activate
export TMPDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build

python build_ic_feb2023.py \
    --pressure_grib era5_feb2023/era5_pressure_2023_febapr.grib \
    --extras_grib era5_feb2023/extras_feb2023.grib \
    --surface_grib_jan era5_feb2023/surface_jan31_2023.grib \
    --surface_grib_feb era5_feb2023/surface_feb01_2023.grib \
    --sample_nc /umbc/rs/pi_halem/users/yashd1/Graphcast-full/dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
    --out_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc

# Self-chain: IC is ready, kick off prepare_truth_feb2023.py next (same
# cluster, chip-cpu -- ordinary --dependency-free sbatch works fine here,
# same pattern as one_month_osse_2023/slurm/run_build_ic_mar2023.sbatch).
sbatch --cluster=chip-cpu \
    /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/slurm/run_prepare_truth_feb2023.sbatch
