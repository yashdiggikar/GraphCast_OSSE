#!/bin/bash
#SBATCH --job-name=gc_prep_truth_feb2023
#SBATCH --cluster=chip-cpu
#SBATCH --partition=2018
#SBATCH --account=pi_halem
#SBATCH --qos=long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/prepare_truth_feb2023_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/prepare_truth_feb2023_%j.err

set -e
cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023

source /umbc/rs/pi_halem/users/yashd1/Graphcast-full/venv_graphcast/bin/activate
export TMPDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build

python prepare_truth_feb2023.py \
    --era5_37level_grib era5_feb2023/era5_pressure_2023_febapr.grib \
    --sample_nc dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
    --num_steps 240 \
    --out_nc truth/truth_leadtime_2023-02-01IC.nc

# Self-chain: IC + truth are both ready now, kick off all 4 GPU rollout jobs
# on chip-gpu -- cross-cluster, hence a plain sbatch call here rather than a
# same-cluster --dependency (same pattern as
# one_month_osse_2023/slurm/run_prepare_truth_mar2023.sbatch).
BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023
sbatch --clusters=chip-gpu "$BASE/control/slurm/run_control_feb2023.sbatch"
sbatch --clusters=chip-gpu "$BASE/osse_t_global/slurm/run_osse_t_global_feb2023.sbatch"
sbatch --clusters=chip-gpu "$BASE/osse_z_global/slurm/run_osse_z_global_feb2023.sbatch"
sbatch --clusters=chip-gpu "$BASE/osse_uv_global/slurm/run_osse_uv_global_feb2023.sbatch"
