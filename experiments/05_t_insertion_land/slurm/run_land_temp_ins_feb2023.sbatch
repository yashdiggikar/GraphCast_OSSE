#!/bin/bash
#SBATCH --job-name=graphcast-land-temp-ins-feb2023
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/land_temp_ins/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/land_temp_ins/logs/%x_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
VDIR=$GD/land_temp_ins
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023
CTRL_CSV=$GD/control/results/csv/rmse_60day_2023-02-01.csv
RUN_LABEL="GraphCast T-insertion, LAND grid points only (land_sea_mask>0.5; 26 levels, matching osse_t_global's fix) [2023-02-01 IC]"

cd "$VDIR"

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export NVIDIA_TF32_OVERRIDE=0
export TMPDIR=$BASE/tmp_build

mkdir -p out/land_temp_ins_2023-02-01
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== rollout ==="
python run_osse_insertion.py \
    --input_nc "$SAMPLE_NC" \
    --params "$BASE/params/GraphCast_full_0.25deg_37levels.npz" \
    --stats_dir "$BASE/stats" \
    --truth_nc "$GD/truth/truth_leadtime_2023-02-01IC.nc" \
    --insert_var_names temperature \
    --insert_levels 125 150 175 200 225 250 300 350 400 450 500 550 600 650 700 750 775 800 825 850 875 900 925 950 975 1000 \
    --insert_mask_var land \
    --land_threshold 0.5 \
    --inject_from_step 120 \
    --num_steps 240 \
    --steps_per_chunk 4 \
    --out_dir out/land_temp_ins_2023-02-01

echo "=== compute_rmse ==="
python compute_rmse.py \
    --pred_dir out/land_temp_ins_2023-02-01 \
    --era5_dir "$ERA5_DIR" \
    --sample_nc "$SAMPLE_NC" \
    --out_csv out/rmse_land_temp_ins_2023-02-01.csv

echo "=== plot (with control overlay) ==="
python plot_rmse.py \
    --csv out/rmse_land_temp_ins_2023-02-01.csv \
    --control_csv "$CTRL_CSV" \
    --out_dir out \
    --out_prefix rmse_vs_leadtime_land_temp_ins \
    --run_label "$RUN_LABEL"

echo "=== compare_control_vs_osse ==="
cd "$BASE"
python compare_control_vs_osse.py \
    --control_csv "$CTRL_CSV" \
    --osse_csv "$VDIR/out/rmse_land_temp_ins_2023-02-01.csv" \
    --out_dir "$VDIR/out" \
    --out_prefix compare_land_temp_ins \
    --osse_label "$RUN_LABEL"

echo "=== reorganize ==="
cd "$VDIR"
mkdir -p full_predictions results/csv results/plots
find out -mindepth 1 -maxdepth 1 -type d -exec mv {} full_predictions/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.csv" -exec mv {} results/csv/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.png" -exec mv {} results/plots/ \;
rmdir out 2>/dev/null || true

echo "DONE: $(ls results/csv | wc -l) csv, $(ls results/plots | wc -l) plots"
