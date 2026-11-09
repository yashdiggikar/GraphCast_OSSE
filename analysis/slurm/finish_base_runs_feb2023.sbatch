#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-finish-base
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/finish_base_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/finish_base_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

echo "=== control: compute_rmse ==="
cd "$GD/control"
python compute_rmse.py \
    --pred_dir out/rollout_60day_2023-02-01 \
    --era5_dir "$ERA5_DIR" \
    --sample_nc "$SAMPLE_NC" \
    --out_csv out/rmse_60day_2023-02-01.csv

echo "=== control: plot (no overlay -- it IS the baseline) ==="
python plot_rmse.py \
    --csv out/rmse_60day_2023-02-01.csv \
    --out_dir out \
    --out_prefix rmse_vs_leadtime \
    --ic_label "2023-02-01T06:00Z"

echo "=== control: reorganize ==="
mkdir -p full_predictions results/csv results/plots
find out -mindepth 1 -maxdepth 1 -type d -exec mv {} full_predictions/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.csv" -exec mv {} results/csv/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.png" -exec mv {} results/plots/ \;
rmdir out 2>/dev/null || true

CTRL_CSV=$GD/control/results/csv/rmse_60day_2023-02-01.csv

declare -A LABELS=(
    [osse_t_global]="GraphCast T-insertion (global, 26 levels, drop top 11 stratospheric) [2023-02-01 IC]"
    [osse_z_global]="GraphCast Z-insertion (global, all 37 levels) [2023-02-01 IC]"
    [osse_uv_global]="GraphCast UV-insertion (global, all 37 levels) [2023-02-01 IC]"
)

for V in osse_t_global osse_z_global osse_uv_global; do
    echo "=== $V: compute_rmse ==="
    cd "$GD/$V"
    python compute_rmse.py \
        --pred_dir "out/${V}_2023-02-01" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --out_csv "out/rmse_${V}_2023-02-01.csv"

    echo "=== $V: plot (with control overlay) ==="
    python plot_rmse.py \
        --csv "out/rmse_${V}_2023-02-01.csv" \
        --control_csv "$CTRL_CSV" \
        --out_dir out \
        --out_prefix "rmse_vs_leadtime_${V}" \
        --run_label "${LABELS[$V]}"

    echo "=== $V: compare_control_vs_osse ==="
    SHORT=$(echo "$V" | sed 's/osse_//')
    cd "$BASE"
    python compare_control_vs_osse.py \
        --control_csv "$CTRL_CSV" \
        --osse_csv "$GD/$V/out/rmse_${V}_2023-02-01.csv" \
        --out_dir "$GD/$V/out" \
        --out_prefix "compare_${SHORT}" \
        --osse_label "${LABELS[$V]}"

    echo "=== $V: reorganize ==="
    cd "$GD/$V"
    mkdir -p full_predictions results/csv results/plots
    find out -mindepth 1 -maxdepth 1 -type d -exec mv {} full_predictions/ \;
    find out -mindepth 1 -maxdepth 1 -type f -name "*.csv" -exec mv {} results/csv/ \;
    find out -mindepth 1 -maxdepth 1 -type f -name "*.png" -exec mv {} results/plots/ \;
    rmdir out 2>/dev/null || true
done

echo "ALL BASE RUNS FINISHED"
for V in control osse_t_global osse_z_global osse_uv_global; do
    echo "-- $V --"
    ls "$GD/$V/results/csv" 2>&1
    ls "$GD/$V/results/plots" | wc -l
done
