#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-finish-t-perturbed-corr
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/T_PERTURBED_CORR/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/T_PERTURBED_CORR/logs/%x_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
VDIR=$GD/T_PERTURBED_CORR
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023
CTRL_CSV=$GD/control/results/csv/rmse_60day_2023-02-01.csv
UNPERTURBED_CSV=$GD/osse_t_global/results/csv/rmse_osse_t_global_2023-02-01.csv
RUN_LABEL="GraphCast T-insertion COP (perturbed, global, 26 levels, matching osse_t_global's fix) [2023-02-01 IC]"

cd "$VDIR"

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

echo "=== compute_rmse ==="
python compute_rmse.py \
    --pred_dir out/t_perturbed_corr_2023-02-01 \
    --era5_dir "$ERA5_DIR" \
    --sample_nc "$SAMPLE_NC" \
    --out_csv out/rmse_t_perturbed_corr_2023-02-01.csv

echo "=== plot (with control overlay) ==="
python plot_rmse.py \
    --csv out/rmse_t_perturbed_corr_2023-02-01.csv \
    --control_csv "$CTRL_CSV" \
    --out_dir out \
    --out_prefix rmse_vs_leadtime_t_perturbed_corr \
    --run_label "$RUN_LABEL"

echo "=== compare vs control (free-run) ==="
cd "$BASE"
python compare_control_vs_osse.py \
    --control_csv "$CTRL_CSV" \
    --osse_csv "$VDIR/out/rmse_t_perturbed_corr_2023-02-01.csv" \
    --out_dir "$VDIR/out" \
    --out_prefix compare_t_perturbed_corr \
    --osse_label "$RUN_LABEL"

echo "=== compare vs osse_t_global (unperturbed hard-replacement) ==="
python compare_control_vs_osse.py \
    --control_csv "$UNPERTURBED_CSV" \
    --osse_csv "$VDIR/out/rmse_t_perturbed_corr_2023-02-01.csv" \
    --out_dir "$VDIR/out" \
    --out_prefix compare_t_perturbed_corr_vs_unperturbed \
    --osse_label "$RUN_LABEL"

echo "=== reorganize ==="
cd "$VDIR"
mkdir -p full_predictions results/csv results/plots
find out -mindepth 1 -maxdepth 1 -type d -exec mv {} full_predictions/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.csv" -exec mv {} results/csv/ \;
find out -mindepth 1 -maxdepth 1 -type f -name "*.png" -exec mv {} results/plots/ \;
rmdir out 2>/dev/null || true

echo "DONE: $(ls results/csv | wc -l) csv, $(ls results/plots | wc -l) plots"
