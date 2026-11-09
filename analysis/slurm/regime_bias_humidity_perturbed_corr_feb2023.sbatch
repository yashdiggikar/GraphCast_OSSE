#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-regime-bias-q-perturbed-corr
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_bias_q_perturbed_corr_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_bias_q_perturbed_corr_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023
ERA5_Q_FILE=$GD/era5_feb2023/era5_q_2023_febapr.grib
TERRAIN_MASK=$BASE/terrain_mask_0.25deg.nc
TCC_NC=/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

export MPLCONFIGDIR="$GD/.mplcache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

mkdir -p "$GD/regime_analysis_results/terrain/csv" \
         "$GD/regime_analysis_results/cloud/csv" \
         "$GD/regime_analysis_results/lat_slice/csv" \
         "$GD/bias_maps/t_perturbed_corr" \
         "$GD/bias_maps/uv_perturbed_corr"

declare -A PRED_DIR=(
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/full_predictions/t_perturbed_corr_2023-02-01"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/full_predictions/uv_perturbed_corr_2023-02-01"
)

cd "$GD"

### ── 1) Cloud / Terrain / Latitude regime RMSE (T/U/V/Z, 300/500/1000hPa) ──
for V in t_perturbed_corr uv_perturbed_corr; do
    PD="${PRED_DIR[$V]}"

    echo "=== $V: terrain ==="
    python compute_terrain_rmse.py \
        --pred_dir "$PD" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --terrain_mask_nc "$TERRAIN_MASK" \
        --out_csv "regime_analysis_results/terrain/csv/rmse_terrain_${V}.csv"

    echo "=== $V: cloud ==="
    python compute_cloud_rmse.py \
        --pred_dir "$PD" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --tcc_nc "$TCC_NC" \
        --out_csv "regime_analysis_results/cloud/csv/rmse_cloud_${V}.csv"

    echo "=== $V: lat_slice ==="
    python compute_latslice_rmse.py \
        --pred_dir "$PD" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --out_csv "regime_analysis_results/lat_slice/csv/rmse_latslice_${V}.csv"

    echo "=== $V: bias maps (day 30-60 mean) ==="
    python compute_bias_maps.py \
        --pred_dir "$PD" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --out_dir "bias_maps/${V}" \
        --mature_start_day 30.0
done

echo "REGIME + BIAS ANALYSIS DONE"

### ── 2) Specific humidity (Q) RMSE -- brand new, control + both COP variants ──
declare -A Q_PRED_DIR=(
    [control]="$GD/control/full_predictions/rollout_60day_2023-02-01"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/full_predictions/t_perturbed_corr_2023-02-01"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/full_predictions/uv_perturbed_corr_2023-02-01"
)
declare -A Q_OUT_CSV=(
    [control]="$GD/control/results/csv/rmse_q_2023-02-01.csv"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/results/csv/rmse_q_t_perturbed_corr_2023-02-01.csv"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/results/csv/rmse_q_uv_perturbed_corr_2023-02-01.csv"
)

for V in control t_perturbed_corr uv_perturbed_corr; do
    echo "=== $V: compute_rmse_q ==="
    python compute_rmse_q.py \
        --pred_dir "${Q_PRED_DIR[$V]}" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --out_csv "${Q_OUT_CSV[$V]}"
done

### ── 3) Q comparison vs control (300/500/850/1000hPa) ──
CTRL_Q_CSV="${Q_OUT_CSV[control]}"

mkdir -p "$GD/T_PERTURBED_CORR/results/q_tmp" "$GD/UV_PERTURBED_CORR/results/q_tmp"

echo "=== compare Q: t_perturbed_corr vs control ==="
cd "$BASE"
python compare_control_vs_osse.py \
    --control_csv "$CTRL_Q_CSV" \
    --osse_csv "${Q_OUT_CSV[t_perturbed_corr]}" \
    --levels 300 500 850 1000 \
    --out_dir "$GD/T_PERTURBED_CORR/results/q_tmp" \
    --out_prefix compare_t_perturbed_corr_q \
    --osse_label "GraphCast T-insertion COP (perturbed) [2023-02-01 IC] -- specific humidity"

echo "=== compare Q: uv_perturbed_corr vs control ==="
python compare_control_vs_osse.py \
    --control_csv "$CTRL_Q_CSV" \
    --osse_csv "${Q_OUT_CSV[uv_perturbed_corr]}" \
    --levels 300 500 850 1000 \
    --out_dir "$GD/UV_PERTURBED_CORR/results/q_tmp" \
    --out_prefix compare_uv_perturbed_corr_q \
    --osse_label "GraphCast UV-insertion COP (perturbed) [2023-02-01 IC] -- specific humidity"

cd "$GD"
find T_PERTURBED_CORR/results/q_tmp -name "*.csv" -exec mv {} T_PERTURBED_CORR/results/csv/ \;
find T_PERTURBED_CORR/results/q_tmp -name "*.png" -exec mv {} T_PERTURBED_CORR/results/plots/ \;
rmdir T_PERTURBED_CORR/results/q_tmp
find UV_PERTURBED_CORR/results/q_tmp -name "*.csv" -exec mv {} UV_PERTURBED_CORR/results/csv/ \;
find UV_PERTURBED_CORR/results/q_tmp -name "*.png" -exec mv {} UV_PERTURBED_CORR/results/plots/ \;
rmdir UV_PERTURBED_CORR/results/q_tmp

echo "ALL DONE"
