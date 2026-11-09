#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-regime-q-expanded
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_q_expanded_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_q_expanded_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_Q_FILE=$GD/era5_feb2023/era5_q_2023_febapr.grib
TERRAIN_MASK=$BASE/terrain_mask_0.25deg.nc
TCC_NC=/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

export MPLCONFIGDIR="$GD/.mplcache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

mkdir -p "$GD/regime_analysis_results/terrain/csv" \
         "$GD/regime_analysis_results/cloud/csv"

cd "$GD"

declare -A PRED_DIR=(
    [control]="$GD/control/full_predictions/rollout_60day_2023-02-01"
    [osse_t_global]="$GD/osse_t_global/full_predictions/osse_t_global_2023-02-01"
    [osse_z_global]="$GD/osse_z_global/full_predictions/osse_z_global_2023-02-01"
    [osse_uv_global]="$GD/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/full_predictions/t_perturbed_corr_2023-02-01"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/full_predictions/uv_perturbed_corr_2023-02-01"
)

### ── 1) Cloud + Terrain Q regime RMSE for all 6 variants ──
for V in control osse_t_global osse_z_global osse_uv_global t_perturbed_corr uv_perturbed_corr; do
    PD="${PRED_DIR[$V]}"

    echo "=== $V: terrain Q ==="
    python compute_terrain_rmse_q.py \
        --pred_dir "$PD" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --terrain_mask_nc "$TERRAIN_MASK" \
        --out_csv "regime_analysis_results/terrain/csv/rmse_terrain_q_${V}.csv"

    echo "=== $V: cloud Q ==="
    python compute_cloud_rmse_q.py \
        --pred_dir "$PD" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --tcc_nc "$TCC_NC" \
        --out_csv "regime_analysis_results/cloud/csv/rmse_cloud_q_${V}.csv"
done

echo "CLOUD+TERRAIN Q REGIME ANALYSIS DONE"

### ── 2) Q RMSE for the 3 remaining normal experiments (control's Q RMSE ──
###      was already computed by job 252642) ──
declare -A Q_PRED_DIR=(
    [osse_t_global]="$GD/osse_t_global/full_predictions/osse_t_global_2023-02-01"
    [osse_z_global]="$GD/osse_z_global/full_predictions/osse_z_global_2023-02-01"
    [osse_uv_global]="$GD/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"
)
declare -A Q_OUT_CSV=(
    [osse_t_global]="$GD/osse_t_global/results/csv/rmse_q_osse_t_global_2023-02-01.csv"
    [osse_z_global]="$GD/osse_z_global/results/csv/rmse_q_osse_z_global_2023-02-01.csv"
    [osse_uv_global]="$GD/osse_uv_global/results/csv/rmse_q_osse_uv_global_2023-02-01.csv"
)

for V in osse_t_global osse_z_global osse_uv_global; do
    echo "=== $V: compute_rmse_q ==="
    python compute_rmse_q.py \
        --pred_dir "${Q_PRED_DIR[$V]}" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --out_csv "${Q_OUT_CSV[$V]}"
done

### ── 3) Q comparison vs control for each of the 3 ──
CTRL_Q_CSV="$GD/control/results/csv/rmse_q_2023-02-01.csv"
if [ ! -f "$CTRL_Q_CSV" ]; then
    echo "FATAL: control Q RMSE CSV not found at $CTRL_Q_CSV -- job 252642 must complete first" >&2
    exit 1
fi

declare -A OUT_PREFIX=(
    [osse_t_global]="compare_t_global_q"
    [osse_z_global]="compare_z_global_q"
    [osse_uv_global]="compare_uv_global_q"
)
declare -A OSSE_LABEL=(
    [osse_t_global]="GraphCast T-insertion (global, 26 levels) [2023-02-01 IC] -- specific humidity"
    [osse_z_global]="GraphCast Z-insertion (global, all 37 levels) [2023-02-01 IC] -- specific humidity"
    [osse_uv_global]="GraphCast UV-insertion (global, all 37 levels) [2023-02-01 IC] -- specific humidity"
)

cd "$BASE"
for V in osse_t_global osse_z_global osse_uv_global; do
    echo "=== compare Q: $V vs control ==="
    mkdir -p "$GD/$V/results/q_tmp"
    python compare_control_vs_osse.py \
        --control_csv "$CTRL_Q_CSV" \
        --osse_csv "${Q_OUT_CSV[$V]}" \
        --levels 300 500 850 1000 \
        --out_dir "$GD/$V/results/q_tmp" \
        --out_prefix "${OUT_PREFIX[$V]}" \
        --osse_label "${OSSE_LABEL[$V]}"
    cd "$GD/$V"
    find results/q_tmp -name "*.csv" -exec mv {} results/csv/ \;
    find results/q_tmp -name "*.png" -exec mv {} results/plots/ \;
    rmdir results/q_tmp
    cd "$BASE"
done

echo "ALL EXPANDED Q + REGIME WORK DONE"
