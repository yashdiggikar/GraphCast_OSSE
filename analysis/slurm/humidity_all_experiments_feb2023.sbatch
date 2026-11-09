#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-humidity-all
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/humidity_all_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/humidity_all_%j.err

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

mkdir -p "$GD/regime_analysis_results/cloud/csv" "$GD/regime_analysis_results/terrain/csv"

declare -A PRED_DIR=(
    [control]="$GD/control/full_predictions/rollout_60day_2023-02-01"
    [osse_t_global]="$GD/osse_t_global/full_predictions/osse_t_global_2023-02-01"
    [osse_uv_global]="$GD/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"
    [osse_z_global]="$GD/osse_z_global/full_predictions/osse_z_global_2023-02-01"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/full_predictions/t_perturbed_corr_2023-02-01"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/full_predictions/uv_perturbed_corr_2023-02-01"
)
declare -A OUT_CSV=(
    [control]="$GD/control/results/csv/rmse_q_2023-02-01.csv"
    [osse_t_global]="$GD/osse_t_global/results/csv/rmse_q_osse_t_global_2023-02-01.csv"
    [osse_uv_global]="$GD/osse_uv_global/results/csv/rmse_q_osse_uv_global_2023-02-01.csv"
    [osse_z_global]="$GD/osse_z_global/results/csv/rmse_q_osse_z_global_2023-02-01.csv"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/results/csv/rmse_q_t_perturbed_corr_2023-02-01.csv"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/results/csv/rmse_q_uv_perturbed_corr_2023-02-01.csv"
)

cd "$GD"

### ── 1) Specific humidity (Q) RMSE, all 6 experiments ──
for V in control osse_t_global osse_uv_global osse_z_global t_perturbed_corr uv_perturbed_corr; do
    echo "=== $V: compute_rmse_q ==="
    mkdir -p "$(dirname "${OUT_CSV[$V]}")"
    python compute_rmse_q.py \
        --pred_dir "${PRED_DIR[$V]}" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --out_csv "${OUT_CSV[$V]}"
done

### ── 2) Q cloud regime RMSE, all 6 experiments ──
for V in control osse_t_global osse_uv_global osse_z_global t_perturbed_corr uv_perturbed_corr; do
    echo "=== $V: cloud Q ==="
    python compute_cloud_rmse_q.py \
        --pred_dir "${PRED_DIR[$V]}" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --tcc_nc "$TCC_NC" \
        --out_csv "regime_analysis_results/cloud/csv/rmse_cloud_q_${V}.csv"
done

### ── 3) Q terrain regime RMSE, all 6 experiments ──
for V in control osse_t_global osse_uv_global osse_z_global t_perturbed_corr uv_perturbed_corr; do
    echo "=== $V: terrain Q ==="
    python compute_terrain_rmse_q.py \
        --pred_dir "${PRED_DIR[$V]}" \
        --era5_q_file "$ERA5_Q_FILE" \
        --sample_nc "$SAMPLE_NC" \
        --terrain_mask_nc "$TERRAIN_MASK" \
        --out_csv "regime_analysis_results/terrain/csv/rmse_terrain_q_${V}.csv"
done

echo "ALL Q RMSE/REGIME COMPUTATION DONE"

### ── 4) Q comparison vs control (300/500/850/1000hPa), 5 non-control experiments ──
CTRL_Q_CSV="${OUT_CSV[control]}"

declare -A CMP_OUT=(
    [osse_t_global]="$GD/osse_t_global/results"
    [osse_uv_global]="$GD/osse_uv_global/results"
    [osse_z_global]="$GD/osse_z_global/results"
    [t_perturbed_corr]="$GD/T_PERTURBED_CORR/results"
    [uv_perturbed_corr]="$GD/UV_PERTURBED_CORR/results"
)
declare -A CMP_LABEL=(
    [osse_t_global]="GraphCast T-insertion (noiseless) [2023-02-01 IC] -- specific humidity"
    [osse_uv_global]="GraphCast UV-insertion (noiseless) [2023-02-01 IC] -- specific humidity"
    [osse_z_global]="GraphCast Z-insertion (noiseless) [2023-02-01 IC] -- specific humidity"
    [t_perturbed_corr]="GraphCast T-insertion COP (perturbed) [2023-02-01 IC] -- specific humidity"
    [uv_perturbed_corr]="GraphCast UV-insertion COP (perturbed) [2023-02-01 IC] -- specific humidity"
)

cd "$BASE"
for V in osse_t_global osse_uv_global osse_z_global t_perturbed_corr uv_perturbed_corr; do
    echo "=== compare Q: $V vs control ==="
    OUTD="${CMP_OUT[$V]}"
    mkdir -p "$OUTD/q_tmp"
    python compare_control_vs_osse.py \
        --control_csv "$CTRL_Q_CSV" \
        --osse_csv "${OUT_CSV[$V]}" \
        --levels 300 500 850 1000 \
        --out_dir "$OUTD/q_tmp" \
        --out_prefix "compare_${V}_q" \
        --osse_label "${CMP_LABEL[$V]}"
    mkdir -p "$OUTD/csv" "$OUTD/plots"
    find "$OUTD/q_tmp" -name "*.csv" -exec mv {} "$OUTD/csv/" \;
    find "$OUTD/q_tmp" -name "*.png" -exec mv {} "$OUTD/plots/" \;
    rmdir "$OUTD/q_tmp"
done

echo "ALL DONE"
