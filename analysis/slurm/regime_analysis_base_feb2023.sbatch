#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-regime-base
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_base_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/regime_base_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023
TERRAIN_MASK=$BASE/terrain_mask_0.25deg.nc
TCC_NC=/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

mkdir -p "$GD/regime_analysis_results/terrain/csv" \
         "$GD/regime_analysis_results/cloud/csv" \
         "$GD/regime_analysis_results/lat_slice/csv"

declare -A PRED_DIR=(
    [osse_t_global]="$GD/osse_t_global/full_predictions/osse_t_global_2023-02-01"
    [osse_z_global]="$GD/osse_z_global/full_predictions/osse_z_global_2023-02-01"
    [osse_uv_global]="$GD/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"
)

cd "$GD"
for V in osse_t_global osse_z_global osse_uv_global; do
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
done

echo "ALL BASE REGIME ANALYSIS DONE"
find "$GD/regime_analysis_results" -name "*.csv" | sort
