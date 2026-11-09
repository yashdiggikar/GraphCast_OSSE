#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-bias-maps
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/bias_maps_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/bias_maps_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
GD=$BASE/GC-DETER-2023
SAMPLE_NC=$GD/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc
ERA5_DIR=$GD/era5_feb2023

module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

declare -A PRED_DIR=(
    [control]="$GD/control/full_predictions/rollout_60day_2023-02-01"
    [osse_t_global]="$GD/osse_t_global/full_predictions/osse_t_global_2023-02-01"
    [osse_z_global]="$GD/osse_z_global/full_predictions/osse_z_global_2023-02-01"
    [osse_uv_global]="$GD/osse_uv_global/full_predictions/osse_uv_global_2023-02-01"
)

cd "$GD"
mkdir -p bias_maps
for V in control osse_t_global osse_z_global osse_uv_global; do
    echo "=== $V: bias maps ==="
    python compute_bias_maps.py \
        --pred_dir "${PRED_DIR[$V]}" \
        --era5_dir "$ERA5_DIR" \
        --sample_nc "$SAMPLE_NC" \
        --out_dir "bias_maps/${V}" \
        --mature_start_day 30.0
done

echo "ALL GC-DETER-2023 BIAS MAPS DONE"
find "$GD/bias_maps" -name "*.npy" | sort
