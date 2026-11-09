#!/bin/bash
#SBATCH --job-name=graphcast-60day-det-feb2023
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/control/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/control/logs/%x_%j.err

set -euo pipefail

cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/control

module purge
module load Python/3.13.5-GCCcore-14.3.0
source /umbc/rs/pi_halem/users/yashd1/Graphcast-full/venv_graphcast/bin/activate

export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export NVIDIA_TF32_OVERRIDE=0

export TMPDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build
mkdir -p out/rollout_60day_2023-02-01

nvidia-smi --query-gpu=name,memory.total --format=csv

python run_graphcast_longrollout.py \
    --input_nc /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc \
    --params /umbc/rs/pi_halem/users/yashd1/Graphcast-full/params/GraphCast_full_0.25deg_37levels.npz \
    --stats_dir /umbc/rs/pi_halem/users/yashd1/Graphcast-full/stats \
    --num_steps 240 \
    --steps_per_chunk 4 \
    --out_dir out/rollout_60day_2023-02-01
