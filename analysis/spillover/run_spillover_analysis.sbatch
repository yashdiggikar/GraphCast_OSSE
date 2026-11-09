#!/bin/bash
#SBATCH --job-name=gc-det-feb2023-spillover
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/spillover_analysis/logs_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/spillover_analysis/logs_%j.err

set -euo pipefail

cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/spillover_analysis

module purge
module load Python/3.13.5-GCCcore-14.3.0
source /umbc/rs/pi_halem/users/yashd1/Graphcast-full/venv_graphcast/bin/activate

export TMPDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build
export MPLCONFIGDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build/mplcache
mkdir -p "$MPLCONFIGDIR"

python spillover_analysis.py
