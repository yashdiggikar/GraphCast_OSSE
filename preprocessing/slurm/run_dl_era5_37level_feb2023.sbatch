#!/bin/bash
#SBATCH --job-name=era5_37lvl_feb2023_dl
#SBATCH --cluster=chip-cpu
#SBATCH --partition=2018
#SBATCH --account=pi_halem
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/dl_era5_37level_feb2023_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/logs/dl_era5_37level_feb2023_%j.err

set -e
cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/era5_feb2023

source /umbc/rs/pi_halem/users/yashd1/venvs/cds/bin/activate
export CDSAPI_RC=/umbc/rs/pi_halem/users/yashd1/.cdsapirc

python /umbc/rs/pi_halem/users/yashd1/Graphcast-full/GC-DETER-2023/download_era5_37level_feb2023.py
