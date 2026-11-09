# Hard-coded paths

The scripts in this repository are released **exactly as they were run** on the UMBC HPC
cluster (chip / `pi_halem` allocation), so they contain absolute paths to the original
locations of data, model weights and outputs. To re-run them elsewhere, replace the
prefixes below with your own locations (e.g. with `sed` or by editing the constants at the
top of each script). No other code changes are needed.

Root prefix used everywhere: `/umbc/rs/pi_halem/users/yashd1` (some legacy Aurora drivers use `/xace/d1/yashd1`).

| Original path prefix | Occurrences | Example file |
|---|---|---|
| `/umbc/rs/pi_halem/users/yashd1/Graphcast-full` | 139 | `analysis/shared/plot_spatial_bias.py` |
| `/umbc/rs/pi_halem/users/yashd1/era5_vs_cams` | 12 | `analysis/compute_cloud_rmse.py` |
| `/umbc/rs/pi_halem/users/yashd1/ALL_ERA5` | 9 | `analysis/compute_cloud_rmse.py` |
| `/umbc/rs/pi_halem/users/yashd1/venvs` | 4 | `preprocessing/slurm/run_dl_era5_37level_feb2023.sbatch` |
| `/umbc/rs/pi_halem/users/yashd1/.cdsapirc` | 4 | `preprocessing/slurm/run_dl_era5_37level_feb2023.sbatch` |
| `/umbc/rs/pi_halem/users/yashd1/spillover_analysis.py` | 2 | `analysis/spillover/spillover_analysis.py` |
| `/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS` | 2 | `experiments/08_t_insertion_perturbed_cop/perturbation.py` |
| `/umbc/rs/pi_halem/users/yashd1/Graphcast deterministic` | 1 | `analysis/slurm/regime_analysis_control_feb2023.sbatch` |
| `/umbc/rs/pi_halem/users/yashd1/CC_aurora` | 1 | `experiments/05_t_insertion_land/extract_propagation_arrays.py` |
