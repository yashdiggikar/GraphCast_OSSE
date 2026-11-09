# GraphCast_OSSE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092593.svg)](https://doi.org/10.5281/zenodo.24092593)

Observing System Simulation Experiment (OSSE) **truth-insertion** framework for Google
DeepMind's **GraphCast** (full operational-resolution checkpoint: 0.25°, 37 pressure levels),
released with the manuscript *An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast* (Diggikar, Halem, and Raman) (submitted to AGU *JGR: Machine Learning
and Computation*).

GraphCast is rolled out autoregressively for 60 days (240 × 6 h steps) from the
**2023-02-01T06:00 UTC** ERA5 initial condition. In each insertion experiment, the first
30 days run free. From step index 120 (0-based, about day 30) onward, selected prognostic
fields are **hard-replaced with ERA5 "truth"** at every 6-hourly step, and the corrected
state is fed back into the model. Forecasts are verified against ERA5.

This is one of five companion repositories:

| Repository | Content |
|---|---|
| `Aurora_OSSE` | Microsoft Aurora (0.25°, 13 levels) |
| `FourCastNetv2_OSSE` | NVIDIA FourCastNet v2-small (earth2mip fork) |
| **`GraphCast_OSSE`** | Google DeepMind GraphCast (0.25°, 37 levels) |
| `PrithviWxC_OSSE` | NASA-IBM Prithvi WxC (MERRA-2) |
| `OSSE_CrossModel_Analysis` | Cross-model comparison and paper figures |

---

## Repository layout

```
GraphCast_OSSE/
├── model/                       Full GraphCast source (fork of google-deepmind/graphcast,
│   └── graphcast/               github.com/yashdiggikar/graphcast-OSSE) + truth-insertion
│                                rollout; package = exact as-run copy used by the control run
├── experiments/                 One folder per experiment (driver, SLURM jobs, RMSE scripts)
│   ├── 01_control/                      free run, no insertion
│   ├── 02_t_insertion_global/           T, global, 26 levels 125–1000 hPa
│   ├── 03_uv_insertion_global/          U+V, global, all 37 levels
│   ├── 04_z_insertion_global/           geopotential, global, all 37 levels
│   ├── 05_t_insertion_land/             T, land points only (LSM > 0.5)
│   ├── 06_t_insertion_ocean/            T, ocean points only
│   ├── 07_tuv_insertion_global/         T + U + V combined
│   ├── 08_t_insertion_perturbed_cop/    T with correlated observation-error perturbation (COP)
│   └── 09_uv_insertion_perturbed_cop/   U+V with correlated, non-divergent perturbation (COP)
│       └── model_overrides/graphcast/   files that differ from model/graphcast for this run
├── preprocessing/               ERA5 download, initial-condition build, truth preparation, masks
├── analysis/                    Bias maps, cloud/terrain/latitude-regime RMSE, humidity, spillover
├── environment/                 Frozen pip environments (model run; CDS download)
├── tools/assemble_experiment.sh Builds a runnable as-run code tree for one experiment
└── docs/                        PROVENANCE.tsv, PATHS.md, method notes, development log
```

### How the per-experiment model code is stored
The original experiments each used their own complete copy of the `graphcast` package. Across
all 9 experiments those copies differ **only** in `graphcast/rollout.py`, which carries the
truth-insertion logic. To avoid storing 9 near-identical copies:

* `model/graphcast/` is the exact package used by `01_control`.
* `experiments/<exp>/model_overrides/graphcast/rollout.py` is the exact `rollout.py` used by
  that experiment.
* `tools/assemble_experiment.sh experiments/<exp> <work_dir>` rebuilds the full, as-run code
  tree for that experiment byte for byte.

Every file's original path and MD5 checksum are listed in `docs/PROVENANCE.tsv`.

---

## Experiments

| Folder | Inserted variable(s) | Levels | Region | Insertion window |
|---|---|---|---|---|
| `01_control` | — | — | — | none (free run) |
| `02_t_insertion_global` | temperature | 26 levels, 125–1000 hPa (top 11 levels ≤ 100 hPa excluded) | global | steps 120–239 |
| `03_uv_insertion_global` | u, v wind | all 37 | global | steps 120–239 |
| `04_z_insertion_global` | geopotential | all 37 | global | steps 120–239 |
| `05_t_insertion_land` | temperature | 26 levels, 125–1000 hPa | land (LSM > 0.5) | steps 120–239 |
| `06_t_insertion_ocean` | temperature | 26 levels, 125–1000 hPa | ocean (LSM ≤ 0.5) | steps 120–239 |
| `07_tuv_insertion_global` | temperature + u, v | see job script | global | steps 120–239 |
| `08_t_insertion_perturbed_cop` | temperature + correlated obs. error | 26 levels, 125–1000 hPa | global | steps 120–239 |
| `09_uv_insertion_perturbed_cop` | u, v + correlated non-divergent error | see job script | global | steps 120–239 |

The exact command-line arguments for each run are in `experiments/<exp>/slurm/*.sbatch`. The
perturbation (COP) method is described in `docs/PERTURBATION_IMPLEMENTATION.md` and
`docs/COP_METHOD_OVERVIEW.md`.

---

## Reproducing the results

### 1. Environment
* Python 3.13 (`module load Python/3.13.5-GCCcore-14.3.0` on the original cluster), JAX with CUDA 12.
* `pip install -r environment/requirements-frozen.txt` (exact versions used).
* GPU memory: the full 37-level checkpoint needs a GPU with more than 24 GB (runs used L40S/H100).
  The deterministic re-run pins XLA deterministic ops (`xla_gpu_deterministic_ops`); see the
  job scripts.

### 2. Model weights and normalization statistics (not redistributed)
From the public DeepMind bucket `gs://dm_graphcast/` (see the upstream README in `model/`):
* `params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz`
  (stored locally as `params/GraphCast_full_0.25deg_37levels.npz`)
* `stats/` (`diffs_stddev_by_level.nc`, `mean_by_level.nc`, `stddev_by_level.nc`)

The weights are licensed by DeepMind under CC BY-NC-SA 4.0.

### 3. Data
ERA5 reanalysis (Hersbach et al., 2020) from the Copernicus Climate Data Store:
```
preprocessing/download_era5_37level_feb2023.py      # 37-level truth, Feb–Apr 2023
preprocessing/download_era5_ic_surface_feb2023.py   # IC surface fields
preprocessing/download_era5_ic_extras_feb2023.py    # IC extra fields
preprocessing/download_era5_q_feb2023.py            # specific humidity (verification)
preprocessing/build_ic_feb2023.py                   # build GraphCast input dataset
preprocessing/prepare_truth_feb2023.py              # lead-time-indexed truth file
preprocessing/shared/build_land_ocean_mask.py       # land/ocean mask (LSM > 0.5)
preprocessing/shared/build_terrain_mask.py          # terrain classes
```
Set your CDS credentials in `~/.cdsapirc` or in the `CDSAPI_KEY` environment variable.

### 4. Run
```bash
tools/assemble_experiment.sh experiments/02_t_insertion_global /scratch/gc_t_ins
cd /scratch/gc_t_ins && sbatch slurm/run_osse_t_global_feb2023.sbatch
```
Scripts contain the original absolute cluster paths; see `docs/PATHS.md` for the list of
prefixes to replace.

### 5. Analysis
* Per-experiment RMSE: `experiments/<exp>/compute_rmse.py`, `plot_rmse.py`
* Spatial bias maps (days 30–60): `analysis/compute_bias_maps.py`
* Regime-stratified RMSE (cloud, terrain, latitude slices; including humidity):
  `analysis/compute_{cloud,terrain,latslice}_rmse*.py`, driven by `analysis/slurm/`
* Land/ocean spillover-distance decay: `analysis/spillover/`

---

## Notes
* `docs/notes/` contains the original development log and the two internal technical analyses
  of the temperature-insertion behaviour (`GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` and
  `T_INSERTION_IMPLEMENTATION_AUDIT.txt`). They are included for transparency; some sections
  refer to an earlier 2022-01-01 IC study that is not part of this release.
* Model outputs (about 78 GB of prediction chunks per experiment) are not included in this repository. See the
  data availability statement in the paper.

## License
The code in `model/` is a modified fork of google-deepmind/graphcast, licensed under Apache-2.0
(`LICENSE`). See `NOTICE` for the modifications. GraphCast model weights are not included and
are subject to DeepMind's CC BY-NC-SA 4.0 terms.

## Citation
Diggikar, Y., Halem, M., & Raman, A. (2026). *GraphCast_OSSE* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24092594
(All versions: https://doi.org/10.5281/zenodo.24092593.) See also `CITATION.cff`. Please also cite GraphCast: Lam, R., et al. (2023), Learning skillful
medium-range global weather forecasting, *Science*, 382, 1416–1421, doi:10.1126/science.adi2336.
