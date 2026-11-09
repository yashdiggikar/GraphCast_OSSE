# GraphCast full — Correlated OSSE Perturbation (COP) Implementation

See `/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md`
for the cross-model method description and full citation list (Errico et al.
2013, Ingleby 2017, Hollingsworth & Lonnberg 1986, Bormann & Bauer 2010,
Bormann et al. 2016, Bannister 2008, Parrish & Derber 1992, Holton). This file
documents the GraphCast-full-specific code, paths, and hyperparameters,
mirroring `PERTURBATION_EXPERIMENTS/{FCN,Aurora,Prithvi}/IMPLEMENTATION.md`.

---

## 1. Location

| | Path |
|---|---|
| Temperature experiment | `T_PERTURBED_CORR/` |
| Wind experiment | `UV_PERTURBED_CORR/` |
| Shared perturbation module | `perturbation.py` (copied into each experiment dir) |
| SLURM job scripts | `{T,UV}_PERTURBED_CORR/slurm/run_{t,uv}_perturbed_corr.sbatch` |

Each experiment has its own full copy of the `graphcast/` package (based on
`osse_z_global/graphcast/`, which already has the timestep-alignment bug fix
from `session_log.md` Section 15), matching this project's existing
per-experiment-copy convention (`osse_t_global/`, `extended_exp/T_Land/`, etc.).
No prior perturbed-insertion baseline existed for GraphCast in this project
(unlike FCN/Aurora/Prithvi, which each had an i.i.d.-noise `T_PERTURBED`
baseline this work upgrades) -- this is GraphCast's first perturbation
experiment of either kind, correlated-noise from the start.

## 2. Where the perturbation logic lives

`graphcast/rollout.py`'s `chunked_prediction_generator()` injection block, in
each experiment's own copy -- same design choice as the reference
implementation (`Aurora/IMPLEMENTATION.md` Section 2): the perturbation-
generation logic differs enough between the T and UV experiments (scalar vs.
streamfunction-derived joint u,v) that each needs its own copy rather than a
shared parameterized function. The actual noise-generation numerics
(`correlated_level_fields`, `streamfunction_to_uv`,
`generate_correlated_wind_perturbation`) live in `perturbation.py`, imported
by each copy's `rollout.py` as `import perturbation as _cop`.

`run_osse_insertion.py` in each directory is **unmodified** from
`osse_t_global/run_osse_insertion.py` / `osse_uv_global/run_osse_insertion.py`
-- it still just passes plain ERA5 `truth_ds` through to
`chunked_prediction_generator`; the noise is added *inside* the injection
loop, not by pre-perturbing a truth file. This matches the reference
implementation's design (perturb on-the-fly inside the model's inference
loop, not a pre-baked perturbed-truth NetCDF).

## 3. Grid and level conventions

- Grid: GraphCast full's native 0.25deg, 721x1440 (includes both poles) --
  no crop/regrid needed, unlike Aurora (720x1440, patch-size-cropped).
- **Deviation from the reference, documented**: the reference implementations
  (FCN/Aurora/Prithvi) all perturb at exactly 13 pressure levels -- genuinely
  all the levels those models have. GraphCast full's insertion experiments in
  this project all operate at its full **37** native levels (this project's
  own established convention, e.g. `osse_t_global/`, `osse_uv_global/`), so
  these perturbed experiments keep that convention for direct comparability
  with this project's own unperturbed runs, rather than reverting to 13
  levels to match the other models exactly. The reference 13-level
  sigma-by-level tables are extended to 37 levels via log-pressure linear
  interpolation (flat-extrapolated above 50 hPa, where no reference
  radiosonde-style curve exists) -- see `perturbation.py`'s module docstring
  for the full statement of this simplification. The vertical *correlation*
  matrix needs no interpolation (it's an analytic function of the pressure
  levels themselves).

## 4. Hyperparameters used (identical to the reference)

```
HORIZ_CORR_LENGTH_KM = 250.0
SIGMA_PX              ~= 9.0 grid cells   (250 / (111 * 0.25))
VERT_L_LOGP           = 0.5
```

Temperature sigma-by-level (K), interpolated from the reference's 13-point
Ingleby (2017)-style table onto GraphCast's 37 levels (log-pressure linear
interpolation, flat above 50 hPa):

```
[1,2,3,5,7,10,20,30,50,...,1000] hPa -> [2.0, 2.0, ..., 2.0 (through 50hPa), 1.76, 1.5, 1.5, 1.34, 1.2, ..., 0.7 (mid-trop min), ..., 1.2 (1000hPa)] K
```//

Wind sigma-by-level (m/s), same interpolation from the reference's
newly-constructed table:

```
[1,...,50] hPa -> flat 1.5 m/s (no reference curve above 50hPa);
peaks ~2.2 m/s near 200hPa (jet core); 1.0 m/s minimum ~700-750hPa;
1.3 m/s at 1000hPa (surface friction/representativeness)
```

(Exact 37-value arrays: `perturbation.SIGMA_T_BY_LEVEL`,
`perturbation.SIGMA_UV_BY_LEVEL`.)

## 5. IC and truth source

Same as every other OSSE experiment in this project: IC
`dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc`
(2022-01-01 06:00 UTC), truth `era5_37level_2022/truth_leadtime_2022-01-01IC.nc`
(all 37 levels, built by `prepare_truth.py`). 240 steps x 6h = 60 days,
steps 0-119 (0-based) free-run, steps 120-239 truth(+noise)-insertion --
matches the "steps 121-240" 1-based convention used throughout this whole
cross-model study.

## 6. Reproducibility

`np.random.default_rng(seed=global_step_1based)`, where
`global_step_1based = global_step (0-based) + 1` -- matches the reference's
per-step-deterministic convention. For the wind experiment, u and v share the
SAME rng draw per step (one streamfunction realization differentiated into
both components) via a per-chunk cache keyed by step, since
`insert_var_names` processes u and v as separate loop passes but they must
not be independently random (see `perturbation.py` module docstring, Section
2.3 of `METHOD_OVERVIEW.md`).

## 7. Validation performed before submitting to GPU

Both injection code paths were unit-tested against a synthetic (non-GraphCast)
predictor function before spending any GPU time (mirroring the region_mask
validation done for `extended_exp/`):
- Temperature: confirmed perturbed values differ from raw truth by noise of
  the expected rough RMS magnitude (~1.6K on a small 40x60 test grid vs. mean
  sigma 1.26K), and are bit-identical across repeated runs with the same
  seeds.
- Wind: confirmed same reproducibility property, AND directly verified (by
  independently recomputing `generate_correlated_wind_perturbation` with the
  same seed) that the u and v injection passes for a given step draw from the
  literally identical streamfunction realization -- confirming the
  cross-variable cache works as designed, not just that each variable's noise
  is individually well-formed.

## 8. SLURM submission

```
sbatch T_PERTURBED_CORR/slurm/run_t_perturbed_corr.sbatch     # job 242858
sbatch UV_PERTURBED_CORR/slurm/run_uv_perturbed_corr.sbatch   # job 242859
```

Same resource request as this project's other OSSE GPU jobs: `--partition=gpu
--qos=chip-gpu --constraint="L40S|h100" --mem=64G --cpus-per-task=8
--time=2-00:00:00`.

## 9. Analysis pipeline (once rollout completes)

Same as every other experiment in this project: `compute_rmse.py` (global
13-level RMSE vs ERA5) -> `compare_control_vs_osse.py` (vs `control/`)
-> curated results copied into a `_results/`-style directory. Also worth
comparing directly against `osse_t_global/`/`osse_uv_global/` (the
unperturbed hard-replacement versions) to see whether observation-error-scale
noise changes the T-insertion instability finding documented in
`GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` -- the core question this
upgrade should let the paper answer, per `METHOD_OVERVIEW.md` Section 6.
