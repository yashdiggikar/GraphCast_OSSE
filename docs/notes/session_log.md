# GraphCast Full Model (0.25°) — Session Log

Working directory: `/umbc/rs/pi_halem/users/yashd1/Graphcast-full/`
Cluster: UMBC pi_halem HPC (SLURM, multi-cluster: `chip-cpu`, `chip-gpu`)
Account: `pi_halem`

Goal: run DeepMind's **full** GraphCast checkpoint (0.25°, 37 pressure levels) — not
GraphCast-Small (1°, 13 levels, already done previously in `gc_pred_files/`) — for a
60-day (240-step, 6-hourly) free-running autoregressive rollout from IC 2022-01-01,
then verify against ERA5 with global RMSE.

---

## 1. Cluster / GPU inventory (as discovered 2026-08-19)

`sinfo -M chip-gpu -o "%N %G %f"`:

| Nodes | GPU | VRAM | count/node |
|---|---|---|---|
| g20-[01,03] | RTX 2080 Ti | 11GB | 8 |
| g20-02 | RTX 2080 | 11GB | 5 |
| g20-04 | RTX 2080 | 11GB | 4 |
| g20-[05-11] | RTX 6000 | 24GB | 8 |
| g20-12 | RTX 8000 | 48GB | 6 |
| g20-13 | RTX 8000 | 48GB | 8 |
| g24-[09-10] | H100 NVL | ~96GB | 2 |
| g24-[01-08,11-12] | L40S | 48GB | 4 |

Partitions: `gpu` (default), `gpu-general` (near-identical node set), `gpu-contrib`.

**`gpu-contrib` is NOT usable by this account** — `sbatch` fails with
`QOS not permitted to use this partition`. Don't waste time trying it again for `pi_halem`.

`gpu` / `gpu-general` only accept `--qos=chip-gpu` (checked via
`scontrol show partition gpu -M chip-gpu`: `AllowQos=chip-gpu,support`). The generic
`normal`/`long`/`medium` QOS names exist on the cluster but are **not** valid for these
GPU partitions — always pass `--qos=chip-gpu` explicitly.

**`sbatch --test-only` start-time estimates are highly volatile.** One job's estimate
swung from "~4 days out" to "starting in under 2 minutes" within a ~5 minute window,
just from other users' jobs churning in the queue. Don't treat these as commitments —
they're only useful as a rough same-order-of-magnitude signal.

### Memory finding (important)

The full 0.25°/37-level checkpoint **does not fit in 24GB (RTX 6000)**. A single
matmul in the mesh2grid GNN alone requested 17.84GB; total OOM. Confirmed working on
**H100 NVL (~96GB)**. Very likely also fits on L40S/RTX 8000 (48GB) but this was never
actually confirmed by a completed run — only inferred from headroom. If picking a GPU
for future runs, prefer ≥48GB and treat 24GB as a hard no.

### Race-submission technique used

For time-sensitive work, submitted the *same* job twice against different GPU
constraints with **different output directories** (to avoid file collisions), then
cancelled whichever loses once one starts running:
- Job A: `--constraint="L40S|h100"` → won, started almost immediately on `g24-09` (H100).
- Job B: `--constraint="RTX_8000"` → queue estimate ballooned to ~11 days out;
  cancelled with `scancel <jobid> -M chip-gpu` once Job A was confirmed running.

---

## 2. Python environment

**Do not use the default `module load python/3.12.4`** — that build's stdlib is
broken (`ModuleNotFoundError: No module named '_ctypes'`), which breaks `jax` at
import time. Use the EasyBuild stack instead:

```bash
module purge
module load Python/3.13.5-GCCcore-14.3.0   # capital P, EasyBuild namespace
```

**`/tmp` is only 2GB** on compute nodes — pip installing `jax[cuda12]` (large NVIDIA
wheels) fails with `No space left on device` unless `TMPDIR` is redirected:
```bash
export TMPDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build
```

**Home directory `/home/yashd1` is a 500MB quota, already 100% full.** Don't write
there (pip cache, etc. — confirm they're redirected to the `pi_halem` filesystem).

### venv

```
Graphcast-full/venv_graphcast/   (built with Python 3.13.5-GCCcore-14.3.0)
```

Installed (roughly in this order):
```
jax[cuda12]          -> jax 0.11.1, confirmed jax.devices() sees CudaDevice
dm-haiku 0.0.17, chex, dm-tree, jraph, numpy 2.5.2, pandas 3.0.5, scipy,
xarray 2026.7.0, typing_extensions, trimesh, absl-py 2.5.0, dask,
netCDF4, h5netcdf, rtree 1.4.1
cfgrib 0.9.15.1, eccodes 2.48.0   (added later, for reading ERA5 GRIB truth data)
```

`ecCodes` modules also exist on the cluster (`module avail` shows `ecCodes/2.31.0-...`)
but were not needed — the pip `eccodes` wheel bundles its own C library.

### The `graphcast` package itself — NOT pip installed from the original repo

The original clone at `/umbc/rs/pi_halem/users/yashd1/GraphCast/graphcast-OSSE/` is
the user's own **heavily customized fork** of DeepMind's repo (already modified for
prior OSSE truth-insertion studies — its `rollout.py` has custom geopotential
truth-injection code and a hand-added `extend_targets_template()` function **not
present in vanilla DeepMind graphcast**). Its `setup.py` also lists fragile
Google-internal deps (`colabtools`, `dinosaur-dycore`) that aren't needed for plain
GraphCast inference and complicate `pip install`.

Instead: copied just the `graphcast/` subpackage into
`Graphcast-full/graphcast/` and import it via script-local path (Python auto-adds
the running script's directory to `sys.path`, so `cd Graphcast-full && python
run_graphcast_longrollout.py` picks up the local copy). This makes the project dir
fully self-contained as requested.

### Bugs found + fixed in the locally-copied `graphcast/rollout.py`

1. **Missing `__init__.py`** — the `cp -r` didn't bring one over cleanly; added an
   empty one so `import graphcast` resolves as a real package, not a broken
   namespace package.
2. **`xr.Dataset(some_existing_dataset)` pattern breaks under xarray 2026.7.0** —
   newer xarray disallows passing a `Dataset` as `data_vars` to the constructor
   (`TypeError: Passing a Dataset as data_vars ... not supported`). This vendored
   code was written against an older xarray. Fixed by changing
   `inputs = xr.Dataset(inputs)` (and same for `targets_template`, `forcings`) to
   `inputs = inputs.copy()` — three lines near the top of
   `chunked_prediction_generator()`.

### Bugs found + fixed in `run_graphcast_longrollout.py` (our script)

1. **`xarray.open_dataset` needs `decode_timedelta=True` explicitly.** The GraphCast
   example-batch NetCDF files encode `time` as CF `units: hours` (a timedelta-style
   encoding, no `since <epoch>`), but current xarray no longer auto-decodes that
   unless asked. Without this flag, `time` stays a raw `int32` and later timedelta
   arithmetic inside `data_utils.py` throws
   `TypeError: Concatenation operation is not implemented for NumPy arrays`.
2. **Synthetic forcings' `batch` dim must be a bare dimension, not an indexed
   coordinate.** The real `inputs` dataset has `batch` as a dimension with no
   coordinate values (`Dimensions without coordinates: batch`). My first attempt at
   building 240-step-ahead synthetic forcings used
   `ds.expand_dims(batch=batch_coord.values)`, which creates an actual indexed
   coordinate — this made `hk.scan`'s carry pytree structure disagree between the
   loop's initial input and its looped output
   (`TypeError: scan body function carry input and carry output must have the same
   pytree structure`). Fixed by using `ds.expand_dims("batch")` (bare dim, no coord
   values) instead.
3. **Missing `rtree`** — `trimesh`'s nearest-triangle lookup (used by
   `grid_mesh_connectivity.py` to build the mesh↔grid graph) needs it;
   `pip install rtree`.

---

## 3. Model checkpoint / data (all public, no auth needed)

Downloaded via direct HTTPS from the public GCS bucket `gs://dm_graphcast`
(`https://storage.googleapis.com/dm_graphcast/...`), listed via the JSON API:
```
curl -s "https://storage.googleapis.com/storage/v1/b/dm_graphcast/o?prefix=params/&fields=items(name,size)"
```

- `params/GraphCast_full_0.25deg_37levels.npz` (~146MB) — the full model, trained on
  ERA5 1979-2017, 0.25°, **37 pressure levels**, mesh refined 2→6,
  precipitation input+output. (Distinct from `GraphCast_operational` [13 levels,
  HRES-tuned] and `GraphCast_small` [1°, 13 levels] — those are smaller/different
  checkpoints in the same bucket, not used here.)
- `stats/{diffs_stddev_by_level,mean_by_level,stddev_by_level}.nc` — tiny
  normalization stat files, required alongside the checkpoint.
- `dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc`
  (13.2GB) — public demo ERA5 example batch. **Only its first 3 timesteps are
  actually used** (2 IC steps + 1 shape template) — see §4.

### Checkpoint's `model_config` / `task_config` (confirmed by loading it)

```
ModelConfig(resolution=0.25, mesh_size=6, latent_size=512, gnn_msg_steps=16,
            hidden_layers=1, ...)
pressure_levels = (1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,
                    400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,
                    950,975,1000)   # 37 levels
input_variables includes: 2m_temperature, mean_sea_level_pressure,
  10m_u/v_component_of_wind, total_precipitation_6hr, temperature, geopotential,
  u/v_component_of_wind, vertical_velocity, specific_humidity,
  toa_incident_solar_radiation, year_progress_sin/cos, day_progress_sin/cos,
  geopotential_at_surface, land_sea_mask
forcing_variables = (toa_incident_solar_radiation, year_progress_sin,
  year_progress_cos, day_progress_sin, day_progress_cos)
```

---

## 4. The memory problem, and how the rollout script solves it

A naive 240-step call (`rollout.chunked_prediction` from vanilla DeepMind code, which
concatenates every chunk into one big in-memory Dataset) would need to materialize a
240-step × 227-field × 721×1440 NaN target template — **on the order of 200GB+** just
for the placeholder, before any real prediction happens. Infeasible.

**`run_graphcast_longrollout.py`** avoids this with three tricks (this is the key
file to read/reuse for any future long rollout):

1. **Lazy target template**: `rollout.extend_targets_template(eval_targets_1,
   required_num_steps)` — this is a function the user's fork already had (NOT in
   vanilla DeepMind graphcast) that builds a `dask.array.zeros`-backed template.
   Slicing+computing one chunk of it is cheap regardless of total length.
2. **Forcings synthesized from pure astronomy, not truth data.** TISR (top-of-atmosphere
   insolation) and year/day-progress are deterministic functions of
   `(datetime, lat, lon)` only — they don't depend on any predicted or ground-truth
   atmospheric state. So all 240 future steps' forcings can be built directly via
   `data_utils.add_derived_vars()` + `data_utils.add_tisr_var()` on a synthetic
   Dataset with the right future timestamps — no need for 60 days of real future
   ERA5. See `build_synthetic_forcings()` in the script.
3. **Streaming generator + write-and-discard.** Uses
   `rollout.chunked_prediction_generator()` (NOT `chunked_prediction()`, which
   accumulates everything) directly, iterating and writing each yielded chunk
   straight to its own NetCDF file (`out/<run>/chunk_NNNN.nc`), then `del`-ing it.
   Peak memory stays at O(steps_per_chunk), not O(num_steps).

### Output level subsetting

The model **runs/rolls forward internally at the checkpoint's native 37 levels**
(required for physical consistency of the autoregressive state) — only what's
**written to disk** is subset, via `--save_levels`. Default matches the Aurora
study's 13-level convention (confirmed from `CC_aurora/down.py`):
```
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000   # hPa
```
This cut output size roughly to 13/37 ≈ 35% (see actual sizes below).

### Script CLI

```bash
python run_graphcast_longrollout.py \
    --input_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
    --params params/GraphCast_full_0.25deg_37levels.npz \
    --stats_dir stats \
    --num_steps 240 \
    --steps_per_chunk 4 \
    --out_dir out/rollout_60day_2022-01-01 \
    --save_levels 50 100 150 200 250 300 400 500 600 700 850 925 1000   # (default)
```

(There's also `run_graphcast_inference.py` — an earlier, simpler non-streaming
script closely mirroring the official demo notebook, using
`rollout.chunked_prediction()` directly. Fine for small `--eval_steps` smoke tests
only; **do not use it for long rollouts**, it will OOM on RAM the same way the naive
approach would. Superseded by `run_graphcast_longrollout.py` for anything beyond a
handful of steps.)

---

## 5. SLURM job scripts

- `slurm/run_60day.sbatch` — **the one that worked.**
  `--partition=gpu --qos=chip-gpu --constraint="L40S|h100" --mem=64G --cpus-per-task=8
  --time=2-00:00:00`, calls `run_graphcast_longrollout.py` with `--num_steps 240
  --steps_per_chunk 4 --out_dir out/rollout_60day_2022-01-01`.
- `slurm/run_60day_rtx8000.sbatch` — identical but `--constraint=RTX_8000`, different
  `--out_dir` (`..._rtx8000`) to avoid collision; used as the race backup, cancelled
  before it ran.
- `slurm/run_inference.sbatch` — earliest smoke-test script, calls the older
  `run_graphcast_inference.py` with `--eval_steps 4`. Superseded; kept for reference.

**Always** `sbatch --test-only <script>` first to catch partition/QOS/constraint
typos before actually queuing (this caught the `gpu-contrib` access problem cheaply).

---

## 6. Result: the completed 60-day rollout

- **SLURM job 237786**, ran on `g24-09` (H100 NVL, 95830MiB reported by
  `nvidia-smi`), **COMPLETED, exit code 0**, `Elapsed 00:06:33`.
- Actual model compute (excluding checkpoint load) was **~325s (~5.5 min)** for all
  240 steps — first chunk included one-time JIT compile overhead (~40s for 4 steps);
  steady-state was ~5-7s per 4-step chunk after that.
- Output: `out/rollout_60day_2022-01-01/chunk_0000.nc` … `chunk_0059.nc`
  (60 files, 4 lead-steps/6h-cadence each), **78GB total** on disk.
- Each chunk's `time` coordinate is **lead-time** (`timedelta64`, e.g. 6h, 12h, ...),
  not absolute datetime — the run never requested `output_datetime` tracking, so to
  get real valid times you must add the IC reference time yourself (see §7).
- Variables per chunk: `2m_temperature, mean_sea_level_pressure,
  10m_u/v_component_of_wind, total_precipitation_6hr, temperature, geopotential,
  u/v_component_of_wind, vertical_velocity, specific_humidity` — pressure-level ones
  at the 13 saved levels only.

### IC reference datetimes (important for any downstream analysis)

From the sample file's first 3 timesteps:
```
index 0: 2022-01-01T00:00:00   (older of the 2 input steps)
index 1: 2022-01-01T06:00:00   (== "ic_last_datetime", the reference point)
index 2: 2022-01-01T12:00:00   (first real target step, lead=6h)
```
So: `valid_time(step_i) = 2022-01-01T06:00:00 + lead_time(step_i)`, for
`i = 1..240`, spanning **2022-01-01T12:00 through 2022-03-02T06:00**.

### A scientific caveat worth remembering

GraphCast's (and every AI weather model's) actual forecast skill degrades sharply
past ~10-15 days. The back half of this 60-day trajectory is dynamical drift/climatology,
not a meaningful "forecast" in the verification sense — expected physics, not a bug,
but relevant when interpreting long-lead RMSE numbers from it.

---

## 7. Verification against ERA5 truth (in progress as of this log)

### Truth data location (already on disk, no download needed)

```
/umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022/
    era5_pressure_2022_01.grib   (16GB, 124 timesteps = all of Jan, 6-hourly)
    era5_pressure_2022_02.grib   (15GB, all of Feb, 6-hourly)
    era5_pressure_2022_03.grib   (2.7GB, confirmed covers 2022-03-01T00 through
                                   2022-03-05T18 — comfortably covers the
                                   2022-03-02T06:00 end of our rollout)
```
Confirmed structure via `cfgrib`: `(time, isobaricInhPa, latitude, longitude)`,
**13 pressure levels** — `1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150,
100, 50` hPa (same 13 as our saved output, just descending vs our ascending order —
harmless, xarray aligns by label not position), 0.25° grid (721×1440), latitude
**descending** (90 → -90, opposite of GraphCast's ascending -90 → 90 — must
`.sortby("latitude")` before comparing), longitude 0 → 359.75 (matches GraphCast's
convention directly, no fix needed). Variables: `z, q, t, u, v` (GRIB short names).
Center: `ecmf` (ECMWF), GRIB edition 1.

Variable name mapping used (`VAR_MAP` in `compute_rmse.py`):
```
temperature          -> t
u_component_of_wind   -> u
v_component_of_wind   -> v
geopotential          -> z
```

### RMSE formula (matches the existing Aurora study convention, for consistency)

Copied from `Aurora_Pred_Files/nrmse/compute_aurora_nrmse.py` — cosine-latitude
weighted global RMSE:
```python
cos_w_2d = cos(deg2rad(lat))[:, None] * ones((1, nlon))
total_w  = cos_w_2d.sum()
rmse(t, level) = sqrt( sum((pred - truth)**2 * cos_w_2d) / total_w )
```
Computed **per variable, per pressure level (all 13), per lead time (all 240 steps)**.

### `compute_rmse.py` — approach

Streams through the 60 prediction chunks like the rollout did (never loads a full
month of ERA5 into memory at once):
1. Opens the 3 monthly GRIB files once each via `cfgrib` (uses
   `backend_kwargs={"indexpath": ""}` to force a clean index rebuild rather than
   trusting possibly-stale pre-existing `.idx` cache files sitting next to the
   `.grib` files from earlier unrelated work — this indexing is the slow part,
   observed to take ~2-3 minutes total for the 3 files combined, one-time cost only).
2. For each of the 60 prediction chunks: compute its `valid_times` (
   `ic_last_datetime + chunk.time`), pull the matching ERA5 slice
   (`.sel(time=valid_times, isobaricInhPa=LEVELS)`), align lat ordering
   (`.sortby("latitude")`) and snap lat/lon coordinate values onto the prediction's
   exact floats (avoids float32-vs-float64 alignment mismatches), then compute the
   weighted RMSE per (variable, level, step) and append rows to a list.
3. Writes one long-format CSV: `out/rmse_60day_2022-01-01.csv` with columns
   `variable, level_hpa, lead_hours, lead_days, valid_time, rmse`.

**Sanity check done before the full run**: manually verified chunk 0, variable
`temperature`, level 500hPa, lead 6h — mean abs diff between prediction and ERA5
truth was **0.23K**, and pred/truth grid mean+std matched closely (251.77±13.49 vs
251.79±13.48). This is physically sensible for GraphCast's known short-lead skill —
treat as a working correctness check if this pipeline is rerun.

### Status: DONE (updated 2026-08-20)

The first background attempt (PID 1823616, under interactive job 237766) died
silently at chunk 27/60 when job 237766's session ended before completion — no CSV,
log just stops after "chunk 27/60 done". Confirmed via `squeue`/`ps` in a fresh
session (job 237766 was gone; a new interactive job 237840 had started instead).
Old partial log preserved at `logs/compute_rmse_attempt1_died_at_chunk27.log`.

Relaunched from scratch (PID 1947848, under interactive job 237840,
`--time=4:00:00`, started 2026-08-20 13:46) with the exact CLI from the "approach"
section above:
```bash
python compute_rmse.py \
  --pred_dir out/rollout_60day_2022-01-01 \
  --era5_dir /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 \
  --sample_nc dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
  --out_csv out/rmse_60day_2022-01-01.csv
```
Completed cleanly in ~17 minutes (matches earlier ETA estimate). Output:
**`out/rmse_60day_2022-01-01.csv`, 12,480 rows** (4 variables x 13 levels x 240
leads, confirmed exact) — columns `variable, level_hpa, lead_hours, lead_days,
valid_time, rmse`. Variables present: `temperature, u_component_of_wind,
v_component_of_wind, geopotential`. Levels: the same 13 as the rollout output
(50...1000 hPa). `lead_days` spans 0.25 to 60.0.

(Harmless non-fatal xarray `FutureWarning`s about `combine coords` defaults appear
in the log from `xarray.concat(slices, dim="time")` at compute_rmse.py:78 — cosmetic
only, safe to ignore, did not affect the run.)

### Plot: RMSE vs lead time (DONE)

`plot_rmse.py` reads the CSV and produces **one PNG per variable** (Temperature, U
wind, V wind, Geopotential), each with **one panel per pressure level**
(300/500/1000 hPa by default, one line per panel — not overlaid), x-axis = lead
time in days, with a dashed vertical guide at 15 days marking the approximate
GraphCast forecast-skill horizon (see §6 caveat). (Originally built as a single 2x2
combined image with overlaid lines per panel; changed to 4 separate per-variable
images with per-level panels per user request.)
```bash
python plot_rmse.py --csv out/rmse_60day_2022-01-01.csv --out_dir out --out_prefix rmse_vs_leadtime
```
Output: **`out/rmse_vs_leadtime_{temperature,u_component_of_wind,v_component_of_wind,geopotential}.png`**
(4 files).

Note: matplotlib was not yet in `venv_graphcast` — installed via
`pip install matplotlib` (pulled in `pillow, kiwisolver, fonttools, cycler,
contourpy, pyparsing` as deps). Also hit the same home-directory-quota issue as
pip/jax earlier (§2) — matplotlib's font cache defaults to `~/.cache/matplotlib`,
which is on the full 500MB `/home/yashd1` quota. Fixed the same way as `TMPDIR`:
```bash
export MPLCONFIGDIR=/umbc/rs/pi_halem/users/yashd1/Graphcast-full/tmp_build/mplcache
```
(First run still succeeded despite a "disk quota exceeded" warning during cache
write — non-fatal — but set `MPLCONFIGDIR` for any future run to avoid the warning
and avoid touching the home quota at all.)

**Read of the result** (for context, not a rigorous analysis): RMSE grows
~monotonically through the ~10-15 day skill horizon across all variables/levels, as
expected. Past that, upper-level (300 hPa) T, U, and Z keep drifting upward over the
back half of the 60 days (consistent with the "dynamical drift, not real forecast
skill" caveat in §6), while V-wind and lower-level (1000 hPa) fields plateau/oscillate
around a climatological-saturation level rather than growing further — also physically
expected (V has intrinsically lower predictability/persistence than the other
fields). Nothing here looked like a bug in the pipeline.

---

## 8. Directory map (as of this log)

```
Graphcast-full/
  session_log.md                      <- this file
  run_graphcast_longrollout.py        <- the working long-rollout script (USE THIS)
  run_graphcast_inference.py          <- older small-eval_steps-only script
  compute_rmse.py                     <- ERA5 verification script (DONE)
  plot_rmse.py                        <- RMSE-vs-lead-time plotting script (DONE)
  graphcast/                          <- locally patched copy of the graphcast package
  venv_graphcast/                     <- Python 3.13.5-based venv, see §2 (+ matplotlib, §7)
  params/GraphCast_full_0.25deg_37levels.npz
  stats/{diffs_stddev_by_level,mean_by_level,stddev_by_level}.nc
  dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc
  slurm/run_60day.sbatch              <- the job script that worked
  slurm/run_60day_rtx8000.sbatch      <- backup/race variant, unused
  slurm/run_inference.sbatch          <- old smoke-test job script
  out/
    smoketest_8step/                  <- early 8-step correctness test (OOM'd on RTX6000,
                                          empty/partial -- superseded, safe to delete)
    rollout_60day_2022-01-01/         <- the real result: chunk_0000.nc .. chunk_0059.nc (78GB)
    rmse_60day_2022-01-01.csv         <- verification result, 12480 rows (DONE)
    rmse_vs_leadtime.png              <- final plot (DONE)
  logs/                               <- sbatch stdout/stderr + compute_rmse.py log
    compute_rmse_attempt1_died_at_chunk27.log  <- first attempt's log, killed when its
                                          parent interactive job ended; kept for reference
  tmp_build/                          <- TMPDIR target for pip (see §2), also holds
                                          mplcache/ (MPLCONFIGDIR for matplotlib, §7);
                                          safe to clean
```

## 9. Open SLURM/session context at time of writing

- Interactive job 237766 (used for the rollout-era work and the first, aborted
  `compute_rmse.py` attempt) ended sometime before 2026-08-20 13:42 — cause unclear
  (likely walltime or manual end of that session), not something diagnosed further
  since a new interactive job had already replaced it.
- Interactive job **237840** (`gpu` partition, `g24-06`, 2 CPU / 8GB RAM,
  `--time=4:00:00`, started 2026-08-20 13:42) is the current session's own
  allocation. Everything in §7's "Status: DONE" section (the successful
  `compute_rmse.py` rerun and `plot_rmse.py`) ran under this job.
- No GPU-related jobs are running — the rollout (job 237786) finished long ago.
  `compute_rmse.py` and `plot_rmse.py` are both CPU-only and have both completed;
  nothing is currently running in the background.

## 10. Audit: was the extended-forcing / long-rollout inference done correctly? (2026-08-20)

User had a hunch something was wrong with the inference/forcing-extension approach.
Ran three independent checks against `run_graphcast_longrollout.py` +
`graphcast/rollout.py`'s `chunked_prediction_generator` / `extend_targets_template`.
**Conclusion: no bug found — the pipeline checks out.**

1. **Direct numeric check** (`verify_forcings.py`, new script, kept in repo): isolated
   `build_synthetic_forcings()` and diffed its output against the *real*
   ERA5-derived forcings (`data_utils.extract_inputs_targets_forcings` on the actual
   `ic_slice`) for the one lead-time with ground truth (6h from IC). Result:
   `year_progress_sin/cos` and `day_progress_sin/cos` matched **exactly** (0 diff).
   `toa_incident_solar_radiation` matched to **mean abs diff 46.3 vs real mean
   1,175,316 (~0.004%)**, max abs diff 1443.6 (~0.12%) — consistent with the 360-bin
   numerical quadrature in `solar_radiation.py`, i.e. floating-point/integration
   noise, not a systematic error. Confirms the astronomy-only forcing synthesis
   (TISR + year/day progress) is computing the right values.
2. **Structural check on the actual saved 60-day output**: concatenated all 240
   steps across the 60 `chunk_*.nc` files — exactly 240 unique, evenly-6h-spaced
   steps, no gaps, no duplicates, spanning 6h through 60d exactly as expected.
3. **Chunk-boundary continuity check** (on `out/rmse_60day_2022-01-01.csv`, T@500hPa):
   with `steps_per_chunk=4`, a broken autoregressive state handoff between chunks
   would show up as a periodic RMSE jump every 4 steps. Mean |RMSE step-to-step
   diff| at chunk-boundary transitions (0.0539) vs interior transitions (0.0552) —
   statistically indistinguishable, no reset/discontinuity artifact.

Traced the "reset time coordinate to the first chunk's labels" trick (present in
both `graphcast/rollout.py::chunked_prediction_generator` and
`graphcast/autoregressive.py`'s internal `hk.scan` loop) and confirmed it's pure
JIT-recompilation-avoidance bookkeeping — the xarray `time` coordinate *label* fed
into `_unflatten_and_expand_time` is never used as a numeric model feature, only
the underlying forcing/input *data arrays* are, and those always carry correct
real-datetime-derived values. This matches vanilla DeepMind GraphCast's own
rollout design, not a local-fork deviation.

**One latent-but-inert fragility found** (did not affect the actual run, since
batch was always 1): in `build_synthetic_forcings()`,
`ic_last_datetime + lead_times` only broadcasts correctly when
`batch_coord.size == 1` (shape `(1,) + (num_steps,)`). For `batch > 1` (e.g. an
ensemble/multi-IC run) this would raise a numpy broadcasting error. Fix before
reusing this script for anything with batch > 1.

`verify_forcings.py` kept in the repo root as a reusable diagnostic if this ever
needs rechecking (e.g. after modifying `build_synthetic_forcings` or upgrading
xarray/jax) — takes a couple minutes to run since it loads the 13GB sample file.

## 11. Status as of 2026-08-20: original scope complete, new phase started

The originally-scoped work is done: 60-day full-resolution GraphCast rollout (§6),
ERA5 verification RMSE table (§7), and the RMSE-vs-lead-time plots (per-variable
PNGs, one panel per level — §7) for T/U/V/Z at 300/500/1000 hPa, and the
extended-forcing correctness audit (§10, no bug found). A new phase (§12) has now
started: bringing GraphCast into the user's existing multi-model OSSE
truth-insertion study.

## 12. NEW PHASE (started 2026-08-20): GraphCast OSSE truth-insertion experiment

### Background / why

The user runs a large, established OSSE (Observing System Simulation Experiment)
research program across **multiple AI weather models** — confirmed present on disk:

- **Aurora**: `CC_aurora/`, `AURORA_DATA/auroa_ins/` (`memory_swath.md`,
  `REPORTS/aurora_ins.txt`) — 5 insertion variants (global UV, near-surface T,
  land-only T, land-only UV, global Z), plus a polar-orbiting-sounder **swath**
  variant (moving longitude-band masks, 25% global coverage each, 6h cadence).
- **Prithvi WxC**: `prithviOSSE/` (`memory.md`) — 6 variants (global T, global UV,
  global Z, land-only T, land-only UV, combined T+UV incl. near-surface T2M/TS/
  U10M/V10M), plus land/ocean/coastal/lower-level/12h-cadence/perturbed
  sub-variants (`t_land`, `t_ocean`, `t_coastal`, `t_lower`, `t_perturbed`,
  `t_perturbed_corr`, etc. — same pattern mirrored under `uv_*`).
- **FourCastNetv2**: `fcn_inference/OSSE-FourCastNetv2/` — same variant pattern as
  Prithvi (t_12h, t_coastal, t_land, t_lower, t_ocean, t_perturbed,
  t_perturbed_corr, tuv_global, uv_12h, uv_coastal, uv_land, uv_lower,
  uv_ocean, uv_perturbed_corr).

**Common design across all of them**: free-run for the first 120 steps (~30 days),
then **hard truth replacement** (no blending) from step 121 onward through the end
of a 240-step (60-day), 6-hourly rollout, using ERA5 "nature run" values inserted
into the model's own autoregressive state so the correction propagates forward.
Variants differ in *which variable(s)* (T / U+V / Z / combined), *where*
(global / land / ocean / coastal / swath), *how often* (every step / 12h), and
whether truth is exact or deliberately perturbed. Measured via `analyze_osse.py` /
`compute_metrics.py` / `extract_propagation_arrays.py` (near-identical scripts
across all three model dirs): RMSE improvement vs. a matched free-running control,
vertical propagation efficiency (does correcting one level bleed into neighbors),
cross-variable propagation (does correcting T affect U/V/Z error), and
lat-band/coastal/ocean/land-stratified skill.

**GraphCast is not yet part of this comparison.** The locally-patched
`graphcast/rollout.py::chunked_prediction_generator` in this project (see §2 — it's
the user's own fork, not vanilla DeepMind) already has geopotential hard-insertion
scaffolding built in from prior work: `truth_ds`, `z_var_name="geopotential"`,
`inject_from_step` args (see the "GEOPOTENTIAL INSERTION (LEAD-TIME MODE)" block,
~line 268-450). **This was NOT exercised in the §6 60-day rollout** — that run
passed neither `truth_ds` nor `inject_from_step`, so it was a pure free-run. This
existing code operates on the model's full **37-level internal state** (not just
the 13 levels that get saved to disk), so any truth used with it must also cover
all 37 levels to avoid the `reindex_like(..., method="nearest")` fallback silently
approximating missing levels from whatever's nearest.

### Plan

1. Download ERA5 truth at the model's native 37 pressure levels for T, U, V, Z
   (in progress — see below).
2. Adapt `run_graphcast_longrollout.py` (or a copy of it) to pass `truth_ds` +
   `inject_from_step` through to `chunked_prediction_generator` for a geopotential
   insertion run (matching the `bhag.py` / `geo_ins_run` pattern from the other
   models — global, all-levels, hard replacement, starting at step 121).
3. Likely also want T and UV insertion variants for GraphCast to match the other
   three models' fuller variant set (the existing rollout.py code only wires up
   `z_var_name` currently — generalizing it to also handle T/U/V would need a small
   extension, not yet done).
4. Run insertion variant(s), compare RMSE against the existing §6/§7 free-running
   baseline (already have full ERA5 truth + RMSE pipeline for that comparison) and
   ideally against the Aurora/Prithvi/FCN insertion results for the same
   variable/insertion-window choices, if a consistent IC/period is used.

**Not yet decided**: exact IC/date window for the insertion run (could reuse the
same 2022-01-01 IC as §6 for direct baseline comparison — this is why the ERA5
download below covers the same Jan-Mar 2022 window), which variable(s) to start
with (geopotential is the one already wired up in code, so likely first), and
whether to replicate ALL of the other models' sub-variants (land/ocean/coastal/
swath/perturbed) or just the core global-insertion ones initially.

### ERA5 37-level download — IN PROGRESS

Submitted as **SLURM job 130646** on cluster `chip-cpu`, partition `2018`,
`--qos=long`, `--time=3-00:00:00` (submitted as a proper batch job specifically
*because* the interactive-job-death failure mode already bit `compute_rmse.py`
once, per §7 — this download needs to survive independently of any interactive
session). Job script: `slurm/run_download_era5_37level.sbatch`. Download script:
`download_era5_37level.py` (exact code the user provided — uses `cdsapi`, venv
`venvs/cds` which already has working CDS credentials at
`/umbc/rs/pi_halem/users/yashd1/.cdsapirc`; note `$HOME/.cdsapirc` does NOT exist,
so the sbatch script sets `export CDSAPI_RC=/umbc/rs/pi_halem/users/yashd1/.cdsapirc`
explicitly before running — needed anywhere this script is rerun outside the
sbatch wrapper).

Request: dataset `reanalysis-era5-pressure-levels`, variables `geopotential,
temperature, u_component_of_wind, v_component_of_wind`, all 37 pressure levels
(`1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,
600,650,700,750,775,800,825,850,875,900,925,950,975,1000` hPa), Jan-Mar 2022,
6-hourly (00/06/12/18 UTC), format `grib`. Output lands in
`era5_37level_2022/` (whatever filename the CDS server assigns via
`download_format: "unarchived"` — check the directory listing, don't assume a
fixed name).

As of 2026-08-20 16:46 the CDS request (ID `4e338446-0df2-411c-90da-f84d7fa44925`)
had reached status **"running"** (i.e., CDS server-side processing, not yet
downloading to our disk). **Size estimate: ~110 GB** — derived from the existing
13-level Jan file (`era5_vs_cams/era5_2022/era5_pressure_2022_01.grib`, confirmed
via eccodes message count: 8060 messages / 16.74 GB = ~2.08 MB/message) scaled to
this request's 53,280 messages (4 vars x 37 levels x 360 timesteps over 90 days).
15TB free on `pi_halem`, no space concern.

**Time estimate** (grounded in the prior 13-level download's actual measured
timing, from `era5_vs_cams/logs/era5_2022_dl-160988.err/.out`): that download
(34.56 GB across 3 separate monthly requests) took ~8 min CDS server-side prep per
request and transferred at a sustained **~7.3 MB/s**. Scaling to this single
~110 GB combined request: prep is the unpredictable part (could be longer than 8
min since this request is ~3x bigger — depends entirely on CDS's current queue
load), but transfer should be roughly **~110GB / 7.3MB/s ≈ 4.2 hours**. **Total
rough estimate: ~4.5-5 hours** from submission. Submitted 2026-08-20 16:45, so
expect substantial progress/completion somewhere around 2026-08-20 21:00-21:45 —
but treat this as a rough guide, not a commitment (same caveat as the SLURM
`--test-only` estimates noted in §1 — external queue/load conditions are volatile).

**To resume/check in a future session**:
```bash
squeue -M chip-cpu -j 130646        # is it still running?
tail logs/era5_37level_download_130646.err   # CDS status: accepted -> running -> successful
ls -la era5_37level_2022/            # does the .grib file exist yet, what size?
```
If job 130646 is gone and no file exists, just resubmit:
`sbatch -M chip-cpu slurm/run_download_era5_37level.sbatch` (don't need
`--test-only` again, already confirmed the partition/QOS combo works).

### Not yet done (next steps, once the download finishes)

- Verify the downloaded 37-level file (level count, variable names, spatial/time
  coverage) the same way §7 verified the 13-level truth files.
- Write the GraphCast insertion-run script (extend or copy
  `run_graphcast_longrollout.py` to pass `truth_ds`/`inject_from_step` into
  `chunked_prediction_generator`).
- Run it, verify output, compute RMSE vs. baseline, compare to the other models.

**Superseded by §13 below** (download confirmed complete, directory restructured,
scripts written, first OSSE run in progress).

---

## 13. Download confirmed complete + repo restructured into control/ + OSSE variants (2026-08-21)

### Download verification

Job 130646 was no longer in `squeue -M chip-cpu` (finished/exited the queue by the
time this session resumed). Checked the raw download log
(`logs/era5_37level_download_130646.err`, a `cdsapi`/`tqdm` progress stream):
final lines show `100%|...| 103G/103G` on all parallel-part progress bars, no
`Error`/`Exception`/`Traceback` anywhere in the file. File on disk:
`era5_37level_2022/ce7b7214a173e25f4fd170f3e7722a2b.grib`, **110,640,608,640 bytes
(103.05 GiB)**, matching the tqdm total almost exactly and consistent with the
pre-download ~110GB estimate. Head/tail byte check confirms a structurally valid,
non-truncated GRIB file (`head(16) == b'GRIB...'`, `tail(16) == b'...7777'` --
`7777` is the standard GRIB end-of-message marker on the last message).

**Caution for future sessions**: a naive full-file message count via
`eccodes.codes_count_in_file()` (called directly, not through cfgrib) is **far too
slow to use as a verification method on this filesystem** -- killed after 17
minutes having read only 5.4MB via ~96,000 tiny (~56 byte) read syscalls (confirmed
via `/proc/<pid>/io`), i.e. it was doing a naive byte-by-byte scan with per-syscall
NFS round-trip latency dominating. **Don't use this method again** on a 100GB+
GRIB file here. `cfgrib`'s own index builder (used by `xarray.open_dataset(...,
engine="cfgrib")`, as `compute_rmse.py` and the new `prepare_truth.py` both use)
is a *different, faster* code path -- confirmed by §7's earlier successful
2-3-minute index build on the 16.74GB monthly file, and by §13's `prepare_truth.py`
run below actually completing its indexing phase. If you need a *fast* sanity
check instead, just check file size against the download log's reported total,
plus a head/tail byte check (as above) -- don't call low-level eccodes counting
functions directly on multi-GB+ files over this network filesystem.

### Repo restructured: control/ + per-OSSE-variant dirs, each with its own graphcast/ copy

Per user request, split the project into a clearly-labeled **`control/`** dir
(the already-completed free-running work from §1-11) and separate **per-OSSE-variant
dirs, each getting its own fresh copy of the `graphcast/` package** (so editing one
variant's insertion logic can never affect another variant or the verified control
baseline). This mirrors the existing convention in `fcn_inference/t_perturbed_corr/`
(own `OSSE-FourCastNetv2/` copy) and the general Aurora/Prithvi/FCNv2 OSSE dir
pattern referenced in §12.

```
Graphcast-full/
  session_log.md
  prepare_truth.py                  <- NEW, shared truth-prep (see below)
  compare_control_vs_osse.py        <- NEW, shared control-vs-OSSE diff/plot script
  download_era5_37level.py, setup_ref.py, gpu_and_env_inspection.log   <- unchanged, shared
  venv_graphcast/                   <- unchanged, shared across control + all variants
  params/, stats/, dataset_samples/ <- unchanged, shared (checkpoint + tiny inputs)
  era5_37level_2022/                <- shared truth GRIB + (new) prepared truth NetCDF
    ce7b7214a173e25f4fd170f3e7722a2b.grib
    truth_leadtime_2022-01-01IC.nc  <- NEW, prepare_truth.py output (in progress)
  tmp_build/                        <- unchanged, shared TMPDIR/MPLCONFIGDIR
  control/                          <- MOVED here, the completed §1-11 free-run work
    graphcast/                      <- the original patched copy (untouched, verified)
    run_graphcast_longrollout.py, run_graphcast_inference.py,
    compute_rmse.py, plot_rmse.py, verify_forcings.py
    slurm/run_60day.sbatch, run_60day_rtx8000.sbatch, run_inference.sbatch
    out/rollout_60day_2022-01-01/ (78GB), rmse_60day_2022-01-01.csv,
        rmse_vs_leadtime_*.png (4), smoketest_8step/ (empty, superseded)
    logs/ (the rollout job's + compute_rmse's logs)
  osse_z_global/                    <- NEW: first OSSE variant (geopotential, global,
                                        all-37-levels, hard insertion from step 120
                                        0-based == "step 121" 1-based, matching the
                                        other models' ~30-day free-run / ~30-day
                                        truth-corrected split)
    graphcast/                      <- FRESH copy of graphcast/, insertion logic
                                        generalized (see below)
    run_osse_insertion.py           <- NEW, adapted from control's long-rollout script
    compute_rmse.py, plot_rmse.py   <- copied from control/ (plot_rmse.py gained a
                                        --run_label flag so its plot titles can say
                                        "OSSE" instead of the control's default text)
    slurm/run_osse_z_global.sbatch  <- NEW
    out/, logs/                     <- empty, populated once the run happens
  (top-level out/, logs/, slurm/ dirs from before this restructure were emptied by
   the move and rmdir'd -- everything that was in them now lives under control/,
   except the era5_37level download's own log/slurm files which stayed at top level
   since they're shared/one-time, not part of "control" specifically)
```

### Generalized the insertion mechanism (in `osse_z_global/graphcast/rollout.py` only)

The pre-existing insertion code in `chunked_prediction_generator` (see §12) only
handled a single hardcoded `z_var_name` (geopotential). Since future variants
(T-only, UV, combined) are expected per §12's plan, generalized this **in the new
osse_z_global copy only** (control's copy is untouched/still the exact code that
produced the verified §6 result):
- `z_var_name: str = "geopotential"` -> `insert_var_names: Sequence[str] =
  ("geopotential",)` (a list/tuple instead of a single name).
- Both the validation block (checks truth_ds/predictions have the needed
  variable(s)) and the injection loop (previously operated on one DataArray) now
  loop over `insert_var_names`, doing the same per-variable hard-replace-then-
  reindex-to-grid logic as before for each one.
- No other behavior changed -- for a single-variable list this is functionally
  identical to the old single-`z_var_name` code path.
- Verified `ast.parse()`-clean; not yet exercised by an actual run (that happens
  when the §13 SLURM job below runs).
- **Future T/UV/combined variants**: copy `osse_z_global/graphcast/` (which already
  has the generalized version) rather than `control/graphcast/` (which doesn't),
  and just change `--insert_var_names` in that variant's run script/sbatch.

### `prepare_truth.py` (NEW, shared) -- builds the lead-time-indexed truth Dataset

Necessary because `chunked_prediction_generator`'s insertion code needs `truth_ds`
with `time` as **lead-time timedeltas** exactly matching `targets_template.time`,
covering the full target range (0..239 steps, i.e. lead 6h..1440h) even though only
steps >= `inject_from_step` actually get used -- see the exact-match validation at
`rollout.py` ~line 356-367. Also needs **all 37 pressure levels** (§12), unlike the
13-level truth used for RMSE verification.

Reads `era5_37level_2022/*.grib` via `cfgrib`, selects the 240 valid times matching
the rollout's IC (`2022-01-01T06:00 + {1..240}*6h`), sorts lat ascending (cfgrib is
descending), renames to GraphCast variable names (z/t/u/v ->
geopotential/temperature/u_component_of_wind/v_component_of_wind) and dims
(latitude/longitude/isobaricInhPa -> lat/lon/level), relabels the time axis from
absolute datetime to lead-time timedelta (keeping absolute time as a secondary
`valid_time` coord for reference/debugging), casts to float32, writes one NetCDF:
`era5_37level_2022/truth_leadtime_2022-01-01IC.nc`. Shared across all future OSSE
variants (T/UV/combined) since the truth data itself doesn't depend on which
variable(s) a given variant chooses to insert -- only `--insert_var_names` in the
run script differs per variant.

Submitted as **SLURM job 152089** on `chip-cpu`/partition `2018`/`--qos=long`
(`slurm/run_prepare_truth.sbatch`, 4 CPU / 32GB / 1-day walltime) -- as a proper
batch job rather than relying on the interactive session, same reasoning as the
§7 `compute_rmse.py` interactive-session-death incident and the original download
job. `sbatch --test-only` confirmed the partition/QOS combo works before real
submission.

**Status as of this log**: job running on node `c18-04`, past the "opening file"
print, into the cfgrib index-build phase on the 103GB GRIB file (the slow part --
expect this to take a while; check `logs/prepare_truth_152089.{out,err}` and
`squeue -M chip-cpu -j 152089` in a future session to see if it's done, and
whether `era5_37level_2022/truth_leadtime_2022-01-01IC.nc` exists yet).

### `run_osse_insertion.py` (NEW, in `osse_z_global/`)

Same memory-safe streaming structure as `control/run_graphcast_longrollout.py`
(lazy target template, astronomy-only forcing synthesis, chunked-generator
write-and-discard -- see that script's docstring / §4 for why). New: loads
`--truth_nc` (prepare_truth.py's output), validates `--insert_var_names` are
present in it, and passes `truth_ds` / `insert_var_names` / `--inject_from_step`
(default 120, 0-based) through to `rollout.chunked_prediction_generator`. Uses this
variant's own local `graphcast/` copy (generalized insertion code above), not
control's.

### `osse_z_global/slurm/run_osse_z_global.sbatch` (NEW) -- not yet submitted for real

Same GPU constraints as the proven control job (`--constraint="L40S|h100"`,
64GB/8cpu/2-day walltime, `--qos=chip-gpu` on `chip-gpu`/`gpu` partition -- see §1's
memory finding: full 37-level checkpoint needs ~48GB+ VRAM, confirmed working on
H100). `sbatch --test-only` confirmed the script itself is schedulable. **Not yet
submitted for real** -- blocked on `prepare_truth.py` (job 152089) finishing first,
since the sbatch script's `--truth_nc` arg points at that job's not-yet-existent
output file. Once that file exists, submit with:
```bash
cd /umbc/rs/pi_halem/users/yashd1/Graphcast-full/osse_z_global
sbatch slurm/run_osse_z_global.sbatch
```
This runs geopotential-only, global, all-levels insertion from step 120 (0-based)
onward -- the variant already wired up in the original fork's code, so first per
§12's plan. T-only, UV, and combined variants can follow the same pattern (new
`osse_*` dir, copy `osse_z_global/graphcast/` which already has the generalized
insertion code, new run script/sbatch just changing `--insert_var_names`).

### `compare_control_vs_osse.py` (NEW, shared, top-level) -- not yet run

Takes the control's RMSE CSV and an OSSE variant's RMSE CSV (both in the same
`compute_rmse.py` long-format), produces: (1) per-variable/per-level overlay plots
(control vs OSSE RMSE-vs-lead-time lines, with a marker at the injection day), (2)
a post-injection-window (`lead_days >= inject_step*6/24`) mean-%-RMSE-improvement
summary CSV + bar chart, matching the `rmse_improvement_fcn.png` convention already
used for the other 3 models (`fcn_inference/rmse_improvement_fcn.png`). Not yet run
-- needs the OSSE rollout + its own `compute_rmse.py` pass to finish first.

### Full remaining pipeline for the Z-global OSSE variant

1. Wait for `prepare_truth.py` (job 152089) to finish ->
   `era5_37level_2022/truth_leadtime_2022-01-01IC.nc` exists.
2. `cd osse_z_global && sbatch slurm/run_osse_z_global.sbatch` -- GPU insertion
   rollout, ~same runtime as control's (§6: ~6-7 min actual compute, submitted job
   may queue longer) -> `out/osse_z_global_2022-01-01/chunk_0000.nc..chunk_0059.nc`.
3. `python compute_rmse.py --pred_dir out/osse_z_global_2022-01-01 --era5_dir
   /umbc/rs/pi_halem/users/yashd1/era5_vs_cams/era5_2022 --sample_nc
   ../dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc
   --out_csv out/rmse_osse_z_global_2022-01-01.csv` (13-level truth is fine here,
   same as control -- this step verifies the *saved* 13-level output, unrelated to
   the 37-level truth used *during* the run for insertion).
4. `python plot_rmse.py --csv out/rmse_osse_z_global_2022-01-01.csv --out_dir out
   --out_prefix rmse_vs_leadtime_osse_z_global --run_label "GraphCast full OSSE
   (Z global insertion, step>=120)"`.
5. From `Graphcast-full/`: `python compare_control_vs_osse.py --control_csv
   control/out/rmse_60day_2022-01-01.csv --osse_csv
   osse_z_global/out/rmse_osse_z_global_2022-01-01.csv --inject_step 120 --out_dir
   osse_z_global/out --out_prefix compare_z_global`.
6. Compare against the Aurora/Prithvi/FCNv2 global-Z (or nearest equivalent)
   insertion results per §12's stated goal, if a consistent IC/period is used
   (not yet checked whether their ICs match this study's 2022-01-01).

---

## 14. All 3 OSSE variants completed + a real bug fix + a genuine physical finding (2026-08-21)

Per user request, ran **3 separate insertion experiments** (not just Z): geopotential
(`osse_z_global/`), temperature (`osse_t_global/`), and combined U+V wind
(`osse_uv_global/`) -- each its own dir with its own fresh `graphcast/` copy (from
`osse_z_global/graphcast/`'s already-generalized `insert_var_names` code, see §13),
own `run_osse_insertion.py`/`compute_rmse.py`/`plot_rmse.py`, own
`slurm/run_osse_{z,t,uv}_global.sbatch`. All three insert at **all 37 native model
levels** (confirmed: insertion code has no level subsetting -- operates on the raw
model prediction array before the save-time 13-level `.sel(level=...)` subset;
`prepare_truth.py`'s truth dataset also covers all 37 levels for every variable --
this was double-checked directly against a user question mid-session).

### `prepare_truth.py` (job 152089) result

Completed: **COMPLETED, exit 0, 59:50 elapsed**. Output
`era5_37level_2022/truth_leadtime_2022-01-01IC.nc`, **137.4 GiB**
(147,513,202,402 bytes), structure verified by direct load: `time: 240` (21600s /
6h through 5184000s / 1440h = 60d, matches exactly), `level: 37`, all 4 variables
present, no NaNs in spot-checked slices, physically sane magnitudes (e.g.
geopotential@1000hPa mean ~844 m^2/s^2, temperature@mid-level mean ~235K).
**Caution repeated from earlier in this log**: don't verify a large GRIB file with
a naive `eccodes.codes_count_in_file()` call -- confirmed unusably slow (killed
after 17min, <10MB read) on this filesystem; `cfgrib`'s own indexing (used here) is
a different, much faster code path.

### Real bug found and fixed: read-only array from `jax.device_get()`

First submission of all 3 GPU jobs (237902/237903/237904) **all failed identically**
at the exact same point -- chunk 30, step 120 (the insertion boundary) -- with
`ValueError: assignment destination is read-only` from
`pred_da[dict(time=ti)] = truth_chunk.isel(time=k)` in `rollout.py`'s injection
loop. Root cause: `jax.device_get()` (called just before the injection block, to
bring `predictions` back to host) can return a **read-only-backed numpy array** --
a known JAX behavior, not something the control run ever exercised since control
never passed `truth_ds`/`inject_from_step` (this exact code path was, per §12,
"NOT yet exercised" before this session). Fix applied to all three variants'
`graphcast/rollout.py` (identical one-line change in each copy, ~line 431):
```python
# jax.device_get() can hand back a read-only-backed numpy array;
# deep-copy so the in-place assignment below doesn't crash.
pred_da = predictions[var_name].copy(deep=True)
```
(previously: `pred_da = predictions[var_name]`, no copy). Verified fix by clearing
the 3 partial output dirs and resubmitting (237906/237907/237908) -- all three now
run cleanly through the injection boundary and to completion. **If writing any
future OSSE variant** (new insertion variable, different model, etc.) that mutates
a `jax.device_get()`-derived array in place, apply the same `.copy(deep=True)`
defensively up front -- this is a general JAX gotcha, not specific to geopotential/
temperature/wind.

One side effect of the fix: a one-time ~2.5min XLA recompile (`input_reduce_fusion`
"very slow compile" warning) occurs exactly once per job, right at the injection
boundary, since the injection code path is a genuinely new JIT trace the control
run's compilation cache never saw. This is a one-time cost, not a per-chunk
slowdown -- confirmed from the per-chunk timing logs (chunks before/after the
boundary both settle back to ~5-9s/chunk).

### All 3 completed successfully

| Job | Variant | Elapsed | Result |
|---|---|---|---|
| 237906 | Z (geopotential) global | 00:10:39 | COMPLETED, 60/60 chunks, 78GB |
| 237907 | T (temperature) global | 00:10:39 | COMPLETED, 60/60 chunks, 78GB |
| 237908 | UV (u+v wind) global | 00:11:27 | COMPLETED, 60/60 chunks, 78GB |

All landed on **L40S (46068 MiB)** GPUs -- this is the first actual confirmation
(§1's memory finding only *inferred* L40S/RTX8000 would fit the full 37-level
checkpoint from headroom, never confirmed by a completed run). Now confirmed: L40S
runs the full model fine, insertion included.

`compute_rmse.py` run for all 3 (as parallel `chip-cpu` batch jobs
154142/154143/154144, ~23min each, all COMPLETED exit 0, 12,480 rows each --
`osse_{z,t,uv}_global/out/rmse_osse_{z,t,uv}_global_2022-01-01.csv`). Then
`compare_control_vs_osse.py` run for all 3 against `control/out/rmse_60day_2022-01-01.csv`
(`--inject_step 120`) -> `osse_{z,t,uv}_global/out/compare_{z,t,uv}_global_*.png/csv`.

### Result: Z and UV insertion behave physically; T-only insertion is unstable

Post-injection-window (day 30-60) mean %-RMSE-improvement vs the free-running
control, from the `compare_*_pct_improvement.csv` files:

- **Z (geopotential) insertion**: geopotential itself improves **75-94%** across
  all 13 saved levels. Cross-variable effects on T/U/V are mild and mixed (small
  improvements at upper levels, small-to-moderate degradation lower down) --
  physically unsurprising (correcting the mass field has a real but secondary
  effect on wind/temperature).
- **UV (u+v wind) insertion**: U and V each improve **46-87%**. Notably, this
  **also improves geopotential (40-52%, all levels except a small -9.6% at 50hPa)
  and temperature (12-43%)** -- physically sensible via geostrophic/thermal-wind
  balance: correcting the wind field pulls the mass/thermal fields toward
  consistency too.
- **Temperature insertion**: temperature itself improves beautifully (**72-94%**,
  RMSE down to ~1.3-1.8K from a 6-23K free-running baseline) -- confirms the
  insertion mechanism itself is working correctly (inserted values do match
  truth). But **geopotential and wind RMSE get dramatically WORSE than the
  free-running control** -- e.g. geopotential@1000hPa: control RMSE 1034 -> OSSE
  RMSE 5004 (~5x worse, "-344% improvement"); U-wind and V-wind show similarly
  large blowups (-30% to -750% depending on level).

**Confirmed this is a genuine physical instability, not a bug**, by directly
inspecting Z500 field values (not just RMSE): right at the injection boundary
(chunk_0030, steps 121-124), T-insertion's Z500 (mean 53722, range
46878-58299) is nearly identical to control's (mean 53714, range 46796-58267) --
both physically sane. By the final chunk (chunk_0059, steps 237-240), T-insertion's
Z500 has drifted far outside climatological range (mean 60175, range
35643-79021) versus control's still-sane (mean 54068, range 42525-59987). No
NaN/Inf anywhere -- this is smooth, progressive divergence, not a crash or a
data/indexing bug. **Interpretation**: hard-replacing temperature alone, every 6h,
for 120 consecutive insertion steps, repeatedly injects a state that's dynamically
*unbalanced* with the model's own evolved geopotential/wind fields (temperature
truth is correct in isolation, but not consistent with the current Z/U/V state via
hydrostatic/geostrophic balance) -- each replacement is a small "shock" that
compounds over 120 steps. This mirrors a well-known effect in real data
assimilation: univariate temperature-only hard updates violate balance
constraints and can excite spurious growth, which is why operational NWP
analysis increments are multivariate/balanced rather than univariate. Z-insertion
and UV-insertion don't show this because Z is a more dynamically-integrative field
(less prone to sharp local imbalance from a single-level nudge) and wind
correction actually *promotes* balance (helps Z/T too) rather than fighting it.

**This is a legitimate, likely-worth-featuring finding for the Ch.5 cross-model
synthesis** (see `[[project_multimodel_study]]` memory) -- not something to "fix".
If a future session wants to probe this further: worth trying a `t_perturbed_corr`-
style softer variant (partial/damped nudging instead of hard replacement, or 12h
cadence instead of every 6h -- matching the `t_perturbed`/`t_12h` sub-variant
pattern already used for the other 3 models in `fcn_inference`/`prithviOSSE`) to
see whether the instability is specifically a hard-replacement artifact that a
gentler insertion avoids.

### Current state of `Graphcast-full/`

All 3 OSSE variants are fully done end-to-end: rollout -> RMSE vs 13-level ERA5
truth -> comparison-vs-control plots/CSVs. Nothing currently running. Next steps
(not yet done, for a future session): (a) the softer T-insertion variant idea
above, if desired; (b) cross-model comparison against Aurora/Prithvi/FCNv2's
equivalent insertion variants for the Ch.5 synthesis, if a consistent IC/period
can be matched (open question already noted in step 6 above); (c) T/UV-only
`plot_rmse.py` per-variant RMSE-vs-leadtime plots were not yet generated in this
session (only the control's were, back in §7) -- straightforward to add if wanted,
same command pattern as §7 with `--run_label` set per variant.

**Superseded by §15 below** (a real timestep-alignment bug was found in the
insertion code that produced this section's numbers -- all 3 variants were rerun
with a fix and the corrected results replace everything above).

---

## 15. Timestep-alignment bug found + fixed in insertion code, all 3 OSSE variants rerun (2026-08-24)

### The bug

User's hunch: after hard truth insertion, the inserted variable's own RMSE
should drop to ~0 and *stay* there (it's literally ERA5), but the §14 plots
instead showed a zigzag/sawtooth pattern post-injection. Investigated and
confirmed a real bug in `chunked_prediction_generator`'s injection block
(`graphcast/rollout.py`, ~line 446, present identically in all 3 variant copies
since they all derive from `osse_z_global/graphcast/`'s generalized version,
§13).

Root cause: for JIT-recompilation-avoidance (same trick noted in §10 as
harmless for the *non-injection* control run), predictions inside the chunk
loop carry a **fixed placeholder time coord** (always chunk 0's lead times,
e.g. `[6h,12h,18h,24h]`) -- the real absolute lead-time labels for the chunk
are only restored at the very end of the loop, *after* injection runs. The
injection code's alignment step,
```python
truth_chunk = truth_chunk.reindex_like(pred_da.isel(time=inject_idx), method="nearest")
```
was intended to align `truth_chunk` to `pred_da` on **lat/lon/level/batch**
only (truth_chunk's time axis was already exactly correct and correctly
ordered from the preceding `.sel(time=leads_to_inject)`). But `reindex_like`
reindexes on **every shared dimension**, including time -- and it reindexed
truth's real, far-future lead times (e.g. 726/732/738/744h for the chunk at
day 30) against `pred_da`'s fake placeholder labels (`[6,12,18,24]`). Since
all 4 real truth times are ~700h away from the placeholder labels, "nearest"
collapsed onto the same single source position (the chunk's earliest lead
time) for all 4 target positions. Net effect: only the *first* sub-step of
each `steps_per_chunk`-sized chunk got the correct truth injected; the other
`steps_per_chunk - 1` sub-steps silently got that same stale value repeated
instead of their own. With `--steps_per_chunk 4` (used by all 3 variants),
this produced an exact period-4 sawtooth in the inserted variable's RMSE --
confirmed directly against `osse_z_global`'s old CSV before fixing anything
(RMSE snapped to `0.0` on chunk boundaries, then rose to 234/370/515 within
each chunk, repeating every 4 steps).

### The fix

Pin `truth_chunk`'s time coord to `pred_da`'s (already positionally 1:1 by
construction) *before* calling `reindex_like`, so the reindex only does its
intended job on lat/lon/level/batch:
```python
target_for_align = pred_da.isel(time=inject_idx)
truth_chunk = truth_chunk.assign_coords(time=target_for_align["time"])
truth_chunk = truth_chunk.reindex_like(target_for_align, method="nearest")
```
Applied identically to all 3 copies: `osse_z_global/graphcast/rollout.py`,
`osse_t_global/graphcast/rollout.py`, `osse_uv_global/graphcast/rollout.py`.
Syntax-checked with `ast.parse()` on all three before rerunning.

**Sanity check before the full rerun**: after applying the fix and rerunning,
directly diffed `osse_z_global`'s chunk_0030 (first injected chunk) against
`truth_leadtime_2022-01-01IC.nc` at all 4 of its lead times (726/732/738/744h)
-- mean abs diff was exactly `0.0000` at **every** sub-step (previously only
the first would have matched). Confirms the fix, not just in theory but
against real output.

### Rerun: all 3 variants, full pipeline

Same GPU/CPU job scripts as §13/§14 (unchanged, still valid --
`sbatch --test-only` confirmed all 3 before submitting), just pointing at the
now-fixed `graphcast/` copies:

| Stage | Jobs | Result |
|---|---|---|
| GPU insertion rollout | 242643 (Z), 242644 (T), 242645 (UV) on `chip-gpu` | All COMPLETED, exit 0, ~10:25-13:35 elapsed, 60/60 chunks each |
| `compute_rmse.py` | 179301 (Z), 179302 (T), 179303 (UV) on `chip-cpu` | All COMPLETED, exit 0, ~20:08 elapsed, 12,480 rows each |
| `plot_rmse.py` + `compare_control_vs_osse.py` | run directly (fast) | All plots/CSVs regenerated, same filenames as §14 (overwritten) |

Post-fix, the injected variable's own RMSE is now **flat `0.0` at every single
post-injection step** (not just chunk boundaries) -- verified directly on the
new `rmse_osse_z_global_2022-01-01.csv`.

### Corrected findings (replaces §14's numbers)

Post-injection-window (day 30-60) mean %-RMSE-improvement vs. the free-running
control, from the corrected `compare_*_pct_improvement.csv` files:

- **Z (geopotential) insertion**: geopotential itself now improves a flat
  **~99.2-99.3%** at every level (was 75-94% under the bug -- the bug had
  itself been injecting corrupted/stale geopotential 3/4 of the time, which
  inflated the *apparent* remaining error). Cross-variable effects on T/U/V
  are still mild/mixed, consistent with §14's physical interpretation.
- **UV (u+v wind) insertion**: U and V both now improve **~99.1-99.2% flat**
  (was 46-87%). Still improves geopotential (**48-67%** at all levels except
  a small -9.2% at 50hPa) and temperature (**-17% to +48%**, negative only at
  50hPa) via geostrophic/thermal-wind balance, same physical story as §14 but
  with correct numbers.
- **Temperature insertion**: temperature itself now improves a flat
  **~99.1-99.2%** (was 72-94%). The T-only instability finding from §14
  **not only survives the fix, it's now substantially more severe**:
  geopotential RMSE is **74-476% worse** than the free-running control
  (previously -344% at its worst single level; now every level is worse, up
  to -476% at 50hPa), and U/V wind RMSE similarly blow up **35-753% worse**
  across most levels. This makes physical sense in hindsight: under the old
  buggy sawtooth, only 1-in-4 steps actually received the destabilizing
  univariate hard-replacement "shock" (the other 3 got a smoother, stale
  repeated value); with the bug fixed, every single 6h step now delivers a
  correct-but-unbalanced temperature update, compounding the dynamical
  imbalance more consistently over the 120 insertion steps. **The core
  physical finding from §14 (univariate hard T-insertion excites imbalance
  and degrades Z/U/V, mirroring known behavior in real data assimilation) is
  confirmed and strengthened by the fix, not undermined by it.**

### `final_results/` updated

`final_results/osse_{z,t,uv}_global/{csv,plots}/` were refreshed in place with
the corrected CSVs and PNGs (same filenames, old buggy versions overwritten).
`final_results/control/` is untouched (control never used the insertion code
path, so it was never affected by this bug -- see §12/§14, this exact code
path was confirmed "NOT yet exercised" by control).

### Status

Nothing currently running. This closes out the bug the user's hunch caught;
the §14 "next steps" list (softer T-insertion variant, cross-model comparison)
is still open for a future session.

**Superseded by §§16-20 below** (cross-model explanation write-up, latitude-
slice follow-up, land/ocean insertion study, and a perturbed-truth robustness
check were all completed in the same extended session).

---

## 16. Cross-model explanation write-up: `GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` (2026-08-24)

User asked for a detailed explanation of *why* GraphCast full's T-insertion
behaves so differently from Aurora/FCN/Prithvi's (those three show T-Ins
*helping* wind/mass fields; full GraphCast shows catastrophic degradation).
Investigated using the existing cross-model OSSE study curated at
`/umbc/rs/pi_halem/users/yashd1/Organized_results/` (coupling matrices for
Aurora/FCN/Prithvi, and the already-written `PHYSICAL_ANALYSIS_T_Ins.txt` /
`PHYSICAL_ANALYSIS.txt` reports, which had already established a "central
asymmetry" -- wind constrains mass more easily than mass constrains wind,
quasi-geostrophic theory -- and documented GraphCast-*Small*'s own equatorial
Wind-Ins blind spot in a prior, separate 1-degree study). Per the user's
explicit correction mid-investigation: `Organized_results/GraphCast/` is
GraphCast-*Small* results and must never be cited as data for full
GraphCast's numbers -- only used as independent prior supporting context for
a shared architectural signature.

Wrote `GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` (top-level, this
directory) with three architectural/theoretical candidate explanations: (1)
GraphCast's local message-passing GNN receptive field vs. Aurora/FCN/
Prithvi's larger-or-global receptive fields (windowed attention, spectral
operator, ViT); (2) no learned geostrophic-adjustment mechanism, evidenced by
GraphCast-Small's documented equatorial Wind-Ins failure and T-Ins temporal
sign-reversal; (3) full GraphCast's 37 forced vertical levels vs. the other
3 models' 13-14 (a genuine, checked configuration difference, not
speculation -- confirmed directly against `Organized_results/{Aurora,
FourCastNet}/coupling/scripts/*.py`'s `LEVELS = [...]` constants).

**User pushback, appropriately skeptical**: asked directly whether this was
"a real explanation I can put in a journal paper." Answer given: the
empirical finding (Section 3) and confound-ruling-out (Section 4) are solid;
the three architectural explanations are literature-consistent HYPOTHESES,
not proven by a controlled ablation (the 4 models differ simultaneously in
architecture, training data/truth source, resolution, IC date/season --
Organized_results itself uses Feb-2023 ICs for Aurora/FCN vs Jan-2022 for
GraphCast/Prithvi). User then asked to make this distinction explicit in the
document itself.

**Response**: added an epistemic-status framework to the top of the
document -- `[ESTABLISHED]` / `[HYPOTHESIS]` / `[NEEDS FOLLOW-UP]` tags on
every major claim and section header, an "at-a-glance claim summary" table,
explicit caveat paragraphs after each hypothesis, and a full new section
(now §10 after later renumbering, see §17-20 below) listing every confounded
variable between the 4 models and stating plainly that this is an
observational comparison, not a controlled ablation. This is the standing
convention for the whole document going forward -- any future addition to
this file should use the same three-tag system.

---

## 17. Latitude-slice stratification follow-up (§8.1 in the analysis file) -- completed 2026-08-24

The analysis file's own §8.1 recommended follow-up: stratify the full-
resolution T/Z/UV-insertion results by the same 4 narrow (+-5deg) NH
latitude slices (2N/14N/40N/60N) already used for Aurora/FCN/Prithvi in
`Organized_results/*/lat_bands/`, to test whether GraphCast full's T-Ins
failure is a Rossby-number/tropical phenomenon (like GraphCast-Small's
documented Wind-Ins blind spot) or something broader.

New shared script: `compute_latslice_rmse.py` (top-level) -- same streaming
chunk/ERA5-loading approach as `compute_rmse.py`, reduced over 4 narrow
cosine-weighted latitude bands instead of the whole globe, restricted to
300/500/1000 hPa. Smoke-tested on 1 chunk before the full run. New output
directory `latslice_analysis/{control,osse_z_global,osse_t_global,
osse_uv_global}/` (180056-180059 SLURM jobs on chip-cpu, ~18 min each, all
COMPLETED) plus `latslice_analysis/latslice_pct_improvement_summary.csv`.

**Result, genuinely new (not what either branch of the original hypothesis
predicted)**: T-insertion degrades U/V/Z at literally EVERY latitude slice
tested (2N through 60N) -- not a tropical-only phenomenon. Z-insertion and
UV-insertion, by contrast, DO reproduce the exact equator-to-pole
(Rossby-number) breakdown signature previously only documented for
GraphCast-Small in a different study -- now confirmed IN-STUDY, in the
full-resolution checkpoint, under two different insertion experiments. Also
cross-checked against Aurora's and Prithvi's own already-computed
latitude-slice values: their T-Ins is positive at every slice tested,
confirming "negative at every latitude slice" is specific to GraphCast full,
not a general property of T-insertion experiments.

Folded into `GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` as new §9 (with
downstream updates to the epistemic tags on §5.2's caveat and the
at-a-glance table -- §5.2's Z-Ins/UV-Ins claim upgraded to
`[ESTABLISHED, IN-STUDY]`; §5.2's T-Ins-specific application flagged as
needing revision, since the failure is broader than the Rossby-number story
that explains Z-Ins/UV-Ins).

---

## 18. Land-only / ocean-only insertion study: `extended_exp/` (2026-08-24)

User asked to replicate, for full GraphCast, the land-only/ocean-only
insertion study already run for Aurora/FCN/Prithvi
(`Organized_results/extended_exp/{Aurora,FCN,Prithvi}/{T_Land,T_Ocean,
UV_Land,UV_Ocean}/`) -- 6 experiments this time (T/UV/Z x Land/Ocean; the
reference study only had T and UV, Z-land/ocean is a genuine extension for
this project).

**Masking approach**: extended `chunked_prediction_generator` (in a new
generalized `graphcast/rollout.py`, based on `osse_z_global`'s already-fixed
copy) with a `region_mask: Optional[np.ndarray]` parameter -- a (lat,lon)
boolean array; True = insert truth there, False = keep the model's own
prediction. Built `build_land_ocean_mask.py` -> `land_sea_mask_0.25deg.nc`
directly from GraphCast's OWN `land_sea_mask` static input field (in the
sample dataset, fractional 0-1, thresholded >0.5 = land -- 33.9% land/66.1%
ocean, matches expected Earth land fraction) rather than sourcing a separate
mask file the way the other models' pipelines do -- keeps the mask exactly
self-consistent with what the model itself was trained on. This is more
efficient than the other models' approach of pre-masking (NaN-ing) the
entire truth file per region (would mean 6x137GB of duplicate truth data);
masking happens at runtime instead.

Unit-tested the merge logic with a synthetic predictor before running
anything on GPU (confirmed masked-in region gets truth, masked-out region
keeps the model's own prediction, and `region_mask=None` exactly preserves
the original global-insertion behavior -- a regression check for the
existing 3 `osse_*_global/` experiments, unaffected by this change).

Created `extended_exp/{T_Land,T_Ocean,UV_Land,UV_Ocean,Z_Land,Z_Ocean}/`,
each with its own `graphcast/` copy + `run_osse_insertion.py` (extended with
`--insert_region {land,ocean}` and `--land_mask_nc`) + `compute_rmse.py` +
`plot_rmse.py` + slurm scripts. All 6 GPU jobs (242833-242838) COMPLETED,
60/60 chunks each, 10-13 min. Verified masking correctness directly against
real output: T_Land matches ERA5 exactly over land (0.0 diff) and diverges
over ocean (2.95K); T_Ocean is the mirror image (0.0 over ocean, 4.1K over
land). RMSE-vs-ERA5 (jobs 180076-180081) and comparison-vs-control all
completed. Curated results copied to `extended_exp_results/{...}/{csv,
plots}/` (mirrors `final_results/`'s structure).

**Result**: partial-coverage T-insertion still catastrophically degrades
Z/U/V -- T_Land: Z -96.8%, U -82.8%, V -120.0%; T_Ocean (worse, since 66% of
the grid forced vs 34%): Z -168.9%, U -173.0%, V -170.5%. Confirms the
instability is not an artifact of forcing 100% global coverage. Z-insertion
and UV-insertion remain mild/positive under both land-only and ocean-only
restriction (e.g. UV_Ocean -> Z +40.9%, UV_Ocean -> U +63.8%), matching
their global-insertion behavior. Not yet folded into
`GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` as of this log -- reported to
user in-conversation, offered to add as a new section, not yet requested.

---

## 19. Correlated OSSE Perturbation (COP) experiments: `T_PERTURBED_CORR/`, `UV_PERTURBED_CORR/` (2026-08-24)

User asked to port the "perturbed truth insertion" experiments already run
for all 3 other models
(`/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/` --
`METHOD_OVERVIEW.md` is the model-agnostic method writeup) to full GraphCast.
Read the reference implementation directly (
`fcn_inference/{t,uv}_perturbed_corr/OSSE-FourCastNetv2/earth2mip/networks/
__init__.py::_correlated_level_fields()` /
`_generate_correlated_wind_perturbation()` / `_streamfunction_to_uv()`) to
get the exact numerics, not just the prose description.

**Method** (full citations in `PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md`
and now also in `GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` §10's
reference list): instead of hard-replacing with exact ERA5 truth, hard-
replace with truth + noise that is (1) horizontally correlated -- Gaussian
Random Field via `scipy.ndimage.gaussian_filter` on white noise, ~250km
correlation length; (2) vertically correlated -- exponential decay in
log-pressure space, Cholesky-mixed across levels; (3) for wind only,
non-divergent -- both u and v derived from ONE random streamfunction
realization (not independent per-component noise, which would inject
spurious divergence). Magnitude calibrated to Ingleby (2017)-style
radiosonde error tables.

**GraphCast-specific port**: wrote `perturbation.py` (numpy port of the
reference's core functions) and extended the reference's 13-level sigma
tables to GraphCast's 37 native levels via log-pressure linear interpolation
(flat-extrapolated above 50hPa, where no reference curve exists -- documented
limitation in `perturbation.py`'s own docstring). The vertical correlation
MATRIX needed no interpolation (analytic function of the pressure levels
themselves). Chose to keep GraphCast's own 37-level insertion convention
(this project's established distinguishing feature) rather than reverting to
13 levels to match the other models exactly.

Created `T_PERTURBED_CORR/` and `UV_PERTURBED_CORR/`, each with a
`graphcast/` copy whose injection block was hand-modified (dedicated copy
per experiment, matching the reference implementation's own design choice --
the perturbation logic differs enough between scalar-T and joint-streamfunction-
wind that a shared parameterized function wasn't worth the refactor).
`run_osse_insertion.py` itself is UNMODIFIED from `osse_t_global`/
`osse_uv_global` -- the noise is generated inside `rollout.py`'s injection
loop at runtime, not baked into a pre-perturbed truth file.

**Real bug found (only after ~7 min of GPU time was spent on each of two
jobs, not caught by unit testing beforehand -- see below)**: first
submission (242858, 242859) BOTH
FAILED after ~7 min with `TypeError: unexpected PRNG key type
<class 'numpy.random._generator.Generator'>` inside `jax.random.split`. Root
cause: the perturbation code's local RNG variable was named `rng`, colliding
with `chunked_prediction_generator`'s own JAX PRNGKey parameter of the SAME
NAME in the SAME function scope -- assigning `rng = np.random.default_rng(...)`
inside the injection block silently clobbered the outer JAX key, which then
broke the NEXT chunk's `rng, this_rng = split_rng_fn(rng)` call. A
single-chunk unit test (used for T_PERTURBED_CORR before first submission)
missed this entirely, since there was no "next chunk" to corrupt. Diagnosed
immediately from the traceback, fixed by renaming to `noise_rng` in both
experiments' `graphcast/rollout.py`, and reverified with a genuinely
multi-chunk (3-chunk) synthetic-predictor test that specifically exercises
injection starting mid-run followed by a subsequent chunk's rng-split --
confirmed no crash and correct injection timing before resubmitting.
**Lesson for future sessions**: any unit test of chunked_prediction_generator
that only uses `num_steps == steps_per_chunk` (i.e. exactly 1 chunk) cannot
catch bugs that only manifest across the chunk-to-chunk boundary (this is
the second time in this project a single-chunk test has missed something --
see also the region_mask work in §18, which happened to not have this
particular class of bug but used the same weak test shape initially).

Resubmitted (242861, 242862), both COMPLETED (15-18 min), 60/60 chunks each.
Verified perturbation magnitude directly against real GPU output: RMS
difference from truth at 500hPa matched the calibrated sigma almost exactly
(T: 0.7006K vs 0.7K target; U: 1.2000 m/s vs 1.2 m/s target), near-zero mean
bias -- confirms correctly-calibrated, unbiased noise, not just "some noise."
RMSE-vs-ERA5 (jobs 180167, 180168) and comparison-vs-control completed.
Curated results in `perturbation_results/{T_PERTURBED_CORR,
UV_PERTURBED_CORR}/{csv,plots}/`. Full method/hyperparameter/validation
writeup in `PERTURBATION_IMPLEMENTATION.md` (top-level, mirrors
`PERTURBATION_EXPERIMENTS/{FCN,Aurora,Prithvi}/IMPLEMENTATION.md`'s
structure).

**Result**: the T-insertion instability and the UV-insertion success are
BOTH essentially unchanged by adding realistic observation-error-scale noise
instead of exact truth (T-Ins -> Z: -199.6% exact vs -196.9% perturbed;
UV-Ins -> U: +99.2% exact vs +89.9% perturbed). Rules out "the instability is
an artifact of forcing the model with an unrealistically perfect signal" as
an explanation -- this is specifically about WHICH field and HOW the
correction is structured (hard, full-strength, every 6h), not about how
clean the forcing is. Does NOT test the separate, still-open damped/partial-
nudging idea (weaker correction strength, analysis file §8.2) -- folded into
`GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` as new §10 (with sections 10+
renumbered accordingly: old §10 Limitations -> §11, old §11 Data Provenance
-> §12), including an explicit note distinguishing this from §8.2.

---

## 20. Status as of 2026-08-24 (end of this extended session)

Nothing running. Completed and documented this session:
- §16: `GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt` written, with an
  epistemic-status (`[ESTABLISHED]`/`[HYPOTHESIS]`/`[NEEDS FOLLOW-UP]`)
  framework -- the standing convention for that file going forward.
- §17: latitude-slice follow-up completed and folded into the analysis file
  as §9 -- a genuinely new finding (T-Ins fails everywhere, Z-Ins/UV-Ins fail
  only in the tropics, reproducing GraphCast-Small's signature in-study).
- §18: 6-experiment land/ocean insertion study (`extended_exp/`) -- results
  computed and curated, NOT yet folded into the analysis file.
- §19: 2-experiment correlated-perturbation study (`T_PERTURBED_CORR/`,
  `UV_PERTURBED_CORR/`) -- results computed, curated, AND folded into the
  analysis file as §10.

**Open for a future session** (from the analysis file's own §8, still
unaddressed): §8.2, a damped/partial-nudging T-insertion variant (weaker
correction strength, not yet tested by anything in this session); §8.3, a
13-level vertical-DOF ablation (repeat T-insertion forcing only 13 of
GraphCast's 37 levels, to isolate the vertical-resolution hypothesis from
the architecture hypothesis -- currently tagged `[NEEDS FOLLOW-UP]` in the
analysis file, unresolved). Also open: folding §18's land/ocean results into
the analysis file (offered to the user, not yet requested); cross-model
comparison of the perturbed/land-ocean results against the equivalent
Aurora/FCN/Prithvi experiments in `Organized_results/`, if a consistent
IC/period can be matched (same open question noted since §12).

### Full directory map addendum (new since §8's map)

```
Graphcast-full/
  GRAPHCAST_T_INSERTION_ANOMALY_ANALYSIS.txt   <- cross-model explanation, epistemic-tagged
  PERTURBATION_IMPLEMENTATION.md               <- COP method port writeup
  perturbation.py                              <- shared COP numerics (37-level)
  build_land_ocean_mask.py, land_sea_mask_0.25deg.nc
  compute_latslice_rmse.py
  latslice_analysis/{control,osse_z_global,osse_t_global,osse_uv_global}/   <- §17
  extended_exp/{T_Land,T_Ocean,UV_Land,UV_Ocean,Z_Land,Z_Ocean}/            <- §18
  extended_exp_results/{...}/{csv,plots}/                                   <- §18, curated
  T_PERTURBED_CORR/, UV_PERTURBED_CORR/                                     <- §19
  perturbation_results/{T_PERTURBED_CORR,UV_PERTURBED_CORR}/{csv,plots}/    <- §19, curated
  final_results/{control,osse_z_global,osse_t_global,osse_uv_global}/{csv,plots}/  <- §15, curated
```

---

## 21. `deterministic_run/` control-overlay + `osse_t_global` corrected to 26-level insertion (2026-09-06)

Picking back up in `deterministic_run/` (the L40S-pinned + `xla_gpu_deterministic_ops`
bit-exact-through-day-30 re-run of the main 4-variant study, IC 2022-01-01,
see §15 for the original curation). Two changes requested this session:

**Control-overlay on the 3-panel `rmse_vs_leadtime_*` plots.** These plots
(one PNG per variable, 3 panels for 300/500/1000hPa) previously showed only
the run's own RMSE curve — unlike the single-panel `compare_*` plots (from
`compare_control_vs_osse.py`), which always overlaid control as a black
dotted line by design. Added an optional `--control_csv` arg to
`osse_t_global/plot_rmse.py` (then copied to `osse_z_global`/`osse_uv_global`/
`dropped_insertion`): when given, overlays control's RMSE as a black dotted
line in every panel, with a legend. Regenerated all 4 variants'
`rmse_vs_leadtime_*` plots via a `replot_with_control_overlay.sbatch` job —
completed cleanly, synced into `GC_FINAL_RESULTS/` (see below).

**`osse_t_global` corrected: full-37-level T-insertion → 26-level (drop top 11
stratospheric levels ≤100hPa)**, to match `dropped_insertion`'s established
convention. Backed up the pre-fix results to
`osse_t_global/{full_predictions,results}_FULL37LEVEL_OLD/` (kept, not
deleted). First resubmit attempt failed immediately: `osse_t_global`'s local
copy of `run_osse_insertion.py`/`graphcast/rollout.py` predates the
`--insert_levels` support that `dropped_insertion`'s copy already had —
patched both files to add it (three additive edits: new `--insert_levels`
CLI arg, a `level_mask`/`.where()` step in `rollout.py`'s injection loop
applied AFTER the truth's `reindex_like` — doing it before would let
`reindex_like(method="nearest")` silently backfill "excluded" levels from
the nearest *available* truth level instead of leaving them uninserted —
and threading `insert_levels` through the `chunked_prediction_generator`
call). Verified via `diff` against `dropped_insertion`'s already-working
implementation (functionally identical, docstrings/arg-order the only
difference) before resubmitting. Resubmitted (job 248698), ran clean,
verified bit-exact vs. control through day 30 (`max abs diff: 0.0` for
temperature/2m_temperature/u_component_of_wind). Reran RMSE + control-overlay
plots + `compare_control_vs_osse.py` + reorganized into
`full_predictions/`+`results/{csv,plots}/`. **Note left for the user**:
`osse_t_global` and `dropped_insertion` are now functionally identical
(same 26-level definition) — the only difference is naming; `dropped_insertion`
may be worth retiring or repurposing since its whole reason for existing
(comparison against a full-37-level baseline) no longer has that baseline.

## 22. `GC_FINAL_RESULTS/` — curated 4-run copy of `deterministic_run/` (2026-09-06)

Created `GC_FINAL_RESULTS/{control,osse_t_global,osse_z_global,osse_uv_global}/
{csv,plots}/`, copied from `deterministic_run/`'s reorganized `results/`
dirs (post-§21 fixes, so `osse_t_global` here is already the corrected
26-level version). `dropped_insertion` deliberately left out (side-experiment,
not one of the "4 main runs" — offered to add, not requested).

## 23. New study: `GC-DETER-2023/` — deterministic re-run on a 2023-02-01 IC (2026-09-06 – 2026-09-08)

User requested the exact same deterministic re-run (all determinism
settings from §14/§21: `--constraint=L40S`, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`,
`NVIDIA_TF32_OVERRIDE=0`, `jax_default_matmul_precision=highest`) but on a
**2023-02-01T06:00Z IC** instead of 2022-01-01 — matching the IC already used
by the sibling `prithvi-feb23` study (see that project's session_log.md) for
eventual cross-model comparison. No existing ERA5/GraphCast-format data
covered this date (the `AURORA_DATA/2023-02-01-*.nc` files found in a search
are Aurora's 13-level convention, not usable) — required a fresh 3-month
(Feb–Apr 2023) CDS download.

Built `GC-DETER-2023/` from scratch, adapting the existing 2022 recipe
scripts one month at a time from `one_month_osse_2023/`'s Mar-2023 pattern
(same day-boundary IC-surface split, same `SHARED_GRIB_IDX` pattern for the
larger 3-month bulk grib):
- `download_era5_37level_feb2023.py`, `download_era5_ic_surface_feb2023.py`,
  `download_era5_ic_extras_feb2023.py` — 3 CDS downloads (chip-cpu)
- `build_ic_feb2023.py` — assembles the 3-timestep 2023-02-01 IC file
- `prepare_truth_feb2023.py` — 240-step (60-day) lead-time-indexed truth
  Dataset (unlike `one_month_osse_2023`'s 30-day one-month scope)
- 4 base-run dirs (`control`, `osse_t_global`, `osse_z_global`,
  `osse_uv_global`), each with its own `graphcast/` package copy +
  `run_osse_insertion.py`/`compute_rmse.py`/`plot_rmse.py` copied from
  `deterministic_run/osse_t_global` (already patched for `--insert_levels`
  and control-overlay per §21) — `osse_t_global` here uses the 26-level
  convention from the start (no separate "old" version needed this time).

Full pipeline self-chained via SLURM `--dependency=afterok`: 3 downloads →
`build_ic` → `prepare_truth` → 4 GPU rollouts. All completed cleanly
(only harmless xarray `FutureWarning`s in the logs, no real errors). Then
`finish_base_runs_feb2023.sbatch` (compute_rmse + control-overlay plot +
`compare_control_vs_osse.py` + reorganize, for all 4 variants) — first
attempt (249063) **failed partway through**: `control/plot_rmse.py`
(copied fresh, never patched) doesn't have the `--run_label`/`--control_csv`
args that `osse_t_global`/`osse_z_global`/`osse_uv_global`'s copies do (those
were patched in §21; control's copy in the *old* `deterministic_run/` study
was never patched either, since control has nothing to overlay against
itself) — fixed by switching that one call to `--ic_label` instead (the
arg control's `plot_rmse.py` actually supports), resubmitted (249094),
completed cleanly (all 4 variants: RMSE CSV, control-overlay plots for the
3 OSSE variants, `compare_*` plots, reorganized into
`full_predictions/`+`results/{csv,plots}/`).

## 24. `GC-DETER-2023` extended experiments: combined T+UV, land-only T, ocean-only T (2026-09-06 – 2026-09-08)

Requested: 3 new experiments mirroring the land/ocean-masked insertion study
already done for Prithvi (`prithviOSSE/{land_temp_ins,ocean_temp_ins}`, see
also §18's own `extended_exp/{T_Land,T_Ocean,...}` for this project's earlier
land/ocean work) — `combined_tuv_ins` (T+U+V together), `land_temp_ins`,
`ocean_temp_ins` — on the new 2023-02-01 IC.

**New capability added to `graphcast/rollout.py`** (`chunked_prediction_generator`),
additively, alongside the existing `insert_levels`:
- `insert_levels_by_var: Optional[dict]` — per-variable level-restriction
  override (e.g. temperature stays 26-level-restricted in the combined T+UV
  run while u/v insert at all 37 levels, matching each variable's own
  established single-var convention instead of forcing one `insert_levels`
  list onto every inserted variable).
- `insert_mask: Optional[xr.DataArray]` — a boolean `(lat, lon)` mask,
  applied via `.where(insert_mask, target_for_align)` **after** the
  level-mask step (same ordering rationale as `insert_levels`: must happen
  after `reindex_like`, against the model's own prediction, not truth,
  at masked-out points).

Threaded both through `run_osse_insertion.py` as `--temp_insert_levels`,
`--insert_mask_var {land,ocean}`, `--land_threshold` (default 0.5). **Land/ocean
mask source: GraphCast's own `land_sea_mask` static field**, already present
in the IC file at native 0.25°/721×1440 resolution (fractional 0–1,
~33.9% of the grid >0.5) — no regridding needed, unlike Prithvi's approach
(which had to interpolate MERRA-2's FRLAND onto its own grid). Verified the
mask's grid matches the prediction chunks' `lat`/`lon` exactly before use.

Copied the patched `run_osse_insertion.py`/`graphcast/` into 3 new dirs
(`combined_tuv_ins`, `land_temp_ins`, `ocean_temp_ins`), each with a single
sbatch doing rollout → compute_rmse → control-overlay plot →
`compare_control_vs_osse.py` → reorganize, dependent on §23's
`finish_base_runs` (needs the control RMSE CSV first).

**Two incidents during this run, both recovered without data loss:**
1. First submission of the 3 new jobs used `--dependency=afterok:$J0` where
   `$J0` was captured as `"249063;chip-gpu"` (the `;cluster` suffix
   `sbatch --parsable` appends on this multi-cluster setup) — `sbatch`
   rejected the dependency string outright. Same class of bug as an earlier
   session's chip-cpu chaining issue; fixed by stripping to the bare numeric
   ID before passing to `--dependency`.
2. A `sed`-based copy of `land_temp_ins`'s sbatch to build `ocean_temp_ins`'s
   used underscore-based patterns (`land_temp_ins` → `ocean_temp_ins`) but
   the `#SBATCH --job-name` line uses **hyphens** (`land-temp-ins`) — silently
   left mislabeled as `graphcast-land-temp-ins-feb2023` even though every
   other part of the script (paths, `--insert_mask_var ocean`, etc.) was
   correctly substituted (confirmed via `diff` before running). Caught by
   inspecting `squeue`'s job-name column; fixed with a direct `Edit`, no
   rerun needed since the job hadn't started yet.
3. **`ocean_temp_ins`'s rollout job (249100) was preempted by the scheduler
   at chunk 43/60**, and again on the immediate retry (249124) — this time
   at chunk 42/60 of `compute_rmse`, i.e. *after* the rollout itself had
   actually finished (60/60 chunks already on disk; the preemption hit
   mid-RMSE, not mid-rollout). Neither preemption auto-requeued (job
   vanished from `squeue` entirely rather than going back to `PD`). First
   retry re-ran the whole thing from scratch (~1.5hr). Second time, wrote a
   `finish_ocean_temp_ins_feb2023.sbatch` that **skips the rollout**
   (asserts exactly 60 `chunk_*.nc` files exist first, aborts loudly if
   not) and resumes directly from `compute_rmse` — avoided re-spending
   another ~1.5hr of GPU time. Completed cleanly (249246): 2 CSVs, 17 plots.

All 3 extended experiments now match the 4 base runs' full deliverable set
(raw chunks in `full_predictions/`, RMSE+plots+compare in `results/{csv,plots}/`).

## 25. `GC-DETER-2023` regime analysis (lat-slice / terrain / cloud) — t/z/uv insertion only (2026-09-07)

Reused this project's **existing** GraphCast regime-analysis scripts from
the original 2022 study — `compute_terrain_rmse.py`, `compute_cloud_rmse.py`,
`compute_latslice_rmse.py` (all three already existed at the top level,
same streaming-chunk architecture as `compute_rmse.py`, restricted to
300/500/1000hPa) — copied into `GC-DETER-2023/`, patched only the hardcoded
`era5_pressure_2022_*.grib` glob → `era5_pressure_2023_*.grib`. `terrain_mask_0.25deg.nc`
needed no changes (grid-invariant, same native 0.25° GraphCast grid
regardless of IC date). Cloud regime needed a 2023 total-cloud-cover
truth file — found one already downloaded from an earlier, unrelated
session: `/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/era5_total_cloud_cover_2023_Feb_Apr_6hourly.nc`
(covers 2023-02-01 through 2023-04-30, comfortably spans the 60-day window).

**Scope note**: originally ran this for all 4 base variants
(control+t/z/uv), but the user then explicitly narrowed scope to **only
the 3 insertion runs (t/z/uv), not control** — cancelled the
control-inclusive job mid-queue (before it started), dropped `control` from
the loop, resubmitted. Also explicitly excluded the 3 extended experiments
(§24) from regime analysis — cancelled that job outright (was queued,
never started). Final scope: `regime_analysis_results/{lat_slice,terrain,
cloud}/csv/rmse_{latslice,terrain,cloud}_osse_{t,z,uv}_global.csv` — 9 CSVs
total, no errors.

## 26. `GC-DETER-2023` spatial bias maps (`.npy`, day 30–60 mean) — control + t/z/uv (2026-09-07)

New script, `compute_bias_maps.py` — same streaming chunk/ERA5-loading
architecture as `compute_rmse.py`/the regime scripts, but accumulates raw
`(pred - truth)` (not squared) over only the **mature/post-injection window**
(`lead_days >= 30.0`, matching this study's `--inject_from_step 120` 0-based
truth-insertion start used everywhere else), then divides by count at the
end. Same mature-phase convention (day 30–60 mean bias) as the reference
cross-model bias study (`Organized_results/GraphCast/bias/scripts/
section_gc_bias_analysis.py`, `Organized_results/Prithvi/bias/scripts/
section_5_8_prithvi_bias_analysis.py`) — but computed at GraphCast-full's own
**native 0.25° grid (721×1440)**, no downscaling to 1° like the older
GraphCast-Small reference study needed.

Saved as `.npy` per variable/level (`bias_<var>_<level>hpa.npy`, T/U/V/Z ×
300/500/1000hPa) plus `lat.npy`/`lon.npy`, one dir per variant:
`bias_maps/{control,osse_t_global,osse_z_global,osse_uv_global}/` — scoped
to the same 4 (control + 3 main insertions, no extended experiments) per the
§25 scoping decision. Ran cleanly, no errors — 12 `.npy` files + lat/lon per
variant, 56 total.

## 27. `GC-DETER-2023/file_paths.txt` (2026-09-07 – 2026-09-08)

Wrote a comprehensive path index at `GC-DETER-2023/file_paths.txt` —
input/truth data, top-level scripts, per-variant (all 7: 4 base + 3
extended) raw-chunk directories and RMSE/plot/comparison outputs, the §25
regime-analysis CSVs, the §26 bias-map `.npy` files, and a pointer to
`GC_FINAL_RESULTS/` (§22, the separate 2022-01-01 study, explicitly noted as
NOT this study). Regenerated once after §24's `ocean_temp_ins` finished, to
pick up its previously-`out/`-only (not yet reorganized) results section.

## 28. Status as of 2026-09-08 (end of this session)

Nothing running. Completed and documented this session:
- §21: `deterministic_run/` control-overlay plots + `osse_t_global` corrected
  to 26-level insertion (old 37-level version preserved as `_FULL37LEVEL_OLD`).
- §22: `GC_FINAL_RESULTS/` curated 4-run copy.
- §23: New `GC-DETER-2023/` study built end-to-end on a 2023-02-01 IC
  (download → build_ic → prepare_truth → 4 base rollouts → finish), matching
  the sibling `prithvi-feb23` study's IC for future cross-model comparison.
- §24: 3 extended experiments (`combined_tuv_ins`, `land_temp_ins`,
  `ocean_temp_ins`) — required new `insert_levels_by_var`/`insert_mask`
  capability in `rollout.py`, land/ocean mask sourced from GraphCast's own
  native `land_sea_mask` field. Survived 2 scheduler preemptions on
  `ocean_temp_ins` (documented recovery approach: skip-rollout finish-only
  resubmission).
- §25: Regime analysis (lat-slice/terrain/cloud), scoped to t/z/uv only.
- §26: Spatial bias maps (`.npy`, day 30–60 mean), scoped to control+t/z/uv.
- §27: `file_paths.txt` written and kept current.

**Open for a future session**: the §21 note about `osse_t_global` and
`dropped_insertion` now being functionally identical (naming/retirement
decision still not made); regime analysis and bias maps were deliberately
NOT extended to the 3 §24 extended experiments (explicit scope decision,
not an oversight — revisit if the user wants that coverage later);
cross-model comparison against `prithvi-feb23`'s equivalent 2023-02-01
results (both studies now share an IC, RMSE convention, and — as of §26 —
a matching day-30-60 spatial bias convention, making this comparison
straightforward whenever wanted).

### Full directory map addendum (new since §20's map)

```
Graphcast-full/
  deterministic_run/osse_t_global/{full_predictions,results}_FULL37LEVEL_OLD/  <- §21, preserved
  GC_FINAL_RESULTS/{control,osse_t_global,osse_z_global,osse_uv_global}/{csv,plots}/  <- §22
  GC-DETER-2023/                                              <- §23-27, new 2023-02-01 IC study
    dataset_samples/, truth/, era5_feb2023/                   <- §23, IC/truth data
    {control,osse_t_global,osse_z_global,osse_uv_global}/     <- §23, 4 base runs
      {full_predictions,results}/{csv,plots}/
    {combined_tuv_ins,land_temp_ins,ocean_temp_ins}/          <- §24, extended experiments
      {full_predictions,results}/{csv,plots}/
    compute_terrain_rmse.py, compute_cloud_rmse.py, compute_latslice_rmse.py  <- §25
    regime_analysis_results/{lat_slice,terrain,cloud}/csv/    <- §25, t/z/uv only
    compute_bias_maps.py                                      <- §26
    bias_maps/{control,osse_t_global,osse_z_global,osse_uv_global}/  <- §26, .npy maps
    file_paths.txt                                             <- §27
```

## 29. `GC-DETER-2023` Control regime analysis backfilled — Latitude/Cloud/Terrain Results (2026-09-09)

User noticed the curated `Graphcast deterministic/Graphcast thesis results/`
copy had Latitude Results, Cloud Results, and Terrain Results for Control
missing entirely — only RMSE and Spatial Bias existed there (§25/§26 scoped
the regime-analysis SLURM jobs to `osse_{t,z,uv}_global` only, never
`control`, so this was a real gap, not a copy oversight).

Wrote `GC-DETER-2023/slurm/regime_analysis_control_feb2023.sbatch` — same
`compute_terrain_rmse.py`/`compute_cloud_rmse.py`/`compute_latslice_rmse.py`
scripts as §25, same `TERRAIN_MASK`/`TCC_NC` inputs, pointed at
`control/full_predictions/rollout_60day_2023-02-01` instead of an OSSE
variant's predictions. Submitted as job **250891** (chip-gpu), ran ~50min,
completed cleanly (11520 rows in the lat-slice CSV alone). Wrote:
- `GC-DETER-2023/regime_analysis_results/terrain/csv/rmse_terrain_control.csv`
- `GC-DETER-2023/regime_analysis_results/cloud/csv/rmse_cloud_control.csv`
- `GC-DETER-2023/regime_analysis_results/lat_slice/csv/rmse_latslice_control.csv`

then auto-copied (same sbatch script) into:
- `Graphcast deterministic/Graphcast thesis results/Control/Terrain Results/CSV/rmse_terrain_control.csv`
- `Graphcast deterministic/Graphcast thesis results/Control/Cloud Results/CSV/rmse_cloud_control.csv`
- `Graphcast deterministic/Graphcast thesis results/Control/Latitude Results/CSV/rmse_latslice_control.csv`

Control's thesis-results folder now has the same 5-subfolder structure
(RMSE, Spatial Bias, Terrain/Cloud/Latitude Results) as every other
experiment in `Graphcast thesis results/`.

## 30. ERA5 specific-humidity (`q`) download for `GC-DETER-2023` (2026-09-09)

User asked whether ERA5/MERRA-2 truth exists for specific humidity (`q`).
Found: the main 60-day verification truth grib
(`era5_feb2023/era5_pressure_2023_febapr.grib`, built by
`download_era5_37level_feb2023.py`) only has `z,t,u,v` — no humidity.
`download_era5_ic_extras_feb2023.py` had downloaded `q`/`w` at all 37
levels, but **only for the 2 IC timesteps** (2023-02-01 00:00/06:00) — that
was for building the initial condition, not for verification.

Wrote `download_era5_q_feb2023.py` (new) — same day/month/time coverage as
the T/Z/U/V truth download (Feb/Mar/Apr 2023, 6-hourly), but only
`specific_humidity` at 4 levels (300/500/850/1000 hPa) to keep the download
small, since those are the levels this study reports on. Wrapper
`slurm/run_dl_era5_q_feb2023.sbatch` (chip-cpu, same CDS credentials pattern
as every other ERA5 download here). Submitted as job **274423**, completed
in ~7min. Output: `GC-DETER-2023/era5_feb2023/era5_q_2023_febapr.grib`
(2.96 GB). No downstream RMSE/analysis script written against this yet —
just the raw truth data, in case Q-based GraphCast analysis is requested
later (Prithvi's own MERRA-2 truth already has full `QV` coverage at all 14
levels across the whole 70-day window, confirmed separately — see the
equivalent `prithvi-feb23/session_log.md` entry, no fix needed there).

## 31. Correlated OSSE Perturbation (COP) ported to `GC-DETER-2023` (2026-09-09)

User requested the same COP method already implemented at the repo root
(§ not in this log — `perturbation.py`, `T_PERTURBED_CORR/`,
`UV_PERTURBED_CORR/`, built for the older Dec2021/`deterministic_run`
study; see `PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md` for the full
cross-model method) ported onto **this** (`GC-DETER-2023`, Feb2023 IC)
study, in dedicated directories, not mixed with any existing experiment.

Created `GC-DETER-2023/T_PERTURBED_CORR/` and `GC-DETER-2023/UV_PERTURBED_CORR/`,
each fully self-contained: own `graphcast/` package copy (from
`osse_t_global/graphcast/` and `osse_uv_global/graphcast/` respectively —
this study's own, newer copies, not the repo-root reference's older ones),
own `perturbation.py` (copied verbatim from the repo root — generic,
grid/level-agnostic beyond the 0.25deg/37-level constants), own
`run_osse_insertion.py` (copied unmodified from `osse_t_global`/
`osse_uv_global`), own `slurm/`, `out/`, `logs/`.

**T experiment — required an adaptation, not a verbatim port.** This
study's `osse_t_global/graphcast/rollout.py` has newer
`insert_levels`/`insert_mask` functionality (added since the repo-root
reference was written) that restricts T-insertion to 26 levels (excludes
the 11 stratospheric levels <=100hPa, this study's own corrected
convention — see §21). The repo-root reference COP port adds noise
unconditionally to the whole truth chunk right before the final assignment
loop, which would be WRONG here: it would add noise to `target_for_align`
(the excluded levels' own model prediction) after the `.where()` mask
already replaced truth with the model's own state there. Fixed by moving
the COP noise-generation block (`2a-cop`) to run immediately after the
`reindex_like` step and BEFORE the `insert_levels`/`insert_mask` `.where()`
restriction — so noise only ever reaches truth values that actually get
inserted. Verified this reasoning with a standalone level-reorder test
(`_cop.LEVELS_37[reorder_idx]` against the 26-level truth-level array) before
submitting to GPU.

**UV experiment — verbatim port.** `GC-DETER-2023/osse_uv_global/graphcast/
rollout.py` is byte-identical to the repo-root reference's own
`osse_uv_global` copy at the injection block (confirmed via `diff`, exit 0),
so the wind COP patch (shared-streamfunction cache for u/v, keyed by global
step) was ported line-for-line with no adaptation needed.

Both `rollout.py` edits compile-checked (`py_compile`) and the
`perturbation.py` numerics (unit-variance per-level fields, 26-level
reorder-index mapping, non-divergent wind construction) were sanity-tested
standalone against a synthetic grid before GPU submission.

New sbatch scripts (mirroring `osse_t_global`/`osse_uv_global`'s own Feb2023
job scripts): `T_PERTURBED_CORR/slurm/run_t_perturbed_corr_feb2023.sbatch`
(same 26-level `--insert_levels` list as `osse_t_global`),
`UV_PERTURBED_CORR/slurm/run_uv_perturbed_corr_feb2023.sbatch` (global, all
37 levels, matches `osse_uv_global`). Submitted:
- **job 250945** — T_PERTURBED_CORR, chip-gpu
- **job 250946** — UV_PERTURBED_CORR, chip-gpu

Both running as of this log (not yet complete). Output will land in
`T_PERTURBED_CORR/out/t_perturbed_corr_2023-02-01/` and
`UV_PERTURBED_CORR/out/uv_perturbed_corr_2023-02-01/` (chunked NetCDF,
same convention as every other variant). Documented in a new
`GC-DETER-2023/PERTURBATION_IMPLEMENTATION.md` (mirrors the repo-root
`PERTURBATION_IMPLEMENTATION.md`'s structure, states the T-experiment
adaptation explicitly). No RMSE/analysis pipeline run yet — pending rollout
completion.

## 32. Status as of 2026-09-09 (this session)

Running (chip-gpu): job 250945 (`T_PERTURBED_CORR`), job 250946
(`UV_PERTURBED_CORR`) — both §31, not yet complete as of this entry.

Completed and documented this session:
- §29: Control regime analysis backfilled (terrain/cloud/lat_slice), closes
  the `Graphcast thesis results/Control/` gap.
- §30: ERA5 `q` truth downloaded (300/500/850/1000hPa, full 60-day window) —
  data only, no analysis script yet.
- §31: COP ported to `GC-DETER-2023` in 2 new dedicated directories.

**Open for a future session**: `T_PERTURBED_CORR`/`UV_PERTURBED_CORR`'s
RMSE/comparison pipeline once rollouts finish (compare against `control/`
AND against `osse_t_global`/`osse_uv_global`, the unperturbed hard-
replacement versions, per `PERTURBATION_IMPLEMENTATION.md` §"Analysis
pipeline"); no Q-based RMSE/bias script written yet against §30's new ERA5
humidity truth; cross-model comparison against `prithvi-feb23` (still open,
carried over from §28).

### Full directory map addendum (new since §28)

```
Graphcast-full/
  GC-DETER-2023/
    era5_feb2023/era5_q_2023_febapr.grib                      <- §30, new
    download_era5_q_feb2023.py, slurm/run_dl_era5_q_feb2023.sbatch  <- §30
    regime_analysis_results/{terrain,cloud,lat_slice}/csv/rmse_*_control.csv  <- §29, new
    slurm/regime_analysis_control_feb2023.sbatch               <- §29
    T_PERTURBED_CORR/                                          <- §31, new, dedicated
      graphcast/, perturbation.py, run_osse_insertion.py, slurm/, out/, logs/
    UV_PERTURBED_CORR/                                         <- §31, new, dedicated
      graphcast/, perturbation.py, run_osse_insertion.py, slurm/, out/, logs/
    PERTURBATION_IMPLEMENTATION.md                             <- §31, new
Graphcast deterministic/Graphcast thesis results/Control/
  {Terrain,Cloud,Latitude} Results/CSV/rmse_*_control.csv       <- §29, new
```

## 33. Spillover-distance decay + propagation-error maps for `land_temp_ins`/`ocean_temp_ins` (2026-09-09)

User pointed at the AMS paper's `AMS updated.docx.txt` (Appendix A.1,
"Spillover-distance decay") and the reference implementation
`/umbc/rs/pi_halem/users/yashd1/spillover_analysis.py` (built for Aurora/
FCNv2/Prithvi's original Jan2022 land/ocean T-insertion runs), and asked for
the same two analyses ported onto **this** study's `land_temp_ins`/
`ocean_temp_ins` (spillover decay, both) and `land_temp_ins` only
(propagation maps).

**Spillover-distance decay.** New `GC-DETER-2023/spillover_analysis/
spillover_analysis.py` -- same method as the reference (great-circle
distance-to-nearest-inserted-cell via `cKDTree` on unit-sphere coords, 15
distance bins, cosine-lat-weighted joint space+time RMSE over the
last-20-day mature window days 41-60, % improvement vs control per bin),
but single-model (GraphCast only -- the reference's Aurora/FCNv2/Prithvi
plotting/looping was dropped since there's no matching GraphCast run in the
original 3-model comparison) and pointed at this study's own paths: land
mask = this study's own `dataset_samples/source-...nc`'s `land_sea_mask`
field (721x1440, no crop, thresholded >0.5 -- the SAME mask/threshold
`land_temp_ins`/`ocean_temp_ins` were actually run with), truth =
`truth/truth_leadtime_2023-02-01IC.nc`, forecasts = `control/`,
`land_temp_ins/`, `ocean_temp_ins/full_predictions/` chunk directories
(T500 selected directly, GraphCast's chunk files already save exactly the
500hPa level, no nearest-level approximation needed). One efficiency fix
made after an initial slow/timing-out smoke test: `iter_chunk_t500_mature()`
was rewritten to select `.sel(level=500)` BEFORE `.load()`, so only that one
variable/level slice is read off disk per chunk instead of all 11 saved
variables x 13 levels.

Submitted as `slurm/run_spillover_analysis.sbatch` (repurposed the standard
osse_t_global-style GPU job template, though the analysis itself is pure
CPU/numpy/xarray) -- **job 250956**, completed cleanly in ~7min. Wrote:
- `spillover_analysis/spillover_decay_T_{LAND,OCEAN}.csv`
- `spillover_analysis/fig_spillover_decay_T_{LAND,OCEAN}.png`
- `spillover_analysis/spillover_decay_methodology.txt`

**Propagation-error snapshot maps (land_temp_ins only).** New
`land_temp_ins/extract_propagation_arrays.py`, ported from
`CC_aurora/extract_propagation_arrays.py` (the reference format used for
Aurora/FCN/old-Prithvi's own land/ocean experiments). NOT RMSE, NOT signed
bias -- unsigned `|forecast - truth|` at a single gridpoint/single valid
time, no time- or space-averaging, at 3 snapshot days: "First Injection"
(this study's own step 121/1-indexed = 0-indexed step 120 = lead 726h =
day 30.25, the first step that actually receives masked land truth --
matches the reference's day-30-ish first-injection convention, exact hour
offset differs slightly by design since it's THIS study's own inject step,
not the reference's Aurora-specific step-122 timing), "Day 45" (0-indexed
step 179, lead 1080h), "Day 60" (0-indexed step 239, lead 1440h, the final
step) -- day-45/day-60 step indices match the Aurora reference exactly
(`DAY45_IDX=179`, `DAY60_IDX=239`) since both share a 6-hourly, 240-step
cadence from a comparable IC. T500 pulled directly from land_temp_ins's own
chunk files (no nearest-level approx needed, same as the spillover script
above); ERA5 truth pulled from `truth/truth_leadtime_2023-02-01IC.nc` at
the 3 matching lead times. `vmax` = 97th-percentile of the 3-snapshot error
array, rounded to 1 decimal (same convention as the reference).

Ran directly (lightweight -- only 3 timesteps touched, no SLURM needed),
completed in seconds. Wrote:
- `land_temp_ins/propagation_error_land_temp.npy` -- float32 `(3, 721, 1440)`
- `land_temp_ins/propagation_meta_land_temp.json` -- `vmax=15.7`,
  `day_labels=["First Injection","Day 45","Day 60"]`

**Explicitly NOT done, flagged not silently skipped**: `ocean_temp_ins`
propagation maps, and any `land_uv`/`ocean_uv` variant for either analysis
(no equivalent land/ocean UV-insertion run exists in `GC-DETER-2023` at
all, unlike the reference's Aurora `UV_LAND`/`UV_OCEAN`) -- user's request
was explicitly scoped to `land_temp_ins` for the propagation maps.

### Directory map addendum (new since §32)

```
Graphcast-full/GC-DETER-2023/
  spillover_analysis/                                          <- §33, new
    spillover_analysis.py, run_spillover_analysis.sbatch
    spillover_decay_T_{LAND,OCEAN}.csv, fig_spillover_decay_T_{LAND,OCEAN}.png
    spillover_decay_methodology.txt
  land_temp_ins/                                                <- §33, additions
    extract_propagation_arrays.py
    propagation_error_land_temp.npy, propagation_meta_land_temp.json
```

## 34. Curated copy into `Graphcast thesis results/`, COP job status, preemption incident (2026-09-09)

**Curated into thesis results (done this entry).** §33's spillover/
propagation outputs were copied into
`/umbc/rs/pi_halem/users/yashd1/Graphcast deterministic/Graphcast thesis
results/Extended Experiments/`, matching that directory's existing
`{Combined T+UV,Land Temperature,Ocean Temperature} Insertion/RMSE/
{CSV,Plots}` convention with two new subfolder types:
- `Land Temperature Insertion/Spillover Analysis/` <-
  `spillover_decay_T_LAND.csv`, `fig_spillover_decay_T_LAND.png`,
  `spillover_decay_methodology.txt` (copied from
  `GC-DETER-2023/spillover_analysis/`)
- `Ocean Temperature Insertion/Spillover Analysis/` <-
  `spillover_decay_T_OCEAN.csv`, `fig_spillover_decay_T_OCEAN.png`,
  `spillover_decay_methodology.txt` (same source dir -- T_LAND measures
  land-inserted spillover into ocean, so it's filed under Land Temperature
  Insertion; T_OCEAN under Ocean Temperature Insertion, by which experiment
  each file's insertion side belongs to)
- `Land Temperature Insertion/Propagation Maps/` <-
  `propagation_error_land_temp.npy`, `propagation_meta_land_temp.json`
  (copied from `GC-DETER-2023/land_temp_ins/`; no Ocean Temperature
  Insertion equivalent exists yet -- §33 explicitly scoped propagation maps
  to land_temp_ins only)

**§31's COP jobs -- status as of this entry.** `T_PERTURBED_CORR` (job
250945) and `UV_PERTURBED_CORR` (job 250946) both **COMPLETED** cleanly
(confirmed via `sacct`, not just the `.out` log's own "Done" line) -- 60/60
chunks each, full 240-step/60-day rollout, no errors beyond benign xarray
`FutureWarning`s already present in every other rollout.py in this project.
Raw output: `T_PERTURBED_CORR/out/t_perturbed_corr_2023-02-01/chunk_{0000..0059}.nc`,
`UV_PERTURBED_CORR/out/uv_perturbed_corr_2023-02-01/chunk_{0000..0059}.nc`.

**NOT yet run**: `compute_rmse.py` for either COP experiment, nor the
control-comparison step, nor any copy into `Graphcast thesis results/` for
COP -- this is genuinely the next open step, not done yet.

## 35. WHERE THE COP RESULTS GO WHEN THE RMSE PIPELINE IS RUN (destination plan, not yet executed)

Documenting the intended destination now so a future session (or this one,
continued) doesn't have to re-derive the convention. **`T_PERTURBED_CORR`/
`UV_PERTURBED_CORR` are a NEW experiment family** -- there is no existing
"Perturbed Temperature/Wind Insertion" folder in `Graphcast thesis
results/` yet. Planned new top-level folders, mirroring the existing
`Temperature Insertion`/`Wind Insertion` folders' own internal structure
(`RMSE/{CSV,Plots}`, `Spatial Bias/Data`, `Cloud Results/CSV`, `Latitude
Results/CSV`, `Terrain Results/CSV`):

```
Graphcast deterministic/Graphcast thesis results/
  Perturbed Temperature Insertion/     <- NEW, once T_PERTURBED_CORR RMSE is run
    RMSE/{CSV,Plots}
    (Spatial Bias / Cloud / Latitude / Terrain Results -- add only if that
     analysis is actually run for this experiment; not assumed by default)
  Perturbed Wind Insertion/            <- NEW, once UV_PERTURBED_CORR RMSE is run
    RMSE/{CSV,Plots}
```

Source paths once computed: `GC-DETER-2023/T_PERTURBED_CORR/results/csv/`
+ `/plots/` (mirroring `osse_t_global/results/`'s own layout, per
`compute_rmse.py`/`plot_rmse.py`/`compare_control_vs_osse.py`'s existing
output convention used everywhere else in this study) ->
`Graphcast thesis results/Perturbed Temperature Insertion/RMSE/{CSV,Plots}`,
and the UV equivalent. This is a PLAN, not yet executed -- the RMSE
pipeline itself has not been run for either COP experiment as of this
entry (see §34).

## 36. COP RMSE/comparison pipeline executed, curated into thesis results (2026-09-14)

Session picked back up after a disconnect (the terminal's own interactive
`bash` job, 250832, had hit its 3h TIMEOUT -- unrelated to any tracked
pipeline job; all 4 COP rollouts from §34 had already finished cleanly
hours before that, confirmed via `sacct`). User asked to execute §35's
plan in full.

**What was built.** `plot_rmse.py` (identical file, confirmed via `diff`)
copied into `T_PERTURBED_CORR/`/`UV_PERTURBED_CORR/` from
`osse_t_global/`/`osse_uv_global/` (had been missed when those dirs were
first set up in §31 -- only `compute_rmse.py`/`perturbation.py`/
`run_osse_insertion.py` were copied then). New `slurm/finish_{t,uv}_
perturbed_corr_feb2023.sbatch` in each dir, mirroring `finish_base_
runs_feb2023.sbatch`'s exact `compute_rmse.py` -> `plot_rmse.py` ->
`compare_control_vs_osse.py` -> reorganize sequence, with one addition
specific to COP: **two** `compare_control_vs_osse.py` invocations instead
of one, per `PERTURBATION_IMPLEMENTATION.md`'s "Analysis pipeline" section
-- `compare_{t,uv}_perturbed_corr` (vs `control/`, the free-running
baseline) and `compare_{t,uv}_perturbed_corr_vs_unperturbed` (vs
`osse_t_global/`/`osse_uv_global/`, the noiseless hard-replacement
baseline for the same variable/levels).

Submitted `sbatch --clusters=chip-gpu T_PERTURBED_CORR/slurm/finish_t_
perturbed_corr_feb2023.sbatch` (**job 252627**) and the UV equivalent
(**job 252628**). Both COMPLETED cleanly (~28min each; stderr only had
the same benign matplotlib `RuntimeWarning: invalid value encountered in
dot` already seen elsewhere in this project, plus a disk-quota
font-cache warning on UV that didn't affect output). Verified real output
existed (not just exit 0): `results/csv/` (3 CSVs) + `results/plots/`
(30 PNGs) each.

**Interpreting the `-inf` rows in the `_vs_unperturbed` CSVs (not a bug).**
The unperturbed hard-replacement run has ~0 RMSE on the directly-inserted
variable during the injection window (it copies truth exactly, no noise),
so `pct_improvement = 100*(0 - X)/0` is mathematically `-inf` for
temperature in `compare_t_perturbed_corr_vs_unperturbed` and for
u/v-wind in the UV equivalent -- expected, not a script defect. The
meaningful signal in that CSV is the OTHER (spillover) variables, where
both denominators are nonzero.

**Headline findings (mean post-injection %% RMSE improvement, `results/
csv/compare_*_pct_improvement.csv`).**

- `T_PERTURBED_CORR` vs control: T improves 83-92%% at every inserted
  level (125-1000hPa); U/V improve modestly (2-33%%); **geopotential
  mostly WORSENS** (-6%% to -81%%, worst at 100/150hPa just above the
  26-level insertion cutoff) -- reproduces this study's established
  "T-Ins degrades Z/U/V" finding (see [[project_graphcast_full]]) under
  observation-noise-scale perturbation too, not just the noiseless
  hard-replacement version.
- `T_PERTURBED_CORR` vs unperturbed: spillover variables (U/V/Z) are
  within +/-2%% of the noiseless version -- COP's noise barely moves the
  spillover result for T-insertion.
- `UV_PERTURBED_CORR` vs control: broad, large improvement -- U/V 80-95%%,
  geopotential 50-66%%, temperature 30-53%% (except -30%% at 50hPa).
- `UV_PERTURBED_CORR` vs unperturbed: spillover variables (T/Z) within
  +/-1.5%% of the noiseless version.
- **Net takeaway**: observation-error-scale noise (COP) barely changes
  the insertion-skill picture relative to the noiseless hard-replacement
  for either experiment's spillover variables -- the OSSE benefit is
  robust to realistic observation noise, not an artifact of injecting
  perfect truth.

**Curated into thesis results**, matching the existing `RMSE/{CSV,Plots}`
convention (confirmed against `Temperature Insertion/RMSE/{CSV,Plots}`):
- `Graphcast thesis results/Perturbed Temperature Insertion/RMSE/{CSV,Plots}`
  <- `T_PERTURBED_CORR/results/{csv,plots}` (3 CSV, 30 plots)
- `Graphcast thesis results/Perturbed Wind Insertion/RMSE/{CSV,Plots}`
  <- `UV_PERTURBED_CORR/results/{csv,plots}` (3 CSV, 30 plots)

These are new top-level experiment folders (no `Spatial Bias`/`Cloud`/
`Latitude`/`Terrain Results` yet -- not run for COP, per §35's plan, only
add if actually run).

**Still open**: cross-model comparison against `prithvi-feb23`'s COP
results (§23 there, done same session) -- carried over, still the
standing open item from §16/§28.

## 37. Cloud/Terrain/Latitude/Spatial-Bias fill-in for COP, plus a new
specific-humidity (Q) pipeline for all 6 experiments (2026-09-15)

Session resumed after an SSH disconnect (terminal connection dropped
mid-work -- unrelated to any tracked SLURM job). User had asked (via a
background fork launched the previous session, before the disconnect)
for the still-missing regime/bias breakdowns to be filled in for both
models' COP experiments, plus a brand-new humidity/specific-humidity
RMSE pipeline for GraphCast (Prithvi's model output already carries
QV natively; GraphCast's `specific_humidity` field had never been
scored by anything before this).

**What the fork had already finished before the disconnect** (job
252642, ran unattended): cloud/terrain/lat-slice regime RMSE + spatial
bias maps (T/U/V/Z, day 30-60 mean) for both `T_PERTURBED_CORR` and
`UV_PERTURBED_CORR`, via the existing `compute_cloud_rmse.py`/
`compute_terrain_rmse.py`/`compute_latslice_rmse.py`/
`compute_bias_maps.py` scripts pointed at the COP output dirs. This
part completed cleanly.

**What crashed, and why (real bug, not a job/scheduler issue).** The
same job then moved on to the new `compute_rmse_q.py` script (Q RMSE,
4-level ERA5 truth at 300/500/850/1000hPa only -- the only levels
`era5_q_2023_febapr.grib` has) and crashed on its very first call
(`control`) with `ValueError: operands could not be broadcast together
with shapes (13,721,1440) (4,721,1440)`. Root cause: the script pulled
`pred["specific_humidity"]` (all 13 standard levels) but never
subset it to `LEVELS = [300, 500, 850, 1000]` before diffing against
the already-4-level truth -- an omission that the sibling
`compute_cloud_rmse_q.py`/`compute_terrain_rmse_q.py` scripts (written
in the same batch) did NOT have; both of those correctly call
`pred_var.sel(level=LEVELS)`. Fixed with the matching one-line
`pred_var = pred_var.sel(level=LEVELS)` in `compute_rmse_q.py`.

Also found on inspection: the failed job's sbatch only ever wired Q
RMSE for `control` + the 2 COP experiments, and never actually
*called* `compute_cloud_rmse_q.py`/`compute_terrain_rmse_q.py` at all
(despite both scripts existing on disk) -- neither matched the user's
ask to cover "all COP, normal experiments, cloud and terrain regime
also."

**Rebuilt as `slurm/humidity_all_experiments_feb2023.sbatch`**,
covering all 6 experiments (`control`, `osse_t_global`,
`osse_uv_global`, `osse_z_global`, `T_PERTURBED_CORR`,
`UV_PERTURBED_CORR`): Q RMSE -> Q cloud regime -> Q terrain regime ->
Q comparison-vs-control (300/500/850/1000hPa, via
`compare_control_vs_osse.py`) for the 5 non-control experiments.
Submitted as **job 253050**; COMPLETED in 1h10m with zero stderr
output across all 18 script invocations + 5 comparisons. Verified
real output (not just exit 0): `rmse_q_*.csv` in each experiment's
`results/csv/`, `rmse_cloud_q_*.csv`/`rmse_terrain_q_*.csv` in
`regime_analysis_results/{cloud,terrain}/csv/`, and
`compare_*_q_pct_improvement.csv` + 6 plots per non-control experiment.

**No Q lat-slice script was written** for GraphCast (unlike Prithvi's
`lat_results` which already had QV@700/850hPa lat-band CSVs from the
prior session's fork) -- the user's specific "Q700/Q850" ask was for
Prithvi only; GraphCast's ask was general "humidity/specific-humidity"
coverage, which RMSE + cloud regime + terrain regime + comparison
satisfies. Flagging as a known scope boundary, not an oversight, in
case a future session wants lat-band Q breakdown for GraphCast too.

**Headline findings (mean post-injection %% RMSE improvement in
specific humidity, `compare_*_q_pct_improvement.csv`, vs control).**
Every insertion experiment improves Q spillover skill relative to
control -- no exceptions, unlike geopotential which frequently
degrades under T-insertion (see §36's T/U/V/Z findings):
- T-insertion (noiseless): 21-32%% (weakest at 850hPa, strongest at
  1000hPa/300hPa)
- UV-insertion (noiseless): 23-38%% -- the strongest and most
  consistent Q improvement of any experiment, echoing UV-insertion's
  strong T/U/V/Z spillover too
- Z-insertion (noiseless): the weakest by far -- only 2.5-6%% at
  300/500/850hPa, though still 25%% at 1000hPa
- T-insertion COP (perturbed): 19-30%%, tracking ~1-2 points below its
  noiseless counterpart (consistent with §36's "COP noise costs a
  small amount of spillover skill" finding)
- UV-insertion COP (perturbed): 23-37%%, essentially matching its
  noiseless counterpart

**Curated into thesis results**: for each of the 6 experiments,
`rmse_cloud_q_*.csv` -> `Cloud Results/CSV/`, `rmse_terrain_q_*.csv`
-> `Terrain Results/CSV/`, raw `rmse_q_*.csv` + (for the 5
non-control) `compare_*_q_pct_improvement.csv`/6 PNGs -> `RMSE/CSV/`
and `RMSE/Plots/`. Combined with this entry's earlier (non-Q)
cloud/terrain/lat-slice/bias-maps curation into `Perturbed Temperature
Insertion/` and `Perturbed Wind Insertion/`, every experiment folder
in `Graphcast thesis results/` now has Cloud Results, Terrain Results,
RMSE (incl. Q), and (except the two Perturbed folders, per §36's
scope note) Latitude Results and Spatial Bias.

**Prithvi side (parallel work, same session)**: the prior session's
fork had already finished everything for Prithvi before the disconnect
-- cloud/terrain/lat-slice regime (incl. QV@700/850hPa) and spatial
bias for all 6 experiments, plus Q comparison plots in every
experiment's own `rmse/` dir. Only curation into
`Prithvi thesis results/` remained; done this session (see
`prithvi-feb23/session_log.md` §24).

**Now closed**: this was the last standing item from §35's original
destination plan. Both thesis-results trees are essentially complete
for every experiment run so far. Still open: the cross-model
GraphCast-vs-Prithvi comparison (carried over from §16/§28/§36,
untouched this session).

## 38. GraphCast added to the tri-model TUV synergy figure (common/
tuv_final/); Q700 clarified as unnecessary (2026-09-15)

User flagged three specific items as "genuinely still missing, not
fabricated" and asked me to verify each rather than assume. See
`prithvi-feb23/session_log.md` §25 for item 1 (Prithvi 500hPa U/V
regime results, fixed there) and the verification that GraphCast's own
Cloud/Terrain Results were NOT actually missing 500hPa (already had
300/500/1000hPa for U/V). This entry covers items 2 and 3, both
GraphCast-specific.

**Item 2 -- GraphCast Q at 700hPa: confirmed real, resolved without new
work.** `download_era5_q_feb2023.py`'s docstring/request explicitly
shows only `["300","500","850","1000"]` was ever requested from CDS --
700hPa was a deliberate exclusion, not a bug, to keep the download
small. Fixing it properly would need a new CDS API download (external,
network-dependent, minutes-to-hours in queue). User's call: skip 700hPa,
use 850/1000hPa instead -- both already fully computed and curated as
of §37, so **no new work was needed** for this item.

**Item 3 -- GraphCast in the cross-model TUV synergy figure
(`common/tuv_final/plot_synergy_z500_q850.py`, Fig 5.27/5.28):
confirmed real, fixed.** This figure predates GraphCast's addition to
the thesis entirely -- `common/tuv_plots/compute_tuv_rmse.py` (the
script that builds its `cache/*.csv`/`*.npy` inputs) is hardcoded to
3 models (`fcn`, `aurora`, `prithvi`) with per-model bespoke truth
loaders (FCN/Aurora: ERA5 grib via `cfgrib` on their own 721x1440/
720x1440 grids; Prithvi: MERRA-2 via `h5py` on a 360x576 grid, with a
documented ascending/descending level-index remapping,
`pri_idx_to_merra_idx()`). Also note: these 3 models' own control/
insertion runs used a DIFFERENT, older IC (Prithvi: 2022-01-01; FCN/
Aurora: 2023-02-01-ish) than this session's GC-DETER-2023 study
(2023-02-01) -- this figure is a separate, earlier cross-model study
(`common/` = the original tri-study), not literally the same
experiment set as `GC-DETER-2023`.

**What was needed for the missing `combined_tuv_ins` (`tuv`)
experiment's Q850:** all 3 other GraphCast experiments (`control` =
`ctrl`, `osse_t_global` = `tins`, `osse_uv_global` = `uins`) already
had Q850 RMSE from §37's job 253050, but `combined_tuv_ins` was never
included in that job (it isn't one of the "6 core experiments" §37's
job targeted). Ran `compute_rmse_q.py` against it directly (one
one-off `sbatch`, **job 253132**, ~5min, COMPLETED clean) --
`combined_tuv_ins/results/csv/rmse_q_combined_tuv_ins_2023-02-01.csv`.

**New bridge script**: `common/tuv_plots/compute_graphcast_tuv_cache.py`
-- NOT a rewrite of `compute_tuv_rmse.py`'s per-model loaders (GraphCast
already has its own complete, working RMSE pipeline in GC-DETER-2023;
duplicating FCN/Aurora's from-scratch ERA5-loading approach would be
pure waste). Instead: (1) the four `graphcast_{ctrl,tins,uins,tuv}_
rmse.csv` timeseries (lead_day, z500, q850) are built by extracting the
already-computed 500hPa geopotential and 850hPa specific-humidity rows
straight out of each experiment's existing `results/csv/rmse_*.csv` /
`rmse_q_*.csv` -- pure pandas filter+merge, no new numerics; (2) the
spatial per-gridpoint synergy maps (`graphcast_synergy_{z500,q850}.npy`,
`graphcast_impr_tuv_{z500,q850}.npy`) are new -- these need per-gridpoint
MSE, not available in the aggregated RMSE CSVs, so a new
`mature_mse_field()` function (mirrors `compute_bias_maps.py`'s
streaming chunk-loading pattern, but squared error instead of raw
bias, and lead-day-(40,60] to match this figure's OWN mature-phase
convention -- NOT the 30+ convention `compute_bias_maps.py` uses
elsewhere in GC-DETER-2023) reads the 4 experiments' raw chunk NetCDFs
once each for z500 and q850. Also saves `graphcast_lat.npy`/
`graphcast_lon.npy` so the plot script reads GraphCast's actual grid
instead of a hardcoded `np.linspace` (unlike the other 3 models, whose
grids are hardcoded in `plot_synergy_z500_q850.py` since those are
fixed/known already) -- avoids a silent lat-order/sign mistake.
Ran as **job 253133** (chained via `--dependency=afterok:253132`),
COMPLETED in 58min, only benign `xarray` `FutureWarning`s in stderr.

**Extended `plot_synergy_z500_q850.py` from 3 models to 4**: added
`"graphcast":"GraphCast"` to `MOD_LABEL`, `"GraphCast":"#009E73"` (next
color in the existing Okabe-Ito-derived palette) to `COLORS`, a
`graphcast` entry to `VAR_MAP` for both Z500/Q850, a `MODELS` list
threaded through both the spatial-map and bar-chart loops (replacing
the hardcoded 3-model lists), widened the Fig 5.27 grid from 2x3 to
2x4 (`figsize` 15->19.5), and rebalanced the Fig 5.28 bar
width/offsets for 4 bars instead of 3. Also moved the bar-chart legend
from `loc="upper right"` (inside the axes, overlapped 2 of the bar
value labels even with 3 models) to outside the axes
(`bbox_to_anchor=(1.01,1.0)`) -- a pre-existing minor readability issue
made worse by the 4th bar, fixed since it's a 1-line change touching
only the earlier-broken behavior and not any model's data/methodology.
Ran via `sbatch` using `venvs/aurora310` (matching the ORIGINAL
`run_tuv_plot.slurm`'s environment, since `venv_graphcast` lacks
cartopy/matplotlib -- first attempt on the wrong venv was caught before
it wasted a queue slot) -- **jobs 253218** (initial) then **253235**
(after the legend fix), both COMPLETED clean. Verified both PNGs
visually: `fig5_27_spatial_synergy_maps_z500_q850.png` (now 4 columns
x 2 rows, GraphCast's Z500 map shows a coherent, heavily negative-
synergy Pacific/Atlantic pattern) and
`fig5_28_synergy_bar_chart_z500_q850.png` (GraphCast: Z500 -28.4%,
Q850 +14.3%, matching the sign/magnitude implied by GraphCast's own
established "T-Ins degrades Z" finding -- UV-insertion alone already
captures most of the Z500 benefit, so combined T+UV insertion actually
UNDERPERFORMS UV-only, a large negative synergy, all 4 labels now
legible).

**Notable finding, not a bug**: GraphCast's Fig 5.27 SUBPLOT TITLE text
(raw spatial `nanmean`/`nanstd`, unclipped) shows extreme numbers --
Z500: -671.5% +/- 1870.9%, Q850: -14.2% +/- 175.3% -- vs. the other 3
models' much tamer title numbers (worst case Prithvi Q850: -14.4% +/-
129.1%). Diagnosed via a latitude histogram of |synergy|>1000%
gridpoints: 100% of the extreme values cluster in -30N to +40N
(peaking 0-10N) -- the deep tropics, where geopotential height has
very low natural variance/dynamic range, so tiny absolute-error
differences between experiments produce enormous *percentage*
improvements at individual gridpoints. This is the same
small-denominator artifact documented for the `_vs_unperturbed` -inf
rows in §36, just continuous instead of infinite, and all 4 models
exhibit it to some degree (it's inherent to any ratio-based synergy
metric applied to Z near the equator) -- GraphCast is simply the most
affected, likely because its native full 0.25deg resolution (vs. the
other 3 models' coarser/different grids) resolves more small-scale
tropical noise. The COLOR MAP itself is unaffected (clipped to +/-15%
by design) and shows a scientifically coherent pattern; only the raw
title-text summary statistic is skewed. Left the statistic (mean+/-std)
unchanged to match the other 3 models' own methodology exactly, rather
than unilaterally changing how ALL 4 models' titles are computed --
flagging this here rather than silently "fixing" it, since redesigning
that statistic is a methodology decision, not a data-completeness gap.

**NOT done, deliberately out of scope**: `common/tuv_final/
PHYSICAL_ANALYSIS.txt` (the ~580-line scientific narrative
interpreting Figs 5.27/5.28, `common/MEMORY.md`'s companion analysis
document) still only discusses FCN/Aurora/Prithvi -- it was not
rewritten to add a GraphCast paragraph. The user's request was
specifically about data/figures being "genuinely missing," which is
now fixed; writing new scientific interpretation prose at the same
depth as the existing document for a 4th model is separate,
substantially larger work that wasn't asked for. Flagging as an open
item for whoever wants that narrative extended.

**Curated**: nothing new to copy into `Graphcast thesis results/` --
this figure lives in `common/tuv_final/`, which is the OLDER
cross-model-synthesis output location (a sibling to, not part of,
either model's individual thesis-results tree), and was updated in
place.
