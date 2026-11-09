# GC-DETER-2023 — Correlated OSSE Perturbation (COP) Implementation

Ports the COP scheme (`../perturbation.py`, `../PERTURBATION_IMPLEMENTATION.md`,
originally built for the Dec2021/`deterministic_run` study) onto this
directory's Feb2023 IC / GC-DETER-2023 pipeline. See
`/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md`
for the full cross-model method (citations, hyperparameters).

## Location

| | Path |
|---|---|
| Temperature experiment | `T_PERTURBED_CORR/` |
| Wind experiment | `UV_PERTURBED_CORR/` |
| Shared perturbation module | `perturbation.py` (copied into each, verbatim from `../perturbation.py`) |
| SLURM job scripts | `{T,UV}_PERTURBED_CORR/slurm/run_{t,uv}_perturbed_corr_feb2023.sbatch` |

Both are dedicated, self-contained directories (own `graphcast/` package copy,
own `run_osse_insertion.py`, own `out/`/`logs/`) — no shared state with
`osse_t_global/`, `osse_uv_global/`, or any other experiment in this study.

## What differs from the Dec2021-study reference port

The reference `T_PERTURBED_CORR`/`UV_PERTURBED_CORR` (repo root) insert at
**all 37 levels, globally**, because that was the only convention that
existed at the time. GC-DETER-2023 has since established a **corrected**
convention (`osse_t_global/`): temperature inserts at only **26 levels**
(excludes the 11 stratospheric levels <=100hPa), while wind (`osse_uv_global/`)
still inserts globally at all 37 levels. This port keeps that same convention
for direct comparability with this study's own unperturbed baselines:

- `T_PERTURBED_CORR/run_osse_insertion.py` is copied from `osse_t_global/`
  (has `--insert_levels`/`--insert_mask` support); the sbatch script passes
  the same 26-level list as `osse_t_global`'s.
- `graphcast/rollout.py`'s injection block places the COP noise-generation
  step **before** the `insert_levels`/`insert_mask` `.where()` restriction
  (a "2a-cop" step, new relative to the reference port), not after — so
  noise is only ever added to truth values that actually get inserted;
  excluded levels/points keep the model's own unperturbed prediction
  (`target_for_align`), never a noised-then-discarded value. This is a
  correctness requirement specific to this port, since the reference
  (all-levels, no restriction) never had to consider it.
- `UV_PERTURBED_CORR` needed **no such adaptation** — `osse_uv_global/graphcast/rollout.py`
  is byte-identical to the Dec2021-study reference's own `osse_uv_global`
  copy at the injection block, so the wind COP patch was ported verbatim.

## IC / truth / params

Same as `osse_t_global`/`osse_uv_global`: IC
`dataset_samples/source-era5_date-2023-02-01_res-0.25_levels-37_steps-12.nc`,
truth `truth/truth_leadtime_2023-02-01IC.nc`, params/stats shared from the
repo root (`../params/`, `../stats/`). 240 steps x 6h, steps 0-119 free-run,
steps 120-239 truth(+noise)-insertion.

## Validated before GPU submission

`perturbation.py`'s numerics (unit-variance per-level fields, level-reorder
mapping for the 26-level truth subset, non-divergent wind construction) were
sanity-checked standalone against a synthetic grid before submitting either
job — see the reference `PERTURBATION_IMPLEMENTATION.md` Section 7 for the
same class of checks (this port re-verified the 26-level reorder specifically,
new relative to that reference).

## SLURM submission

```
sbatch --clusters=chip-gpu T_PERTURBED_CORR/slurm/run_t_perturbed_corr_feb2023.sbatch    # job 250945
sbatch --clusters=chip-gpu UV_PERTURBED_CORR/slurm/run_uv_perturbed_corr_feb2023.sbatch   # job 250946
```

## Analysis pipeline (once rollout completes)

Same as every other GC-DETER-2023 experiment: `compute_rmse.py` (copied in,
unmodified) -> compare against `../control/` and `../osse_t_global/`/
`../osse_uv_global/` (the unperturbed hard-replacement versions) to see
whether observation-error-scale noise changes the insertion-skill result.
