# Correlated OSSE Perturbation (COP) Scheme — Cross-Model Method Overview

**Status as of 2026-07-22**: implemented and running across all three models
(FourCastNetv2, Aurora, Prithvi WxC); 6 experiments submitted (temperature +
wind, per model). See `FCN/IMPLEMENTATION.md`, `Aurora/IMPLEMENTATION.md`,
`Prithvi/IMPLEMENTATION.md` for per-model code-level detail. This file is the
model-agnostic method description — the part meant to become the paper's
methods-section text directly.

---

## 1. Motivation

All three OSSE (Observing System Simulation Experiment) pipelines in this
study previously perturbed "truth" reanalysis fields before insertion using
spatially-uncorrelated Gaussian noise: at every grid point and every level,
an independent draw from `N(0, sigma_l^2)`, with `sigma_l` varying by
pressure level following Ingleby (2017)-style radiosonde error statistics
(Errico et al. 2013 OSSE framework). This is a legitimate published baseline,
but it has two well-documented shortcomings relative to how real observation
and representativeness error actually behaves:

1. **Real error is spatially correlated.** Representativeness error and
   instrument error both have finite horizontal correlation length scales —
   neighboring grid points' errors are not independent (Hollingsworth &
   Lönnberg 1986; Bormann & Bauer 2010; Bormann, Bonavita, et al. 2016).
   Purely independent-per-pixel noise is the most commonly criticized
   simplification in OSSE observation-error design (see the review in
   Tellus, "The importance of simulated errors in observing system
   simulation experiments").
2. **Independent wind-component noise is dynamically inconsistent.**
   Perturbing u and v independently injects spurious small-scale divergence
   and vorticity that has no physical basis — operational data-assimilation
   systems avoid this by representing wind background/observation error via
   streamfunction and velocity potential rather than raw (u,v) components
   (Parrish & Derber 1992; multivariate background-error covariance reviews,
   e.g. Bannister 2008 Part II).

The Correlated OSSE Perturbation (COP) scheme below adds both of these
structures while keeping the magnitude calibration (Ingleby 2017-style
per-level sigma tables) and the overall Errico et al. (2013) OSSE framing
unchanged — this is explicitly an upgrade to the *noise structure*, not a
change to the OSSE experimental design (240 autoregressive 6h steps, steps
1-120 free-run, steps 121-240 truth-insertion, hard replacement in physical
space, same as every other experiment in this study).

---

## 2. The three correlation structures

### 2.1 Horizontal spatial correlation

Each level's perturbation field is constructed as a realization of a
**Gaussian Random Field**: draw independent standard-normal white noise on
the model's native grid, then convolve with an isotropic Gaussian smoothing
kernel. This is one of the standard constructions of a GRF with an
approximately-Gaussian covariance function (see reviews of methods for
generating spatially correlated random fields, e.g. spectral/Cholesky/
kriging-based generators). Implemented via `scipy.ndimage.gaussian_filter`,
longitude axis treated periodically (`mode="wrap"`), latitude axis not
(`mode="nearest"`).

**Horizontal correlation length: 250 km** — chosen to sit within the range
commonly cited for radiosonde/representativeness error horizontal
correlation scales (Hollingsworth-Lönnberg-style estimates; Bormann et al.
2016 assessments of interchannel/spatial observation error correlation).

**Known simplification, stated explicitly**: the smoothing kernel's pixel
width is computed from the correlation length and the grid's degree spacing,
applied isotropically in *grid-cell* space, not true great-circle distance.
Near the poles, this means the *effective* physical correlation length in
the zonal direction is smaller than 250 km (because longitude circles
shrink), a known limitation of simple lat/lon-grid smoothing shared by many
practical OSSE implementations. Not corrected in this implementation;
flagged here for the paper's limitations discussion.

### 2.2 Vertical correlation across levels

Errors at adjacent pressure/model levels are correlated, not independent —
modeled with an **exponential decay in log-pressure space**, a standard
functional form for vertical background/observation-error correlation
(Bannister 2008 Part II review of forecast-error covariance modelling;
vertical interchannel correlation estimation via the Desroziers diagnostic,
e.g. for ATMS/IASI in Bormann et al.):

```
corr(l_i, l_j) = exp( -| ln(p_i) - ln(p_j) | / L_vert )
```

with `L_vert = 0.5` (natural-log-pressure units) chosen so that adjacent
standard levels (e.g. 850/700 hPa, `|ln(850/700)| = 0.194`) correlate at
`exp(-0.194/0.5) ≈ 0.68`, while levels far apart (e.g. 1000/50 hPa,
`|ln(1000/50)| = 3.0`) correlate at `exp(-3.0/0.5) ≈ 0.002` (~uncorrelated).
The resulting `N x N` correlation matrix is Cholesky-factored
(`C = A A^T`), and independently-smoothed per-level GRFs are linearly mixed
through `A` (`mixed[l] = sum_j A[l,j] * smoothed[j]`) — a linear combination
of already-spatially-smooth fields remains spatially smooth, so this step
adds vertical correlation without destroying the horizontal structure from
step 2.1. Each output level is renormalized to unit variance, then scaled by
its physical target sigma (Ingleby 2017-style table, unchanged from the
uncorrelated baseline).

### 2.3 Non-divergent (streamfunction-derived) wind perturbation — wind
experiments only

Instead of drawing u' and v' independently, both are derived from a single
random **streamfunction** field `psi` (itself built with sections 2.1+2.2's
horizontal+vertical correlation), via the standard non-divergent/rotational
wind relation (Holton, *An Introduction to Dynamic Meteorology*):

```
u' = -(1/R) * d(psi)/d(phi)
v' =  (1/(R * cos(phi))) * d(psi)/d(lambda)
```

(`R` = Earth radius, `phi` = latitude, `lambda` = longitude). Longitude
derivative uses a periodic central difference (global grid wraparound);
latitude derivative uses `np.gradient` against the true latitude coordinate
array. `cos(phi)` is clipped at `cos(85 deg)` to avoid the pole singularity.

Because differentiating a unit-variance streamfunction produces a raw
`u'`/`v'` whose physical magnitude depends on the correlation length (not
directly controllable), a **calibration step** rescales each level's raw
`u'`, `v'` so their empirical global standard deviation matches the target
physical sigma (m/s) from the wind-error-by-level table, independently for u
and v.

**Documented limitation**: `cos(phi)` clipping causes `v'`'s raw
(pre-rescale) magnitude to be systematically larger at high latitudes than
at the equator; because rescaling is done with one *global* std rather than
a latitude-dependent one, the final perturbation is somewhat under-scaled
near the equator and over-scaled near the poles relative to the nominal
target sigma. This is a genuine simplification of the current
implementation (a latitude-banded or fully local rescaling would fix it) —
stated here rather than hidden, since it is directly relevant to any
per-latitude error analysis in the paper.

**Also documented**: u and v are assumed to share the same sigma-by-level
magnitude table (isotropic turbulence assumption) in all three models'
implementations — real wind observation error is not perfectly isotropic,
but no per-component (u vs v) error table was available to differentiate
them, and this is a standard simplifying assumption in OSSE literature.

---

## 3. Reproducibility

All randomness remains **per-step-deterministic**: `np.random.default_rng
(seed=step)`, unchanged from the uncorrelated baseline experiments. Every
perturbed run is therefore exactly reproducible given the step index, and
temperature and wind perturbations at the same step (where both are
generated, e.g. the surface-vs-upper-air draws within a single model's wind
injection) use distinguishable seeds (`step` and `step + 1_000_000`) so they
are independent of each other rather than accidentally correlated through
seed reuse.

---

## 4. What is unchanged from the pre-existing (uncorrelated) perturbed
experiments

- Overall OSSE design: 240 steps x 6h = 60 days, steps 1-120 free-run, steps
  121-240 truth-insertion, hard replacement (not blended/Kalman-gain-scaled)
  in physical space, immediately after each forward pass and before the
  corrected state is fed into the next autoregressive step.
- IC sources, dates, and grids: identical to each model's existing
  perturbed-baseline experiment (`t_perturbed`/`T_PERTURBED`/
  `perturbed_temp_ins`) — see per-model IMPLEMENTATION.md files.
- Magnitude calibration: same Ingleby (2017)-style sigma-by-level tables
  (temperature); a newly-constructed but methodologically analogous
  wind-error-by-level table (no prior wind-perturbation baseline existed to
  reuse, since only T was perturbed with i.i.d. noise before this work).
- Channel indices / variable names touched: identical to each model's
  existing non-perturbed insertion experiments (land/ocean/coastal), just
  applied globally with correlated noise instead of hard truth-replacement
  or i.i.d.-noise truth.

---

## 5. The six new experiments

| Model | Temperature | Wind (U+V) |
|---|---|---|
| FourCastNetv2 | `fcn_inference/t_perturbed_corr/` | `fcn_inference/uv_perturbed_corr/` |
| Aurora | `CC_aurora/T_PERTURBED_CORR/` | `CC_aurora/UV_PERTURBED_CORR/` |
| Prithvi WxC | `prithviOSSE/perturbed_corr_temp_ins/` | `prithviOSSE/perturbed_corr_uv_ins/` |

All six were submitted as SLURM jobs on 2026-07-22 (UMBC `chip-gpu`
cluster). Outputs are mirrored (via symlink, not copy — files are tens of
GB each) into this directory's `FCN/`, `Aurora/`, `Prithvi/` subfolders.

---

## 6. Suggested analysis / validation for the paper (not yet run)

Before reporting results, the following sanity checks are recommended
(mirroring how the original token-shuffle-bug and other correctness issues
were caught elsewhere in this project — verify empirically, not just by
code review):

1. **Empirical correlation length check**: compute the lag-N spatial
   autocorrelation of one perturbation field (any level, any step) and
   confirm it decays smoothly and matches the ~250 km target, the same way
   `extract_embeddings_notes.txt`'s lag-1 autocorrelation check validated
   the Prithvi WxC token-unshuffle fix elsewhere in this project.
2. **Vertical correlation check**: compute the empirical correlation
   between perturbation fields at adjacent vs. distant levels and compare
   against the theoretical `VERT_CORR` matrix (already spot-checked during
   development — see Section 7).
3. **Non-divergence check (wind only)**: compute the divergence
   (`du/dx + dv/dy`) of a perturbation field and confirm it is small
   relative to its vorticity, verifying the streamfunction construction
   actually produced a near-non-divergent field in practice, not just in
   the idealized math.
4. **Comparison against the uncorrelated baseline** (`t_perturbed`/
   `T_PERTURBED`/`perturbed_temp_ins`): the core scientific question this
   upgrade should let the paper answer is whether spatially/vertically
   structured (vs. i.i.d.) observation error changes the forecast-skill
   impact of insertion — this requires running `analyze_osse.py`-style RMSE
   comparisons between the old and new perturbed experiments once both have
   completed output.

---

## 7. Development-time validation already performed

Before submitting any of the 6 SLURM jobs, the core numerical
perturbation-generation logic (spatial smoothing + vertical Cholesky-mixing
+ streamfunction differentiation) was unit-tested standalone (not inside
the full pipeline) on a representative 721x1440 grid:

- Per-level output fields: unit variance confirmed (std ≈ 1.0000 ± 1e-6).
- Horizontal correlation: lag-1 ≈ 0.997, lag-9 (~1 correlation-length-worth
  of pixels) ≈ 0.78, lag-20 ≈ 0.31 — confirms smooth decay with distance,
  not independent noise.
- Vertical correlation: empirical correlation between level 0/1 (adjacent)
  ≈ 0.25 vs. theoretical 0.250; level 0/12 (far apart) ≈ -0.03 vs.
  theoretical 0.0025 — matches the designed `VERT_CORR` matrix closely
  (small deviation expected from a single-realization empirical estimate).
- Streamfunction-derived u,v: finite everywhere (no NaN/Inf), confirmed
  non-trivial magnitude difference between raw u and raw v before rescaling
  (expected, from the `1/cos(phi)` factor in the v equation — corrected by
  the per-level rescaling step, see Section 2.3's documented limitation).

---

## Citations

- Errico, R. M., et al. (2013). Development and validation of observing-system
  simulation experiments at NASA's Global Modeling and Assimilation Office.
  *Quarterly Journal of the Royal Meteorological Society*.
- "The importance of simulated errors in observing system simulation
  experiments." *Tellus A: Dynamic Meteorology and Oceanography*, 2021.
  https://tellusjournal.org/articles/10.1080/16000870.2021.1886795
- Ingleby, B. (2017). An assessment of different radiosonde types 2015/2016.
  ECMWF Technical Memorandum (contract report for Vaisala Oyj).
- Hollingsworth, A., & Lönnberg, P. (1986). The statistical structure of
  short-range forecast errors as determined from radiosonde data. Part I:
  The wind field. *Tellus A*.
- Bormann, N., & Bauer, P. (2010). Estimates of spatial and interchannel
  observation-error characteristics for current sounder radiances. *QJRMS*.
- Bormann, N., et al. (2016). Estimating and accounting for correlated
  observation error — assessments for infrared sounders. Convergence issues
  in the estimation of interchannel correlated observation errors,
  *Monthly Weather Review*, 146(10), 2018.
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). Diagnosis of
  observation, background and analysis-error statistics in observation
  space. *QJRMS*.
- Bannister, R. N. (2008). A review of forecast error covariance statistics
  in atmospheric variational data assimilation. II: Modelling the forecast
  error covariance statistics. *QJRMS*.
- Parrish, D. F., & Derber, J. C. (1992). The National Meteorological
  Center's spectral statistical-interpolation analysis system. *Monthly
  Weather Review*.
- Holton, J. R. *An Introduction to Dynamic Meteorology* (any edition) —
  streamfunction / non-divergent wind relation.
- Magnusson, L., Nycander, J., & Källén, E. (2009). Flow-dependent
  perturbation growth in ensemble prediction. (background — flow-dependent
  perturbation methods, relevant to a possible future Tier-4 upgrade, not
  used in this implementation).

See also, for the pre-existing (uncorrelated) perturbation baseline this
work upgrades: `fcn_inference/t_perturbed/`, `CC_aurora/T_PERTURBED/`,
`prithviOSSE/perturbed_temp_ins/`, and the shared cross-model design note
`OBSERVATION_ERROR_PERTURBATION.txt` (present in `fcn_inference/t_perturbed/`
and `t_12h/`).
