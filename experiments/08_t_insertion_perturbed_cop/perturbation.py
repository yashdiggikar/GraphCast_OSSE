"""Correlated OSSE Perturbation (COP) scheme, ported to full GraphCast (37 native
pressure levels) from the reference implementation shared by FourCastNetv2, Aurora,
and Prithvi WxC -- see
/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/METHOD_OVERVIEW.md for the
full model-agnostic method writeup (citations, hyperparameter justification) and
this file's docstring below for what's specific to the GraphCast full port.

Two correlation structures (three counting the wind-specific one):
  1. HORIZONTAL: each level's noise field is a Gaussian Random Field -- white
     noise smoothed with an isotropic Gaussian kernel, correlation length
     ~250 km (Hollingsworth & Lonnberg 1986; Bormann & Bauer 2010; Bormann
     et al. 2016).
  2. VERTICAL: exponential decay in log-pressure space
     (corr(p_i,p_j) = exp(-|ln(p_i)-ln(p_j)|/L), L=0.5) across pressure
     levels (Bannister 2008 Part II), applied via Cholesky-mixing of
     independently-smoothed per-level fields.
  3. WIND ONLY: u,v derived from a single random streamfunction field (itself
     built with 1+2) via the non-divergent/rotational wind relation, instead
     of perturbing u and v independently (Parrish & Derber 1992; Holton).

Ported (not just copied) from
fcn_inference/{t,uv}_perturbed_corr/OSSE-FourCastNetv2/earth2mip/networks/__init__.py
-- same core numerics (_correlated_level_fields, _streamfunction_to_uv,
_generate_correlated_wind_perturbation are near-verbatim), same hyperparameters
(HORIZ_CORR_LENGTH_KM=250, VERT_L_LOGP=0.5), same reproducibility convention
(np.random.default_rng(seed=step), 1-based step matching "steps 121-240" insertion
convention used throughout this whole cross-model study).

GraphCast-FULL-SPECIFIC DEVIATION FROM THE REFERENCE (documented, not hidden):
the reference implementations (Aurora/FCN/Prithvi) all perturb at exactly 13
native pressure levels -- that is genuinely all the vertical levels those models
have. Full GraphCast's insertion experiments in this project all operate at its
full 37 native levels (see osse_{t,z,uv}_global/), not a 13-level subset -- that
is this project's own established, documented convention (GraphCast full's
defining feature vs "small"/"operational" checkpoints), so this perturbed
experiment keeps that convention rather than reverting to 13 levels, to stay
directly comparable to this project's own unperturbed T-Global/UV-Global runs.
Consequence: the reference SIGMA-by-level tables (built for exactly the 13
standard levels, an Ingleby 2017-style radiosonde error curve) are extended to
GraphCast's 37 levels via log-pressure LINEAR INTERPOLATION (np.interp on
log(pressure), which flat-extrapolates outside the reference table's 50-1000 hPa
range) -- so all levels from 30 hPa up to 1 hPa (outside the reference curve's
range) get held at the 50 hPa sigma value. This is a real simplification (no
radiosonde-style error curve exists that high) and is stated here for the paper's
limitations discussion, exactly as the reference implementation documents its own
simplifications (cos(phi) clipping, grid-cell-space smoothing near the poles,
isotropic u/v sigma assumption -- all still apply unchanged here too).

The vertical CORRELATION matrix itself needs no interpolation -- it is an
analytic function of the pressure levels themselves (exp(-|delta ln p|/L)), so
it is computed directly for GraphCast's own 37 levels.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# GraphCast full's 37 native pressure levels (ascending, hPa) -- matches
# TARGET_LEVELS_T/TARGET_LEVELS_UV's ascending-pressure ordering convention in
# the reference implementation.
LEVELS_37 = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                       225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                       775, 800, 825, 850, 875, 900, 925, 950, 975, 1000], dtype=np.float64)
N_LEVELS = len(LEVELS_37)  # 37

# ── Reference (13-level) sigma tables, verbatim from
# fcn_inference/{t,uv}_perturbed_corr/OSSE-FourCastNetv2/earth2mip/networks/__init__.py ──
_REF_LEVELS = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                        dtype=np.float64)
_REF_SIGMA_T = np.array([2.0, 1.5, 1.5, 1.2, 1.0, 0.8, 0.7, 0.7, 0.8, 0.8, 1.0, 1.0, 1.2],
                         dtype=np.float32)   # K, Ingleby (2017)-style radiosonde error
_REF_SIGMA_UV = np.array([1.5, 1.6, 1.8, 2.2, 2.0, 1.8, 1.4, 1.2, 1.1, 1.0, 1.0, 1.1, 1.3],
                          dtype=np.float32)  # m/s, same physical reasoning (larger near jet core)

# Log-pressure linear interpolation onto GraphCast's 37 levels (flat-extrapolates
# for the 8 levels above 50 hPa, where no reference radiosonde curve exists --
# see module docstring).
_logp37 = np.log(LEVELS_37)
_logp_ref = np.log(_REF_LEVELS)
SIGMA_T_BY_LEVEL = np.interp(_logp37, _logp_ref, _REF_SIGMA_T).astype(np.float32)     # (37,)
SIGMA_UV_BY_LEVEL = np.interp(_logp37, _logp_ref, _REF_SIGMA_UV).astype(np.float32)   # (37,)

# ── Correlation-structure hyperparameters (identical to the reference) ──
HORIZ_CORR_LENGTH_KM = 250.0
GRID_SPACING_DEG = 0.25          # GraphCast full's native grid -- matches Aurora/FCN
KM_PER_DEG_LAT = 111.0
SIGMA_PX = HORIZ_CORR_LENGTH_KM / (KM_PER_DEG_LAT * GRID_SPACING_DEG)   # ~9.0 grid cells

VERT_L_LOGP = 0.5   # natural-log-pressure decay scale
_vert_dist = np.abs(_logp37[:, None] - _logp37[None, :])
VERT_CORR = np.exp(-_vert_dist / VERT_L_LOGP)                 # (37, 37), analytic -- no interpolation needed
VERT_CHOL = np.linalg.cholesky(VERT_CORR).astype(np.float32)  # (37, 37)

EARTH_RADIUS_M = 6371000.0


def correlated_level_fields(n_lev, shape_hw, vert_chol, rng):
    """Return (n_lev, H, W) fields, each unit-variance, spatially smooth
    (Gaussian-filtered white noise, correlation length SIGMA_PX pixels) AND
    correlated across levels via vert_chol (Cholesky factor of the vertical
    exponential-decay correlation matrix). Longitude axis (last) is treated
    periodically (mode="wrap"); latitude axis is not (mode="nearest").
    Verbatim port of the reference _correlated_level_fields()."""
    H, W = shape_hw
    raw = np.stack([
        gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=SIGMA_PX, mode=("nearest", "wrap"))
        for _ in range(n_lev)
    ], axis=0)
    raw = raw / raw.std(axis=(1, 2), keepdims=True)
    mixed = np.einsum("ij,jhw->ihw", vert_chol, raw)
    mixed = mixed / mixed.std(axis=(1, 2), keepdims=True)
    return mixed.astype(np.float32)


def streamfunction_to_uv(psi, lat_deg, lon_deg):
    """psi: (n_lev, H, W) random streamfunction field (arbitrary units).
    Returns non-divergent (u, v) via u = -(1/R) dpsi/dphi,
    v = (1/(R cos(phi))) dpsi/dlambda (Holton, streamfunction/rotational-wind
    relation). Longitude derivative uses a periodic central difference;
    latitude derivative uses np.gradient with the true latitude coordinate.
    Verbatim port of the reference _streamfunction_to_uv()."""
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    dlambda = float(np.mean(np.diff(lon_rad)))

    dpsi_dlambda = (np.roll(psi, -1, axis=2) - np.roll(psi, 1, axis=2)) / (2.0 * dlambda)
    dpsi_dphi = np.gradient(psi, lat_rad, axis=1)

    cosphi = np.clip(np.cos(lat_rad), np.cos(np.deg2rad(85.0)), None)[None, :, None]

    u = -(1.0 / EARTH_RADIUS_M) * dpsi_dphi
    v = (1.0 / (EARTH_RADIUS_M * cosphi)) * dpsi_dlambda
    return u.astype(np.float32), v.astype(np.float32)


def generate_correlated_wind_perturbation(n_lev, lat_deg, lon_deg, vert_chol, sigma_by_level, rng):
    """Full pipeline: correlated random streamfunction -> non-divergent u,v ->
    rescale per level so empirical std matches sigma_by_level (m/s). Verbatim
    port of the reference _generate_correlated_wind_perturbation()."""
    psi = correlated_level_fields(n_lev, (len(lat_deg), len(lon_deg)), vert_chol, rng)
    u_raw, v_raw = streamfunction_to_uv(psi, lat_deg, lon_deg)
    u_std = u_raw.std(axis=(1, 2), keepdims=True)
    v_std = v_raw.std(axis=(1, 2), keepdims=True)
    u = (u_raw / u_std) * sigma_by_level[:, None, None]
    v = (v_raw / v_std) * sigma_by_level[:, None, None]
    return u.astype(np.float32), v.astype(np.float32)
