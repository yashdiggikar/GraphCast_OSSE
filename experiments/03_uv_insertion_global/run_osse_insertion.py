"""GraphCast (full, 0.25deg, 37-level) OSSE truth-insertion rollout: U+V wind,
global, all-levels, hard replacement -- matches the free-run 60-day rollout in
`control/run_graphcast_longrollout.py`, except from `--inject_from_step` (0-based;
default 120 == "step 121" in the other models' 1-based convention, ~30 days in)
onward, the named insertion variable(s) are hard-replaced with ERA5 truth at every
6-hourly step through the end of the 240-step (60-day) rollout, and the replaced
values are fed back into the model's own autoregressive state -- matching the
Aurora/Prithvi/FCNv2 OSSE study's "free-run first half, then truth-corrected second
half" design (see session_log.md Sec.12).

Same memory-safe streaming approach as the control script (lazy target template,
astronomy-only forcing synthesis, chunked_prediction_generator + write-and-discard)
-- see control/run_graphcast_longrollout.py's docstring for why that's needed.
Only new piece here: loads a pre-built lead-time-indexed truth Dataset (from
../prepare_truth.py) and passes it through to `rollout.chunked_prediction_generator`'s
truth-insertion args.

Usage:
    python run_osse_insertion.py \
        --input_nc ../dataset_samples/source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc \
        --params ../params/GraphCast_full_0.25deg_37levels.npz \
        --stats_dir ../stats \
        --truth_nc ../era5_37level_2022/truth_leadtime_2022-01-01IC.nc \
        --insert_var_names u_component_of_wind v_component_of_wind \
        --inject_from_step 120 \
        --num_steps 240 \
        --steps_per_chunk 4 \
        --out_dir out/osse_uv_global_2022-01-01
"""
import argparse
import dataclasses
import functools
import os
import time

import haiku as hk
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np
import xarray

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_nc", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--stats_dir", required=True)
    p.add_argument("--truth_nc", required=True,
                    help="Lead-time-indexed truth Dataset from prepare_truth.py "
                         "(all 37 levels, geopotential/temperature/u_.../v_...)")
    p.add_argument("--insert_var_names", nargs="+",
                    default=["u_component_of_wind", "v_component_of_wind"],
                    help="Which GraphCast variable name(s) to hard-replace with truth "
                         "from --inject_from_step onward.")
    p.add_argument("--inject_from_step", type=int, default=120,
                    help="0-based global step index. Default 120 == 'step 121' in the "
                         "other models' 1-based convention (~30 days free-run, then "
                         "truth-corrected for the remaining ~30 days).")
    p.add_argument("--num_steps", type=int, default=240,
                    help="Total 6h autoregressive steps (240 = 60 days)")
    p.add_argument("--steps_per_chunk", type=int, default=4,
                    help="Must evenly divide num_steps")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_levels", type=int, nargs="+",
                    default=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                    help="Pressure levels (hPa) to keep in saved output "
                         "(model still runs/rolls forward at full 37 levels; "
                         "this only subsets what's written to disk). "
                         "Default matches the Aurora study's 13-level convention.")
    return p.parse_args()


def build_synthetic_forcings(lat, lon, batch_coord, ic_last_datetime,
                              timestep, num_steps, forcing_variables):
    """Builds an all-future forcings Dataset via pure astronomy (no truth data).

    NOTE: only broadcasts correctly for batch_coord.size == 1 (see session_log.md
    Sec.10's "latent-but-inert fragility" note) -- fine here since these OSSE runs
    are single-IC, batch=1, matching the control run.
    """
    lead_times = (np.arange(1, num_steps + 1) * timestep)
    datetimes = ic_last_datetime + lead_times

    coords = dict(
        time=("time", lead_times),
        lat=("lat", lat.values),
        lon=("lon", lon.values),
    )
    ds = xarray.Dataset(coords=coords)

    if batch_coord is not None:
        ds = ds.expand_dims("batch")  # bare size-1 dim, no coordinate values
        dt_2d = np.broadcast_to(datetimes, (batch_coord.size, num_steps))
        ds = ds.assign_coords(datetime=(("batch", "time"), dt_2d))
    else:
        ds = ds.assign_coords(datetime=("time", datetimes))

    data_utils.add_derived_vars(ds)
    data_utils.add_tisr_var(ds)

    return ds[list(forcing_variables)]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading checkpoint:", args.params, flush=True)
    with open(args.params, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model config:", model_config, flush=True)
    print("Forcing variables:", task_config.forcing_variables, flush=True)

    diffs_stddev_by_level = xarray.load_dataset(f"{args.stats_dir}/diffs_stddev_by_level.nc")
    mean_by_level = xarray.load_dataset(f"{args.stats_dir}/mean_by_level.nc")
    stddev_by_level = xarray.load_dataset(f"{args.stats_dir}/stddev_by_level.nc")

    print("Opening input file (lazy) and slicing first 3 timesteps for IC:", args.input_nc, flush=True)
    full_ds = xarray.open_dataset(args.input_nc, decode_timedelta=True)
    ic_slice = full_ds.isel(time=slice(0, 3)).load()  # 2 inputs + 1 target step
    full_ds.close()

    assert model_config.resolution in (0, 360.0 / ic_slice.sizes["lon"])

    eval_inputs, eval_targets_1, eval_forcings_1 = data_utils.extract_inputs_targets_forcings(
        ic_slice,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    print("IC inputs:", dict(eval_inputs.dims), flush=True)

    timestep = eval_targets_1.coords["time"].values[0]
    ic_last_datetime = ic_slice.coords["datetime"].isel(time=1).values

    print("Building lazy target template for", args.num_steps, "steps...", flush=True)
    targets_template = rollout.extend_targets_template(eval_targets_1, args.num_steps)

    print("Synthesizing forcings (TISR + year/day progress) for", args.num_steps,
          "future steps...", flush=True)
    batch_coord = eval_inputs.coords["batch"] if "batch" in eval_inputs.dims else None
    forcings = build_synthetic_forcings(
        lat=eval_inputs.coords["lat"],
        lon=eval_inputs.coords["lon"],
        batch_coord=batch_coord,
        ic_last_datetime=ic_last_datetime,
        timestep=timestep,
        num_steps=args.num_steps,
        forcing_variables=task_config.forcing_variables,
    )
    print("Forcings dims:", dict(forcings.dims), flush=True)

    print("Loading truth dataset for insertion:", args.truth_nc, flush=True)
    truth_ds = xarray.open_dataset(args.truth_nc, decode_timedelta=True)
    missing = [v for v in args.insert_var_names if v not in truth_ds.data_vars]
    if missing:
        raise KeyError(f"--insert_var_names has variable(s) not in truth_nc: {missing}. "
                        f"Available: {list(truth_ds.data_vars)}")
    print(f"Insertion: variables={args.insert_var_names}, "
          f"from step {args.inject_from_step} (0-based) through {args.num_steps - 1} "
          f"(== 1-based steps {args.inject_from_step + 1}..{args.num_steps})", flush=True)

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    print(f"Starting streaming OSSE insertion rollout: {args.num_steps} steps, "
          f"{args.steps_per_chunk} steps/chunk -> "
          f"{args.num_steps // args.steps_per_chunk} chunks", flush=True)

    t0 = time.time()
    chunk_idx = 0
    for prediction_chunk in rollout.chunked_prediction_generator(
        predictor_fn=run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=targets_template,
        forcings=forcings,
        num_steps_per_chunk=args.steps_per_chunk,
        verbose=True,
        truth_ds=truth_ds,
        insert_var_names=args.insert_var_names,
        inject_from_step=args.inject_from_step,
    ):
        out_path = os.path.join(args.out_dir, f"chunk_{chunk_idx:04d}.nc")
        # Model rolls forward internally at the checkpoint's native 37 levels;
        # only the on-disk output is subset, to keep file size manageable.
        to_save = prediction_chunk.sel(level=args.save_levels) if "level" in prediction_chunk.dims else prediction_chunk
        to_save.to_netcdf(out_path)
        elapsed = time.time() - t0
        step_num = (chunk_idx + 1) * args.steps_per_chunk
        print(f"[{elapsed:8.1f}s] wrote {out_path} "
              f"(through step {step_num}/{args.num_steps}, "
              f"lead {step_num * 6}h = {step_num * 6 / 24:.1f}d)", flush=True)
        del prediction_chunk
        chunk_idx += 1

    print(f"Done. {chunk_idx} chunks written to {args.out_dir} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
