"""Compares an OSSE truth-insertion run's RMSE against the free-running control,
for GraphCast full (0.25deg). Both inputs are long-format CSVs produced by
`compute_rmse.py` (either the one in control/ or the one copied into an OSSE
variant dir) -- columns: variable, level_hpa, lead_hours, lead_days, valid_time, rmse.

Produces:
  1. <out_prefix>_<VARCODE><level>.png -- one PNG per (variable, level) pair
     (e.g. compare_z_global_T500.png), a single panel with just 2 curves:
     control (free-running, black dotted) vs the OSSE run (dark red). No
     cross-variable/cross-level grouping -- one plot, one field, one level.
  2. <out_prefix>_pct_improvement.csv -- long-format, one row per
     (variable, level), with mean RMSE improvement over the POST-INJECTION window
     only (lead_days >= inject_day), since the whole point of the comparison is
     the insertion window, not the free-run window where control==OSSE by design.
  3. <out_prefix>_pct_improvement.png -- bar chart of that summary, matching the
     Aurora/Prithvi/FCNv2 "rmse_improvement" convention.

Usage:
    python compare_control_vs_osse.py \
        --control_csv control/out/rmse_60day_2022-01-01.csv \
        --osse_csv osse_z_global/out/rmse_osse_z_global_2022-01-01.csv \
        --inject_step 120 \
        --out_dir osse_z_global/out \
        --out_prefix compare_z_global
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VAR_ORDER = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential",
             "specific_humidity"]
VAR_LABEL = {
    "temperature": "Temperature (K)",
    "u_component_of_wind": "U wind (m/s)",
    "v_component_of_wind": "V wind (m/s)",
    "geopotential": "Geopotential (m^2/s^2)",
    "specific_humidity": "Specific humidity (kg/kg)",
}
VAR_CODE = {
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "geopotential": "Z",
    "specific_humidity": "Q",
}
DEFAULT_LEVELS = [300, 500, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--control_csv", required=True)
    p.add_argument("--osse_csv", required=True)
    p.add_argument("--inject_step", type=int, default=120,
                    help="0-based global step index insertion started at "
                         "(used to compute inject_day = inject_step * 6 / 24 for "
                         "the post-injection RMSE-improvement window).")
    p.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_prefix", default="compare")
    p.add_argument("--osse_label", default="OSSE (truth insertion)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    inject_day = args.inject_step * 6.0 / 24.0

    ctrl = pd.read_csv(args.control_csv)
    osse = pd.read_csv(args.osse_csv)

    # ---- 1) One single-panel plot per (variable, level): control vs OSSE, 2 curves ----
    for var in VAR_ORDER:
        c_sub = ctrl[ctrl["variable"] == var]
        o_sub = osse[osse["variable"] == var]
        if c_sub.empty or o_sub.empty:
            continue

        for lvl in args.levels:
            cs = c_sub[c_sub["level_hpa"] == lvl].sort_values("lead_days")
            os_ = o_sub[o_sub["level_hpa"] == lvl].sort_values("lead_days")
            if cs.empty or os_.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(cs["lead_days"], cs["rmse"], color="black", linestyle=":", linewidth=1.5, label="Control (free-run)")
            ax.plot(os_["lead_days"], os_["rmse"], color="darkred", linewidth=1.3, label=args.osse_label)
            ax.axvline(inject_day, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_xlabel("Lead time (days)")
            ax.set_ylabel("RMSE")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)

            var_code = VAR_CODE.get(var, var)
            ax.set_title(
                f"{VAR_LABEL.get(var, var)} @ {lvl} hPa ({var_code}{lvl}) -- Control vs {args.osse_label}\n"
                f"global cos(lat)-weighted RMSE, IC 2022-01-01T06:00Z "
                f"(dashed line at day {inject_day:.0f} marks insertion start)",
                fontsize=10,
            )
            fig.tight_layout()

            out_png = os.path.join(args.out_dir, f"{args.out_prefix}_{var_code}{lvl}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"Wrote {out_png}")

    # ---- 2/3) Post-injection % improvement summary ----
    ctrl_post = ctrl[ctrl["lead_days"] >= inject_day]
    osse_post = osse[osse["lead_days"] >= inject_day]

    merge_keys = ["variable", "level_hpa", "lead_hours"]
    merged = pd.merge(
        ctrl_post[merge_keys + ["rmse"]].rename(columns={"rmse": "rmse_control"}),
        osse_post[merge_keys + ["rmse"]].rename(columns={"rmse": "rmse_osse"}),
        on=merge_keys, how="inner",
    )
    merged["pct_improvement"] = 100.0 * (merged["rmse_control"] - merged["rmse_osse"]) / merged["rmse_control"]

    summary = (
        merged.groupby(["variable", "level_hpa"])
        .agg(mean_rmse_control=("rmse_control", "mean"),
             mean_rmse_osse=("rmse_osse", "mean"),
             mean_pct_improvement=("pct_improvement", "mean"))
        .reset_index()
    )
    summary_csv = os.path.join(args.out_dir, f"{args.out_prefix}_pct_improvement.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # Bar chart: one group per variable, bars per level, post-injection mean % improvement
    plot_summary = summary[summary["level_hpa"].isin(args.levels)]
    variables_present = [v for v in VAR_ORDER if v in plot_summary["variable"].unique()]
    x = np.arange(len(variables_present))
    width = 0.8 / max(len(args.levels), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lvl in enumerate(args.levels):
        vals = [
            plot_summary[(plot_summary["variable"] == v) & (plot_summary["level_hpa"] == lvl)]
            ["mean_pct_improvement"].mean()
            for v in variables_present
        ]
        ax.bar(x + i * width, vals, width, label=f"{lvl} hPa")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(args.levels) - 1) / 2)
    ax.set_xticklabels([VAR_LABEL.get(v, v) for v in variables_present], rotation=15, ha="right")
    ax.set_ylabel("Mean RMSE improvement (%)")
    ax.set_title(f"{args.osse_label} vs control -- post-injection (lead >= {inject_day:.0f}d) mean RMSE improvement")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, f"{args.out_prefix}_pct_improvement.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
