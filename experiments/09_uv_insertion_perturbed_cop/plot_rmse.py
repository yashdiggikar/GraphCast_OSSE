"""Plot RMSE vs lead time (GraphCast full 60-day rollout vs ERA5).

One PNG per variable (T, U, V, Z); within each PNG, one panel per pressure
level (300/500/1000 hPa by default).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VAR_ORDER = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential"]
VAR_LABEL = {
    "temperature": "Temperature (K)",
    "u_component_of_wind": "U wind (m/s)",
    "v_component_of_wind": "V wind (m/s)",
    "geopotential": "Geopotential (m^2/s^2)",
}
LEVELS = [300, 500, 1000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="out/rmse_60day_2022-01-01.csv")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--out_prefix", default="rmse_vs_leadtime")
    p.add_argument("--levels", type=int, nargs="+", default=LEVELS)
    p.add_argument("--run_label", default="GraphCast full (0.25deg) 60-day rollout",
                    help="Short description of this run, shown in the plot title "
                         "(e.g. to distinguish an OSSE insertion run from the control).")
    p.add_argument("--control_csv", default=None,
                    help="Optional control-run RMSE CSV (same format as --csv). "
                         "When given, overlays the control's RMSE as a black "
                         "dotted line in every panel for reference.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["level_hpa"].isin(args.levels)]

    ctrl_df = None
    if args.control_csv:
        ctrl_df = pd.read_csv(args.control_csv)
        ctrl_df = ctrl_df[ctrl_df["level_hpa"].isin(args.levels)]

    for var in VAR_ORDER:
        sub = df[df["variable"] == var]
        if sub.empty:
            continue
        ctrl_sub = ctrl_df[ctrl_df["variable"] == var] if ctrl_df is not None else None

        fig, axes = plt.subplots(1, len(args.levels), figsize=(5 * len(args.levels), 4.5),
                                  sharey=True)
        if len(args.levels) == 1:
            axes = [axes]

        for ax, lvl in zip(axes, args.levels):
            s = sub[sub["level_hpa"] == lvl].sort_values("lead_days")
            ax.plot(s["lead_days"], s["rmse"], color="tab:blue", linewidth=1.3, label=args.run_label)
            if ctrl_sub is not None:
                cs = ctrl_sub[ctrl_sub["level_hpa"] == lvl].sort_values("lead_days")
                if not cs.empty:
                    ax.plot(cs["lead_days"], cs["rmse"], color="black", linestyle=":",
                            linewidth=1.3, label="Control (free-run)")
            ax.set_title(f"{lvl} hPa")
            ax.set_xlabel("Lead time (days)")
            ax.axvline(15, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(alpha=0.3)
        if ctrl_sub is not None:
            axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_ylabel("RMSE")

        fig.suptitle(
            f"{VAR_LABEL.get(var, var)} -- {args.run_label} vs ERA5\n"
            "global cos(lat)-weighted RMSE, IC 2022-01-01T06:00Z "
            "(dashed line at 15d marks approximate GraphCast skill horizon)",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        out_png = os.path.join(args.out_dir, f"{args.out_prefix}_{var}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
