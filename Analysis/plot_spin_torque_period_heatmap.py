#!/usr/bin/env python3
"""2D heatmap: spin period vs torque-align angle (Sobol samples + interpolation)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.interpolate import griddata


def load_outputs(csv_path: Path) -> dict[str, np.ndarray]:
    spin: list[float] = []
    align: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    run_id: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            run_id.append(int(row["run_id"]))
            spin.append(float(row["apophis_spin_period"]))
            align.append(float(row["apophis_spin_torque_align_deg"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"]))
    return {
        "run_id": np.asarray(run_id),
        "spin": np.asarray(spin),
        "align": np.asarray(align),
        "disp": np.asarray(disp),
        "unbound": np.asarray(unbound),
    }


def _interp_field(
    align: np.ndarray,
    spin: np.ndarray,
    values: np.ndarray,
    align_lim: tuple[float, float],
    spin_lim: tuple[float, float],
    n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_align = np.linspace(align_lim[0], align_lim[1], n)
    grid_spin = np.linspace(spin_lim[0], spin_lim[1], n)
    xi, yi = np.meshgrid(grid_align, grid_spin)
    zi = griddata((align, spin), values, (xi, yi), method="linear")
    return xi, yi, zi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260611_172636_flyby_spin_torque_period_kmin_obj/"
            "sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "sobol_mass_runs/plots/flyby_spin_torque_period_kmin_obj_heatmap.png"
        ),
    )
    parser.add_argument("--spin-min", type=float, default=1.3)
    parser.add_argument("--spin-max", type=float, default=2.5)
    parser.add_argument("--align-min", type=float, default=0.0)
    parser.add_argument("--align-max", type=float, default=180.0)
    parser.add_argument(
        "--disp-cap",
        type=float,
        default=1.15,
        help="Cap dispersion colour scale for bound-deformation panel.",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = (repo / args.csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    d = load_outputs(csv_path)
    spin_lim = (args.spin_min, args.spin_max)
    align_lim = (args.align_min, args.align_max)

    disp_cap = args.disp_cap
    disp_show = np.minimum(d["disp"], disp_cap)
    catastrophic = d["disp"] > 10.0

    ub_min = float(np.floor(d["unbound"].min() * 1000) / 1000)
    ub_max = float(np.ceil(d["unbound"].max() * 1000) / 1000)

    xi, yi, zi_ub = _interp_field(
        d["align"], d["spin"], d["unbound"], align_lim, spin_lim
    )
    _, _, zi_disp = _interp_field(
        d["align"], d["spin"], disp_show, align_lim, spin_lim
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)

    ax = axes[0]
    im0 = ax.pcolormesh(
        xi,
        yi,
        zi_ub,
        shading="auto",
        cmap="YlOrRd",
        vmin=ub_min,
        vmax=ub_max,
    )
    ax.scatter(
        d["align"],
        d["spin"],
        c=d["unbound"],
        s=55,
        edgecolors="k",
        linewidths=0.6,
        cmap="YlOrRd",
        vmin=ub_min,
        vmax=ub_max,
        zorder=3,
    )
    intact = d["disp"] <= 1.05
    if np.any(intact):
        ax.scatter(
            d["align"][intact],
            d["spin"][intact],
            s=90,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.2,
            zorder=4,
            label=r"intact ($R_\mathrm{gyr}/R_0 \leq 1.05$)",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel(r"Torque-align angle (deg; $0°=+\hat{h}$, $180°=-\hat{h}$)")
    ax.set_ylabel("Spin period (hours)")
    ax.set_title("Peak unbound mass fraction (main body)")
    ax.set_xlim(align_lim)
    ax.set_ylim(spin_lim)
    cbar0 = fig.colorbar(im0, ax=ax, pad=0.02)
    cbar0.set_label("unbound_fraction")

    ax = axes[1]
    im1 = ax.pcolormesh(
        xi,
        yi,
        zi_disp,
        shading="auto",
        cmap="viridis",
        vmin=1.0,
        vmax=disp_cap,
    )
    ax.scatter(
        d["align"][~catastrophic],
        d["spin"][~catastrophic],
        c=disp_show[~catastrophic],
        s=55,
        edgecolors="k",
        linewidths=0.6,
        cmap="viridis",
        vmin=1.0,
        vmax=disp_cap,
        zorder=3,
    )
    if np.any(catastrophic):
        ax.scatter(
            d["align"][catastrophic],
            d["spin"][catastrophic],
            s=120,
            marker="*",
            c="red",
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label="catastrophic (off scale)",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel(r"Torque-align angle (deg; $0°=+\hat{h}$, $180°=-\hat{h}$)")
    ax.set_ylabel("Spin period (hours)")
    ax.set_title(r"Peak dispersion ratio (capped at %.2f)" % disp_cap)
    ax.set_xlim(align_lim)
    ax.set_ylim(spin_lim)
    cbar1 = fig.colorbar(im1, ax=ax, pad=0.02)
    cbar1.set_label("dispersion_ratio")

    fig.suptitle(
        r"Earth flyby OBJ DEM ($n_p=500$, $k_c=3.5\times10^6$ dyne cm$^{-1}$, "
        r"$t_\mathrm{max}=4.5$ d)" + "\n"
        "64 Sobol samples; shaded field = linear interpolation (illustrative)",
        fontsize=11,
        y=1.02,
    )

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    out_log = out.with_name(out.stem + "_log_disp.png")
    fig2, ax2 = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    log_disp = np.log10(np.maximum(d["disp"], 1.0))
    _, _, zi_log = _interp_field(d["align"], d["spin"], log_disp, align_lim, spin_lim)
    vmax_log = float(np.ceil(log_disp.max()))
    im2 = ax2.pcolormesh(
        xi,
        yi,
        zi_log,
        shading="auto",
        cmap="magma",
        norm=colors.Normalize(vmin=0.0, vmax=vmax_log),
    )
    ax2.scatter(
        d["align"],
        d["spin"],
        c=log_disp,
        s=55,
        edgecolors="k",
        linewidths=0.6,
        cmap="magma",
        norm=colors.Normalize(vmin=0.0, vmax=vmax_log),
        zorder=3,
    )
    ax2.set_xlabel(r"Torque-align angle (deg; $0°=+\hat{h}$, $180°=-\hat{h}$)")
    ax2.set_ylabel("Spin period (hours)")
    ax2.set_title(r"$\log_{10}$(peak dispersion ratio)")
    ax2.set_xlim(align_lim)
    ax2.set_ylim(spin_lim)
    fig2.colorbar(im2, ax=ax2, label=r"$\log_{10}$(dispersion_ratio)")
    fig2.savefig(out_log, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_log}")


if __name__ == "__main__":
    main()
