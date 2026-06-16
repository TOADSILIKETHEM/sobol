#!/usr/bin/env python3
"""2D heatmap of Priority 3 flyby sweep: spin period vs kc (Sobol samples + interpolation)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata


def load_outputs(csv_path: Path) -> dict[str, np.ndarray]:
    spin: list[float] = []
    kc: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    run_id: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            run_id.append(int(row["run_id"]))
            spin.append(float(row["apophis_spin_period"]))
            kc.append(float(row["kc_cgs"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"]))
    return {
        "run_id": np.asarray(run_id),
        "spin": np.asarray(spin),
        "kc": np.asarray(kc),
        "disp": np.asarray(disp),
        "unbound": np.asarray(unbound),
        "log_kc": np.log10(np.maximum(np.asarray(kc), 1.0)),
    }


def _interp_field(
    spin: np.ndarray,
    log_kc: np.ndarray,
    values: np.ndarray,
    spin_lim: tuple[float, float],
    log_kc_lim: tuple[float, float],
    n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_spin = np.linspace(spin_lim[0], spin_lim[1], n)
    grid_log_kc = np.linspace(log_kc_lim[0], log_kc_lim[1], n)
    xi, yi = np.meshgrid(grid_spin, grid_log_kc)
    zi = griddata(
        (spin, log_kc),
        values,
        (xi, yi),
        method="linear",
    )
    return xi, yi, zi


def _kc_tick_formatter(val: float, _pos: float) -> str:
    if val < 0.05:
        return "0"
    return f"{10**val:.0e}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260601_103753_flyby_spin_kc_p3/sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/flyby_spin_kc_p3_heatmap.png"),
    )
    parser.add_argument(
        "--spin-min",
        type=float,
        default=1.0,
        help="Axis lower limit for spin period (hours).",
    )
    parser.add_argument(
        "--spin-max",
        type=float,
        default=3.0,
        help="Axis upper limit for spin period (hours).",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = (repo / args.csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    d = load_outputs(csv_path)
    spin_lim = (args.spin_min, args.spin_max)
    log_kc_lim = (0.0, 8.0)

    # Capped dispersion for bound-deformation structure (run 4 is orders of magnitude larger).
    disp_cap = 1.15
    disp_show = np.minimum(d["disp"], disp_cap)

    xi, yi, zi_ub = _interp_field(
        d["spin"], d["log_kc"], d["unbound"], spin_lim, log_kc_lim
    )
    _, _, zi_disp = _interp_field(
        d["spin"], d["log_kc"], disp_show, spin_lim, log_kc_lim
    )

    catastrophic = d["disp"] > 10.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    # --- unbound fraction ---
    ax = axes[0]
    im0 = ax.pcolormesh(
        xi,
        yi,
        zi_ub,
        shading="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    sc0 = ax.scatter(
        d["spin"],
        d["log_kc"],
        c=d["unbound"],
        s=55,
        edgecolors="k",
        linewidths=0.6,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        zorder=3,
    )
    for rid, s, lk, u in zip(d["run_id"], d["spin"], d["log_kc"], d["unbound"], strict=True):
        if u > 0.05:
            ax.annotate(
                str(rid),
                (s, lk),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel(r"$\log_{10}(k_c\,/\,\mathrm{dyne\,cm}^{-1})$")
    ax.set_title("Peak unbound mass fraction")
    ax.set_xlim(spin_lim)
    ax.set_ylim(log_kc_lim)
    ax.yaxis.set_major_formatter(FuncFormatter(_kc_tick_formatter))
    cbar0 = fig.colorbar(im0, ax=ax, pad=0.02)
    cbar0.set_label("unbound_fraction")

    # --- dispersion (capped) ---
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
        d["spin"][~catastrophic],
        d["log_kc"][~catastrophic],
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
            d["spin"][catastrophic],
            d["log_kc"][catastrophic],
            s=120,
            marker="*",
            c="red",
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label="catastrophic (off scale)",
        )
        for s, lk, disp_val, rid in zip(
            d["spin"][catastrophic],
            d["log_kc"][catastrophic],
            d["disp"][catastrophic],
            d["run_id"][catastrophic],
            strict=True,
        ):
            ax.annotate(
                f"run {rid}\n$R_\\mathrm{{gyr}}$={disp_val:.0e}",
                (s, lk),
                fontsize=7,
                color="darkred",
                ha="left",
                va="center",
                xytext=(8, 0),
                textcoords="offset points",
            )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel(r"$\log_{10}(k_c\,/\,\mathrm{dyne\,cm}^{-1})$")
    ax.set_title(r"Peak dispersion ratio (capped at %.2f)" % disp_cap)
    ax.set_xlim(spin_lim)
    ax.set_ylim(log_kc_lim)
    ax.yaxis.set_major_formatter(FuncFormatter(_kc_tick_formatter))
    cbar1 = fig.colorbar(im1, ax=ax, pad=0.02)
    cbar1.set_label("dispersion_ratio")

    fig.suptitle(
        "Cohesion and spin near the Earth flyby boundary (sphere DEM)\n"
        r"$t_\mathrm{max}=4.5$ d, $n_p=500$, 16 Sobol samples; shaded field = linear interpolation",
        fontsize=11,
        y=1.02,
    )

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # Optional second figure: log dispersion including catastrophic run
    out_log = out.with_name(out.stem + "_log_disp.png")
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    log_disp = np.log10(np.maximum(d["disp"], 1.0))
    _, _, zi_log = _interp_field(d["spin"], d["log_kc"], log_disp, spin_lim, log_kc_lim)
    im2 = ax2.pcolormesh(
        xi,
        yi,
        zi_log,
        shading="auto",
        cmap="magma",
        norm=colors.Normalize(vmin=0.0, vmax=np.ceil(log_disp.max())),
    )
    ax2.scatter(
        d["spin"],
        d["log_kc"],
        c=log_disp,
        s=55,
        edgecolors="k",
        linewidths=0.6,
        cmap="magma",
        norm=colors.Normalize(vmin=0.0, vmax=np.ceil(log_disp.max())),
        zorder=3,
    )
    ax2.set_xlabel("Spin period (hours)")
    ax2.set_ylabel(r"$\log_{10}(k_c\,/\,\mathrm{dyne\,cm}^{-1})$")
    ax2.set_title(r"$\log_{10}$(peak dispersion ratio)")
    ax2.set_xlim(spin_lim)
    ax2.set_ylim(log_kc_lim)
    ax2.yaxis.set_major_formatter(FuncFormatter(_kc_tick_formatter))
    fig2.suptitle(
        "Cohesion and spin near the Earth flyby boundary (sphere DEM)\n"
        r"$t_\mathrm{max}=4.5$ d, $n_p=500$ — log scale shows catastrophic run 4",
        fontsize=10,
    )
    fig2.colorbar(im2, ax=ax2, label=r"$\log_{10}$(dispersion_ratio)")
    fig2.savefig(out_log, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_log}")


if __name__ == "__main__":
    main()
