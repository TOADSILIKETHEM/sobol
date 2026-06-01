#!/usr/bin/env python3
"""Plot dispersion_ratio vs spin period: Earth flyby vs apophis_only control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    spins: list[float] = []
    disp: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            spins.append(float(row["apophis_spin_period"]))
            disp.append(float(row["dispersion_ratio"]))
    order = np.argsort(spins)
    return np.asarray(spins)[order], np.asarray(disp)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--earth-csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260531_172510_flyby_spin_earth/sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "--noearth-csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260531_185537_flyby_spin_noearth_ctrl/sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/flyby_spin_earth_vs_noearth_dispersion.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    earth_csv = (repo / args.earth_csv).resolve()
    noearth_csv = (repo / args.noearth_csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    spin_e, disp_e = load_batch(earth_csv)
    spin_n, disp_n = load_batch(noearth_csv)

    fig, (ax_main, ax_diff) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_main.plot(
        spin_e,
        disp_e,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=7,
        label="With Earth (flyby_spin_earth)",
    )
    ax_main.plot(
        spin_n,
        disp_n,
        "s--",
        color="#ff7f0e",
        linewidth=2,
        markersize=6,
        label="No Earth — apophis_only (flyby_spin_noearth_ctrl)",
    )
    ax_main.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_main.set_ylabel("Peak dispersion ratio")
    ax_main.set_title(
        "DEM rubble-pile deformation vs spin period\n"
        r"$k_c = 0$, $t_\mathrm{max} = 4.5$ days, same 16 Sobol spin samples"
    )
    ax_main.legend(loc="upper right", framealpha=0.95)
    ax_main.set_ylim(bottom=0.995)
    ax_main.grid(True, alpha=0.3)

    # Match spins (same samples); difference = Earth − control
    if not np.allclose(spin_e, spin_n):
        raise SystemExit("Spin periods differ between batches — cannot pair runs.")
    delta = disp_e - disp_n
    colors = np.where(delta >= 0, "#1f77b4", "#d62728")
    ax_diff.bar(spin_e, delta, width=0.06, color=colors, alpha=0.75, edgecolor="none")
    ax_diff.axhline(0.0, color="gray", linewidth=1)
    ax_diff.set_xlabel("Spin period (hours)")
    ax_diff.set_ylabel(r"$\Delta$ dispersion\n(Earth − no Earth)")
    ax_diff.grid(True, alpha=0.3)

    real_spin_hr = 30.0
    for ax in (ax_main, ax_diff):
        ax.axvline(
            real_spin_hr,
            color="green",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label="Real Apophis (~30 h)" if ax is ax_main else None,
        )
    handles, labels = ax_main.get_legend_handles_labels()
    handles.append(
        plt.Line2D([0], [0], color="green", linestyle="-.", linewidth=1.2, label="Real Apophis (~30 h)")
    )
    ax_main.legend(handles=handles, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
