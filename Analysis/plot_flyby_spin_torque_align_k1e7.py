#!/usr/bin/env python3
"""Plot flyby torque-align campaign: spin-orbit angle vs breakup metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch(csv_path: Path) -> dict[str, np.ndarray]:
    align: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    run_id: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            run_id.append(int(row["run_id"]))
            align.append(float(row["apophis_spin_torque_align_deg"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"]))
    order = np.argsort(align)
    return {
        "run_id": np.asarray(run_id)[order],
        "align": np.asarray(align)[order],
        "disp": np.asarray(disp)[order],
        "unbound": np.asarray(unbound)[order],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260601_140230_flyby_spin_torque_align_k1e7/"
            "sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/flyby_spin_torque_align_k1e7.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = (repo / args.csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = load_batch(csv_path)
    align = data["align"]
    disp = data["disp"]
    unbound = data["unbound"]

    breakup = disp > 2.0

    fig, (ax_disp, ax_unb) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )

    ax_disp.plot(align, disp, "o-", color="#2ca02c", linewidth=2, markersize=7, zorder=2)
    if np.any(breakup):
        ax_disp.scatter(
            align[breakup],
            disp[breakup],
            s=120,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.5,
            zorder=3,
            label="Substantial breakup",
        )
    ax_disp.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_disp.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_disp.axvline(180, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_disp.set_ylabel("Peak size ratio")
    ax_disp.set_yscale("log")
    ax_disp.set_ylim(0.98, max(disp) * 1.4)
    ax_disp.grid(True, which="both", alpha=0.3)
    ax_disp.legend(loc="upper left", framealpha=0.95)

    ax_unb.plot(align, unbound * 100, "s-", color="#9467bd", linewidth=2, markersize=7)
    if np.any(breakup):
        ax_unb.scatter(
            align[breakup],
            unbound[breakup] * 100,
            s=100,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.5,
            zorder=3,
        )
    ax_unb.set_xlabel(r"Torque-align angle (degrees; 0° = +$\hat{h}$, 180° = $-\hat{h}$)")
    ax_unb.set_ylabel("Peak unbound fraction (%)")
    ax_unb.grid(True, alpha=0.3)
    ax_unb.set_ylim(bottom=-1)

    fig.suptitle(
        "Flyby spin–orbit coupling: torque-align vs breakup\n"
        r"Sphere DEM, $P = 1.52$ hr, $k_c = 10^7$ dyne/cm, $n_p = 500$, CA $\approx$ 38,000 km",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
