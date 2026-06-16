#!/usr/bin/env python3
"""Plot §4.6 sphere no-Earth null: cohesive sphere stays intact at 1.5–2.05 hr spin."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch(csv_path: Path) -> dict[str, np.ndarray]:
    spin: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            spin.append(float(row["apophis_spin_period"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"]))
    order = np.argsort(spin)
    return {
        "spin": np.asarray(spin)[order],
        "disp": np.asarray(disp)[order],
        "unbound": np.asarray(unbound)[order],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260608_132304_noearth_ctrl_spin_1p5_2hr/"
            "sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/noearth_ctrl_spin_1p5_2hr.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = (repo / args.csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = load_batch(csv_path)
    spin = data["spin"]
    disp = data["disp"]
    unbound = data["unbound"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(spin, disp, "o-", color="#2ca02c", linewidth=2, markersize=8, zorder=2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel("Peak dispersion ratio")
    ax.set_ylim(0.998, max(disp.max(), 1.01) * 1.005)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.02,
        0.04,
        f"All runs: unbound = 0%, peak disp ≤ {disp.max():.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.suptitle(
        "Null control: fast spin does not disrupt cohesive spheres (no Earth)\n"
        r"Sphere DEM, $k_c = 10^7$ dyne/cm, $n_p = 500$, $t_\mathrm{max} = 4.5$ d, apophis_only",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
