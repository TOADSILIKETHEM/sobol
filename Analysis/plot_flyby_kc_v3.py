#!/usr/bin/env python3
"""Plot §4.2 flyby_kc_v3: cohesion null at real Apophis spin (~30 hr)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kc: list[float] = []
    disp: list[float] = []
    ca: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            kc.append(float(row["kc_cgs"]))
            disp.append(float(row["dispersion_ratio"]))
            ca.append(float(row["closest_approach_km"]))
    order = np.argsort(kc)
    return (
        np.asarray(kc)[order],
        np.asarray(disp)[order],
        np.asarray(ca)[order],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260531_153147_flyby_kc_v3/sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/flyby_kc_v3_dispersion_vs_kc.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = (repo / args.csv).resolve()
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    kc, disp, ca = load_batch(csv_path)
    log_kc = np.log10(kc)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(log_kc, disp, "o-", color="#1f77b4", linewidth=2, markersize=8, zorder=2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}(k_c)$ [dyne/cm]")
    ax.set_ylabel("Peak dispersion ratio")
    ax.set_ylim(1.0, max(disp) * 1.02)
    ax.grid(True, alpha=0.3)

    ca_mean = float(np.mean(ca))
    ax.text(
        0.02,
        0.04,
        f"All runs: unbound = 0%, CA ≈ {ca_mean:,.0f} km",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.suptitle(
        "Flyby baseline at real Apophis spin: cohesion sensitivity null\n"
        r"Earth flyby, $P \approx 30$ hr, sphere DEM, $n_p = 500$, "
        r"$k_c \approx 3.5\times10^6$–$10^8$ dyne/cm (Sobol)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
