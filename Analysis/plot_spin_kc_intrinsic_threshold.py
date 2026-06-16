#!/usr/bin/env python3
"""Plot §4.3 intrinsic spin threshold: kc=0 vs kc=10^7, no Earth, 12 hr tmax."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_batch(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    spin: list[float] = []
    disp: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            spin.append(float(row["apophis_spin_period"]))
            disp.append(float(row["dispersion_ratio"]))
    order = np.argsort(spin)
    return np.asarray(spin)[order], np.asarray(disp)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kc0-csv",
        type=Path,
        default=None,
        help="spin_kc0_12hr outputs CSV (auto-detect latest if omitted)",
    )
    parser.add_argument(
        "--kc1e7-csv",
        type=Path,
        default=None,
        help="spin_kc1e7_12hr outputs CSV (auto-detect latest if omitted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/spin_kc_intrinsic_threshold_12hr.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    runs_root = repo / "sobol_mass_runs"

    def resolve_batch(label: str, explicit: Path | None) -> Path:
        if explicit is not None:
            return (repo / explicit).resolve()
        matches = sorted(runs_root.glob(f"sobol_*_{label}/sobol_mass_outputs.csv"))
        if not matches:
            raise SystemExit(f"No batch found for label {label!r} under {runs_root}")
        return matches[-1].resolve()

    kc0_csv = resolve_batch("spin_kc0_12hr", args.kc0_csv)
    kc1e7_csv = resolve_batch("spin_kc1e7_12hr", args.kc1e7_csv)
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    spin0, disp0 = load_batch(kc0_csv)
    spin1, disp1 = load_batch(kc1e7_csv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        spin0,
        disp0,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=7,
        label=r"$k_c = 0$ (cohesionless)",
    )
    ax.plot(
        spin1,
        disp1,
        "s--",
        color="#ff7f0e",
        linewidth=2,
        markersize=6,
        label=r"$k_c = 10^7$ dyne/cm",
    )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel("Peak dispersion ratio")
    ax.set_yscale("log")
    ymax = max(np.max(disp0), np.max(disp1)) * 1.3
    ax.set_ylim(0.95, max(ymax, 2.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle(
        "Intrinsic spin threshold without Earth (sphere DEM)\n"
        r"$n_p = 500$, $t_\mathrm{max} = 12$ hr, apophis_only — qualitative threshold only",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"  kc=0:   {kc0_csv}")
    print(f"  kc=1e7: {kc1e7_csv}")


if __name__ == "__main__":
    main()
