#!/usr/bin/env python3
"""Plot §4.3 intrinsic spin threshold: kc=0 vs kc=10^7, no Earth, 12 hr tmax."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "dispersion_ratio": {
        "ylabel": "Peak dispersion ratio",
        "yscale": "log",
        "ref_lines": [(1.0, "gray", ":")],
        "ylim_floor": 0.95,
        "scale": 1.0,
    },
    "unbound_fraction": {
        "ylabel": "Peak unbound fraction (%)",
        "yscale": "linear",
        "ref_lines": [(1.0, "gray", ":"), (30.0, "gray", ":")],
        "ylim_floor": 0.0,
        "scale": 100.0,
    },
}


def load_batch(csv_path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    spin: list[float] = []
    values: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            spin.append(float(row["apophis_spin_period"]))
            values.append(float(row[column]))
    order = np.argsort(spin)
    return np.asarray(spin)[order], np.asarray(values)[order]


def resolve_batch(runs_root: Path, label: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    matches = sorted(runs_root.glob(f"sobol_*_{label}/sobol_mass_outputs.csv"))
    if not matches:
        raise SystemExit(f"No batch found for label {label!r} under {runs_root}")
    return matches[-1].resolve()


def plot_threshold(
    spin0: np.ndarray,
    y0: np.ndarray,
    spin1: np.ndarray,
    y1: np.ndarray,
    metric: str,
    output: Path,
) -> None:
    cfg = METRICS[metric]
    y0 = y0 * cfg["scale"]
    y1 = y1 * cfg["scale"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        spin0,
        y0,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=7,
        label=r"$k_c = 0$ (cohesionless)",
    )
    ax.plot(
        spin1,
        y1,
        "s--",
        color="#ff7f0e",
        linewidth=2,
        markersize=6,
        label=r"$k_c = 10^7$ dyne/cm",
    )
    for yref, color, style in cfg["ref_lines"]:
        ax.axhline(yref, color=color, linestyle=style, linewidth=1, alpha=0.8)

    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_yscale(cfg["yscale"])
    ymax = max(np.max(y0), np.max(y1)) * 1.15
    if cfg["yscale"] == "log":
        ax.set_ylim(cfg["ylim_floor"], max(ymax, 2.0))
    else:
        ax.set_ylim(cfg["ylim_floor"], max(ymax, 5.0))

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.suptitle(
        "Intrinsic spin threshold without Earth (sphere DEM)\n"
        r"$n_p = 500$, $t_\mathrm{max} = 12$ hr, apophis_only — qualitative threshold only",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


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
        "--metric",
        choices=sorted(METRICS),
        default="dispersion_ratio",
        help="Output metric column (default: dispersion_ratio)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --metric)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    runs_root = repo / "sobol_mass_runs"

    kc0_csv = resolve_batch(runs_root, "spin_kc0_12hr", args.kc0_csv)
    kc1e7_csv = resolve_batch(runs_root, "spin_kc1e7_12hr", args.kc1e7_csv)
    if args.output is None:
        if args.metric == "dispersion_ratio":
            out = repo / "sobol_mass_runs/plots/spin_kc_intrinsic_threshold_12hr.png"
        else:
            out = repo / "sobol_mass_runs/plots/spin_kc_intrinsic_threshold_12hr_unbound.png"
    else:
        out = (repo / args.output).resolve()

    spin0, y0 = load_batch(kc0_csv, args.metric)
    spin1, y1 = load_batch(kc1e7_csv, args.metric)
    plot_threshold(spin0, y0, spin1, y1, args.metric, out)
    print(f"  kc=0:   {kc0_csv}")
    print(f"  kc=1e7: {kc1e7_csv}")


if __name__ == "__main__":
    main()
