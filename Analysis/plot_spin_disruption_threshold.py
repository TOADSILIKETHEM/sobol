#!/usr/bin/env python3
"""Four-regime spin disruption threshold plot (thesis §4.8.4).

Combines sphere vs OBJ, cohesion, and Earth/no-Earth campaigns to show how
bilobed geometry and spin–orbit alignment shift the instability boundary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_float(val: str) -> "float | None":
    s = val.strip()
    return float(s) if s else None


def load_spin_disp(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    spin: list[float] = []
    disp: list[float] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            s = _to_float(row["apophis_spin_period"])
            d = _to_float(row["dispersion_ratio"])
            if s is None or d is None:
                continue
            spin.append(s)
            disp.append(d)
    order = np.argsort(spin)
    return np.asarray(spin)[order], np.asarray(disp)[order]


def load_obj_earth_opposite(repo: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fixed ~177° Earth OBJ runs (np=1000) plus near-opposite Sobol samples."""
    spin: list[float] = []
    disp: list[float] = []

    fine_dt_suffixes = [
        "torque_align_obj_fine_dt_p155_np1000",
        "torque_align_obj_fine_dt_p1p6hr_np1000",
        "torque_align_obj_fine_dt_p2hr_np1000",
    ]
    runs_root = repo / "sobol_mass_runs"
    for suffix in fine_dt_suffixes:
        matches = sorted(runs_root.glob(f"sobol_*_{suffix}/sobol_mass_outputs.csv"))
        if not matches:
            continue
        csv_path = matches[-1]
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                s = _to_float(row["apophis_spin_period"])
                d = _to_float(row["dispersion_ratio"])
                if s is None or d is None:
                    continue
                spin.append(s)
                disp.append(d)

    kmin_csv = (
        repo
        / "sobol_mass_runs/sobol_20260611_172636_flyby_spin_torque_period_kmin_obj/"
        "sobol_mass_outputs.csv"
    )
    if kmin_csv.exists():
        with kmin_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                a = _to_float(row["apophis_spin_torque_align_deg"])
                if a is None or not (160.0 <= a <= 180.0):
                    continue
                s = _to_float(row["apophis_spin_period"])
                d = _to_float(row["dispersion_ratio"])
                if s is None or d is None:
                    continue
                spin.append(s)
                disp.append(d)

    order = np.argsort(spin)
    return np.asarray(spin)[order], np.asarray(disp)[order]


def resolve_batch(repo: Path, label: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return (repo / explicit).resolve()
    matches = sorted(
        (repo / "sobol_mass_runs").glob(f"sobol_*_{label}/sobol_mass_outputs.csv")
    )
    if not matches:
        raise SystemExit(f"No batch found for label {label!r}")
    return matches[-1].resolve()


def plot_series(
    ax: plt.Axes,
    spin: np.ndarray,
    disp: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
    linewidth: float = 2.0,
    markersize: float = 7,
    zorder: int = 2,
    alpha: float = 1.0,
) -> None:
    ax.plot(
        spin,
        disp,
        f"{marker}{linestyle}",
        color=color,
        linewidth=linewidth,
        markersize=markersize,
        label=label,
        zorder=zorder,
        alpha=alpha,
    )
    breakup = disp > 2.0
    if np.any(breakup):
        ax.scatter(
            spin[breakup],
            disp[breakup],
            s=90,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2,
            zorder=zorder + 1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sphere-kc1e7-csv",
        type=Path,
        default=Path(
            "sobol_mass_runs/sobol_20260608_132304_noearth_ctrl_spin_1p5_2hr/"
            "sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument("--sphere-kc0-csv", type=Path, default=None)
    parser.add_argument(
        "--obj-noearth-csv",
        type=Path,
        default=None,
        help="OBJ no-Earth control CSV (default: latest noearth_obj_ctrl_spin_1p5_2hr batch)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/spin_disruption_threshold_four_regime.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sphere_kc1e7 = (repo / args.sphere_kc1e7_csv).resolve()
    sphere_kc0 = resolve_batch(repo, "spin_kc0_12hr", args.sphere_kc0_csv)
    obj_noearth = resolve_batch(repo, "noearth_obj_ctrl_spin_1p5_2hr", args.obj_noearth_csv)

    spin_s1, disp_s1 = load_spin_disp(sphere_kc1e7)
    spin_s0, disp_s0 = load_spin_disp(sphere_kc0)
    spin_o0, disp_o0 = load_spin_disp(obj_noearth)
    spin_oe, disp_oe = load_obj_earth_opposite(repo)

    fig, ax = plt.subplots(figsize=(9, 6))

    plot_series(
        ax,
        spin_s1,
        disp_s1,
        label=r"Sphere, no Earth, $k_c = 10^7$ ($t_\mathrm{max}=4.5$ d)",
        color="#2ca02c",
        marker="o",
        linestyle="-",
    )
    plot_series(
        ax,
        spin_s0,
        disp_s0,
        label=r"Sphere, no Earth, $k_c = 0$ ($t_\mathrm{max}=12$ h)",
        color="#1f77b4",
        marker="s",
        linestyle="--",
        alpha=0.9,
    )
    plot_series(
        ax,
        spin_oe,
        disp_oe,
        label=r"OBJ, Earth, $\hat{s} \approx -\hat{h}$ ($n_p=1000$)",
        color="#d62728",
        marker="^",
        linestyle="-",
        linewidth=2.2,
    )
    plot_series(
        ax,
        spin_o0,
        disp_o0,
        label=r"OBJ, no Earth, $\hat{s} \approx -\hat{h}$ ($n_p=1000$)",
        color="#9467bd",
        marker="D",
        linestyle="-",
    )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.axvspan(1.45, 2.05, color="#ff9896", alpha=0.12, zorder=0)
    ax.annotate(
        r"Real Apophis ($P \approx 30$ h) $\rightarrow$",
        xy=(5.0, 1.0),
        xytext=(3.6, 1.35),
        fontsize=9,
        color="#17becf",
        arrowprops=dict(arrowstyle="->", color="#17becf", lw=1.2),
    )

    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel("Peak dispersion ratio")
    ax.set_yscale("log")
    ymax = max(
        np.max(disp_s0),
        np.max(disp_s1),
        np.max(disp_o0),
        np.max(disp_oe),
    )
    ax.set_ylim(0.95, ymax * 1.5)
    ax.set_xlim(0.6, 5.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    ax.text(
        0.02,
        0.03,
        "Shaded band: danger window (1.45–2.05 h)\n"
        r"Real Apophis ($\sim$30 h) sits far below all thresholds shown",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    fig.suptitle(
        "Spin-disruption threshold for Apophis DEM rubble piles\n"
        r"Cohesion stabilises spheres; bilobed OBJ at unfavourable spin–orbit alignment "
        r"disrupts near $P \approx 1.55$ hr",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
