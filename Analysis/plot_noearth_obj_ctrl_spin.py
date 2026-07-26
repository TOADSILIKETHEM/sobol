#!/usr/bin/env python3
"""Plot Priority 2 no-Earth OBJ control: spin period vs breakup metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_float(val: str) -> "float | None":
    s = val.strip()
    return float(s) if s else None


def load_batch(csv_path: Path) -> dict[str, np.ndarray]:
    spin: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    run_id: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            s = _to_float(row["apophis_spin_period"])
            d = _to_float(row["dispersion_ratio"])
            u = _to_float(row["unbound_fraction"])
            if s is None or d is None or u is None:
                continue
            run_id.append(int(row["run_id"]))
            spin.append(s)
            disp.append(d)
            unbound.append(u)
    order = np.argsort(spin)
    return {
        "run_id": np.asarray(run_id)[order],
        "spin": np.asarray(spin)[order],
        "disp": np.asarray(disp)[order],
        "unbound": np.asarray(unbound)[order],
    }


def resolve_batch_csv(repo: Path, label: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return (repo / explicit).resolve()
    matches = sorted(
        (repo / "sobol_mass_runs").glob(f"sobol_*_{label}/sobol_mass_outputs.csv")
    )
    if not matches:
        raise SystemExit(f"No batch found for label {label!r}")
    return matches[-1].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Batch CSV (default: latest noearth_obj_ctrl_spin_1p5_2hr)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/noearth_obj_ctrl_spin_1p5_2hr.png"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = resolve_batch_csv(repo, "noearth_obj_ctrl_spin_1p5_2hr", args.csv)
    out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = load_batch(csv_path)
    spin = data["spin"]
    disp = data["disp"]
    unbound = data["unbound"]

    fig, (ax_disp, ax_unb) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )

    ax_disp.plot(spin, disp, "o-", color="#2ca02c", linewidth=2, markersize=8, zorder=2)
    outlier = disp > 2.0
    if np.any(outlier):
        ax_disp.scatter(
            spin[outlier],
            disp[outlier],
            s=120,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.5,
            zorder=3,
            label="Substantial breakup",
        )
    ax_disp.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_disp.set_ylabel("Peak size ratio")
    ax_disp.set_yscale("log")
    ax_disp.set_ylim(0.98, max(disp) * 1.4)
    ax_disp.grid(True, which="both", alpha=0.3)
    ax_disp.legend(loc="upper right", framealpha=0.95)

    ax_unb.plot(spin, unbound * 100, "s-", color="#9467bd", linewidth=2, markersize=7)
    if np.any(outlier):
        ax_unb.scatter(
            spin[outlier],
            unbound[outlier] * 100,
            s=100,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=2.5,
            zorder=3,
        )
    ax_unb.set_xlabel("Spin period (hours)")
    ax_unb.set_ylabel("Peak unbound fraction (%)")
    ax_unb.grid(True, alpha=0.3)
    ax_unb.set_ylim(bottom=-0.5)

    fig.suptitle(
        "Priority 2: matched OBJ no-Earth control\n"
        r"$n_p = 1000$, $k_c = 10^7$ dyne/cm, spin axis ~177° (obl/az), $t_\mathrm{max} = 4.5$ d",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
