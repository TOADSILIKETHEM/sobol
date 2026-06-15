#!/usr/bin/env python3
"""Plot Priority 2 no-Earth OBJ control: spin period vs breakup metrics."""

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
    run_id: list[int] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            run_id.append(int(row["run_id"]))
            spin.append(float(row["apophis_spin_period"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"]))
    order = np.argsort(spin)
    return {
        "run_id": np.asarray(run_id)[order],
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
            "sobol_mass_runs/sobol_20260614_171414_noearth_obj_ctrl_spin_1p5_2hr/"
            "sobol_mass_outputs.csv"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sobol_mass_runs/plots/noearth_obj_ctrl_spin_1p5_2hr.png"),
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
    ax_disp.set_ylabel("Peak dispersion ratio")
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
