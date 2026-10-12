#!/usr/bin/env python3
"""Ablation of the np10k rp 8000 km breakup: baseline, realistic and the three one-change runs.

Runs (latest batch matching each pattern under sobol_mass_runs/, or pass --runs label=dir ...):
  baseline   frictionless, frictionless settle, no boulders   (run_breakup_np10k_rp8000.py)
  A          friction, frictional settle, no boulders          (run_ablation_np10k_rp8000.py A)
  B          frictionless, frictionless settle, boulders       (... B)
  C          friction, frictionless settle, no boulders        (... C)
  realistic  friction, frictional settle, boulders             (run_breakup_np10k_fric_boulders.py)
Panel 1: clump count vs time from pericentre (plot_breakup_clumps.clump_frame, same FoF rule).
Panel 2: largest-clump mass fraction.
Summary per run: breakup onset (first dump after pericentre with >1 clump), final clump count and
mass fractions, and for boulder runs bonds kept and boulders intact (plot_breakup_compare.bond_frame).
Writes sobol_mass_runs/plots/breakup_np10k_rp8000_ablation.{png,csv} and _summary.csv. Needs kept dumps.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import plot_breakup_clumps as pbc  # noqa: E402
import plot_breakup_compare as pbcmp  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# label: (batch glob, legend, colour, has boulders)
RUNS = {
    "baseline": ("sobol_*_breakup_np10k_rp8000", "Baseline: no friction, no boulders", pbc.INK_2, False),
    "A": ("sobol_*_breakup_np10k_ablA_fric_rp8000", "A: friction, loose pack", pbc.BANDS[0][1], False),
    "C": ("sobol_*_breakup_np10k_ablC_fric_densepack_rp8000", "C: friction, dense pack", pbc.BANDS[2][1],
          False),
    "B": ("sobol_*_breakup_np10k_ablB_boulders_rp8000", "B: boulders, no friction", pbc.BANDS[1][1], True),
    "realistic": ("sobol_*_breakup_np10k_fric_boulders_rp8000", "Realistic: friction + boulders",
                  "#7c4dbd", True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], metavar="LABEL=DIR",
                    help=f"override a run dir; labels: {', '.join(RUNS)}")
    ap.add_argument("--earth-sink-row", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=pbc.REPO / "sobol_mass_runs" / "plots")
    args = ap.parse_args()
    override = dict(kv.split("=", 1) for kv in args.runs)

    data = {}
    for label, (pattern, *_rest) in RUNS.items():
        run_dir = Path(override[label]).resolve() if label in override else pbcmp.latest(pattern).resolve()
        th, rows = pbcmp.series(run_dir, args.earth_sink_row, bonds=RUNS[label][3])
        data[label] = (run_dir, th, rows)

    out = args.out_dir / "breakup_np10k_rp8000_ablation"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "t_from_peri_hr", "n_grains", "n_clumps", "f_largest", "f_second", "f_other", "f_loose"])
        for label, (_, th, rows) in data.items():
            for t, r in zip(th, rows):
                w.writerow([label, f"{t:.3f}", r["n_grains"], r["n_clumps"], f"{r['f_largest']:.4f}",
                            f"{r['f_second']:.4f}", f"{r['f_other']:.4f}", f"{r['f_loose']:.4f}"])
    print(f"[INFO] wrote {out}.csv")

    with open(f"{out}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "batch", "n_grains", "onset_hr", "final_clumps", "f_largest", "f_second", "f_other",
                    "f_loose", "bond_frac", "lobes_intact", "random_intact"])
        for label, (run_dir, th, rows) in data.items():
            after = [(t, r) for t, r in zip(th, rows) if t >= 0 and r["n_clumps"] > 1]
            onset = f"{after[0][0]:.2f}" if after else ""
            last = rows[-1]
            bond = lobes = rand = ""
            if RUNS[label][3]:
                bond = f"{last['bond_ends'] / pbcmp.first_bonds(rows):.4f}"
                lobes = f"{last['lobes_intact']}/{last['n_lobes']}"
                rand = f"{last['random_intact']}/{last['n_random']}"
            w.writerow([label, run_dir.parent.name, last["n_grains"], onset, last["n_clumps"],
                        f"{last['f_largest']:.4f}", f"{last['f_second']:.4f}", f"{last['f_other']:.4f}",
                        f"{last['f_loose']:.4f}", bond, lobes, rand])
    print(f"[INFO] wrote {out}_summary.csv")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": pbc.INK_2, "axes.labelcolor": pbc.INK,
                         "xtick.color": pbc.INK_2, "ytick.color": pbc.INK_2, "text.color": pbc.INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
                                   facecolor=pbc.SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(pbc.SURFACE)
        ax.grid(axis="y", color=pbc.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color=pbc.INK_2, linewidth=1, linestyle=(0, (4, 3)))

    for label, (_, th, rows) in data.items():
        _, legend, colour, _ = RUNS[label]
        style = (0, (5, 3)) if label == "baseline" else "-"
        ax1.step(th, [r["n_clumps"] for r in rows], where="mid", color=colour, linewidth=1.8, linestyle=style,
                 label=legend)
        ax2.plot(th, [r["f_largest"] for r in rows], color=colour, linewidth=1.8, linestyle=style)
    ax1.set_ylabel(f"Clumps (≥{pbc.MIN_CLUMP} grains)")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_ylim(0, None)
    # panel 1 is crowded at the top; panel 2 is empty bottom-left (all runs whole before pericentre)
    ax2.legend(*ax1.get_legend_handles_labels(), loc="lower left", fontsize=8, frameon=True,
               facecolor=pbc.SURFACE, edgecolor=pbc.GRID)
    ax1.set_title("Tidal breakup at rp 8,000 km: one change at a time", color=pbc.INK, fontsize=11, loc="left")
    ax2.set_ylabel("Largest clump mass fraction")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Time from pericentre (hr)")
    t_all = np.concatenate([th for _, th, _ in data.values()])
    ax2.set_xlim(t_all.min(), t_all.max())

    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor=pbc.SURFACE)
    print(f"[INFO] wrote {out}.png")


if __name__ == "__main__":
    main()
