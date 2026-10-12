#!/usr/bin/env python3
"""Two np10k breakup runs side by side, plus boulder survival for the bonded one.

Panel 1: clump count vs time from pericentre (plot_breakup_clumps.clump_frame, same FoF rule) for a
baseline run (frictionless, no boulders) and a bonded run (friction + phantomclumps boulders).
Panel 2 (bonded run only): fraction of the first dump's bonds still present, and fraction of boulders
still in one bonded piece (connected components of the bond graph within each iclump), lobes
(iclump 1, 2 = phantomclumps contact-binary cores) and random boulders separately.

Defaults: latest sobol_*_breakup_np10k_rp8000 vs latest sobol_*_breakup_np10k_fric_boulders_rp8000.
Writes sobol_mass_runs/plots/<bonded batch>_compare.{png,csv}. Needs kept dumps.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import plot_breakup_clumps as pbc  # noqa: E402
from particle_dem import _read_dump, list_full_dumps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

N_LOBES = 2  # phantomclumps iplaced = 1 makes the two lobes first: iclump 1 and 2


def latest(pattern: str) -> Path:
    hits = sorted((pbc.REPO / "sobol_mass_runs").glob(f"{pattern}/run_0001"))
    if not hits:
        raise SystemExit(f"no {pattern}/run_0001 found; pass the run dir")
    return hits[-1]


def bond_frame(dump: Path) -> dict:
    """Bond ends present, and boulders (iclump > 0) still in one bond-connected piece."""
    sdf, _ = _read_dump(dump)
    clump = sdf["iclump"].to_numpy()
    iorig = sdf["iorig"].to_numpy().astype(np.int64)
    partners = sdf[[c for c in sdf.columns if c.startswith("ibond")]].to_numpy().astype(np.int64)
    idx = np.full(iorig.max() + 1, -1)
    idx[iorig] = np.arange(len(iorig))
    i, k = np.nonzero(partners > 0)
    j = idx[partners[i, k]]
    ok = j >= 0
    i, j = i[ok], j[ok]
    n = len(iorig)
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    intact = {}
    for c in np.unique(clump[clump > 0]):
        intact[int(c)] = len(np.unique(lab[clump == c])) == 1
    lobes = [intact[c] for c in range(1, N_LOBES + 1) if c in intact]
    rand = [v for c, v in intact.items() if c > N_LOBES]
    return {"bond_ends": int((partners > 0).sum()), "lobes_intact": int(sum(lobes)), "n_lobes": len(lobes),
            "random_intact": int(sum(rand)), "n_random": len(rand)}


def first_bonds(rows: list[dict]) -> int:
    """Bond ends in the first dump that has any (dump 0 predates bond formation)."""
    return next(r["bond_ends"] for r in rows if r["bond_ends"] > 0)


def series(run_dir: Path, earth_row: int, bonds: bool) -> tuple[np.ndarray, list[dict]]:
    dumps = list_full_dumps(run_dir, "sobol")
    if not dumps:
        raise SystemExit(f"no sobol_* dumps in {run_dir}")
    t_peri, _ = pbc.pericentre_from_orbit(dumps[0], earth_row)
    rows = []
    for d in dumps:
        r = pbc.clump_frame(d, earth_row)
        if bonds:
            r.update(bond_frame(d))
        rows.append(r)
        print(f"{run_dir.parent.name} {d.name}: {r['n_clumps']} clumps", flush=True)
    th = (np.array([r["t_s"] for r in rows]) - t_peri) / 3600.0
    return th, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=None, help="run dir (default latest breakup_np10k_rp8000)")
    ap.add_argument("--bonded", type=Path, default=None,
                    help="run dir (default latest breakup_np10k_fric_boulders_rp8000)")
    ap.add_argument("--earth-sink-row", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=pbc.REPO / "sobol_mass_runs" / "plots")
    args = ap.parse_args()

    base = (args.baseline or latest("sobol_*_breakup_np10k_rp8000")).resolve()
    bond = (args.bonded or latest("sobol_*_breakup_np10k_fric_boulders_rp8000")).resolve()
    tb, rb = series(base, args.earth_sink_row, bonds=False)
    tf, rf = series(bond, args.earth_sink_row, bonds=True)

    out = args.out_dir / f"{bond.parent.name}_compare"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "t_from_peri_hr", "n_clumps", "f_largest", "bond_frac", "lobes_intact",
                    "random_intact_frac"])
        for t, r in zip(tb, rb):
            w.writerow(["baseline", f"{t:.3f}", r["n_clumps"], f"{r['f_largest']:.4f}", "", "", ""])
        b0 = first_bonds(rf)
        for t, r in zip(tf, rf):
            if r["bond_ends"] == 0:  # dump 0: bonds form on the first force evaluation
                continue
            w.writerow(["bonded", f"{t:.3f}", r["n_clumps"], f"{r['f_largest']:.4f}",
                        f"{r['bond_ends'] / b0:.4f}", r["lobes_intact"],
                        f"{r['random_intact'] / max(r['n_random'], 1):.4f}"])
    print(f"[INFO] wrote {out}.csv")

    c_base, c_bond, c3 = pbc.BANDS[0][1], pbc.BANDS[1][1], pbc.BANDS[2][1]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": pbc.INK_2, "axes.labelcolor": pbc.INK,
                         "xtick.color": pbc.INK_2, "ytick.color": pbc.INK_2, "text.color": pbc.INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.12},
                                   facecolor=pbc.SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(pbc.SURFACE)
        ax.grid(axis="y", color=pbc.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color=pbc.INK_2, linewidth=1, linestyle=(0, (4, 3)))

    ax1.step(tb, [r["n_clumps"] for r in rb], where="mid", color=c_base, linewidth=2,
             label="Frictionless, no boulders")
    ax1.step(tf, [r["n_clumps"] for r in rf], where="mid", color=c_bond, linewidth=2,
             label="Friction + bonded boulders")
    ax1.set_ylabel(f"Clumps (≥{pbc.MIN_CLUMP} grains)")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_ylim(0, None)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=pbc.SURFACE, edgecolor=pbc.GRID)
    ax1.set_title("Tidal breakup at rp 8,000 km: frictionless vs friction + boulders", color=pbc.INK,
                  fontsize=11, loc="left")

    b0 = first_bonds(rf)
    keep = [i for i, r in enumerate(rf) if r["bond_ends"] > 0]
    tk = tf[keep]
    bond_frac = [rf[i]["bond_ends"] / b0 for i in keep]
    rand_frac = [rf[i]["random_intact"] / max(rf[i]["n_random"], 1) for i in keep]
    ax2.plot(tk, rand_frac, color=c3, linewidth=2, label="Random boulders in one piece")
    ax2.plot(tk, bond_frac, color=c_bond, linewidth=2, linestyle=(0, (5, 3)), label="Bonds remaining")
    lo = min(min(bond_frac), min(rand_frac))
    ax2.set_ylim(0 if lo < 0.9 else 0.9, 1.01)  # zoom when nearly everything survives
    ax2.set_ylabel("Fraction (bonded run)")
    ax2.set_xlabel("Time from pericentre (hr)")
    ax2.set_xlim(min(tb[0], tf[0]), max(tb[-1], tf[-1]))
    lob = rf[-1]
    ax2.text(0.99, 0.06, f"Lobe cores in one piece at end: {lob['lobes_intact']} of {lob['n_lobes']}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8.5, color=pbc.INK_2)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=pbc.SURFACE, edgecolor=pbc.GRID)

    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor=pbc.SURFACE)
    print(f"[INFO] wrote {out}.png")


if __name__ == "__main__":
    main()
