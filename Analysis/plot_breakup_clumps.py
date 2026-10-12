#!/usr/bin/env python3
"""Clump count and clump mass fractions vs time for one particle-DEM breakup run.

Clumps = friends-of-friends groups of DEM grains (linking length = LINK_FACTOR x median
nearest-neighbour distance, per dump). A clump has >= MIN_CLUMP grains; smaller groups
count as loose grains. Time is relative to pericentre (minimum Earth-sink to grain-CoM
distance over the dumps). Needs kept dumps (--no-cleanup / keep_dumps).

Default run: latest sobol_mass_runs/sobol_*_breakup_np10k_rp8000/run_0001.
Writes sobol_mass_runs/plots/<batch>_clumps.{png,csv}.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_SOBOL_DIR = Path(__file__).resolve().parents[1]
if str(_SOBOL_DIR) not in sys.path:
    sys.path.insert(0, str(_SOBOL_DIR))
from particle_dem import IDEM, _read_dump, list_full_dumps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO = _SOBOL_DIR.parent
LINK_FACTOR = 1.5
MIN_CLUMP = 50
KM_PER_CM = 1e-5
R_EARTH_KM = 6371.0
G_CGS = 6.674e-8

# Reference categorical slots 1-4 (dataviz palette.md, light mode; adjacent pairs validated)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BANDS = [
    ("Largest clump", "#2a78d6"),
    ("2nd largest clump", "#eb6834"),
    (f"Other clumps (≥{MIN_CLUMP} grains)", "#1baf7a"),
    (f"Loose grains (groups <{MIN_CLUMP})", "#eda100"),
]


def default_run_dir() -> Path:
    hits = sorted((REPO / "sobol_mass_runs").glob("sobol_*_breakup_np10k_rp8000/run_0001"))
    if not hits:
        raise SystemExit("no sobol_*_breakup_np10k_rp8000/run_0001 found; pass --run-dir")
    return hits[-1]


def clump_frame(dump: Path, earth_row: int) -> dict:
    sdf, sinks = _read_dump(dump)
    keep = sdf["h"].to_numpy() > 0.0
    if "itype" in sdf.columns:
        keep &= sdf["itype"].to_numpy() == IDEM
    x = np.column_stack([sdf["x"], sdf["y"], sdf["z"]])[keep]
    m = (sdf["m"].to_numpy()[keep] if "m" in sdf.columns
         else np.full(len(x), float(sdf.params[f"massoftype_{IDEM}"])))
    com = (x * m[:, None]).sum(0) / m.sum()

    tree = cKDTree(x)
    nn, _ = tree.query(x, k=2)
    pairs = tree.query_pairs(LINK_FACTOR * np.median(nn[:, 1]), output_type="ndarray")
    n = len(x)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    group_mass = np.bincount(labels, weights=m)
    big = np.sort(group_mass[np.bincount(labels) >= MIN_CLUMP])[::-1] / m.sum()

    udist = float(sdf.params.get("udist", 1.0))
    earth = sinks.iloc[earth_row][["x", "y", "z"]].to_numpy(dtype=float)
    return {
        "t_s": float(sdf.params["time"]) * float(sdf.params.get("utime", 1.0)),
        "earth_dist_km": float(np.linalg.norm(com - earth)) * udist * KM_PER_CM,
        "n_grains": n,
        "n_clumps": int(len(big)),
        "f_largest": float(big[0]) if len(big) > 0 else 0.0,
        "f_second": float(big[1]) if len(big) > 1 else 0.0,
        "f_other": float(big[2:].sum()),
        "f_loose": float(1.0 - big.sum()),
    }


def pericentre_from_orbit(dump: Path, earth_row: int) -> tuple[float, float]:
    """(t_peri_s, rp_km) of the grain CoM about Earth, two-body hyperbola from one early dump.

    Dumps are dtmax apart, so the nearest dump can miss pericentre by thousands of km.
    """
    sdf, sinks = _read_dump(dump)
    keep = sdf["h"].to_numpy() > 0.0
    if "itype" in sdf.columns:
        keep &= sdf["itype"].to_numpy() == IDEM
    p = sdf.params
    udist, utime, umass = float(p["udist"]), float(p["utime"]), float(p["umass"])
    pos = np.column_stack([sdf["x"], sdf["y"], sdf["z"]])[keep].mean(0)  # equal-mass grains
    vel = np.column_stack([sdf["vx"], sdf["vy"], sdf["vz"]])[keep].mean(0)
    e_row = sinks.iloc[earth_row]
    r = (pos - e_row[["x", "y", "z"]].to_numpy(dtype=float)) * udist
    v = (vel - e_row[["vx", "vy", "vz"]].to_numpy(dtype=float)) * udist / utime
    mu = G_CGS * float(e_row["m"]) * umass
    rn = np.linalg.norm(r)
    a = 1.0 / (2.0 / rn - v @ v / mu)  # < 0 for a hyperbola
    if a >= 0:
        raise SystemExit("grain CoM is bound to Earth; not a hyperbolic flyby")
    e = np.linalg.norm(((v @ v - mu / rn) * r - (r @ v) * v) / mu)
    F = np.arccosh((1.0 - rn / a) / e)
    t_to_peri = (e * np.sinh(F) - F) / np.sqrt(mu / (-a) ** 3)
    if r @ v > 0:
        t_to_peri = -t_to_peri  # already past pericentre
    t0 = float(p["time"]) * utime
    return t0 + t_to_peri, a * (1.0 - e) * KM_PER_CM


def measure(run_dir: Path, earth_row: int) -> tuple[list[dict], float, float]:
    dumps = list_full_dumps(run_dir, "sobol")
    if not dumps:
        raise SystemExit(f"no sobol_* dumps in {run_dir} (run with keep_dumps / --no-cleanup)")
    t_peri_s, rp_km = pericentre_from_orbit(dumps[0], earth_row)
    rows = []
    for d in dumps:
        rows.append(clump_frame(d, earth_row))
        print(f"{d.name}: {rows[-1]['n_clumps']} clumps", flush=True)
    return rows, t_peri_s, rp_km


def plot(rows: list[dict], t_peri_s: float, rp_km: float, out_png: Path, title: str) -> None:
    th = (np.array([r["t_s"] for r in rows]) - t_peri_s) / 3600.0
    nclump = np.array([r["n_clumps"] for r in rows])
    stack = np.array([[r["f_largest"], r["f_second"], r["f_other"], r["f_loose"]] for r in rows]).T

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.12},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))

    ax1.step(th, nclump, where="mid", color=BANDS[0][1], linewidth=2)
    ax1.set_ylabel(f"Clumps (≥{MIN_CLUMP} grains)")
    ax1.set_ylim(0, max(nclump.max() + 1, 2))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # left of the pericentre line: before pericentre the count is 1, so the space above is free
    ax1.text(-0.3, 0.92 * ax1.get_ylim()[1],
             f"pericentre\n{rp_km:,.0f} km, {rp_km / R_EARTH_KM:.2f} R$_\\oplus$",
             color=INK_2, va="top", ha="right", fontsize=9)
    ax1.set_title(title, color=INK, fontsize=11, loc="left")

    polys = ax2.stackplot(th, stack, colors=[c for _, c in BANDS], labels=[n for n, _ in BANDS],
                          edgecolor=SURFACE, linewidth=1.0)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(th[0], th[-1])
    ax2.set_ylabel("Fraction of grain mass")
    ax2.set_xlabel("Time from pericentre (hr)")
    # direct labels at the right edge where a band is thick enough to hold text
    cum = np.cumsum(stack[:, -1])
    for i, (name, _) in enumerate(BANDS):
        lo = cum[i - 1] if i else 0.0
        if stack[i, -1] >= 0.06:
            ax2.text(th[-1] - 0.3, (lo + cum[i]) / 2, f"{name.split(' (')[0]}  {stack[i, -1]:.0%}",
                     ha="right", va="center", fontsize=8.5, color=INK,
                     bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    ax2.legend(handles=polys, loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE,
               edgecolor=GRID)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[INFO] wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--earth-sink-row", type=int, default=0,
                    help="0-based row of Earth in the dump's sink table (hyperbola: 0, the only sink)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "sobol_mass_runs" / "plots")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dir = (args.run_dir or default_run_dir()).resolve()
    batch = run_dir.parent.name
    rows, t_peri_s, rp_km = measure(run_dir, args.earth_sink_row)
    print(f"[INFO] pericentre (two-body, from first dump): t = {t_peri_s / 3600:.3f} hr, rp = {rp_km:,.0f} km")

    out_csv = args.out_dir / f"{batch}_clumps.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] wrote {out_csv}")

    n = rows[0]["n_grains"]
    title = args.title or f"Tidal breakup, {n:,} grains: clumps after pericentre"
    plot(rows, t_peri_s, rp_km, args.out_dir / f"{batch}_clumps.png", title)


if __name__ == "__main__":
    main()
