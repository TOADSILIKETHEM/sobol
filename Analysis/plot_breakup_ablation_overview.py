#!/usr/bin/env python3
"""One-figure overview of the np10k rp 8000 km breakup ablation, from plot_breakup_ablation.py's CSVs.

Left: largest-clump mass fraction vs time from pericentre, direct-labelled, onset marked.
Right: final mass split per run (largest / 2nd / other clumps / loose), sorted by largest.
Colour by what changed: boulder runs orange, friction-only runs blue, baseline grey.
Reads sobol_mass_runs/plots/breakup_np10k_rp8000_ablation{.csv,_summary.csv} (no dumps needed);
writes breakup_np10k_rp8000_ablation_overview.png beside them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import plot_breakup_clumps as pbc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# label: (short name, what changed, colour, line style)
RUNS = {
    "baseline": ("Baseline", "no friction, no boulders", "#8a8984", (0, (5, 3))),
    "A": ("A", "friction, loose pack", "#2a78d6", "-"),
    "C": ("C", "friction, dense pack", "#8fbdf0", "-"),
    "B": ("B", "boulders, no friction", "#eb6834", "-"),
    "realistic": ("Realistic", "friction + boulders", "#a3401a", "-"),
}
SEGMENTS = [("f_largest", "Largest", 1.0), ("f_second", "2nd largest", 0.55),
            ("f_other", "Other clumps", 0.28), ("f_loose", "Loose grains", None)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots-dir", type=Path, default=pbc.REPO / "sobol_mass_runs" / "plots")
    args = ap.parse_args()
    base = args.plots_dir / "breakup_np10k_rp8000_ablation"

    series = defaultdict(lambda: ([], []))
    with open(f"{base}.csv") as f:
        for r in csv.DictReader(f):
            series[r["run"]][0].append(float(r["t_from_peri_hr"]))
            series[r["run"]][1].append(float(r["f_largest"]))
    with open(f"{base}_summary.csv") as f:
        summary = {r["run"]: r for r in csv.DictReader(f)}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": pbc.INK_2, "axes.labelcolor": pbc.INK,
                         "xtick.color": pbc.INK_2, "ytick.color": pbc.INK_2, "text.color": pbc.INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=pbc.SURFACE,
                                   gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    for ax in (ax1, ax2):
        ax.set_facecolor(pbc.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # left: largest piece over time
    ax1.grid(axis="y", color=pbc.GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.axvline(0, color=pbc.INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax1.text(0.2, 0.04, "pericentre", color=pbc.INK_2, fontsize=8, rotation=90, va="bottom")
    t_end = max(max(t) for t, _ in series.values())
    ends = []
    for label, (name, _, colour, style) in RUNS.items():
        t, f = series[label]
        ax1.plot(t, f, color=colour, linewidth=2.2, linestyle=style)
        onset = summary[label]["onset_hr"]
        if onset:
            to = float(onset)
            fo = f[min(range(len(t)), key=lambda i: abs(t[i] - to))]
            ax1.plot(to, fo, "o", color=colour, markersize=5, zorder=3)
        ends.append([f[-1], name, colour])
    # direct labels at the right end, nudged apart so they don't overlap
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.055)
    for y, name, colour in ends:
        ax1.text(t_end + 0.4, y, f"{name}", color=colour, fontsize=9, fontweight="bold", va="center")
    ax1.set_xlim(-3, t_end)
    ax1.set_ylim(0, 1.03)
    ax1.set_xlabel("Time from pericentre (hr)")
    ax1.set_ylabel("Mass fraction in largest piece")
    ax1.set_title("Largest piece over time  (dot = breakup onset)", loc="left", fontsize=10.5)

    # right: final mass split, sorted by largest piece
    order = sorted(RUNS, key=lambda k: float(summary[k]["f_largest"]))
    for y, label in enumerate(order):
        name, change, colour, _ = RUNS[label]
        left = 0.0
        for key, _, alpha in SEGMENTS:
            w = max(float(summary[label][key]), 0.0)
            ax2.barh(y, w, left=left, height=0.62, color=colour if alpha else pbc.GRID,
                     alpha=alpha if alpha else 1.0, edgecolor=pbc.SURFACE, linewidth=0.8)
            left += w
        fl = float(summary[label]["f_largest"])
        ax2.text(fl - 0.02, y, f"{fl:.0%}", ha="right", va="center", fontsize=9, fontweight="bold",
                 color="white" if label in ("A", "B", "realistic") else pbc.INK)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{RUNS[k][0]}\n{RUNS[k][1]}" for k in order], fontsize=9)
    for tick, k in zip(ax2.get_yticklabels(), order):
        tick.set_color(RUNS[k][2] if k != "C" else "#4a90d9")
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("Share of body mass at end of run")
    ax2.set_title("Where the mass ends up", loc="left", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=pbc.INK_2 if a else pbc.GRID, alpha=a if a else 1.0)
               for _, _, a in SEGMENTS]
    ax2.legend(handles, [s[1] for s in SEGMENTS], loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=4,
               fontsize=8, frameon=False, handlelength=1.2, columnspacing=1.0)

    fig.suptitle("Bonded boulders keep ~70% of the body in one piece; friction alone gives 31–64%",
                 x=0.07, ha="left", fontsize=12, fontweight="bold")
    n = {k: int(summary[k]["n_grains"]) for k in RUNS}
    fig.text(0.07, -0.08,
             f"Tidal breakup, rp 8,000 km, one seed per run. Frictional settle packs looser: A and Realistic "
             f"have {n['A']:,} grains, the others {n['baseline']:,}. Run ends {t_end:.0f} hr after pericentre.",
             fontsize=8, color=pbc.INK_2)

    out = f"{base}_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=pbc.SURFACE)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
