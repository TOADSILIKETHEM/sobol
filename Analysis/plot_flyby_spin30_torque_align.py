#!/usr/bin/env python3
"""Plot spin30 torque-align campaign: orientation vs tidal spin change."""

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
    align, intr, app, post, disp, unb = [], [], [], [], [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            a = _to_float(row["apophis_spin_torque_align_deg"])
            i = _to_float(row["intrinsic_spin_period_hr"])
            ap = _to_float(row["approach_spin_period_hr"])
            po = _to_float(row["post_flyby_spin_period_hr"])
            d = _to_float(row["dispersion_ratio"])
            u = _to_float(row["unbound_fraction"])
            if any(v is None for v in (a, i, ap, po, d, u)):
                continue
            align.append(a)
            intr.append(i)
            app.append(ap)
            post.append(po)
            disp.append(d)
            unb.append(u)
    order = np.argsort(align)
    return {
        k: np.asarray(v)[order]
        for k, v in {
            "align": align,
            "intr": intr,
            "app": app,
            "post": post,
            "disp": disp,
            "unb": unb,
        }.items()
    }


def resolve_batch_csv(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return (repo / explicit).resolve()
    matches = sorted(
        (repo / "sobol_mass_runs").glob(
            "sobol_*_flyby_spin30_torque_align_k1e7/sobol_mass_outputs.csv"
        )
    )
    if not matches:
        raise SystemExit(
            "No flyby_spin30_torque_align_k1e7 batch found under sobol_mass_runs/"
        )
    return matches[-1].resolve()


def _plot_sanity_panel(ax_sanity: plt.Axes, d: dict[str, np.ndarray]) -> None:
    ax_sanity.plot(d["align"], d["disp"], "o-", color="#2ca02c", linewidth=2, markersize=7, label="Size ratio")
    ax_sanity2 = ax_sanity.twinx()
    ax_sanity2.plot(d["align"], d["unb"] * 100, "s--", color="#9467bd", linewidth=1.5, markersize=6, label="Unbound %")
    ax_sanity.set_ylabel("Peak size ratio")
    ax_sanity2.set_ylabel("Peak unbound (%)")
    ax_sanity.set_xlabel("Torque-align angle (degrees)")
    ax_sanity.set_ylim(1.0, max(d["disp"]) * 1.05)
    unb_pct = d["unb"] * 100
    ub_ymax = max(5.0, float(np.max(unb_pct)) * 1.2)
    ax_sanity2.set_ylim(0, ub_ymax)
    ax_sanity.grid(True, alpha=0.3)
    lines1, labels1 = ax_sanity.get_legend_handles_labels()
    lines2, labels2 = ax_sanity2.get_legend_handles_labels()
    ax_sanity.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.95)


def plot_change(fig: plt.Figure, ax_spin: plt.Axes, d: dict[str, np.ndarray]) -> None:
    d_total = (d["post"] - d["intr"]) / d["intr"] * 100
    d_app = (d["app"] - d["intr"]) / d["intr"] * 100
    ax_spin.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_spin.plot(d["align"], d_total, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Post − intrinsic")
    ax_spin.plot(d["align"], d_app, "s--", color="#ff7f0e", linewidth=1.5, markersize=6, label="Approach − intrinsic")
    ax_spin.set_ylabel("Spin period change (%)")
    ax_spin.grid(True, alpha=0.3)
    ax_spin.legend(loc="upper right", framealpha=0.95)
    fig.suptitle(
        "Real Apophis spin (P = 30 hr): torque-align vs tidal spin change\n"
        r"Sphere DEM, $k_c = 10^7$ dyne/cm, $n_p = 500$, CA $\approx$ 38,000 km",
        fontsize=11,
    )


def plot_periods(fig: plt.Figure, ax_spin: plt.Axes, d: dict[str, np.ndarray]) -> None:
    ax_spin.axhline(30.0, color="gray", linestyle=":", linewidth=1, alpha=0.8, label="Input P = 30 hr")
    ax_spin.plot(d["align"], d["app"], "s--", color="#ff7f0e", linewidth=1.5, markersize=6, label="Approach spin period")
    ax_spin.plot(d["align"], d["post"], "o-", color="#1f77b4", linewidth=2, markersize=7, label="Post-flyby spin period")
    ymin = min(float(np.min(d["app"])), float(np.min(d["post"]))) - 0.15
    ymax = max(float(np.max(d["app"])), float(np.max(d["post"]))) + 0.15
    ax_spin.set_ylim(ymin, ymax)
    ax_spin.set_ylabel("Spin period (hr)")
    ax_spin.grid(True, alpha=0.3)
    ax_spin.legend(loc="upper right", framealpha=0.95)
    fig.suptitle(
        "Real Apophis spin (P = 30 hr): approach and post-flyby spin period\n"
        r"Sphere DEM, $k_c = 10^7$ dyne/cm, $n_p = 500$, CA $\approx$ 38,000 km",
        fontsize=11,
    )


def plot_periods_all(fig: plt.Figure, ax_spin: plt.Axes, d: dict[str, np.ndarray]) -> None:
    ax_spin.axhline(30.0, color="gray", linestyle=":", linewidth=1, alpha=0.8, label="Input P = 30 hr")
    ax_spin.plot(
        d["align"],
        d["intr"],
        "^-.",
        color="#2ca02c",
        linewidth=1.5,
        markersize=6,
        label="Intrinsic spin period",
    )
    ax_spin.plot(
        d["align"],
        d["app"],
        "s--",
        color="#ff7f0e",
        linewidth=1.5,
        markersize=6,
        label="Approach spin period",
    )
    ax_spin.plot(
        d["align"],
        d["post"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=7,
        label="Post-flyby spin period",
    )
    ymin = min(float(np.min(d["intr"])), float(np.min(d["app"])), float(np.min(d["post"]))) - 0.15
    ymax = max(float(np.max(d["intr"])), float(np.max(d["app"])), float(np.max(d["post"]))) + 0.15
    ax_spin.set_ylim(ymin, ymax)
    ax_spin.set_ylabel("Spin period (hr)")
    ax_spin.grid(True, alpha=0.3)
    ax_spin.legend(loc="upper center", framealpha=0.95, fontsize=9)
    fig.suptitle(
        "Real Apophis spin (P = 30 hr): intrinsic, approach, and post-flyby spin period\n"
        r"Sphere DEM, $k_c = 10^7$ dyne/cm, $n_p = 500$, CA $\approx$ 38,000 km",
        fontsize=11,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="sobol_mass_outputs.csv (default: latest flyby_spin30_torque_align_k1e7 batch)",
    )
    parser.add_argument(
        "--mode",
        choices=("change", "periods", "periods_all"),
        default="change",
        help="change: percent delta vs intrinsic; periods: approach/post hours; periods_all: all three windows",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: flyby_spin30_torque_align_k1e7.png or _periods.png)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_path = resolve_batch_csv(repo, args.csv)
    if args.output is None:
        stems = {
            "change": "flyby_spin30_torque_align_k1e7.png",
            "periods": "flyby_spin30_torque_align_k1e7_periods.png",
            "periods_all": "flyby_spin30_torque_align_k1e7_periods_all.png",
        }
        out = (repo / "sobol_mass_runs/plots" / stems[args.mode]).resolve()
    else:
        out = (repo / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    d = load_batch(csv_path)

    fig, (ax_spin, ax_sanity) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )

    if args.mode == "periods":
        plot_periods(fig, ax_spin, d)
    elif args.mode == "periods_all":
        plot_periods_all(fig, ax_spin, d)
    else:
        plot_change(fig, ax_spin, d)

    _plot_sanity_panel(ax_sanity, d)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
