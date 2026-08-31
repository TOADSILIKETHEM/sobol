#!/usr/bin/env python3
"""Plot np_apophis resolution study (v2 campaign + legacy v1 reference)."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
from csv_columns import kt_cgs_from_row

import matplotlib.pyplot as plt
import numpy as np

# --- v2 batch suffixes (latest timestamp wins per suffix) ---
V2_EARTH_OPP_DENSE = "np_sens_v2_obj_earth_opposite_dense"
V2_EARTH_OPP_SIGMAC = "np_sens_v2_obj_earth_opposite_sigmac"
V2_NOEARTH_SIGMAC = "np_sens_v2_obj_noearth_p155_sigmac"
V2_NOEARTH_DENSE = "np_sens_v2_obj_noearth_p155_dense"
V2_SPHERE_P155 = "np_sens_v2_sphere_noearth_p155"
V2_EARTH_ALIGNED = "np_sens_v2_obj_earth_aligned"
V2_EARTH_OPP_NP2000 = "np_sens_v2_obj_earth_opposite_np2000"

# Legacy v1 (optional reference)
V1_SPHERE_P200 = "np_sens_sphere_noearth_p200"


@dataclass(frozen=True)
class BatchData:
    label: str
    path: Path
    np: np.ndarray
    disp: np.ndarray
    unbound: np.ndarray
    intrinsic: np.ndarray
    post_flyby: np.ndarray
    kc: Optional[np.ndarray] = None


def latest_outputs_csv(repo: Path, batch_suffix: str) -> Path:
    matches = sorted((repo / "sobol_mass_runs").glob(f"sobol_*_{batch_suffix}/sobol_mass_outputs.csv"))
    if not matches:
        raise SystemExit(f"No batch found for suffix: {batch_suffix}")
    return matches[-1].resolve()


def load_batch(csv_path: Path, label: str = "") -> BatchData:
    np_vals: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    intrinsic: list[float] = []
    post_flyby: list[float] = []
    kc: list[float] = []
    has_kc = False
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            np_vals.append(float(row["np_apophis"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row["unbound_fraction"] or 0))
            intr = row.get("intrinsic_spin_period_hr", "").strip()
            intrinsic.append(float(intr) if intr else np.nan)
            post = row.get("post_flyby_spin_period_hr", "").strip()
            post_flyby.append(float(post) if post else np.nan)
            kc_val = kt_cgs_from_row(row)
            if kc_val is not None:
                has_kc = True
                kc.append(kc_val)
    order = np.argsort(np_vals)
    return BatchData(
        label=label or csv_path.parent.name,
        path=csv_path,
        np=np.asarray(np_vals)[order],
        disp=np.asarray(disp)[order],
        unbound=np.asarray(unbound)[order],
        intrinsic=np.asarray(intrinsic)[order],
        post_flyby=np.asarray(post_flyby)[order],
        kc=np.asarray(kc)[order] if has_kc else None,
    )


def _mark_np500(ax, np_vals: np.ndarray, y: np.ndarray, *, color: str = "#d62728") -> None:
    idx = np.where(np_vals == 500)[0]
    if idx.size:
        ax.scatter(np_vals[idx], y[idx], s=130, facecolors="none", edgecolors=color, linewidths=2.2, zorder=4)


def _style_np_axis(ax) -> None:
    ax.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.75, label=r"Default $n_p=500$")
    ax.grid(True, which="both", alpha=0.3)


def plot_dual_metric_ax(
    ax,
    data: BatchData,
    *,
    log_disp: bool = True,
    title: str = "",
    show_legend: bool = True,
) -> None:
    unb_pct = data.unbound * 100
    ax.plot(data.np, data.disp, "o-", color="#2ca02c", linewidth=2, markersize=7, label="Dispersion ratio", zorder=2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    _mark_np500(ax, data.np, data.disp)
    if log_disp:
        ax.set_yscale("log")
        lo = min(0.95, float(np.min(data.disp)) * 0.8)
        ax.set_ylim(lo, float(np.max(data.disp)) * 1.6)
    else:
        ax.set_ylim(0.998, max(1.01, float(np.max(data.disp)) * 1.05))
    ax.set_ylabel("Peak dispersion ratio")
    _style_np_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(data.np, unb_pct, "s--", color="#9467bd", linewidth=1.5, markersize=6, label="Unbound %")
    _mark_np500(ax2, data.np, unb_pct)
    ax2.set_ylabel("Peak unbound (%)")
    ax2.set_ylim(0, max(8.0, float(np.max(unb_pct)) * 1.2))

    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if show_legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.95)


def plot_breakup_panels_v2(repo: Path, out: Path) -> None:
    """Replace legacy 3-panel figure with v2 primary regimes."""
    panels = [
        (V2_EARTH_OPP_DENSE, "Earth flyby, torque-align ~177.5°", r"$P=1.52$ hr, fixed $k_c=10^7$ (17 $n_p$)"),
        (V2_NOEARTH_DENSE, "OBJ, no Earth", r"$P=1.55$ hr, fixed $k_c=10^7$"),
        (V2_SPHERE_P155, "Sphere, no Earth", r"$P=1.55$ hr null (marginal spin)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    for ax, (suffix, head, sub) in zip(axes, panels, strict=True):
        data = load_batch(latest_outputs_csv(repo, suffix), suffix)
        log_disp = suffix != V2_SPHERE_P155 or float(np.max(data.disp)) > 2
        plot_dual_metric_ax(ax, data, log_disp=log_disp, title=f"{head}\n{sub}")
        ax.set_xticks(data.np)
        ax.set_xticklabels([str(int(v)) for v in data.np], fontsize=8)
    axes[-1].set_xlabel(r"$n_p$ (Apophis grains)")
    fig.suptitle(
        r"np sensitivity v2: breakup metrics vs grain count ($t_\mathrm{max}=4.5$ d)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_combined_overview_v2(repo: Path, out: Path) -> None:
    """Replace legacy combined figure — all v2 breakup arms on shared axes."""
    specs = [
        (V2_EARTH_OPP_DENSE, "Earth opposite (dense, fixed $k_c$)", "#ff7f0e", "s"),
        (V2_EARTH_OPP_SIGMAC, "Earth opposite ($\\sigma_c$ const.)", "#ff7f0e", "o"),
        (V2_NOEARTH_DENSE, "OBJ no-Earth (fixed $k_c$)", "#1f77b4", "s"),
        (V2_NOEARTH_SIGMAC, "OBJ no-Earth ($\\sigma_c$ const.)", "#1f77b4", "o"),
        (V2_EARTH_ALIGNED, "Earth near-aligned (~11°)", "#9467bd", "^"),
        (V2_SPHERE_P155, "Sphere no-Earth $P=1.55$ hr", "#2ca02c", "D"),
    ]
    fig, (ax_d, ax_u) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for suffix, label, color, mk in specs:
        data = load_batch(latest_outputs_csv(repo, suffix), suffix)
        ls = "-" if "fixed" in label or "dense" in label else "--"
        ax_d.plot(data.np, data.disp, f"{mk}{ls}", color=color, linewidth=1.8, markersize=6, label=label, alpha=0.9)
        ax_u.plot(data.np, data.unbound * 100, f"{mk}{ls}", color=color, linewidth=1.4, markersize=5, alpha=0.9)

    # np=2000 anchor
    d2k = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_NP2000), V2_EARTH_OPP_NP2000)
    ax_d.scatter(d2k.np, d2k.disp, s=160, marker="*", color="#d62728", zorder=5, label=r"$n_p=2000$ (Earth opposite)")
    ax_u.scatter(d2k.np, d2k.unbound * 100, s=120, marker="*", color="#d62728", zorder=5)

    ax_d.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_d.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Peak dispersion ratio")
    ax_d.set_ylim(0.9, 400)
    ax_d.legend(loc="upper right", fontsize=7, framealpha=0.95, ncol=2)
    ax_d.grid(True, which="both", alpha=0.3)

    ax_u.set_xlabel(r"$n_p$ (Apophis grains)")
    ax_u.set_ylabel("Peak unbound (%)")
    ax_u.grid(True, alpha=0.3)

    fig.suptitle("v2 combined: no single $n_p$ converges; geometry and $k_c$ prescription matter", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_intrinsic_spin_v2(repo: Path, out: Path) -> None:
    """Replace legacy intrinsic-spin figure for Earth flyby arms."""
    specs = [
        (V2_EARTH_OPP_DENSE, 1.52, "Earth opposite (dense)", "#ff7f0e"),
        (V2_EARTH_ALIGNED, 1.52, "Earth aligned (~11°)", "#9467bd"),
        (V2_NOEARTH_DENSE, 1.55, "OBJ no-Earth", "#1f77b4"),
        (V2_SPHERE_P155, 1.55, "Sphere no-Earth", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for suffix, nom, label, color in specs:
        data = load_batch(latest_outputs_csv(repo, suffix), suffix)
        mask = np.isfinite(data.intrinsic)
        drift = (data.intrinsic[mask] - nom) / nom * 100
        ax.plot(data.np[mask], drift, "o-", color=color, linewidth=2, markersize=6, label=f"{label} (nom. {nom:.2f} hr)")

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$n_p$ (Apophis grains)")
    ax.set_ylabel("Intrinsic spin drift vs input (%)")
    ax.set_title(
        "Positive drift = slower spin (longer period) vs setup input",
        fontsize=9,
        loc="left",
        pad=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    fig.suptitle("Settled (pre-flyby) spin vs $n_p$ — DEM rearrangement, not tides", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_earth_opposite_dense_detail(repo: Path, out: Path) -> None:
    """New: full 17-point dense scan + np=2000 extension."""
    data = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_DENSE), "dense")
    d2k = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_NP2000), "np2000")

    fig, (ax_d, ax_u) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax_d.plot(data.np, data.disp, "o-", color="#ff7f0e", linewidth=2, markersize=7, label="Dense scan")
    ax_d.scatter(d2k.np, d2k.disp, s=200, marker="*", color="#d62728", zorder=5, label=r"$n_p=2000$")
    _mark_np500(ax_d, data.np, data.disp)
    ax_d.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Peak dispersion ratio")
    ax_d.set_title(r"Earth opposite flyby — dense $n_p$ grid (fixed $k_c=10^7$, $P=1.52$ hr)", loc="left")
    ax_d.legend(loc="upper right")
    ax_d.grid(True, which="both", alpha=0.3)

    ax_u.plot(data.np, data.unbound * 100, "s-", color="#9467bd", linewidth=2, markersize=6)
    ax_u.scatter(d2k.np, d2k.unbound * 100, s=160, marker="*", color="#d62728", zorder=5)
    _mark_np500(ax_u, data.np, data.unbound * 100)
    ax_u.set_xlabel(r"$n_p$ (Apophis grains)")
    ax_u.set_ylabel("Peak unbound (%)")
    ax_u.set_title("High unbound at ~425–650 despite low dispersion — mass-loss without geometric spreading", loc="left", fontsize=9)
    ax_u.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_kc_vs_sigmac(repo: Path, out: Path, *, earth: bool) -> None:
    """New: fixed $k_c$ vs $\\sigma_c$-constant at matched coarse $n_p$."""
    if earth:
        fixed = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_DENSE), "fixed")
        scaled = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_SIGMAC), "sigmac")
        title = r"Earth opposite: fixed $k_c=10^7$ vs $\sigma_c$-constant scaling"
        subtitle = r"$k_c(n_p)=10^7(500/n_p)^{1/3}$; torque-align $\approx 177.5°$, $P=1.52$ hr"
    else:
        fixed = load_batch(latest_outputs_csv(repo, V2_NOEARTH_DENSE), "fixed")
        scaled = load_batch(latest_outputs_csv(repo, V2_NOEARTH_SIGMAC), "sigmac")
        title = r"OBJ no-Earth: fixed $k_c$ vs $\sigma_c$-constant scaling"
        subtitle = r"$P=1.55$ hr, apophis_only"

    # Use only shared np on both arms for fair comparison.
    shared = sorted(set(fixed.np.astype(int)) & set(scaled.np.astype(int)))
    fi = {int(n): (d, u) for n, d, u in zip(fixed.np, fixed.disp, fixed.unbound)}
    si = {int(n): (d, u) for n, d, u in zip(scaled.np, scaled.disp, scaled.unbound)}
    np_shared = np.asarray(shared)

    fig, (ax_d, ax_u) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fd = np.asarray([fi[n][0] for n in shared])
    fu = np.asarray([fi[n][1] for n in shared]) * 100
    sd = np.asarray([si[n][0] for n in shared])
    su = np.asarray([si[n][1] for n in shared]) * 100

    ax_d.plot(np_shared, fd, "s-", color="#1f77b4", linewidth=2, markersize=8, label=r"Fixed $k_c=10^7$")
    ax_d.plot(np_shared, sd, "o--", color="#d62728", linewidth=2, markersize=8, label=r"$\sigma_c$ constant ($k_c \propto n_p^{-1/3}$)")
    ax_d.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax_d.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Peak dispersion ratio")
    ax_d.legend(loc="best")
    ax_d.grid(True, which="both", alpha=0.3)

    ax_u.plot(np_shared, fu, "s-", color="#1f77b4", linewidth=2, markersize=8)
    ax_u.plot(np_shared, su, "o--", color="#d62728", linewidth=2, markersize=8)
    ax_u.set_xlabel(r"$n_p$ (Apophis grains)")
    ax_u.set_ylabel("Peak unbound (%)")
    ax_u.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_spin_axis_comparison(repo: Path, out: Path) -> None:
    """New: near-aligned vs opposite at matched $n_p$."""
    aligned = load_batch(latest_outputs_csv(repo, V2_EARTH_ALIGNED), "aligned")
    dense = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_DENSE), "opposite")
    shared = sorted(set(int(n) for n in dense.np) & set(int(n) for n in aligned.np))
    if not shared:
        raise SystemExit("No shared np between opposite dense and aligned batches")
    oi = {int(n): (d, u) for n, d, u in zip(dense.np, dense.disp, dense.unbound)}
    ai = {int(n): (d, u) for n, d, u in zip(aligned.np, aligned.disp, aligned.unbound)}
    np_s = np.asarray(shared)
    od = np.asarray([oi[n][0] for n in shared])
    ou = np.asarray([oi[n][1] for n in shared]) * 100
    ad = np.asarray([ai[n][0] for n in shared])
    au = np.asarray([ai[n][1] for n in shared]) * 100

    x = np.arange(len(shared))
    w = 0.35
    fig, (ax_d, ax_u) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_d.bar(x - w / 2, od, width=w, color="#ff7f0e", label="Opposite (~177.5°)")
    ax_d.bar(x + w / 2, ad, width=w, color="#9467bd", label="Near-aligned (~11°)")
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Peak dispersion ratio")
    ax_d.set_xticks(x, [str(n) for n in shared])
    ax_d.set_xlabel(r"$n_p$")
    ax_d.legend()
    ax_d.grid(True, axis="y", alpha=0.3)

    ax_u.bar(x - w / 2, ou, width=w, color="#ff7f0e", label="Opposite")
    ax_u.bar(x + w / 2, au, width=w, color="#9467bd", label="Near-aligned")
    ax_u.set_ylabel("Peak unbound (%)")
    ax_u.set_xticks(x, [str(n) for n in shared])
    ax_u.set_xlabel(r"$n_p$")
    ax_u.legend()
    ax_u.grid(True, axis="y", alpha=0.3)

    fig.suptitle(r"Spin-axis orientation vs $n_p$ at $P=1.52$ hr (OBJ, Earth, $k_c=10^7$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_sphere_marginal_null(repo: Path, out: Path) -> None:
    """New: sphere at $P=1.55$ hr vs legacy $P=2.0$ hr null."""
    p155 = load_batch(latest_outputs_csv(repo, V2_SPHERE_P155), "P1.55")
    try:
        p200 = load_batch(latest_outputs_csv(repo, V1_SPHERE_P200), "P2.0")
        has_v1 = True
    except SystemExit:
        has_v1 = False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(p155.np, p155.disp, "o-", color="#d62728", linewidth=2, markersize=8, label=r"Sphere $P=1.55$ hr (v2)")
    if has_v1:
        ax.plot(p200.np, p200.disp, "^--", color="#2ca02c", linewidth=1.8, markersize=7, label=r"Sphere $P=2.0$ hr (v1 null)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$n_p$ (Apophis grains)")
    ax.set_ylabel("Peak dispersion ratio")
    ax.set_title(r"Sphere no-Earth: $P=1.55$ hr is not resolution-independent ($n_p=300$ disrupts)", loc="left")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_tidal_spin_earth(repo: Path, out: Path) -> None:
    """New: post-flyby spin change vs $n_p$ for Earth opposite dense scan."""
    data = load_batch(latest_outputs_csv(repo, V2_EARTH_OPP_DENSE), "dense")
    nom = 1.52
    mask = np.isfinite(data.post_flyby)
    pct = (data.post_flyby[mask] - nom) / nom * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data.np[mask], pct, "o-", color="#1f77b4", linewidth=2, markersize=7)
    _mark_np500(ax, data.np[mask], pct)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.axvline(500, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$n_p$ (Apophis grains)")
    ax.set_ylabel("Post-flyby spin period change vs input (%)")
    ax.set_title(r"Earth tidal spin-up vs $n_p$ (opposite alignment, dense scan)", loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("sobol_mass_runs/plots"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out_dir = (repo / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Replaced legacy v1 figures (same filenames, v2 content)
    plot_breakup_panels_v2(repo, out_dir / "np_sensitivity_breakup_panels.png")
    plot_combined_overview_v2(repo, out_dir / "np_sensitivity_combined.png")
    plot_intrinsic_spin_v2(repo, out_dir / "np_sensitivity_intrinsic_spin.png")

    # New v2 figures
    plot_earth_opposite_dense_detail(repo, out_dir / "np_sensitivity_v2_earth_opposite_dense.png")
    plot_kc_vs_sigmac(repo, out_dir / "np_sensitivity_v2_kc_vs_sigmac_earth.png", earth=True)
    plot_kc_vs_sigmac(repo, out_dir / "np_sensitivity_v2_kc_vs_sigmac_noearth.png", earth=False)
    plot_spin_axis_comparison(repo, out_dir / "np_sensitivity_v2_spin_axis_comparison.png")
    plot_sphere_marginal_null(repo, out_dir / "np_sensitivity_v2_sphere_marginal_null.png")
    plot_tidal_spin_earth(repo, out_dir / "np_sensitivity_v2_tidal_spin_vs_np.png")


if __name__ == "__main__":
    main()
