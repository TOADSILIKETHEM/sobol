#!/usr/bin/env python3
"""Plots and classic-sensitivity bar charts for np × spin-period grid campaign (Jun 2026).

Batches:
  P1 — np_spin_p1_noearth_obj_kc       (OBJ no-Earth, fixed kc)
  P2 — np_spin_p2_noearth_obj_sigmac    (OBJ no-Earth, σ_c-constant)
  P3 — np_spin_p3_earth_opposite_kc     (OBJ Earth opposite ~177°)
"""

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
from scipy.interpolate import griddata

# Batch suffixes (latest timestamp wins)
P1_SUFFIX = "np_spin_p1_noearth_obj_kc"
P2_SUFFIX = "np_spin_p2_noearth_obj_sigmac"
P3_SUFFIX = "np_spin_p3_earth_opposite_kc"

NP_COLORS = {
    400: "#1f77b4",
    500: "#d62728",
    600: "#2ca02c",
    750: "#ff7f0e",
    1000: "#9467bd",
}

STABLE_SPIN_MIN = 1.85
PCRIT_THRESHOLDS = (1.5, 2.0)


@dataclass(frozen=True)
class GridBatch:
    label: str
    title: str
    path: Path
    np_vals: np.ndarray
    spin: np.ndarray
    disp: np.ndarray
    unbound: np.ndarray
    intrinsic: np.ndarray
    post_flyby: np.ndarray
    kc: Optional[np.ndarray] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def latest_csv(repo: Path, suffix: str) -> Path:
    matches = sorted((repo / "sobol_mass_runs").glob(f"sobol_*_{suffix}/sobol_mass_outputs.csv"))
    if not matches:
        raise SystemExit(f"No batch found for suffix: {suffix}")
    return matches[-1].resolve()


def load_batch(csv_path: Path, label: str, title: str) -> GridBatch:
    np_vals: list[float] = []
    spin: list[float] = []
    disp: list[float] = []
    unbound: list[float] = []
    intrinsic: list[float] = []
    post: list[float] = []
    kc: list[float] = []
    has_kc = False
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            np_vals.append(float(row["np_apophis"]))
            spin.append(float(row["apophis_spin_period"]))
            disp.append(float(row["dispersion_ratio"]))
            unbound.append(float(row.get("unbound_fraction") or 0))
            intr = (row.get("intrinsic_spin_period_hr") or "").strip()
            intrinsic.append(float(intr) if intr else np.nan)
            pf = (row.get("post_flyby_spin_period_hr") or "").strip()
            post.append(float(pf) if pf else np.nan)
            kc_val = kt_cgs_from_row(row)
            if kc_val is not None:
                has_kc = True
                kc.append(kc_val)
    return GridBatch(
        label=label,
        title=title,
        path=csv_path,
        np_vals=np.asarray(np_vals),
        spin=np.asarray(spin),
        disp=np.asarray(disp),
        unbound=np.asarray(unbound),
        intrinsic=np.asarray(intrinsic),
        post_flyby=np.asarray(post),
        kc=np.asarray(kc) if has_kc else None,
    )


def unique_sorted(vals: np.ndarray) -> np.ndarray:
    return np.sort(np.unique(vals))


def subset_mask(batch: GridBatch, np_val: float, tol: float = 0.5) -> np.ndarray:
    return np.isclose(batch.np_vals, np_val, rtol=0, atol=tol)


def curve_for_np(batch: GridBatch, np_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = subset_mask(batch, np_val)
    order = np.argsort(batch.spin[m])
    return (
        batch.spin[m][order],
        batch.disp[m][order],
        batch.unbound[m][order],
    )


def _interp_np_spin(
    batch: GridBatch,
    values: np.ndarray,
    n_spin: int = 80,
    n_np: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np_u = unique_sorted(batch.np_vals)
    sp_u = unique_sorted(batch.spin)
    grid_np = np.linspace(float(np_u.min()), float(np_u.max()), n_np)
    grid_spin = np.linspace(float(sp_u.min()), float(sp_u.max()), n_spin)
    xi, yi = np.meshgrid(grid_spin, grid_np)
    zi = griddata(
        (batch.spin, batch.np_vals),
        values,
        (xi, yi),
        method="linear",
    )
    return xi, yi, zi


def compute_pcrit(
    batch: GridBatch,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First spin period where disp exceeds threshold; nan if never crossed."""
    np_u = unique_sorted(batch.np_vals)
    pcrit: list[float] = []
    for n in np_u:
        spin, disp, _ = curve_for_np(batch, float(n))
        above = np.where(disp > threshold)[0]
        if above.size == 0:
            pcrit.append(float("nan"))
        else:
            pcrit.append(float(spin[int(above[0])]))
    return np_u, np.asarray(pcrit)


def plot_disp_curves_panels(
    batches: list[GridBatch],
    out: Path,
    *,
    log_y: bool = True,
) -> None:
    fig, axes = plt.subplots(1, len(batches), figsize=(5.2 * len(batches), 4.8), sharey=True)
    if len(batches) == 1:
        axes = [axes]
    for ax, batch in zip(axes, batches, strict=True):
        for n in unique_sorted(batch.np_vals):
            spin, disp, _ = curve_for_np(batch, float(n))
            c = NP_COLORS.get(int(n), "#333333")
            lw = 2.6 if int(n) == 500 else 1.8
            ax.plot(spin, disp, "o-", color=c, linewidth=lw, markersize=5, label=rf"$n_p={int(n)}$")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(1.5, color="#d62728", linestyle="--", linewidth=1, alpha=0.5)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(0.98, max(batch.disp.max() * 1.3, 2.0))
        ax.set_xlabel("Spin period (hours)")
        ax.set_title(batch.title, loc="left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axes[0].set_ylabel("Peak dispersion ratio")
    fig.suptitle(
        r"OBJ ~177°: disruption vs spin period at each $n_p$ ($k_c=10^7$, $t_\mathrm{max}=4.5$ d)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_heatmap(
    batch: GridBatch,
    out: Path,
    *,
    field: str = "disp",
) -> None:
    if field == "disp":
        values = np.log10(np.maximum(batch.disp, 1.0))
        cmap = "magma"
        label = r"$\log_{10}$(dispersion ratio)"
        title_extra = "peak dispersion ratio (log scale)"
    else:
        values = batch.unbound * 100
        cmap = "YlOrRd"
        label = "Peak unbound (%)"
        title_extra = "peak unbound mass fraction"

    xi, yi, zi = _interp_np_spin(batch, values)
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    im = ax.pcolormesh(xi, yi, zi, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        batch.spin,
        batch.np_vals,
        c=values,
        s=45,
        edgecolors="k",
        linewidths=0.5,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )
    del sc
    ax.set_xlabel("Spin period (hours)")
    ax.set_ylabel(r"$n_p$ (Apophis grains)")
    ax.set_title(f"{batch.label}: {title_extra}", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, label=label)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_pcrit(batch: GridBatch, out: Path, pcrit_csv: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    rows: list[list] = [["np_apophis", "P_crit_disp_1.5", "P_crit_disp_2.0"]]
    for thr, ls, marker in zip(PCRIT_THRESHOLDS, ("-", "--"), ("o", "s"), strict=True):
        np_u, pcrit = compute_pcrit(batch, thr)
        ax.plot(np_u, pcrit, f"{marker}{ls}", linewidth=2, markersize=7, label=rf"disp $>{thr}$")
        for n, p in zip(np_u, pcrit, strict=True):
            rows.append([int(n), f"{p:.4g}" if np.isfinite(p) else "", ""])
    # fix csv — write both thresholds properly
    rows = [["np_apophis", "P_crit_disp_1.5", "P_crit_disp_2.0"]]
    np_u = unique_sorted(batch.np_vals)
    p15 = compute_pcrit(batch, 1.5)[1]
    p20 = compute_pcrit(batch, 2.0)[1]
    for n, a, b in zip(np_u, p15, p20, strict=True):
        rows.append([
            int(n),
            f"{a:.4g}" if np.isfinite(a) else "",
            f"{b:.4g}" if np.isfinite(b) else "",
        ])
    with pcrit_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    for thr, ls, marker in zip(PCRIT_THRESHOLDS, ("-", "--"), ("o", "s"), strict=True):
        np_u, pcrit = compute_pcrit(batch, thr)
        ax.plot(np_u, pcrit, f"{marker}{ls}", linewidth=2, markersize=7, label=rf"disp $>{thr}$")
    ax.axvline(500, color="#888", linestyle=":", alpha=0.7, label=r"default $n_p=500$")
    ax.set_xlabel(r"$n_p$ (Apophis grains)")
    ax.set_ylabel(r"$P_\mathrm{crit}$ (hours)")
    ax.set_title(f"Spin-disruption threshold vs resolution — {batch.label}", loc="left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {pcrit_csv}")


def plot_stable_band(batch: GridBatch, out: Path) -> None:
    m = batch.spin >= STABLE_SPIN_MIN
    np_u = unique_sorted(batch.np_vals[m])
    disp_means: list[float] = []
    disp_max: list[float] = []
    unb_means: list[float] = []
    for n in np_u:
        sub = m & subset_mask(batch, float(n))
        disp_means.append(float(np.mean(batch.disp[sub])))
        disp_max.append(float(np.max(batch.disp[sub])))
        unb_means.append(float(np.mean(batch.unbound[sub]) * 100))
    x = np.arange(len(np_u))
    fig, (ax_d, ax_u) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    ax_d.bar(x, disp_means, color="#2ca02c", alpha=0.85, label="mean")
    ax_d.scatter(x, disp_max, color="#d62728", s=50, zorder=3, label="max")
    ax_d.axhline(1.1, color="gray", linestyle=":", alpha=0.8)
    ax_d.set_ylabel("Dispersion ratio")
    ax_d.set_ylim(0.99, max(max(disp_max) * 1.1, 1.15))
    ax_d.legend(fontsize=8)
    ax_d.grid(True, axis="y", alpha=0.3)
    ax_u.bar(x, unb_means, color="#9467bd", alpha=0.85)
    ax_u.set_ylabel("Mean unbound (%)")
    ax_u.set_xticks(x)
    ax_u.set_xticklabels([str(int(n)) for n in np_u])
    ax_u.set_xlabel(r"$n_p$")
    ax_u.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"Stable band (spin $\\geq$ {STABLE_SPIN_MIN} hr) — {batch.label}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_kc_sigmac_overlay(p1: GridBatch, p2: GridBatch, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey="row")
    np_show = [400, 500, 750, 1000]
    for ax, n in zip(axes.flat, np_show, strict=True):
        s1, d1, _ = curve_for_np(p1, float(n))
        s2, d2, _ = curve_for_np(p2, float(n))
        ax.plot(s1, d1, "o-", color="#1f77b4", linewidth=2, markersize=5, label="fixed $k_c$")
        ax.plot(s2, d2, "s--", color="#ff7f0e", linewidth=2, markersize=5, label=r"$\sigma_c$-constant")
        ax.set_yscale("log")
        ax.set_ylim(0.98, max(p1.disp.max(), p2.disp.max()) * 1.2)
        ax.axhline(1.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(rf"$n_p={n}$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("Spin period (hours)")
    axes[0, 0].set_ylabel("Peak dispersion ratio")
    axes[1, 0].set_ylabel("Peak dispersion ratio")
    fig.suptitle("Fixed $k_c$ vs $\\sigma_c$-constant scaling (OBJ no-Earth ~177°)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_earth_vs_noearth(p1: GridBatch, p3: GridBatch, out: Path) -> None:
    np_shared = sorted(set(unique_sorted(p1.np_vals).astype(int)) & set(unique_sorted(p3.np_vals).astype(int)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes_flat = axes.flat
    for ax, n in zip(axes_flat, np_shared, strict=True):
        s1, d1, _ = curve_for_np(p1, float(n))
        s3, d3, _ = curve_for_np(p3, float(n))
        ax.plot(s1, d1, "o-", color="#2ca02c", linewidth=2, markersize=5, label="no Earth")
        ax.plot(s3, d3, "s--", color="#d62728", linewidth=2, markersize=5, label="Earth opposite")
        ax.set_yscale("log")
        ax.set_ylim(0.98, max(p1.disp.max(), p3.disp.max()) * 1.2)
        ax.axhline(1.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(rf"$n_p={n}$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("Spin period (hours)")
    axes[0, 0].set_ylabel("Peak dispersion ratio")
    axes[1, 0].set_ylabel("Peak dispersion ratio")
    fig.suptitle("Intrinsic (no-Earth) vs Earth flyby amplification (OBJ ~177°)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_intrinsic_spin(batch: GridBatch, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for n in unique_sorted(batch.np_vals):
        m = subset_mask(batch, float(n))
        valid = np.isfinite(batch.intrinsic[m])
        if not np.any(valid):
            continue
        c = NP_COLORS.get(int(n), "#333")
        ax.scatter(
            batch.spin[m][valid],
            batch.intrinsic[m][valid],
            s=50,
            color=c,
            label=rf"$n_p={int(n)}$",
            alpha=0.85,
        )
    spin_all = unique_sorted(batch.spin)
    ax.plot(spin_all, spin_all, "k:", linewidth=1, alpha=0.5, label="1:1 input")
    ax.set_xlabel("Input spin period (hours)")
    ax.set_ylabel("Intrinsic (settled) spin period (hours)")
    ax.set_title(f"Settled spin vs input — {batch.label}", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_tidal_spin_p3(p3: GridBatch, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for n in unique_sorted(p3.np_vals):
        m = subset_mask(p3, float(n))
        valid = np.isfinite(p3.post_flyby[m])
        if not np.any(valid):
            continue
        inp = p3.spin[m][valid]
        post = p3.post_flyby[m][valid]
        pct = (post - inp) / inp * 100
        c = NP_COLORS.get(int(n), "#333")
        ax.plot(inp, pct, "o-", color=c, linewidth=1.8, markersize=5, label=rf"$n_p={int(n)}$")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Input spin period (hours)")
    ax.set_ylabel("Post-flyby spin change vs input (%)")
    ax.set_title("Tidal spin modification (Earth opposite, bound main body)", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def load_sensitivity_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            param = row.get("parameter", "")
            eta = row.get("eta2_bins", "").strip()
            if param and eta:
                out[param] = float(eta)
    return out


def plot_sensitivity_bars(sens_dir: Path, out: Path) -> None:
    """Bar chart of η² from classic Analysis.py outputs."""
    configs = [
        ("P1 disp", sens_dir / "p1_dispersion_ratio_sensitivity.csv", "dispersion_ratio"),
        ("P1 unb", sens_dir / "p1_unbound_fraction_sensitivity.csv", "unbound_fraction"),
        ("P2 disp", sens_dir / "p2_dispersion_ratio_sensitivity.csv", "dispersion_ratio"),
        ("P3 disp", sens_dir / "p3_dispersion_ratio_sensitivity.csv", "dispersion_ratio"),
        ("P3 unb", sens_dir / "p3_unbound_fraction_sensitivity.csv", "unbound_fraction"),
    ]
    fig, axes = plt.subplots(1, len(configs), figsize=(3.2 * len(configs), 4.2), sharey=True)
    params = ["apophis_spin_period", "np_apophis"]
    param_labels = ["Spin period", r"$n_p$"]
    x = np.arange(len(params))
    width = 0.65
    for ax, (title, csv_path, _) in zip(axes, configs, strict=True):
        eta = load_sensitivity_csv(csv_path)
        vals = [eta.get(p, 0.0) for p in params]
        ax.bar(x, vals, width, color=["#1f77b4", "#ff7f0e"])
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel(r"$\eta^2$ (classic marginal sensitivity)")
    fig.suptitle(
        "Exploratory sensitivity on structured $n_p \\times$ spin grid (not Saltelli S1/ST)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sobol_mass_runs/plots/np_spin_grid"),
    )
    parser.add_argument(
        "--sens-dir",
        type=Path,
        default=Path("sobol_mass_runs/plots/np_spin_grid/sensitivity"),
        help="Directory with classic *_sensitivity.csv files from Analysis.py",
    )
    args = parser.parse_args()
    root = repo_root()
    out_dir = (root / args.out_dir).resolve()
    sens_dir = (root / args.sens_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = load_batch(
        latest_csv(root, P1_SUFFIX),
        "P1",
        "P1: OBJ no-Earth, fixed $k_c$",
    )
    p2 = load_batch(
        latest_csv(root, P2_SUFFIX),
        "P2",
        r"P2: OBJ no-Earth, $\sigma_c$-constant",
    )
    p3 = load_batch(
        latest_csv(root, P3_SUFFIX),
        "P3",
        "P3: OBJ Earth opposite",
    )

    # A1
    plot_disp_curves_panels([p1, p2, p3], out_dir / "np_spin_grid_disp_vs_period_panels.png")

    # A2 heatmaps
    for batch, tag in [(p1, "p1"), (p2, "p2"), (p3, "p3")]:
        plot_heatmap(batch, out_dir / f"np_spin_grid_heatmap_disp_{tag}.png", field="disp")
        plot_heatmap(batch, out_dir / f"np_spin_grid_heatmap_unbound_{tag}.png", field="unbound")

    # A3 P_crit (P1 primary)
    plot_pcrit(p1, out_dir / "np_spin_grid_pcrit_vs_np.png", out_dir / "np_spin_grid_pcrit.csv")

    # A4 stable band (all three)
    for batch, tag in [(p1, "p1"), (p2, "p2"), (p3, "p3")]:
        plot_stable_band(batch, out_dir / f"np_spin_grid_stable_band_{tag}.png")

    # A5
    plot_kc_sigmac_overlay(p1, p2, out_dir / "np_spin_grid_kc_vs_sigmac_overlay.png")

    # A6
    plot_earth_vs_noearth(p1, p3, out_dir / "np_spin_grid_earth_vs_noearth.png")

    # A7
    plot_intrinsic_spin(p1, out_dir / "np_spin_grid_intrinsic_spin_p1.png")
    plot_tidal_spin_p3(p3, out_dir / "np_spin_grid_tidal_spin_p3.png")

    # B1 sensitivity bars (if CSVs exist)
    if sens_dir.is_dir() and any(sens_dir.glob("*_sensitivity.csv")):
        plot_sensitivity_bars(sens_dir, out_dir / "np_spin_grid_sensitivity_bars.png")
    else:
        print(f"[WARN] No sensitivity CSVs in {sens_dir}; run Analysis.py first, then re-run plots.")


if __name__ == "__main__":
    main()
