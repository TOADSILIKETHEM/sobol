#!/usr/bin/env python3
"""Run classic marginal sensitivity on all sobol_mass_outputs.csv batches and plot results."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sobol.Analysis.Analysis import (  # noqa: E402
    RESPONSE_CANDIDATES,
    _parse_numeric_cell,
    compute_classic_sensitivity_rows,
    load_table,
    write_sensitivity_csv,
)

# Human-readable axis labels for plot titles.
PARAM_LABELS: Dict[str, str] = {
    "mass_input_kg": "Mass (kg)",
    "scale_vel": "Velocity scale",
    "scale_pos": "Position scale",
    "scale_r_apophis": "Radius scale",
    "scale_rho": "Density scale",
    "apophis_spin_period": "Spin period (hr)",
    "apophis_spin_obliquity": "Spin obliquity (deg)",
    "apophis_spin_azimuth": "Spin azimuth (deg)",
    "apophis_spin_torque_align_deg": "Torque-align angle (deg)",
    "kt_cgs": r"$k_t$ (g/s$^2$/cm)",
    "kc_cgs": r"$k_t$ (g/s$^2$/cm)",  # alias for archived CSVs
    "np_apophis": r"$n_p$",
    "use_dem": "use_dem",
    "use_shape_crop": "use_shape_crop",
    "apophis_only": "apophis_only",
}

RESPONSE_LABELS: Dict[str, str] = {
    "dispersion_ratio": "Dispersion ratio",
    "unbound_fraction": "Unbound fraction",
    "closest_approach_km": "Closest approach (km)",
    "closest_approach_au": "Closest approach (AU)",
    "intrinsic_spin_period_hr": "Intrinsic spin period (hr)",
    "approach_spin_period_hr": "Approach spin period (hr)",
    "post_flyby_spin_period_hr": "Post-flyby spin period (hr)",
    "settled_spin_period_hr": "Settled spin period (hr)",
}


_CAMPAIGN_RE = re.compile(r"^sobol_\d{8}_\d{6}_(.+)$")


def campaign_suffix(batch_dir_name: str) -> str:
    """Strip sobol_YYYYMMDD_HHMMSS_ prefix; return campaign slug for deduplication."""
    m = _CAMPAIGN_RE.match(batch_dir_name)
    if m:
        return m.group(1)
    return batch_dir_name.replace("sobol_", "", 1)


def discover_output_csvs(root: Path, *, latest_only: bool = True) -> List[Path]:
    all_csvs = sorted(root.glob("sobol_*/sobol_mass_outputs.csv"))
    if not latest_only:
        return all_csvs
    by_campaign: Dict[str, List[Path]] = defaultdict(list)
    for path in all_csvs:
        by_campaign[campaign_suffix(path.parent.name)].append(path)
    return sorted(max(paths, key=lambda p: p.parent.name) for paths in by_campaign.values())


def varying_responses(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    ok_only: bool,
) -> List[str]:
    """Return response columns with >=3 finite, varying values after filtering."""
    use_ok = ok_only and "status" in fieldnames
    candidates = list(RESPONSE_CANDIDATES) + ["settled_spin_period_hr"]
    out: List[str] = []
    for col in candidates:
        if col not in fieldnames:
            continue
        vals: List[float] = []
        for r in rows:
            if use_ok and r.get("status", "").strip().lower() != "ok":
                continue
            v = _parse_numeric_cell(r.get(col, ""))
            if v is None or not math.isfinite(v):
                continue
            vals.append(v)
        if len(vals) < 3:
            continue
        if np.ptp(np.asarray(vals, dtype=float)) <= 0.0:
            continue
        out.append(col)
    return out


def batch_slug(csv_path: Path) -> str:
    return csv_path.parent.name.replace("sobol_", "", 1)


def run_all(
    runs_root: Path,
    plots_dir: Path,
    *,
    bootstrap: int,
    bins: int,
    seed: int,
    log_dispersion: bool,
    latest_only: bool,
) -> Tuple[Path, List[Path]]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    master_path = plots_dir / "sensitivity_master.csv"
    plot_paths: List[Path] = []
    master_rows: List[Dict[str, object]] = []

    csv_paths = discover_output_csvs(runs_root, latest_only=latest_only)
    n_all = len(list(runs_root.glob("sobol_*/sobol_mass_outputs.csv")))
    if latest_only and n_all > len(csv_paths):
        print(
            f"[INFO] Using {len(csv_paths)} latest batch CSVs "
            f"({n_all - len(csv_paths)} older duplicates skipped)"
        )
    else:
        print(f"[INFO] Found {len(csv_paths)} batch CSVs under {runs_root}")

    ok_count = 0
    skip_count = 0

    for csv_path in csv_paths:
        slug = batch_slug(csv_path)
        try:
            fieldnames, rows = load_table(csv_path)
        except ValueError as exc:
            print(f"[SKIP] {slug}: {exc}")
            skip_count += 1
            continue

        responses = varying_responses(rows, fieldnames, ok_only=True)
        if not responses:
            print(f"[SKIP] {slug}: no varying response columns")
            skip_count += 1
            continue

        for response in responses:
            use_log = log_dispersion and response == "dispersion_ratio"
            try:
                result_rows, n_used, input_names = compute_classic_sensitivity_rows(
                    csv_path,
                    response,
                    ok_only=True,
                    log_response=use_log,
                    bins=bins,
                    bootstrap=bootstrap,
                    seed=seed,
                )
            except ValueError as exc:
                print(f"[SKIP] {slug} / {response}: {exc}")
                skip_count += 1
                continue

            sens_name = f"sobol_mass_outputs_sensitivity__{response}.csv"
            sens_path = csv_path.parent / sens_name
            write_sensitivity_csv(sens_path, result_rows)

            for row in result_rows:
                master_rows.append(
                    {
                        "batch": slug,
                        "response": response,
                        "log_response": int(use_log),
                        "n_rows": n_used,
                        "inputs": ",".join(input_names),
                        **row,
                    }
                )

            plot_path = plot_sensitivity_bar(
                result_rows,
                plots_dir / f"{slug}__{response}.png",
                title=slug,
                response_label=RESPONSE_LABELS.get(response, response)
                + (" (log10)" if use_log else ""),
            )
            plot_paths.append(plot_path)
            ok_count += 1
            print(
                f"[OK] {slug} / {response}: n={n_used} inputs={len(input_names)} "
                f"-> {plot_path.name}"
            )

    if master_rows:
        with master_path.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "batch",
                "response",
                "log_response",
                "n_rows",
                "inputs",
                "parameter",
                "eta2_bins",
                "eta2_ci95_halfwidth",
                "r2_pearson",
                "r2_spearman",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in master_rows:
                w.writerow({k: row.get(k, "") for k in fields})
        print(f"[INFO] Wrote master table: {master_path} ({len(master_rows)} rows)")

    print(f"[INFO] Completed: {ok_count} analyses, {skip_count} skipped")
    return master_path, plot_paths


def plot_sensitivity_bar(
    result_rows: Sequence[Dict[str, object]],
    out_path: Path,
    *,
    title: str,
    response_label: str,
) -> Path:
    if not result_rows:
        raise ValueError("no result rows to plot")

    params = [str(r["parameter"]) for r in result_rows]
    labels = [PARAM_LABELS.get(p, p) for p in params]
    eta2 = np.array([float(r["eta2_bins"]) for r in result_rows], dtype=float)
    r2p = np.array([float(r["r2_pearson"]) for r in result_rows], dtype=float)
    r2s_raw = [r.get("r2_spearman", "") for r in result_rows]
    r2s = np.array(
        [float(v) if v != "" and math.isfinite(float(v)) else np.nan for v in r2s_raw],
        dtype=float,
    )
    ci = np.array(
        [
            float(r["eta2_ci95_halfwidth"])
            if r.get("eta2_ci95_halfwidth", "") != ""
            and math.isfinite(float(r["eta2_ci95_halfwidth"]))
            else np.nan
            for r in result_rows
        ],
        dtype=float,
    )

    n = len(params)
    y = np.arange(n)
    height = 0.25

    fig_h = max(3.5, 1.2 + 0.55 * n)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    bars_eta = ax.barh(y + height, eta2, height=height, label=r"$\eta^2$ (bins)", color="#1f77b4")
    ax.barh(y, r2p, height=height, label=r"$R^2$ Pearson", color="#ff7f0e")
    ax.barh(y - height, r2s, height=height, label=r"$R^2$ Spearman", color="#2ca02c")

    for i, (bar, err) in enumerate(zip(bars_eta, ci)):
        if math.isfinite(err) and err > 0:
            x = float(bar.get_width())
            ax.errorbar(
                x,
                bar.get_y() + bar.get_height() / 2,
                xerr=err,
                fmt="none",
                ecolor="#1f77b4",
                capsize=3,
                linewidth=1,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Sensitivity (0 = none, 1 = perfect marginal association)")
    ax.set_title(f"{title}\nResponse: {response_label}", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_heatmap(master_path: Path, out_path: Path) -> Optional[Path]:
    """Top parameters by eta2 across all batches (eta2 heatmap)."""
    if not master_path.is_file():
        return None

    with master_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    # batch+response on rows, parameter on cols
    keys = sorted({(r["batch"], r["response"]) for r in rows})
    params = sorted({r["parameter"] for r in rows})
    mat = np.full((len(keys), len(params)), np.nan)
    for i, key in enumerate(keys):
        for j, par in enumerate(params):
            matches = [
                float(r["eta2_bins"])
                for r in rows
                if (r["batch"], r["response"]) == key and r["parameter"] == par
            ]
            if matches:
                mat[i, j] = matches[0]

    if not np.any(np.isfinite(mat)):
        return None

    fig_w = max(10, 0.35 * len(params) + 4)
    fig_h = max(6, 0.22 * len(keys) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([PARAM_LABELS.get(p, p) for p in params], rotation=45, ha="right")
    ylabels = [f"{b}\n({resp})" for b, resp in keys]
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(r"Marginal sensitivity $\eta^2$ across batches")
    fig.colorbar(im, ax=ax, label=r"$\eta^2$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("sobol_mass_runs"),
        help="Directory containing sobol_* batch folders",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("sobol_mass_runs/plots/sobolplots"),
        help="Output directory for sensitivity bar charts",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, metavar="B")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dispersion",
        action="store_true",
        help="Analyse log10(dispersion_ratio) when that response is used",
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Include every timestamped batch (default: latest per campaign suffix only)",
    )
    args = parser.parse_args()

    repo = _REPO
    runs_root = (repo / args.runs_root).resolve()
    plots_dir = (repo / args.plots_dir).resolve()

    master_path, _ = run_all(
        runs_root,
        plots_dir,
        bootstrap=args.bootstrap,
        bins=args.bins,
        seed=args.seed,
        log_dispersion=args.log_dispersion,
        latest_only=not args.all_batches,
    )
    summary = plot_summary_heatmap(master_path, plots_dir / "_summary_eta2_heatmap.png")
    if summary:
        print(f"[INFO] Wrote summary heatmap: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
