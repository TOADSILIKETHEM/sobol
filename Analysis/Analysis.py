#!/usr/bin/env python3
"""
Sensitivity summaries for PHANTOM sweep CSVs (e.g. sobol_mass_outputs.csv).

**Classic mode** (default): reads one evaluation per row (plain Sobol/Monte Carlo layout).
  - Correlation ratio (eta^2) via quantile bins (nonlinear marginal).
  - Squared Pearson / Spearman correlations.

**Saltelli mode**: variance-based **Sobol indices** (S1, ST, optional S2) from outputs in
**Saltelli evaluation order**. Requires ``saltelli_problem.json``, ``saltelli_Y.csv`` (or .npy),
and optionally ``saltelli_meta.json`` — produced by ``run_mass_sobol_phantom.py --saltelli-n``.

You cannot compute Saltelli Sobol indices from a classic ``sobol_mass_outputs.csv`` alone;
that design has only N evaluations instead of N*(D+2) or N*(2*D+2).

Recognised input parameters (varied dimensions):
  mass_input_kg, scale_vel, scale_pos, scale_r_apophis, scale_rho,
  apophis_spin_period, apophis_spin_obliquity, apophis_spin_azimuth,
  apophis_spin_torque_align_deg, kc_cgs, np_apophis,
  use_dem, use_shape_crop, apophis_only.

Recognised response columns (--response / --saltelli-y-column):
  closest_approach_km, closest_approach_au  — orbital closest-approach distance.
  dispersion_ratio                           — DEM only; peak radius-of-gyration ratio (>=1).
  unbound_fraction                           — DEM only; peak unbound mass fraction [0,1].
  intrinsic_spin_period_hr                   — early-plateau bound-rubble spin before tidal ramp-up (hours).
  approach_spin_period_hr                    — mean spin in last 24 h before closest approach (hours).
  post_flyby_spin_period_hr                  — mean bound-rubble spin period after closest approach (hours).

Examples:
  python3 Analysis.py --method classic --csv sobol_mass_runs/.../sobol_mass_outputs.csv \\
      --response closest_approach_au

  python3 Analysis.py --method classic --csv sobol_mass_runs/.../sobol_mass_outputs.csv \\
      --response dispersion_ratio

  python3 Analysis.py --method saltelli \\
      --sobol-problem-json batch/saltelli_problem.json \\
      --saltelli-meta-json batch/saltelli_meta.json \\
      --saltelli-y-csv batch/saltelli_Y.csv \\
      --saltelli-y-column dispersion_ratio

Classic mode writes ``<input_stem>_sensitivity.csv`` by default. Saltelli mode writes
``saltelli_sobol_indices.csv`` next to the problem JSON unless ``--output-sobol-csv`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print(
        "[ERROR] Analysis.py requires numpy (and scipy for Spearman). "
        "Install: pip install numpy scipy",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from SALib.analyze import sobol as sobol_analyze

    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False

# Columns treated as model inputs if present and non-constant after filtering (classic mode).
INPUT_CANDIDATES: Tuple[str, ...] = (
    "mass_input_kg",
    "scale_vel",
    "scale_pos",
    "scale_r_apophis",
    "scale_rho",
    # Spin orientation/rate — patched into setup; only affect DEM runs.
    "apophis_spin_period",
    "apophis_spin_obliquity",
    "apophis_spin_azimuth",
    "apophis_spin_torque_align_deg",
    "kc_cgs",
    # Particle count (varies in --np-apophis-list sweeps).
    "np_apophis",
)

BOOL_INPUTS: Tuple[str, ...] = ("use_dem", "use_shape_crop", "apophis_only")

# All response columns the runner can produce; used only for documentation / help text.
RESPONSE_CANDIDATES: Tuple[str, ...] = (
    "closest_approach_km",
    "closest_approach_au",
    "dispersion_ratio",
    "unbound_fraction",
    "intrinsic_spin_period_hr",
    "approach_spin_period_hr",
    "post_flyby_spin_period_hr",
)

RESULT_CSV_FIELDS = (
    "parameter",
    "eta2_bins",
    "eta2_ci95_halfwidth",
    "r2_pearson",
    "r2_spearman",
)

SOBOL_INDEX_CSV_FIELDS = (
    "parameter",
    "S1",
    "S1_conf",
    "ST",
    "ST_conf",
)


def expected_saltelli_num_evals(num_vars: int, base_n: int, calc_second_order: bool) -> int:
    """Row count from SALib.sample.sobol.sample (must match runner)."""
    if calc_second_order:
        return base_n * (2 * num_vars + 2)
    return base_n * (num_vars + 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Classic: eta^2 + correlations from sobol_mass_outputs.csv. "
            "Saltelli: Sobol S1/ST from SALib bundle (see run_mass_sobol_phantom.py --saltelli-n)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method",
        choices=("classic", "saltelli", "both"),
        default="classic",
        help="classic: CSV marginal metrics; saltelli: Sobol indices; both: run classic then saltelli.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to sobol_mass_outputs.csv (required for classic / both).",
    )
    p.add_argument(
        "--response",
        default="closest_approach_au",
        help=(
            "Output column to analyse in classic mode; also the default for --saltelli-y-column. "
            "Valid options from the runner: closest_approach_km, closest_approach_au, "
            "dispersion_ratio (DEM only), unbound_fraction (DEM only), "
            "intrinsic_spin_period_hr, approach_spin_period_hr, post_flyby_spin_period_hr "
            "(multi-sink Apophis; approach/post need Earth flyby)."
        ),
    )
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows even when status is not ok (classic mode only).",
    )
    p.add_argument(
        "--log-response",
        action="store_true",
        help="Apply log10 to positive response values before classic analysis.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=10,
        metavar="K",
        help="Quantile bins for eta^2 (classic).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="B",
        help="Bootstrap resamples for approximate 95%% CI on eta^2 (classic; 0 = skip).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap (classic).")

    p.add_argument(
        "--sobol-problem-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="SALib problem dict JSON (saltelli / both).",
    )
    p.add_argument(
        "--saltelli-meta-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional metadata from runner (validates num_evals / second-order flag).",
    )
    p.add_argument(
        "--saltelli-y-npy",
        type=Path,
        default=None,
        metavar="PATH",
        help="Model outputs Y as a 1D numpy .npy file in Saltelli order.",
    )
    p.add_argument(
        "--saltelli-y-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="CSV with model outputs (e.g. saltelli_Y.csv from runner).",
    )
    p.add_argument(
        "--saltelli-y-column",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Numeric column to use as Y from the saltelli Y CSV (default: same as --response). "
            "The runner writes closest_approach_km, closest_approach_au, dispersion_ratio, "
            "unbound_fraction, intrinsic_spin_period_hr, approach_spin_period_hr, "
            "and post_flyby_spin_period_hr "
            "to saltelli_Y.csv."
        ),
    )
    p.add_argument(
        "--saltelli-calc-second-order",
        action="store_true",
        help="Must match the Saltelli sample used (enables S2 matrix output).",
    )
    p.add_argument(
        "--sobol-conf-level",
        type=float,
        default=0.95,
        help="Confidence level for Sobol index intervals (saltelli).",
    )

    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Classic sensitivity CSV (default: <input_stem>_sensitivity.csv beside --csv).",
    )
    p.add_argument(
        "--no-output-csv",
        action="store_true",
        help="Skip writing classic sensitivity CSV.",
    )
    p.add_argument(
        "--output-sobol-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Sobol indices CSV (default: saltelli_sobol_indices.csv beside problem JSON).",
    )
    p.add_argument(
        "--no-output-sobol-csv",
        action="store_true",
        help="Skip writing Sobol CSV (still prints table).",
    )
    return p.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.method in ("classic", "both"):
        if args.csv is None:
            raise ValueError(f"--method {args.method} requires --csv")
    if args.method in ("saltelli", "both"):
        if not HAS_SALIB:
            raise ValueError("Saltelli analysis requires SALib (pip install SALib)")
        if args.sobol_problem_json is None:
            raise ValueError("--method saltelli/both requires --sobol-problem-json")
        if args.saltelli_y_npy is None and args.saltelli_y_csv is None:
            raise ValueError("Saltelli mode requires --saltelli-y-npy or --saltelli-y-csv")


def load_problem_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("num_vars", "names", "bounds"):
        if key not in data:
            raise ValueError(f"problem JSON missing {key!r}")
    if len(data["names"]) != int(data["num_vars"]) or len(data["bounds"]) != int(data["num_vars"]):
        raise ValueError("problem JSON num_vars inconsistent with names/bounds")
    return data


def load_meta_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_saltelli_y(
    args: argparse.Namespace,
    problem: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
) -> np.ndarray:
    y_col = args.saltelli_y_column or args.response
    D = int(problem["num_vars"])

    if args.saltelli_y_npy is not None:
        y = np.load(args.saltelli_y_npy)
        y = np.asarray(y, dtype=float).flatten()
    else:
        assert args.saltelli_y_csv is not None
        with args.saltelli_y_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("saltelli Y CSV has no header")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
        if not rows:
            raise ValueError("saltelli Y CSV is empty")
        if y_col not in fieldnames:
            raise ValueError(f"Column {y_col!r} not in saltelli Y CSV: {fieldnames}")
        if "eval_index" in fieldnames:
            rows.sort(key=lambda r: int(r["eval_index"]))
        y_list: List[float] = []
        for r in rows:
            raw = (r.get(y_col) or "").strip()
            if raw == "":
                raise ValueError(f"Empty Y at eval_index={r.get('eval_index', '?')}")
            v = float(raw)
            if not math.isfinite(v):
                raise ValueError(
                    f"Non-finite Y in column {y_col!r} at eval_index={r.get('eval_index', '?')} "
                    "(Sobol analysis needs finite values for all Saltelli runs)."
                )
            y_list.append(v)
        y = np.array(y_list, dtype=float)

    if meta is not None:
        expected = int(meta["num_evals"])
        second = bool(meta["calc_second_order"])
        if expected != len(y):
            raise ValueError(f"len(Y)={len(y)} but saltelli_meta.json has num_evals={expected}")
        if second != args.saltelli_calc_second_order:
            raise ValueError(
                "Mismatch: --saltelli-calc-second-order does not match saltelli_meta.json "
                f"(meta has calc_second_order={second})."
            )
        formula = expected_saltelli_num_evals(D, int(meta["base_n"]), second)
        if formula != len(y):
            raise ValueError(f"Internal inconsistency: meta num_evals vs formula ({formula} vs {len(y)})")
    else:
        denom = (D + 2) if not args.saltelli_calc_second_order else (2 * D + 2)
        if len(y) % denom != 0:
            raise ValueError(
                f"len(Y)={len(y)} not divisible by {denom} for D={D} and second_order="
                f"{args.saltelli_calc_second_order}; supply --saltelli-meta-json to validate."
            )

    return y


def _parse_numeric_cell(raw: str) -> Optional[float]:
    raw = (raw or "").strip()
    if raw == "":
        return None
    lu = raw.upper()
    if lu in ("T", "TRUE", "1"):
        return 1.0
    if lu in ("F", "FALSE", "0"):
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return None


def load_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def rows_to_arrays(
    rows: Sequence[Dict[str, str]],
    fieldnames: Sequence[str],
    response: str,
    ok_only: bool,
    log_response: bool,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    if response not in fieldnames:
        raise ValueError(f"Response column {response!r} not in CSV headers: {fieldnames}")

    use_ok = ok_only and "status" in fieldnames

    kept: List[Tuple[Dict[str, str], float]] = []
    for r in rows:
        if use_ok and r.get("status", "").strip().lower() != "ok":
            continue
        yv = _parse_numeric_cell(r.get(response, ""))
        if yv is None or not math.isfinite(yv):
            continue
        kept.append((r, yv))

    if not kept:
        raise ValueError("No usable rows after filtering (check status and finite response).")

    if log_response:
        logs: List[float] = []
        for _r, yv in kept:
            if yv <= 0:
                raise ValueError("--log-response requires strictly positive finite response values.")
            logs.append(math.log10(yv))
        y_arr = np.array(logs, dtype=float)
    else:
        y_arr = np.array([yv for _r, yv in kept], dtype=float)

    input_names: List[str] = []
    col_arrays: Dict[str, np.ndarray] = {}

    for name in list(INPUT_CANDIDATES) + list(BOOL_INPUTS):
        if name not in fieldnames or name == response:
            continue
        vals: List[float] = []
        ok = True
        for r, _yv in kept:
            v = _parse_numeric_cell(r.get(name, ""))
            if v is None or not math.isfinite(v):
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        x = np.array(vals, dtype=float)
        if np.ptp(x) <= 0.0:
            continue
        input_names.append(name)
        col_arrays[name] = x

    if not input_names:
        raise ValueError("No varying numeric input columns found (check CSV schema).")

    return y_arr, input_names, col_arrays


def correlation_ratio_quantile(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> float:
    n = len(x)
    if n < 3 or n_bins < 2:
        return float("nan")
    n_bins = min(n_bins, n // 2)
    n_bins = max(n_bins, 2)

    order = np.argsort(x)
    y_s = y[order]
    bin_assign = np.clip((np.arange(n) * n_bins) // n, 0, n_bins - 1)

    y_mean = float(np.mean(y_s))
    ss_tot = float(np.sum((y_s - y_mean) ** 2))
    if ss_tot <= 0.0:
        return 0.0

    ss_between = 0.0
    for b in range(n_bins):
        mask = bin_assign == b
        if not np.any(mask):
            continue
        yb = y_s[mask]
        nj = int(np.sum(mask))
        ss_between += nj * (float(np.mean(yb)) - y_mean) ** 2

    return float(ss_between / ss_tot)


def squared_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if not math.isfinite(r):
        return 0.0
    return float(r * r)


def squared_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if not HAS_SCIPY:
        return float("nan")
    if np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return 0.0
    rho, _ = stats.spearmanr(x, y)
    if not math.isfinite(rho):
        return 0.0
    return float(rho * rho)


def bootstrap_eta_sq(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    if len(x) != len(y):
        raise ValueError(f"bootstrap_eta_sq: x and y must have the same length ({len(x)} vs {len(y)})")
    rng = np.random.default_rng(seed)
    stat = correlation_ratio_quantile(x, y, n_bins)
    if n_boot <= 0:
        return stat, float("nan")
    draws: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws.append(correlation_ratio_quantile(x[idx], y[idx], n_bins))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    half = float(hi - lo) / 2.0
    return stat, half


def write_sensitivity_csv(path: Path, result_rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_CSV_FIELDS))
        writer.writeheader()
        for row in result_rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_CSV_FIELDS})


def write_sobol_indices_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SOBOL_INDEX_CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SOBOL_INDEX_CSV_FIELDS})


def write_sobol_s2_csv(path: Path, names: Sequence[str], s2: np.ndarray, s2_conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wf = csv.writer(f)
        wf.writerow(["parameter_i", "parameter_j", "S2", "S2_conf"])
        d = len(names)
        for i in range(d):
            for j in range(d):
                wf.writerow(
                    [
                        names[i],
                        names[j],
                        float(s2[i, j]),
                        float(s2_conf[i, j]),
                    ]
                )


def compute_classic_sensitivity_rows(
    csv_path: Path,
    response: str,
    *,
    ok_only: bool = True,
    log_response: bool = False,
    bins: int = 10,
    bootstrap: int = 0,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], int, List[str]]:
    """Run classic sensitivity on one CSV; return (result_rows, n_rows_used, input_names)."""
    fieldnames, rows = load_table(csv_path)
    y, input_names, cols = rows_to_arrays(
        rows, fieldnames, response, ok_only=ok_only, log_response=log_response
    )
    result_rows: List[Dict[str, object]] = []
    for name in input_names:
        x = cols[name]
        eta2 = correlation_ratio_quantile(x, y, bins)
        if bootstrap > 0:
            _, half_width = bootstrap_eta_sq(x, y, bins, bootstrap, seed)
            ci_cell: object = half_width if math.isfinite(half_width) else ""
        else:
            ci_cell = ""
        r2p = squared_pearson(x, y)
        r2s = squared_spearman(x, y)
        result_rows.append(
            {
                "parameter": name,
                "eta2_bins": eta2,
                "eta2_ci95_halfwidth": ci_cell,
                "r2_pearson": r2p,
                "r2_spearman": r2s if HAS_SCIPY and math.isfinite(r2s) else "",
            }
        )
    return result_rows, len(y), input_names


def run_classic_analysis(args: argparse.Namespace) -> None:
    ok_only = not args.include_failed
    assert args.csv is not None
    fieldnames, rows = load_table(args.csv)
    y, input_names, cols = rows_to_arrays(
        rows, fieldnames, args.response, ok_only=ok_only, log_response=args.log_response
    )

    print(f"[INFO] CSV: {args.csv.resolve()}")
    print(f"[INFO] Rows used: {len(y)}  Response: {args.response}" + (" (log10)" if args.log_response else ""))
    print(f"[INFO] Inputs: {', '.join(input_names)}")
    print()

    header = (
        f"{'parameter':<18} {'eta2_bins':>12} {'eta2_CI95±':>12} {'R2_pearson':>12} {'R2_spearman':>12}"
    )
    print(header)
    print("-" * len(header))

    result_rows: List[Dict[str, object]] = []
    for name in input_names:
        x = cols[name]
        eta2 = correlation_ratio_quantile(x, y, args.bins)
        if args.bootstrap > 0:
            _, half_width = bootstrap_eta_sq(x, y, args.bins, args.bootstrap, args.seed)
            ci_str = f"{half_width:.6f}" if math.isfinite(half_width) else "n/a"
            ci_cell: object = half_width if math.isfinite(half_width) else ""
        else:
            ci_str = "—"
            ci_cell = ""
        r2p = squared_pearson(x, y)
        r2s = squared_spearman(x, y)
        if not HAS_SCIPY:
            r2s_str = "n/a"
            r2s_cell: object = ""
        elif math.isfinite(r2s):
            r2s_str = f"{r2s:.6f}"
            r2s_cell = r2s
        else:
            r2s_str = "n/a"
            r2s_cell = ""

        print(f"{name:<18} {eta2:12.6f} {ci_str:>12} {r2p:12.6f} {r2s_str:>12}")
        result_rows.append(
            {
                "parameter": name,
                "eta2_bins": eta2,
                "eta2_ci95_halfwidth": ci_cell,
                "r2_pearson": r2p,
                "r2_spearman": r2s_cell,
            }
        )

    if not args.no_output_csv:
        out_path = (
            args.output_csv.resolve()
            if args.output_csv is not None
            else (args.csv.parent / f"{args.csv.stem}_sensitivity.csv").resolve()
        )
        write_sensitivity_csv(out_path, result_rows)
        print()
        print(f"[INFO] Wrote sensitivity CSV: {out_path}")

    print()
    print(
        "Interpretation (classic): eta^2 is the correlation ratio from quantile bins on each input "
        "(marginal, nonlinear). R^2 Pearson/Spearman are linear/rank-linear. "
        "These are not Saltelli Sobol indices. "
        "Spin inputs (apophis_spin_period/obliquity/azimuth) only vary in DEM sweeps; "
        "dispersion_ratio and unbound_fraction responses are blank for non-DEM runs and are "
        "excluded automatically when not finite."
    )
    if not HAS_SCIPY:
        print("[NOTE] Install scipy for Spearman R^2 (pip install scipy).")


def run_saltelli_sobol_analysis(args: argparse.Namespace) -> None:
    assert args.sobol_problem_json is not None
    assert HAS_SALIB

    problem = load_problem_json(args.sobol_problem_json)
    meta: Optional[Dict[str, Any]] = None
    if args.saltelli_meta_json is not None:
        meta = load_meta_json(args.saltelli_meta_json)

    y = load_saltelli_y(args, problem, meta)

    Si = sobol_analyze.analyze(
        problem,
        y,
        calc_second_order=args.saltelli_calc_second_order,
        conf_level=args.sobol_conf_level,
        print_to_console=False,
        seed=args.seed,
    )

    names = list(problem["names"])
    print(f"[INFO] Sobol analysis: D={len(names)} evals={len(y)} response={args.saltelli_y_column or args.response}")
    print()
    hdr = f"{'parameter':<18} {'S1':>12} {'S1_conf':>12} {'ST':>12} {'ST_conf':>12}"
    print(hdr)
    print("-" * len(hdr))

    out_rows: List[Dict[str, object]] = []
    for i, name in enumerate(names):
        print(
            f"{name:<18} {float(Si['S1'][i]):12.6f} {float(Si['S1_conf'][i]):12.6f} "
            f"{float(Si['ST'][i]):12.6f} {float(Si['ST_conf'][i]):12.6f}"
        )
        out_rows.append(
            {
                "parameter": name,
                "S1": float(Si["S1"][i]),
                "S1_conf": float(Si["S1_conf"][i]),
                "ST": float(Si["ST"][i]),
                "ST_conf": float(Si["ST_conf"][i]),
            }
        )

    if not args.no_output_sobol_csv:
        out_p = (
            args.output_sobol_csv.resolve()
            if args.output_sobol_csv is not None
            else (args.sobol_problem_json.parent / "saltelli_sobol_indices.csv").resolve()
        )
        write_sobol_indices_csv(out_p, out_rows)
        print()
        print(f"[INFO] Wrote Sobol indices CSV: {out_p}")

        if args.saltelli_calc_second_order and "S2" in Si:
            s2_path = out_p.with_name(out_p.stem + "_S2.csv")
            write_sobol_s2_csv(s2_path, names, Si["S2"], Si["S2_conf"])
            print(f"[INFO] Wrote second-order Sobol CSV: {s2_path}")

    print()
    print(
        "Interpretation (Saltelli): S1 is first-order Sobol index; ST is total-order. "
        "See SALib documentation for estimator details."
    )


def main() -> int:
    args = parse_args()
    validate_cli(args)

    if args.method in ("classic", "both"):
        run_classic_analysis(args)
        if args.method == "both":
            print()

    if args.method in ("saltelli", "both"):
        run_saltelli_sobol_analysis(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
