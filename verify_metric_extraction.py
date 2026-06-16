#!/usr/bin/env python3
"""Compare DEM metric extraction on a completed run directory.

Usage:
  python3 sobol/verify_metric_extraction.py --run-dir sobol_mass_runs/.../run_0001
  python3 sobol/verify_metric_extraction.py --run-dir ... --prefix sobol --apophis-sink-id 11

Recomputes metrics with the current (dump-time) path. Pass ``--legacy`` to also run a slow
substep-dense comparison (``METRICS_LEGACY_SUBSTEPS=1``) for regression against pre-2026 CSVs.
Optionally compares to a row in sobol_mass_outputs.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sobol.run_mass_sobol_phantom import (  # noqa: E402
    _apophis_time_groups,
    _earth_apophis_closest_approach,
    _extract_dem_metrics_bundle,
    _extract_mean_spin_period_hr,
    _main_body_mask,
    _mass_weighted_rg,
    _parse_spin_axis_from_setup_log,
    _parse_utime_from_phantom_log,
    _use_fast_metrics_defaults,
    extract_breakup_metrics,
)


def _finite_close(a: float, b: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    if a == b:
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)


def _metrics_for_run(
    run_dir: Path,
    prefix: str,
    earth_sink_id: int,
    apophis_sink_id: int,
    *,
    apophis_only: bool,
    legacy_substeps: bool,
) -> Dict[str, Any]:
    if legacy_substeps:
        os.environ["METRICS_LEGACY_SUBSTEPS"] = "1"
    else:
        _use_fast_metrics_defaults()

    groups, time_of_key, n_sinks = _apophis_time_groups(run_dir, prefix, apophis_sink_id)
    out: Dict[str, Any] = {"n_dump_groups": len(groups), "n_sinks": n_sinks}

    if n_sinks < 2:
        return out

    t_ca: Optional[float] = None
    if not apophis_only:
        closest_km, closest_au, t_ca = _earth_apophis_closest_approach(
            run_dir, prefix, earth_sink_id, apophis_sink_id
        )
        out["closest_approach_km"] = closest_km
        out["closest_approach_au"] = closest_au

    disp, unbound, intrinsic, approach, post = _extract_dem_metrics_bundle(
        run_dir,
        prefix,
        apophis_sink_id,
        apophis_only=apophis_only,
        earth_sink_id=None if apophis_only else earth_sink_id,
        t_ca=t_ca,
        _groups=groups,
        _time_of_key=time_of_key,
        _n_sinks=n_sinks,
    )
    out["dispersion_ratio"] = disp
    out["unbound_fraction"] = unbound
    out["intrinsic_spin_period_hr"] = intrinsic
    out["approach_spin_period_hr"] = approach
    out["post_flyby_spin_period_hr"] = post

    # Standalone extractors (public API smoke test)
    disp2, unbound2 = extract_breakup_metrics(
        run_dir, prefix, apophis_sink_id, _groups=groups, _time_of_key=time_of_key
    )
    out["dispersion_ratio_standalone"] = disp2
    out["unbound_fraction_standalone"] = unbound2
    grp_kw = dict(_groups=groups, _time_of_key=time_of_key, _n_sinks=n_sinks)
    if apophis_only:
        out["intrinsic_spin_standalone"] = _extract_mean_spin_period_hr(
            run_dir,
            prefix,
            apophis_sink_id,
            time_relation="all",
            skip_earliest_in_window=True,
            **grp_kw,
        )
    else:
        out["intrinsic_spin_standalone"] = _extract_mean_spin_period_hr(
            run_dir,
            prefix,
            apophis_sink_id,
            earth_sink_id=earth_sink_id,
            time_relation="intrinsic_early",
            skip_earliest_in_window=True,
            t_ca=t_ca,
            **grp_kw,
        )
        out["approach_spin_standalone"] = _extract_mean_spin_period_hr(
            run_dir,
            prefix,
            apophis_sink_id,
            earth_sink_id=earth_sink_id,
            time_relation="approach_pre_ca",
            skip_earliest_in_window=False,
            t_ca=t_ca,
            **grp_kw,
        )
        out["post_flyby_spin_standalone"] = _extract_mean_spin_period_hr(
            run_dir,
            prefix,
            apophis_sink_id,
            earth_sink_id=earth_sink_id,
            time_relation="after_ca",
            skip_earliest_in_window=False,
            t_ca=t_ca,
            **grp_kw,
        )
    return out


def _load_csv_expectations(csv_path: Path, run_dir: Path) -> Dict[str, float]:
    run_dir_s = str(run_dir.resolve())
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_dir", "").rstrip("/") == run_dir_s.rstrip("/"):
                out: Dict[str, float] = {}
                for key in (
                    "closest_approach_km",
                    "dispersion_ratio",
                    "unbound_fraction",
                    "intrinsic_spin_period_hr",
                    "approach_spin_period_hr",
                    "post_flyby_spin_period_hr",
                ):
                    val = row.get(key, "").strip()
                    if val:
                        out[key] = float(val)
                return out
    return {}


def _spin_grain_fractions_per_dump(
    run_dir: Path,
    prefix: str,
    apophis_sink_id: int,
    groups: Dict[str, np.ndarray],
    time_of_key: Dict[str, float],
) -> List[Tuple[float, float]]:
    """Per-dump (time_code, spin_grain_fraction) using the same masks as metric extraction."""
    utime = _parse_utime_from_phantom_log(run_dir / "phantom.log")
    if utime is None:
        return []
    rg_initial: Optional[float] = None
    out: List[Tuple[float, float]] = []
    for key in sorted(groups, key=lambda k: time_of_key[k]):
        arr = groups[key]
        if len(arr) < 2:
            continue
        pos = arr[:, :3]
        mass = arr[:, 3]
        vel = arr[:, 4:7]
        m_tot = float(mass.sum())
        if m_tot <= 0.0:
            continue
        rg = _mass_weighted_rg(pos, mass)
        if rg_initial is None:
            rg_initial = rg
            rg_ratio = 1.0
        elif rg_initial > 0.0:
            rg_ratio = rg / rg_initial
        else:
            rg_ratio = 1.0
        mb_all = _main_body_mask(pos, mass, rg_ratio=rg_ratio)
        rcom = (pos * mass[:, None]).sum(0) / m_tot
        dv = vel - (vel * mass[:, None]).sum(0) / m_tot
        dr = pos - rcom
        r = np.sqrt((dr * dr).sum(1))
        v2 = (dv * dv).sum(1)
        safe_r = np.where(r > 0.0, r, 1.0)
        eps = np.where(r > 0.0, 0.5 * v2 - m_tot / safe_r, -1.0)
        bound_mask = eps <= 0.0
        n_spin = int(mb_all[bound_mask].sum())
        out.append((time_of_key[key], n_spin / len(pos)))
    return out


def _print_dump_stats(
    run_dir: Path,
    prefix: str,
    apophis_sink_id: int,
    groups: Dict[str, np.ndarray],
    time_of_key: Dict[str, float],
) -> Dict[str, float]:
    fracs = _spin_grain_fractions_per_dump(
        run_dir, prefix, apophis_sink_id, groups, time_of_key
    )
    if not fracs:
        print("  dump-stats: (no dumps)")
        return {}
    utime = _parse_utime_from_phantom_log(run_dir / "phantom.log")
    values = np.asarray([f for _, f in fracs], dtype=np.float64)
    sample_idx = np.linspace(0, len(fracs) - 1, min(6, len(fracs)), dtype=int)
    print("  dump-stats spin_grain_fraction (sampled dumps):")
    for i in sample_idx:
        t, frac = fracs[int(i)]
        t_hr = t * utime / 3600.0 if utime else float("nan")
        print(f"    t={t_hr:.2f} hr  spin_N/N={frac:.3f}")
    stats = {
        "spin_grain_frac_min": float(values.min()),
        "spin_grain_frac_max": float(values.max()),
        "spin_grain_frac_median": float(np.median(values)),
    }
    print(
        f"  spin_grain_fraction min/median/max: "
        f"{stats['spin_grain_frac_min']:.3f} / "
        f"{stats['spin_grain_frac_median']:.3f} / "
        f"{stats['spin_grain_frac_max']:.3f}"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DEM metric extraction on a run dir.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sobol")
    parser.add_argument("--earth-sink-id", type=int, default=4)
    parser.add_argument("--apophis-sink-id", type=int, default=11)
    parser.add_argument("--apophis-only", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="sobol_mass_outputs.csv row to compare")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also run slow substep-dense metrics (METRICS_LEGACY_SUBSTEPS=1) for regression",
    )
    parser.add_argument(
        "--dump-stats",
        action="store_true",
        help="Print per-dump spin grain fraction audit (main-body mask after bound gate)",
    )
    parser.add_argument(
        "--expect-intact",
        action="store_true",
        help="Assert unbound_fraction < 0.05 and median spin grain fraction >= 0.80",
    )
    parser.add_argument(
        "--expect-unbound-min",
        type=float,
        default=None,
        help="Assert unbound_fraction >= this value (breakup regression runs)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 1

    apophis_only = args.apophis_only
    if not apophis_only:
        setup = run_dir / f"{args.prefix}.setup"
        if setup.is_file() and "apophis_only = T" in setup.read_text(encoding="utf-8", errors="replace"):
            apophis_only = True

    new = _metrics_for_run(
        run_dir,
        args.prefix,
        args.earth_sink_id,
        args.apophis_sink_id,
        apophis_only=apophis_only,
        legacy_substeps=False,
    )
    print(f"run_dir: {run_dir}")
    print(f"dump groups: {new.get('n_dump_groups')}  apophis sinks: {new.get('n_sinks')}")
    for key in (
        "closest_approach_km",
        "dispersion_ratio",
        "unbound_fraction",
        "intrinsic_spin_period_hr",
        "approach_spin_period_hr",
        "post_flyby_spin_period_hr",
    ):
        if key in new:
            print(f"  {key}: {new[key]:.12g}")

    if not _finite_close(new["dispersion_ratio"], new["dispersion_ratio_standalone"], 1e-12):
        print("ERROR: bundle vs standalone dispersion_ratio mismatch", file=sys.stderr)
        return 1
    if not _finite_close(new["unbound_fraction"], new["unbound_fraction_standalone"], 1e-12):
        print("ERROR: bundle vs standalone unbound_fraction mismatch", file=sys.stderr)
        return 1
    if not _finite_close(
        new["intrinsic_spin_period_hr"], new["intrinsic_spin_standalone"], 1e-12
    ):
        print("ERROR: bundle vs standalone intrinsic_spin mismatch", file=sys.stderr)
        return 1
    if not apophis_only:
        if not _finite_close(
            new["approach_spin_period_hr"], new["approach_spin_standalone"], 1e-12
        ):
            print("ERROR: bundle vs standalone approach_spin mismatch", file=sys.stderr)
            return 1
        if not _finite_close(
            new["post_flyby_spin_period_hr"], new["post_flyby_spin_standalone"], 1e-12
        ):
            print("ERROR: bundle vs standalone post_flyby_spin mismatch", file=sys.stderr)
            return 1
    print("OK: bundle matches standalone extractors")

    groups, time_of_key, _ = _apophis_time_groups(
        run_dir, args.prefix, args.apophis_sink_id
    )
    dump_stats: Dict[str, float] = {}
    if args.dump_stats or args.expect_intact:
        dump_stats = _print_dump_stats(
            run_dir, args.prefix, args.apophis_sink_id, groups, time_of_key
        )

    if args.expect_intact:
        unb = new.get("unbound_fraction", float("nan"))
        if not (math.isfinite(unb) and unb < 0.05):
            print(f"ERROR: --expect-intact unbound_fraction={unb} (want < 0.05)", file=sys.stderr)
            return 1
        med = dump_stats.get("spin_grain_frac_median", 0.0)
        if med < 0.80:
            print(
                f"ERROR: --expect-intact median spin_grain_fraction={med:.3f} (want >= 0.80)",
                file=sys.stderr,
            )
            return 1
        print("OK: --expect-intact passed")

    if args.expect_unbound_min is not None:
        unb = new.get("unbound_fraction", float("nan"))
        if not (math.isfinite(unb) and unb >= args.expect_unbound_min):
            print(
                f"ERROR: unbound_fraction={unb} (want >= {args.expect_unbound_min})",
                file=sys.stderr,
            )
            return 1
        print(f"OK: unbound_fraction >= {args.expect_unbound_min}")

    if args.csv:
        expected = _load_csv_expectations(args.csv.resolve(), run_dir)
        if not expected:
            print(f"WARN: no CSV row for {run_dir}", file=sys.stderr)
        else:
            tol = {
                "closest_approach_km": 1e-6,
                "dispersion_ratio": 1e-3,
                "unbound_fraction": 1e-3,
                "intrinsic_spin_period_hr": 0.02,
                "approach_spin_period_hr": 0.02,
                "post_flyby_spin_period_hr": 0.02,
            }
            for key, rtol in tol.items():
                if key not in expected or key not in new:
                    continue
                if not _finite_close(expected[key], new[key], rtol):
                    print(
                        f"WARN: {key} CSV {expected[key]:.12g} vs new {new[key]:.12g} "
                        f"(rtol={rtol})",
                        file=sys.stderr,
                    )
                else:
                    print(f"OK: {key} matches CSV within rtol={rtol}")

    if args.legacy:
        legacy = _metrics_for_run(
            run_dir,
            args.prefix,
            args.earth_sink_id,
            args.apophis_sink_id,
            apophis_only=apophis_only,
            legacy_substeps=True,
        )
        print(f"legacy substep groups: {legacy.get('n_dump_groups')}")
        print(
            f"  legacy dispersion_ratio: {legacy.get('dispersion_ratio', float('nan')):.12g}  "
            f"unbound: {legacy.get('unbound_fraction', float('nan')):.12g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
