#!/usr/bin/env python3
"""Compare DEM metric extraction on a completed run directory.

Usage:
  python3 sobol/verify_metric_extraction.py --run-dir sobol_mass_runs/.../run_0001
  python3 sobol/verify_metric_extraction.py --run-dir ... --prefix sobol --apophis-sink-id 11

Recomputes metrics with the current (dump-time) path and, unless --skip-legacy,
with METRICS_LEGACY_SUBSTEPS=1. Optionally compares to a row in sobol_mass_outputs.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sobol.run_mass_sobol_phantom import (  # noqa: E402
    _apophis_time_groups,
    _earth_apophis_closest_approach,
    _extract_dem_metrics_bundle,
    _extract_mean_spin_period_hr,
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
        os.environ.pop("METRICS_LEGACY_SUBSTEPS", None)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DEM metric extraction on a run dir.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sobol")
    parser.add_argument("--earth-sink-id", type=int, default=4)
    parser.add_argument("--apophis-sink-id", type=int, default=11)
    parser.add_argument("--apophis-only", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="sobol_mass_outputs.csv row to compare")
    parser.add_argument("--skip-legacy", action="store_true")
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

    if not args.skip_legacy:
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
