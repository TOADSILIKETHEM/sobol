#!/usr/bin/env python3
"""Re-extract DEM spin metrics into sobol_mass_outputs.csv when run .ev files remain.

Only updates rows whose run directories still contain Apophis sink .ev files.
Batches with auto-cleanup (no .ev) must be re-run for new intrinsic_spin_period_hr values.

Usage:
  python3 sobol/reextract_spin_metrics.py --batch-dir sobol_mass_runs/<batch>
  python3 sobol/reextract_spin_metrics.py --batch-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sobol.run_mass_sobol_phantom import (  # noqa: E402
    APOPHIS_SINK_ID_DEFAULT,
    EARTH_SINK_ID_DEFAULT,
    _apophis_time_groups,
    _earth_apophis_closest_approach,
    _extract_dem_metrics_bundle,
    _use_fast_metrics_defaults,
)


def _run_has_ev(run_dir: Path, prefix: str, apophis_sink_id: int) -> bool:
    return any(run_dir.glob(f"{prefix}Sink{apophis_sink_id:04d}N*.ev"))


def _apophis_only_from_setup(run_dir: Path, prefix: str) -> bool:
    setup = run_dir / f"{prefix}.setup"
    if not setup.is_file():
        return False
    return "apophis_only = T" in setup.read_text(encoding="utf-8", errors="replace")


def reextract_batch(
    batch_dir: Path,
    *,
    prefix: str,
    earth_sink_id: int,
    apophis_sink_id: int,
    dry_run: bool,
) -> int:
    csv_path = batch_dir / "sobol_mass_outputs.csv"
    if not csv_path.is_file():
        print(f"ERROR: missing {csv_path}", file=sys.stderr)
        return 1

    _use_fast_metrics_defaults()
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    updated = 0
    skipped = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for col in (
            "intrinsic_spin_period_hr",
            "approach_spin_period_hr",
            "post_flyby_spin_period_hr",
        ):
            if col not in fieldnames:
                fieldnames.append(col)
        for row in reader:
            if row.get("status") != "ok":
                rows.append(row)
                continue
            run_dir = Path(row.get("run_dir", ""))
            if not run_dir.is_dir() or not _run_has_ev(run_dir, prefix, apophis_sink_id):
                skipped += 1
                rows.append(row)
                continue

            apophis_only = _apophis_only_from_setup(run_dir, prefix)
            groups, time_of_key, n_sinks = _apophis_time_groups(
                run_dir, prefix, apophis_sink_id
            )
            t_ca = None
            if not apophis_only:
                _, _, t_ca = _earth_apophis_closest_approach(
                    run_dir, prefix, earth_sink_id, apophis_sink_id
                )

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

            def _fmt(v: float) -> str:
                return f"{v:.12g}" if math.isfinite(v) else ""

            row["dispersion_ratio"] = _fmt(disp)
            row["unbound_fraction"] = _fmt(unbound)
            row["intrinsic_spin_period_hr"] = _fmt(intrinsic)
            row["approach_spin_period_hr"] = _fmt(approach)
            row["post_flyby_spin_period_hr"] = _fmt(post)
            rows.append(row)
            updated += 1
            print(
                f"run_{int(row['run_id']):04d}: intrinsic={intrinsic:.4g} hr "
                f"(apophis_only={apophis_only})"
            )

    if dry_run:
        print(f"dry-run: would update {updated} rows, skip {skipped} (no .ev)")
        return 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path} ({updated} updated, {skipped} skipped)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-extract spin metrics into batch CSV")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sobol")
    parser.add_argument("--earth-sink-id", type=int, default=EARTH_SINK_ID_DEFAULT)
    parser.add_argument("--apophis-sink-id", type=int, default=APOPHIS_SINK_ID_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return reextract_batch(
        args.batch_dir.resolve(),
        prefix=args.prefix,
        earth_sink_id=args.earth_sink_id,
        apophis_sink_id=args.apophis_sink_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
