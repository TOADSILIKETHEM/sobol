#!/usr/bin/env python3
"""Check whether intrinsic_spin_period_hr spread collapsed after the sphere
lattice principal-axis fix (docs/SPHERE_LATTICE_FIX.md).

Usage:
    python3 sobol/Analysis/verify_spin_lattice_fix.py <batch_dir> [--input-period-hr P]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_intrinsic_periods(batch_dir: Path) -> list[tuple[float, float]]:
    """Return (torque_align_deg, intrinsic_spin_period_hr) pairs for a batch."""
    csv_path = batch_dir / "sobol_mass_outputs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"no sobol_mass_outputs.csv under {batch_dir}")
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            angle_s = row.get("apophis_spin_torque_align_deg", "")
            period_s = row.get("intrinsic_spin_period_hr", "")
            if not angle_s or not period_s:
                continue
            try:
                angle = float(angle_s)
                period = float(period_s)
            except ValueError:
                continue
            if period != period:  # NaN
                continue
            rows.append((angle, period))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--input-period-hr", type=float, default=None,
                         help="Fixed apophis_spin_period used for the batch "
                              "(if omitted, uses the median observed period).")
    args = parser.parse_args()

    rows = load_intrinsic_periods(args.batch_dir)
    if not rows:
        print(f"No intrinsic_spin_period_hr rows found in {args.batch_dir}", file=sys.stderr)
        return 1

    periods = sorted(p for _, p in rows)
    n = len(periods)
    median = periods[n // 2] if n % 2 else 0.5 * (periods[n // 2 - 1] + periods[n // 2])
    input_period = args.input_period_hr if args.input_period_hr is not None else median

    max_dev_pct = max(abs(p - input_period) / input_period * 100.0 for _, p in rows)
    spread_pct = (max(periods) - min(periods)) / input_period * 100.0

    print(f"batch: {args.batch_dir}")
    print(f"runs with intrinsic_spin_period_hr: {n}")
    print(f"input period (hr): {input_period:.6g}")
    print(f"observed range (hr): {periods[0]:.6f} .. {periods[-1]:.6f}")
    print(f"peak-to-peak spread: {spread_pct:.3f}% of input period")
    print(f"max |deviation| from input: {max_dev_pct:.3f}%")

    if spread_pct < 3.0:
        print("PASS: spread well below the pre-fix ~3% baseline "
              "(docs/SPHERE_LATTICE_FIX.md acceptance criterion).")
    else:
        print("WARN: spread is not below the pre-fix ~3% baseline — "
              "investigate before treating this batch as fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
