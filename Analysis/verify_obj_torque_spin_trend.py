#!/usr/bin/env python3
"""Acceptance checks for OBJ torque-align tidal spin campaigns.

Usage:
    python3 sobol/Analysis/verify_obj_torque_spin_trend.py <batch_dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _f(val: str) -> float | None:
    s = (val or "").strip()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return None if x != x else x


def load_rows(batch_dir: Path) -> list[dict[str, float]]:
    csv_path = batch_dir / "sobol_mass_outputs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    out: list[dict[str, float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            a = _f(row.get("apophis_spin_torque_align_deg", ""))
            intr = _f(row.get("intrinsic_spin_period_hr", ""))
            app = _f(row.get("approach_spin_period_hr", ""))
            post = _f(row.get("post_flyby_spin_period_hr", ""))
            disp = _f(row.get("dispersion_ratio", ""))
            unb = _f(row.get("unbound_fraction", ""))
            if a is None or disp is None or unb is None:
                continue
            if intr is None or app is None or post is None:
                continue
            out.append(
                {
                    "align": a,
                    "intr": intr,
                    "app": app,
                    "post": post,
                    "disp": disp,
                    "unb": unb,
                    "post_app_pct": (post - app) / app * 100.0,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--min-intact", type=int, default=20)
    parser.add_argument("--max-intrinsic-spread-pct", type=float, default=8.0)
    parser.add_argument("--min-endpoint-delta-pct", type=float, default=0.3)
    args = parser.parse_args()

    rows = load_rows(args.batch_dir)
    if not rows:
        print(f"No complete spin rows in {args.batch_dir}", file=sys.stderr)
        return 1

    intact = [r for r in rows if r["disp"] < 1.15 and r["unb"] < 0.01]
    intrinsics = [r["intr"] for r in rows]
    spread_pct = (max(intrinsics) - min(intrinsics)) / min(intrinsics) * 100.0

    rows_sorted = sorted(rows, key=lambda r: r["align"])
    low = rows_sorted[0]["post_app_pct"]
    high = rows_sorted[-1]["post_app_pct"]
    endpoint_delta = abs(high - low)
    opposite_signs = (low < 0.0 and high > 0.0) or (low > 0.0 and high < 0.0)

    print(f"batch: {args.batch_dir}")
    print(f"rows with spin metrics: {len(rows)}")
    print(f"intact (disp<1.15, unb<1%): {len(intact)} / {len(rows)}")
    print(f"intrinsic spread: {spread_pct:.3f}%")
    print(f"post-approach at min angle ({rows_sorted[0]['align']:.1f}°): {low:+.3f}%")
    print(f"post-approach at max angle ({rows_sorted[-1]['align']:.1f}°): {high:+.3f}%")

    ok = True
    if len(intact) < args.min_intact:
        print(f"FAIL: intact count {len(intact)} < {args.min_intact}")
        ok = False
    if spread_pct >= args.max_intrinsic_spread_pct:
        print(f"FAIL: intrinsic spread {spread_pct:.3f}% >= {args.max_intrinsic_spread_pct}%")
        ok = False
    if endpoint_delta < args.min_endpoint_delta_pct:
        print(f"FAIL: endpoint |Δpost-approach| {endpoint_delta:.3f}% < {args.min_endpoint_delta_pct}%")
        ok = False
    if not opposite_signs:
        print("FAIL: post-approach endpoints do not have opposite signs")
        ok = False

    if ok:
        print("PASS: suitable for thesis tidal spin-orbit plot")
        return 0
    print("WARN: batch failed acceptance — adjust spin period or interpret with caveats")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
