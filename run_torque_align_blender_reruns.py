#!/usr/bin/env python3
"""Re-run the two torque-align extremes (near +h and near -h) with dumps kept for Blender."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Repo root = parent of sobol/
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "sobol") not in sys.path:
    sys.path.insert(0, str(_REPO / "sobol"))

from run_mass_sobol_phantom import (  # noqa: E402
    RunRecord,
    RunSample,
    preflight,
    resolve_default_shape_file,
    run_one_case,
    sample_column_order,
    write_samples_csv,
    write_summary_csv,
)

# Original batch sobol_20260601_140230_flyby_spin_torque_align_k1e7 runs 4 and 19.
CASES: Tuple[Tuple[str, float], ...] = (
    ("aligned_near_h", 11.1870651506),
    ("opposite_near_h", 177.504590414),
)

SPIN_PERIOD_HR = 1.52
KT_CGS = 1e7
DN_COHES = 0.1
NP_APOPHIS = 500
# Default template uses 30 min; fine Blender re-runs use 5 min (~6× more keyframes).
DTMAX_HOURS_FINE = 5.0 / 60.0
OPPOSITE_ALIGN_DEG = 177.504590414


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=_REPO / "sobol")
    p.add_argument("--output-root", type=Path, default=_REPO / "sobol_mass_runs")
    p.add_argument(
        "--batch-label",
        default="torque_align_blender_vis",
        help="Subfolder name under output-root (default: torque_align_blender_vis).",
    )
    p.add_argument("--prefix", default="sobol")
    p.add_argument("--phantom-dir", type=Path, default=_REPO / "sobol")
    p.add_argument("--ephemeris-cache-dir", type=Path, default=_REPO / "sobol")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--use-shape-crop",
        action="store_true",
        help="Enable literature Apophis mesh cropping (Shapes/apophis.shape + scale_r_apophis).",
    )
    p.add_argument(
        "--shape-file",
        type=Path,
        default=None,
        help="Shape config or OBJ (default: Shapes/apophis.shape when --use-shape-crop).",
    )
    p.add_argument(
        "--dtmax-hours",
        type=float,
        default=None,
        help="Dump interval in hours (e.g. 0.083333 for 5 min). Default: template (30 min).",
    )
    p.add_argument(
        "--case",
        choices=("both", "aligned", "opposite"),
        default="both",
        help="Which torque-align case(s) to run (default: both).",
    )
    p.add_argument(
        "--spin-period",
        type=float,
        default=None,
        metavar="HR",
        help=f"Apophis spin period in hours (default: {SPIN_PERIOD_HR}).",
    )
    p.add_argument(
        "--np-apophis",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of DEM particles (default: {NP_APOPHIS}).",
    )
    return p


def _select_cases(which: str) -> Tuple[Tuple[str, float], ...]:
    if which == "both":
        return CASES
    if which == "aligned":
        return (CASES[0],)
    return (CASES[1],)


def main() -> int:
    args = build_parser().parse_args()
    base_dir = args.base_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() / f"sobol_{timestamp}_{args.batch_label}"
    output_root.mkdir(parents=True, exist_ok=True)

    shape_path: Optional[Path] = None
    if args.use_shape_crop:
        shape_path = (
            args.shape_file.resolve()
            if args.shape_file is not None
            else resolve_default_shape_file()
        )
        if not shape_path.is_file():
            raise FileNotFoundError(f"shape file not found: {shape_path}")

    spin_period = args.spin_period if args.spin_period is not None else SPIN_PERIOD_HR
    np_apophis = args.np_apophis if args.np_apophis is not None else NP_APOPHIS

    cases = _select_cases(args.case)
    samples: List[RunSample] = []
    for _label, align_deg in cases:
        samples.append(
            RunSample(
                use_dem=True,
                np_apophis=np_apophis,
                apophis_spin_period=spin_period,
                apophis_spin_torque_align_deg=align_deg,
                kt_cgs=KT_CGS,
                coh_gap_max_cgs=runner.coh_gap_max_cgs_from_dn(dn=DN_COHES, np_apophis=np_apophis),
                use_shape_crop=True if args.use_shape_crop else None,
                dtmax_hours=args.dtmax_hours,
            )
        )

    col_order = ["apophis_spin_torque_align_deg", "apophis_spin_period", "np_apophis"]
    if args.dtmax_hours is not None:
        col_order.append("dtmax_hours")
    if args.use_shape_crop:
        col_order.extend(["use_shape_crop", "scale_r_apophis"])
    write_samples_csv(output_root / "sobol_mass_samples.csv", samples, col_order)
    print(f"[INFO] Output directory: {output_root}")
    print(f"[INFO] Cases: {', '.join(f'{lab} ({deg:.4g}°)' for lab, deg in cases)}")
    print(f"[INFO] spin_period = {spin_period:.4g} hr, np_apophis = {np_apophis}")
    if args.dtmax_hours is not None:
        print(f"[INFO] dtmax_in = {args.dtmax_hours:.6g} hr ({args.dtmax_hours * 60:.4g} min between dumps)")
    else:
        print("[INFO] dtmax_in: template default (30 min)")
    if args.use_shape_crop:
        print(f"[INFO] Shape crop: {shape_path}")
    else:
        print("[INFO] Shape crop: off (sphere lattice)")

    base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(
        argparse.Namespace(
            prefix=args.prefix,
            phantom_dir=str(args.phantom_dir),
            dry_run=args.dry_run,
        ),
        base_dir,
        output_root,
    )
    ephem = args.ephemeris_cache_dir.resolve() if args.ephemeris_cache_dir else None

    results: List[RunRecord] = []
    n_cases = len(cases)
    for run_id, (label, _deg) in enumerate(cases, start=1):
        print(f"[INFO] Run {run_id}/{n_cases} ({label}) — no post-run cleanup (dumps kept)", flush=True)
        # run_one_case uses --maxp=4000 when use_shape_crop (mesh lattice >2000 sites).
        rec = run_one_case(
            run_id=run_id,
            sample=samples[run_id - 1],
            base_setup=base_setup,
            base_input=base_input,
            output_root=output_root,
            prefix=args.prefix,
            phantomsetup_bin=phantomsetup_bin,
            phantom_bin=phantom_bin,
            ref_mass_kg=None,
            dry_run=args.dry_run,
            earth_sink_id=4,
            apophis_sink_id=11,
            ephemeris_cache_dir=ephem,
            shape_file=str(shape_path) if shape_path else None,
        )
        results.append(rec)
        print(f"[INFO]   status={rec.status} run_dir={rec.run_dir}", flush=True)
        if rec.status == "ok" and not args.dry_run:
            n_dumps = len(list(Path(rec.run_dir).glob(f"{args.prefix}_*")))
            print(f"[INFO]   kept {n_dumps} binary dump file(s) for Blender", flush=True)

    write_summary_csv(output_root / "sobol_mass_outputs.csv", results, col_order)
    manifest = output_root / "README_blender.txt"
    manifest.write_text(
        "\n".join(
            [
                "Torque-align Blender re-runs (original batch runs 4 and 19).",
                f"spin_period = {spin_period} hr, kt_cgs = {KT_CGS:g}, np_apophis = {np_apophis}",
                f"shape_crop = {args.use_shape_crop}"
                + (f", shape_file = {shape_path.name}" if shape_path else ""),
                f"case filter = {args.case}",
                f"dtmax_hours = {args.dtmax_hours}" if args.dtmax_hours is not None else "dtmax_hours = template (30 min)",
                "Post-run cleanup disabled — sobol_* dumps and *.ev files are kept.",
                "",
                "run_0001  aligned_near_h   (~11° from +h, intact in original batch)",
                "run_0002  opposite_near_h  (~177° from +h, breakup in original batch)",
                "(When --case opposite only, sole run is run_0001 labelled opposite_near_h.)",
                "",
                "Point Windows DEMtoCSV.py / Blender scripts at this directory.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Summary: {output_root / 'sobol_mass_outputs.csv'}")
    print(f"[INFO] Manifest: {manifest}")
    failed = [r for r in results if r.status != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
