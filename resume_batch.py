#!/usr/bin/env python3
"""Resume a partially completed sweep batch in an existing output directory.

Detects missing ``run_id`` rows in ``sobol_mass_outputs.csv`` (vs ``sobol_mass_samples.csv``),
re-runs only those cases with the same sample mapping, then merges into the summary CSV.

Usage (flyby_spin30 example — pass the same flags as the original campaign):

  OMP_NUM_THREADS=2 python3 sobol/resume_batch.py \\
    --batch-dir sobol_mass_runs/sobol_20260616_144429_flyby_spin30_torque_align_k1e7 \\
    --base-dir sobol --prefix sobol --num-samples 24 --seed 43 \\
    --use-dem-fixed true --np-apophis 500 --kc-fixed 1e7 --spin-period-fixed 30 \\
    --spin-torque-align-min 0 --spin-torque-align-max 180 \\
    --tmax-hours 108 --dtmax-hours 0.5 --ephemeris-cache-dir sobol --jobs 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import sys
from pathlib import Path
from typing import List, Set

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sobol.run_mass_sobol_phantom import (  # noqa: E402
    RunRecord,
    RunSample,
    RunWorkerPayload,
    _active_scale_variations,
    _apply_fixed_run_sample_overrides,
    _cleanup_run_dir,
    _execute_run_worker,
    _mass_bounds_active,
    _print_run_progress,
    build_parser,
    build_np_list_samples,
    build_np_spin_grid_samples,
    build_run_samples,
    preflight,
    sample_column_order,
    validate_args,
    write_summary_csv,
)


def _ffloat(val: str) -> float:
    val = (val or "").strip()
    return float(val) if val else float("nan")


def _load_ok_run_ids(summary_csv: Path) -> Set[int]:
    if not summary_csv.is_file():
        return set()
    ids: Set[int] = set()
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                ids.add(int(row["run_id"]))
    return ids


def _load_records(summary_csv: Path, col_order: List[str]) -> List[RunRecord]:
    if not summary_csv.is_file():
        return []
    secondary = [c for c in col_order if c != "mass_input_kg"]
    records: List[RunRecord] = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            param_columns = {c: row.get(c, "") for c in secondary}
            records.append(
                RunRecord(
                    run_id=int(row["run_id"]),
                    mass_input_kg=_ffloat(row.get("mass_input_kg", "")),
                    run_dir=row.get("run_dir", ""),
                    status=row.get("status", ""),
                    closest_approach_km=_ffloat(row.get("closest_approach_km", "")),
                    closest_approach_au=_ffloat(row.get("closest_approach_au", "")),
                    error=row.get("error", ""),
                    dispersion_ratio=_ffloat(row.get("dispersion_ratio", "")),
                    unbound_fraction=_ffloat(row.get("unbound_fraction", "")),
                    intrinsic_spin_period_hr=_ffloat(row.get("intrinsic_spin_period_hr", "")),
                    approach_spin_period_hr=_ffloat(row.get("approach_spin_period_hr", "")),
                    post_flyby_spin_period_hr=_ffloat(row.get("post_flyby_spin_period_hr", "")),
                    param_columns=param_columns,
                )
            )
    return records


def _load_samples_for_resume(
    samples_csv: Path, args: argparse.Namespace, n: int
) -> List[RunSample]:
    """Rebuild RunSample list from the batch samples CSV when possible (np_apophis_list batches)."""
    with samples_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n:
        return build_run_samples(n, args)

    if getattr(args, "np_apophis_list", None) is not None:
        if getattr(args, "spin_period_list", None):
            return build_np_spin_grid_samples(args)
        return build_np_list_samples(args)

    if rows and "np_apophis" in rows[0] and rows[0].get("np_apophis", "").strip():
        out: List[RunSample] = []
        for row in rows:
            spin_s = row.get("apophis_spin_period", "").strip()
            s = RunSample(
                np_apophis=int(row["np_apophis"]),
                apophis_spin_period=float(spin_s) if spin_s else None,
            )
            if args.use_dem_fixed is not None:
                s.use_dem = args.use_dem_fixed == "true"
            if args.use_shape_crop_fixed is not None:
                s.use_shape_crop = args.use_shape_crop_fixed == "true"
            if getattr(args, "apophis_only_fixed", None) is not None:
                s.apophis_only = args.apophis_only_fixed == "true"
            _apply_fixed_run_sample_overrides(s, args, np_val=int(row["np_apophis"]))
            if spin_s:
                s.apophis_spin_period = float(spin_s)
            out.append(s)
        return out

    return build_run_samples(n, args)


def main() -> int:
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Existing batch folder containing sobol_mass_samples.csv",
    )
    resume_parser.add_argument(
        "--run-ids",
        type=int,
        nargs="*",
        default=None,
        help="Explicit run IDs to execute (default: all missing ok rows)",
    )
    resume_args, remainder = resume_parser.parse_known_args()

    batch_dir = resume_args.batch_dir.resolve()
    if not batch_dir.is_dir():
        print(f"[ERROR] Not a directory: {batch_dir}", file=sys.stderr)
        return 1

    samples_csv = batch_dir / "sobol_mass_samples.csv"
    summary_csv = batch_dir / "sobol_mass_outputs.csv"
    if not samples_csv.is_file():
        print(f"[ERROR] Missing {samples_csv}", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(remainder)
    try:
        validate_args(args)
    except (ValueError, SystemExit) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    with samples_csv.open(newline="", encoding="utf-8") as f:
        expected_ids = [int(row["run_id"]) for row in csv.DictReader(f)]
    if args.num_samples != len(expected_ids):
        print(
            f"[WARN] --num-samples {args.num_samples} != {len(expected_ids)} rows in samples CSV",
            file=sys.stderr,
        )

    samples = _load_samples_for_resume(samples_csv, args, len(expected_ids))
    col_order = sample_column_order(args)
    existing = _load_records(summary_csv, col_order)
    done_ids = _load_ok_run_ids(summary_csv)

    if resume_args.run_ids:
        todo_ids = [rid for rid in resume_args.run_ids if rid in expected_ids]
    else:
        todo_ids = [rid for rid in expected_ids if rid not in done_ids]

    if not todo_ids:
        print(f"[INFO] Nothing to resume in {batch_dir} ({len(done_ids)} ok runs on disk).")
        return 0

    print(
        f"[INFO] Resuming {batch_dir.name}: {len(todo_ids)} run(s) to execute "
        f"({len(done_ids)} already ok)"
    )
    print(f"[INFO] Run IDs: {todo_ids}")

    base_dir = Path(args.base_dir).resolve()
    base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(args, base_dir, batch_dir)

    ephemeris_cache = None
    if args.ephemeris_cache_dir:
        ephemeris_cache = Path(args.ephemeris_cache_dir).resolve()

    ref_mass_kg = args.apophis_ref_mass_kg if _mass_bounds_active(args) else None
    literature_scale_r_allowed = "scale_r_apophis" not in {
        p for p, _, _, _ in _active_scale_variations(args)
    }

    payloads = [
        RunWorkerPayload(
            run_id=rid,
            sample=samples[rid - 1],
            base_setup=str(base_setup),
            base_input=str(base_input),
            output_root=str(batch_dir),
            prefix=args.prefix,
            phantomsetup_bin=str(phantomsetup_bin),
            phantom_bin=str(phantom_bin),
            ref_mass_kg=ref_mass_kg,
            dry_run=args.dry_run,
            earth_sink_id=args.sink_earth_id,
            apophis_sink_id=args.sink_apophis_id,
            ephemeris_cache_dir=str(ephemeris_cache) if ephemeris_cache else None,
            shape_file=args.shape_file if args.shape_file else None,
            literature_scale_r_allowed=literature_scale_r_allowed,
        )
        for rid in todo_ids
    ]

    total = len(expected_ids)
    new_results: List[RunRecord] = []
    if args.jobs == 1:
        for p in payloads:
            result = _execute_run_worker(p)
            new_results.append(result)
            _print_run_progress(result, samples, total)
            merged = {r.run_id: r for r in existing}
            for r in new_results:
                merged[r.run_id] = r
            write_summary_csv(summary_csv, list(merged.values()), col_order)
            print(
                f"[INFO] Partial results saved: {len(merged)}/{total} runs → {summary_csv}",
                flush=True,
            )
            if not args.no_cleanup and result.status == "ok" and result.run_dir:
                _cleanup_run_dir(Path(result.run_dir), p.prefix)
    else:
        payload_by_id = {p.run_id: p for p in payloads}
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_execute_run_worker, p) for p in payloads]
            merged = {r.run_id: r for r in existing}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                new_results.append(result)
                merged[result.run_id] = result
                _print_run_progress(result, samples, total)
                write_summary_csv(summary_csv, list(merged.values()), col_order)
                print(
                    f"[INFO] Partial results saved: {len(merged)}/{total} runs → {summary_csv}",
                    flush=True,
                )
                p = payload_by_id[result.run_id]
                if not args.no_cleanup and result.status == "ok" and result.run_dir:
                    _cleanup_run_dir(Path(result.run_dir), p.prefix)

    merged = {r.run_id: r for r in existing}
    for r in new_results:
        merged[r.run_id] = r
    write_summary_csv(summary_csv, list(merged.values()), col_order)
    n_ok = sum(1 for r in merged.values() if r.status == "ok")
    print(f"[INFO] Resume complete: {n_ok}/{total} ok → {summary_csv}")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
