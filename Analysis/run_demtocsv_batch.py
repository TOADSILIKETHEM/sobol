#!/usr/bin/env python3
"""Run Windows CSVconvert/DEMtoCSV.py for one or more PHANTOM run directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DEMTOCSV = Path(
    "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/CSVconvert/DEMtoCSV.py"
)
DEFAULT_BASE_OUT = Path(
    "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/DEMCSVs"
)


def run_demtocsv(
    input_dir: Path,
    demtocsv_path: Path,
    base_output_dir: Path,
    min_dem_grains: int,
) -> None:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    code_lines = demtocsv_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "INPUT_DIR": f'INPUT_DIR    = r"{input_dir}"',
        "BASE_OUTPUT_DIR": f'BASE_OUTPUT_DIR = r"{base_output_dir}"',
        "MIN_DEM_GRAINS": f"MIN_DEM_GRAINS = {min_dem_grains}",
    }
    for i, line in enumerate(code_lines):
        for key, new_line in replacements.items():
            if line.strip().startswith(key):
                code_lines[i] = new_line
                break
    print(f"\n=== DEMtoCSV: {input_dir.name} ===", flush=True)
    exec(compile("\n".join(code_lines), str(demtocsv_path), "exec"), {"__name__": "__main__"})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path, help="PHANTOM run directories with sobol_* dumps")
    p.add_argument("--demtocsv", type=Path, default=DEFAULT_DEMTOCSV)
    p.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUT)
    p.add_argument(
        "--min-dem-grains",
        type=int,
        default=60,
        help="Skip mini-dumps with fewer DEM sinks (default 60; use ~450 for np_apophis=500)",
    )
    args = p.parse_args()
    if not args.demtocsv.is_file():
        print(f"[ERROR] DEMtoCSV.py not found: {args.demtocsv}", file=sys.stderr)
        return 1
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in args.run_dirs:
        run_demtocsv(run_dir, args.demtocsv, args.base_output_dir, args.min_dem_grains)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
