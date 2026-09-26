#!/usr/bin/env python3
"""Run Code/CSVconvert/DEMDumpConvert.py for one or more PHANTOM run directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DEM_DUMP_CONVERT = Path("/home/mboyle/Honours/Code/CSVconvert/DEMDumpConvert.py")
DEFAULT_BASE_OUT = Path(
    "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code/DEMCSVs"
)


def _min_dem_grains_from_setup(run_dir: Path) -> int | None:
    setup = run_dir / "sobol.setup"
    if not setup.is_file():
        return None
    for line in setup.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.split("!", 1)[0].strip()
        if not stripped.lower().startswith("np_apophis"):
            continue
        _, _, val = stripped.partition("=")
        val = val.strip()
        if not val:
            return None
        np_apophis = int(float(val))
        return max(10, np_apophis - max(50, np_apophis // 10))
    return None


def run_dem_dump_convert(
    input_dir: Path,
    dump_convert_path: Path,
    base_output_dir: Path,
    min_dem_grains: int | None,
) -> None:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    code_lines = dump_convert_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "INPUT_DIR": f'INPUT_DIR    = r"{input_dir}"',
        "BASE_OUTPUT_DIR": f'BASE_OUTPUT_DIR = r"{base_output_dir}"',
        "MIN_DEM_GRAINS": f"MIN_DEM_GRAINS = {min_dem_grains if min_dem_grains is not None else 'None'}",
    }
    for i, line in enumerate(code_lines):
        for key, new_line in replacements.items():
            if line.strip().startswith(key) and not line[:1].isspace():
                code_lines[i] = new_line
                break
    print(f"\n=== DEMDumpConvert: {input_dir.name} ===", flush=True)
    exec(compile("\n".join(code_lines), str(dump_convert_path), "exec"), {"__name__": "__main__"})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path, help="PHANTOM run directories with sobol_* dumps")
    p.add_argument("--dump-convert", type=Path, default=DEFAULT_DEM_DUMP_CONVERT)
    p.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUT)
    p.add_argument(
        "--min-dem-grains",
        type=int,
        default=None,
        help="Skip mini-dumps with fewer DEM sinks (default: auto from np_apophis in sobol.setup)",
    )
    args = p.parse_args()
    if not args.dump_convert.is_file():
        print(f"[ERROR] DEMDumpConvert.py not found: {args.dump_convert}", file=sys.stderr)
        return 1
    args.base_output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in args.run_dirs:
        min_grains = args.min_dem_grains
        if min_grains is None:
            min_grains = _min_dem_grains_from_setup(run_dir.resolve())
        run_dem_dump_convert(run_dir, args.dump_convert, args.base_output_dir, min_grains)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
