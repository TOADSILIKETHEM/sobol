#!/usr/bin/env python3
"""Sequential Sobol mass sweep runner for PHANTOM solarsystem runs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple


AU_IN_KM = 149_597_870.7
EARTH_SINK_ID = 4
APOPHIS_SINK_ID = 11


@dataclass
class RunRecord:
    run_id: int
    mass_input_kg: float
    run_dir: str
    status: str
    closest_approach_km: float
    closest_approach_au: float
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sobol mass samples and run PHANTOM sequentially."
    )
    parser.add_argument("--prefix", default="solarsystem1", help="PHANTOM file prefix.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing <prefix>.in and <prefix>.setup.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of Sobol samples to generate.",
    )
    parser.add_argument(
        "--mass-min-kg",
        type=float,
        default=1.0e10,
        help="Lower bound for Apophis mass in kg.",
    )
    parser.add_argument(
        "--mass-max-kg",
        type=float,
        default=1.0e11,
        help="Upper bound for Apophis mass in kg.",
    )
    parser.add_argument(
        "--mass-unit",
        choices=("kg", "g", "msun"),
        default="kg",
        help="Unit string written to m_apophis_in (default: kg).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scramble seed for Sobol sequence.",
    )
    parser.add_argument(
        "--output-root",
        default="sobol_mass_runs",
        help="Directory for run folders and summary outputs.",
    )
    parser.add_argument(
        "--phantom-dir",
        default=os.environ.get("PHANTOM_DIR", "/home/mboyle/phantom"),
        help="PHANTOM installation root containing bin/phantomsetup and bin/phantom.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run directories and setup files without executing PHANTOM.",
    )
    return parser.parse_args()


def sobol_1d_samples(n: int, seed: int) -> List[float]:
    """Generate 1D scrambled Sobol-like samples in [0, 1)."""
    if n <= 0:
        return []
    max_bits = max(1, math.ceil(math.log2(n + 1)))
    direction = [1 << (32 - i) for i in range(1, max_bits + 1)]

    # Simple digital shift scramble for reproducibility.
    scramble = (seed * 2654435761) & 0xFFFFFFFF
    x = scramble
    out: List[float] = []
    for i in range(1, n + 1):
        c = (i & -i).bit_length() - 1
        x ^= direction[c]
        out.append((x & 0xFFFFFFFF) / 2**32)
    return out


def generate_sobol_masses(num_samples: int, mmin: float, mmax: float, seed: int) -> List[float]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if mmin <= 0 or mmax <= 0 or mmax <= mmin:
        raise ValueError("mass bounds must satisfy 0 < min < max")

    unit_samples = sobol_1d_samples(num_samples, seed)
    return [mmin + x * (mmax - mmin) for x in unit_samples]


def format_mass_token(mass_kg: float, mass_unit: str) -> str:
    if mass_unit == "kg":
        return f"{(mass_kg * 1.0e3):.10g}*g"
    if mass_unit == "g":
        return f"{(mass_kg * 1.0e3):.10g}*g"
    if mass_unit == "msun":
        # 2018 CODATA-compatible value used for deterministic conversion.
        return f"{(mass_kg / 1.98847e30):.10g}*msun"
    raise ValueError(f"Unsupported mass unit: {mass_unit}")


def validate_mass_line(setup_text: str, expected_mass_token: str) -> None:
    pattern = re.compile(r"^\s*m_apophis_in\s*=\s*([^!]*?)\s*(?:!.*)?$", re.MULTILINE)
    match = pattern.search(setup_text)
    if not match:
        raise RuntimeError("m_apophis_in key missing after setup update")
    assigned = match.group(1).strip()
    if not assigned:
        raise RuntimeError("m_apophis_in is empty after setup update")
    if assigned != expected_mass_token:
        raise RuntimeError(
            f"m_apophis_in mismatch after setup update (expected '{expected_mass_token}', got '{assigned}')"
        )


def replace_mass_in_setup(setup_path: Path, mass_kg: float, mass_unit: str) -> None:
    setup_text = setup_path.read_text(encoding="utf-8")
    mass_token = format_mass_token(mass_kg, mass_unit)
    pattern = re.compile(r"^(\s*m_apophis_in\s*=\s*)([^!]*)(!.*)?$", re.MULTILINE)
    match = pattern.search(setup_text)
    if not match:
        raise RuntimeError(f"m_apophis_in key not found in setup file: {setup_path}")
    comment = match.group(3) if match.group(3) is not None else ""
    updated_line = f"{match.group(1)}{mass_token}"
    if comment:
        updated_line += f" {comment.strip()}"
    setup_text = pattern.sub(updated_line, setup_text, count=1)
    validate_mass_line(setup_text, mass_token)
    setup_path.write_text(setup_text, encoding="utf-8")


def run_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_sink_rows(path: Path) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    rows_t: List[float] = []
    rows_xyz: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                continue
            rows_t.append(t)
            rows_xyz.append((x, y, z))
    if not rows_t:
        raise RuntimeError(f"No numeric sink rows found in {path}")
    return rows_t, rows_xyz


def extract_closest_approach(run_dir: Path, prefix: str) -> Tuple[float, float]:
    earth_candidates = sorted(run_dir.glob(f"{prefix}Sink{EARTH_SINK_ID:04d}N*.ev"))
    apophis_candidates = sorted(run_dir.glob(f"{prefix}Sink{APOPHIS_SINK_ID:04d}N*.ev"))

    if not earth_candidates or not apophis_candidates:
        raise RuntimeError(
            "Could not find Earth/Apophis sink files "
            f"(expected IDs {EARTH_SINK_ID} and {APOPHIS_SINK_ID})."
        )

    t_earth, xyz_earth = parse_sink_rows(earth_candidates[-1])
    t_apophis, xyz_apophis = parse_sink_rows(apophis_candidates[-1])

    n = min(len(t_earth), len(t_apophis))
    if n == 0:
        raise RuntimeError("Sink files have no overlapping rows.")

    closest_km = float("inf")
    for idx in range(n):
        dx = xyz_apophis[idx][0] - xyz_earth[idx][0]
        dy = xyz_apophis[idx][1] - xyz_earth[idx][1]
        dz = xyz_apophis[idx][2] - xyz_earth[idx][2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < closest_km:
            closest_km = d

    if not math.isfinite(closest_km):
        raise RuntimeError("Failed to compute closest approach from sink rows.")
    return closest_km, closest_km / AU_IN_KM


def preflight(args: argparse.Namespace, base_dir: Path, output_root: Path) -> Tuple[Path, Path, Path, Path]:
    base_setup = base_dir / f"{args.prefix}.setup"
    base_input = base_dir / f"{args.prefix}.in"
    if not base_setup.is_file():
        raise FileNotFoundError(f"Missing setup file: {base_setup}")
    if not base_input.is_file():
        raise FileNotFoundError(f"Missing input file: {base_input}")

    phantomsetup_bin = Path(args.phantom_dir) / "bin" / "phantomsetup"
    phantom_bin = Path(args.phantom_dir) / "bin" / "phantom"
    if not args.dry_run:
        if not phantomsetup_bin.is_file():
            raise FileNotFoundError(f"Missing PHANTOM binary: {phantomsetup_bin}")
        if not phantom_bin.is_file():
            raise FileNotFoundError(f"Missing PHANTOM binary: {phantom_bin}")

    output_root.mkdir(parents=True, exist_ok=True)
    return base_setup, base_input, phantomsetup_bin, phantom_bin


def write_samples_csv(path: Path, masses: List[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "mass_kg"])
        for idx, mass in enumerate(masses, start=1):
            writer.writerow([idx, f"{mass:.12g}"])


def append_summary(path: Path, row: RunRecord, write_header: bool) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "run_id",
                    "mass_input_kg",
                    "run_dir",
                    "status",
                    "closest_approach_km",
                    "closest_approach_au",
                    "error",
                ]
            )
        writer.writerow(
            [
                row.run_id,
                f"{row.mass_input_kg:.12g}",
                row.run_dir,
                row.status,
                f"{row.closest_approach_km:.12g}" if not math.isnan(row.closest_approach_km) else "",
                f"{row.closest_approach_au:.12g}" if not math.isnan(row.closest_approach_au) else "",
                row.error,
            ]
        )


def run_one_case(
    run_id: int,
    mass_kg: float,
    base_setup: Path,
    base_input: Path,
    output_root: Path,
    prefix: str,
    phantomsetup_bin: Path,
    phantom_bin: Path,
    mass_unit: str,
    dry_run: bool,
) -> RunRecord:
    run_dir = output_root / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_setup = run_dir / f"{prefix}.setup"
    run_input = run_dir / f"{prefix}.in"
    shutil.copy2(base_setup, run_setup)
    shutil.copy2(base_input, run_input)
    replace_mass_in_setup(run_setup, mass_kg, mass_unit)

    if dry_run:
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_kg,
            run_dir=str(run_dir),
            status="prepared_only",
            closest_approach_km=float("nan"),
            closest_approach_au=float("nan"),
            error="",
        )

    try:
        run_command([str(phantomsetup_bin), prefix], cwd=run_dir, log_path=run_dir / "setup.log")
        run_command([str(phantom_bin), f"{prefix}.in"], cwd=run_dir, log_path=run_dir / "phantom.log")
        closest_km, closest_au = extract_closest_approach(run_dir, prefix)
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_kg,
            run_dir=str(run_dir),
            status="ok",
            closest_approach_km=closest_km,
            closest_approach_au=closest_au,
            error="",
        )
    except Exception as exc:  # pragma: no cover - runtime path
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_kg,
            run_dir=str(run_dir),
            status="failed",
            closest_approach_km=float("nan"),
            closest_approach_au=float("nan"),
            error=str(exc),
        )


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (Path(args.output_root).resolve() / f"{args.prefix}_{timestamp}")

    try:
        masses = generate_sobol_masses(args.num_samples, args.mass_min_kg, args.mass_max_kg, args.seed)
        base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(args, base_dir, output_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    samples_csv = output_root / "sobol_mass_samples.csv"
    summary_csv = output_root / "sobol_mass_outputs.csv"
    write_samples_csv(samples_csv, masses)
    print(f"[INFO] Wrote Sobol sample table: {samples_csv}")

    for idx, mass in enumerate(masses, start=1):
        print(f"[INFO] Run {idx}/{len(masses)} mass={mass:.6e} kg")
        result = run_one_case(
            run_id=idx,
            mass_kg=float(mass),
            base_setup=base_setup,
            base_input=base_input,
            output_root=output_root,
            prefix=args.prefix,
            phantomsetup_bin=phantomsetup_bin,
            phantom_bin=phantom_bin,
            mass_unit=args.mass_unit,
            dry_run=args.dry_run,
        )
        append_summary(summary_csv, result, write_header=(idx == 1))
        print(f"[INFO]   status={result.status} run_dir={result.run_dir}")

    print(f"[INFO] Summary written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
