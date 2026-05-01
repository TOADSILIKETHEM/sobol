#!/usr/bin/env python3
"""Sequential Sobol parameter sweep runner for PHANTOM solarsystem-style setups (mass + optional scales)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


AU_IN_KM = 149_597_870.7
EARTH_SINK_ID_DEFAULT = 4
APOPHIS_SINK_ID_DEFAULT = 11

# Auto-generated batch folder suffix (after "{prefix}_{timestamp}_"); keep paths portable.
_BATCH_SLUG_DEFAULT_MAX_LEN = 120
_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Scale parameters varied via CLI min/max: (RunSample/.setup attribute, argparse lo/hi attrs, batch slug token).
# Order MUST match Sobol dimension ordering used in build_run_samples and CSV columns.
_SCALE_VARIATION_SPEC: Tuple[Tuple[str, str, str, str], ...] = (
    ("scale_vel", "scale_vel_min", "scale_vel_max", "sv"),
    ("scale_pos", "scale_pos_min", "scale_pos_max", "sp"),
    ("scale_r_apophis", "scale_r_apophis_min", "scale_r_apophis_max", "sra"),
    ("scale_rho", "scale_rho_min", "scale_rho_max", "srho"),
)


def _active_scale_variations(args: argparse.Namespace) -> List[Tuple[str, float, float, str]]:
    """Scale dimensions that are active (both bounds set), in canonical order."""
    out: List[Tuple[str, float, float, str]] = []
    for param, lo_attr, hi_attr, slug_tok in _SCALE_VARIATION_SPEC:
        lo = getattr(args, lo_attr)
        hi = getattr(args, hi_attr)
        if lo is not None and hi is not None:
            out.append((param, lo, hi, slug_tok))
    return out


# ---------------------------------------------------------------------------
# SciPy multi-dimensional Sobol (preferred) vs independent 1D fallback
# ---------------------------------------------------------------------------
def _sobol_nd_scipy(n: int, dim: int, seed: int) -> List[List[float]]:
    from scipy.stats import qmc

    engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
    raw = engine.random(n=n)
    # scipy returns [0,1); keep as list of rows
    return [list(map(float, row)) for row in raw]


def _sobol_nd_fallback(n: int, dim: int, seed: int) -> List[List[float]]:
    """
    Independent 1D quasi-random sequences per dimension (different seeds).
    Joint space coverage is weaker than true multi-d Sobol; install scipy for better designs.
    """
    cols: List[List[float]] = []
    for d in range(dim):
        cols.append(sobol_1d_samples(n, seed + 1_000_003 * (d + 1)))
    return [[cols[d][i] for d in range(dim)] for i in range(n)]


def sobol_nd_unit_samples(n: int, dim: int, seed: int) -> List[List[float]]:
    if n <= 0:
        return []
    if dim <= 0:
        return [[] for _ in range(n)]
    try:
        return _sobol_nd_scipy(n, dim, seed)
    except (ImportError, ValueError) as exc:
        # SciPy missing or Sobol engine rejected parameters — use weaker independent 1D fallback.
        label = "SciPy not installed" if isinstance(exc, ImportError) else f"Sobol engine: {exc}"
        print(f"[WARN] {label}; using independent 1D quasi-random fallback.", file=sys.stderr)
        return _sobol_nd_fallback(n, dim, seed)


@dataclass
class RunRecord:
    run_id: int
    mass_input_kg: float
    run_dir: str
    status: str
    closest_approach_km: float
    closest_approach_au: float
    error: str
    # Values written for CSV secondary columns (setup-related names e.g. scale_vel; mass uses mass_input_kg).
    param_columns: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunSample:
    """Per-run values: None means leave the copied template unchanged for that key."""

    mass_kg: Optional[float] = None
    scale_vel: Optional[float] = None
    scale_pos: Optional[float] = None
    scale_r_apophis: Optional[float] = None
    scale_rho: Optional[float] = None
    use_dem: Optional[bool] = None
    apophis_only: Optional[bool] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Sobol samples over Apophis mass and optional setup_solarsystem.f90 "
            "parameters, then run PHANTOM sequentially. Multi-d Sobol uses scipy when installed; "
            "otherwise independent 1D sequences per dimension are used (weaker joint coverage — "
            "see pip install -r requirements.txt)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prefix", default="sobol", help="PHANTOM file prefix.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing <prefix>.in and <prefix>.setup.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of Sobol samples (runs) to generate.",
    )
    parser.add_argument(
        "--no-vary-mass",
        action="store_true",
        help="Do not sample or overwrite m_apophis_in; template .setup mass line is kept.",
    )
    parser.add_argument(
        "--mass-min-kg",
        type=float,
        default=1.0e10,
        help="Lower bound for Apophis mass in kg (when mass is varied).",
    )
    parser.add_argument(
        "--mass-max-kg",
        type=float,
        default=1.0e11,
        help="Upper bound for Apophis mass in kg (when mass is varied).",
    )
    parser.add_argument(
        "--mass-unit",
        choices=("kg", "g", "msun"),
        default="kg",
        help=(
            "Unit token written to m_apophis_in. Sample mass is always drawn in kg; "
            "'kg' and 'g' both convert that mass to grams for *g in the setup file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scramble seed for Sobol sequence.",
    )
    # Real scalings (optional Sobol dimensions when both min and max are set)
    for key, helpt in (
        ("scale-vel", "scale_vel (Apophis velocity scale)"),
        ("scale-pos", "scale_pos (Apophis initial position scale)"),
        ("scale-r-apophis", "scale_r_apophis (Apophis radius scale)"),
        ("scale-rho", "scale_rho (bulk density scale when mass is density-derived)"),
    ):
        parser.add_argument(
            f"--{key}-min",
            type=float,
            default=None,
            help=f"Lower bound for {helpt}; omit with max to leave unvaried.",
        )
        parser.add_argument(
            f"--{key}-max",
            type=float,
            default=None,
            help=f"Upper bound for {helpt}; omit with min to leave unvaried.",
        )
    parser.add_argument(
        "--vary-use-dem",
        action="store_true",
        help="Sample use_dem (logical) with one Sobol dimension (u>=0.5 -> T).",
    )
    parser.add_argument(
        "--vary-apophis-only",
        action="store_true",
        help=(
            "Sample apophis_only with one Sobol dimension. When True for a run, Earth is not in "
            "the setup and closest-approach extraction is skipped (NaN). See also --sink-*-id."
        ),
    )
    parser.add_argument(
        "--sink-earth-id",
        type=int,
        default=EARTH_SINK_ID_DEFAULT,
        help="Sink index for Earth in .ev filenames (default matches full solar system).",
    )
    parser.add_argument(
        "--sink-apophis-id",
        type=int,
        default=APOPHIS_SINK_ID_DEFAULT,
        help="Sink index for Apophis in .ev filenames (default matches full solar system).",
    )
    parser.add_argument(
        "--output-root",
        default="sobol_mass_runs",
        help="Directory for run folders and summary outputs.",
    )
    parser.add_argument(
        "--batch-label",
        default=None,
        metavar="TEXT",
        help=(
            "If set, use this string as the sweep suffix (sanitized) instead of the auto slug "
            "built from varied parameters and bounds. Batch folder: <prefix>_<timestamp>_<suffix>."
        ),
    )
    parser.add_argument(
        "--batch-slug-max-len",
        type=int,
        default=_BATCH_SLUG_DEFAULT_MAX_LEN,
        metavar="N",
        help=(
            "Maximum length of the sweep suffix for auto slugs (and for --batch-label after "
            "sanitization); longer values are truncated with a short hash appended."
        ),
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


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("num-samples must be >= 1")
    if args.batch_slug_max_len < 9:
        raise ValueError("batch-slug-max-len must be >= 9 (room for hash suffix when truncating)")
    if not args.no_vary_mass:
        if args.mass_min_kg <= 0 or args.mass_max_kg <= 0 or args.mass_max_kg <= args.mass_min_kg:
            raise ValueError("mass bounds must satisfy 0 < mass-min-kg < mass-max-kg (or use --no-vary-mass)")
    for param, lo_attr, hi_attr, _ in _SCALE_VARIATION_SPEC:
        lo = getattr(args, lo_attr)
        hi = getattr(args, hi_attr)
        if (lo is None) ^ (hi is None):
            raise ValueError(f"{param}: set both min and max, or neither")
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"{param}: require min < max")

    dim = count_dimensions(args)
    if dim == 0:
        raise ValueError(
            "No varying dimensions: enable mass (--no-vary-mass off) and/or pass scale */ "
            "vary-use-dem / vary-apophis-only."
        )


def count_dimensions(args: argparse.Namespace) -> int:
    n = 0
    if not args.no_vary_mass:
        n += 1
    n += len(_active_scale_variations(args))
    if args.vary_use_dem:
        n += 1
    if args.vary_apophis_only:
        n += 1
    return n


def sobol_1d_samples(n: int, seed: int) -> List[float]:
    """Generate 1D scrambled Sobol-like samples in [0, 1)."""
    if n <= 0:
        return []
    max_bits = max(1, math.ceil(math.log2(n + 1)))
    direction = [1 << (32 - i) for i in range(1, max_bits + 1)]

    scramble = (seed * 2654435761) & 0xFFFFFFFF
    x = scramble
    out: List[float] = []
    for i in range(1, n + 1):
        c = (i & -i).bit_length() - 1
        x ^= direction[c]
        out.append((x & 0xFFFFFFFF) / 2**32)
    return out


def format_mass_token(mass_kg: float, mass_unit: str) -> str:
    # Internal masses are always kg from Sobol sampling; *g line uses grams.
    if mass_unit in ("kg", "g"):
        return f"{(mass_kg * 1.0e3):.10g}*g"
    if mass_unit == "msun":
        return f"{(mass_kg / 1.98847e30):.10g}*msun"
    raise ValueError(f"Unsupported mass unit: {mass_unit}")


def format_logical_token(val: bool) -> str:
    """Right-padded T/F for list-directed Fortran logical reads in .setup files."""
    ch = "T" if val else "F"
    return f"{ch:>10}"


def format_real_token(val: float) -> str:
    return f"{val:.10g}"


def replace_setup_assignment(setup_text: str, key: str, value_str: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^!]*)(!.*)?$", re.MULTILINE)
    matches = list(pattern.finditer(setup_text))
    if not matches:
        raise RuntimeError(f"Setup key {key!r} not found in .setup file")
    if len(matches) > 1:
        raise RuntimeError(
            f"Setup key {key!r} appears on {len(matches)} lines; expected exactly one assignment."
        )
    match = matches[0]
    comment = match.group(3) if match.group(3) is not None else ""
    updated_line = f"{match.group(1)}{value_str}"
    if comment:
        updated_line += f" {comment.strip()}"
    return pattern.sub(updated_line, setup_text, count=1)


def validate_assignment(setup_text: str, key: str, expected: str) -> None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^!]*?)\s*(?:!.*)?$", re.MULTILINE)
    match = pattern.search(setup_text)
    if not match:
        raise RuntimeError(f"{key} missing after setup update")
    assigned = match.group(1).strip()
    if assigned != expected.strip():
        raise RuntimeError(f"{key} mismatch after setup update (expected {expected!r}, got {assigned!r})")


def apply_run_sample_to_setup(setup_path: Path, sample: RunSample, mass_unit: str) -> Dict[str, str]:
    """Patch setup file; returns map of column_name -> value string for CSV."""
    text = setup_path.read_text(encoding="utf-8")
    columns: Dict[str, str] = {}

    if sample.mass_kg is not None:
        tok = format_mass_token(sample.mass_kg, mass_unit)
        text = replace_setup_assignment(text, "m_apophis_in", tok)
        validate_assignment(text, "m_apophis_in", tok)
    for param, _, _, _ in _SCALE_VARIATION_SPEC:
        v = getattr(sample, param)
        if v is not None:
            tok = format_real_token(v)
            text = replace_setup_assignment(text, param, tok)
            validate_assignment(text, param, tok)
            columns[param] = f"{v:.12g}"

    if sample.use_dem is not None:
        tok = format_logical_token(sample.use_dem)
        text = replace_setup_assignment(text, "use_dem", tok)
        validate_assignment(text, "use_dem", tok)
        columns["use_dem"] = "T" if sample.use_dem else "F"

    if sample.apophis_only is not None:
        tok = format_logical_token(sample.apophis_only)
        text = replace_setup_assignment(text, "apophis_only", tok)
        validate_assignment(text, "apophis_only", tok)
        columns["apophis_only"] = "T" if sample.apophis_only else "F"

    setup_path.write_text(text, encoding="utf-8")
    return columns


def build_run_samples(num_samples: int, args: argparse.Namespace) -> List[RunSample]:
    dim = count_dimensions(args)
    unit_rows = sobol_nd_unit_samples(num_samples, dim, args.seed)
    out: List[RunSample] = []
    for row in unit_rows:
        di = 0
        s = RunSample()
        if not args.no_vary_mass:
            s.mass_kg = args.mass_min_kg + row[di] * (args.mass_max_kg - args.mass_min_kg)
            di += 1
        for param, lo, hi, _ in _active_scale_variations(args):
            setattr(s, param, lo + row[di] * (hi - lo))
            di += 1
        if args.vary_use_dem:
            s.use_dem = row[di] >= 0.5
            di += 1
        if args.vary_apophis_only:
            s.apophis_only = row[di] >= 0.5
            di += 1
        if di != dim:
            raise RuntimeError("internal error: dimension index mismatch")
        out.append(s)
    return out


def sample_column_order(args: argparse.Namespace) -> List[str]:
    """Stable CSV column order for varied parameters."""
    order: List[str] = []
    if not args.no_vary_mass:
        order.append("mass_input_kg")
    for param, _, _, _ in _active_scale_variations(args):
        order.append(param)
    if args.vary_use_dem:
        order.append("use_dem")
    if args.vary_apophis_only:
        order.append("apophis_only")
    return order


def _fmt_slug_float(x: float) -> str:
    """Format a float for use in batch directory names (letters, digits, ., -, _)."""
    return _SLUG_SAFE_RE.sub("", f"{x:.10g}")


def canonical_sweep_descriptor(args: argparse.Namespace) -> str:
    """Stable string for hashing when the auto slug must be truncated."""
    parts = [
        f"num_samples={args.num_samples}",
        f"seed={args.seed}",
        f"no_vary_mass={args.no_vary_mass}",
    ]
    if not args.no_vary_mass:
        parts.append(f"mass_min_kg={args.mass_min_kg}")
        parts.append(f"mass_max_kg={args.mass_max_kg}")
    for param, lo, hi, _ in _active_scale_variations(args):
        parts.append(f"{param}={lo}:{hi}")
    parts.append(f"vary_use_dem={args.vary_use_dem}")
    parts.append(f"vary_apophis_only={args.vary_apophis_only}")
    return "|".join(parts)


def build_auto_batch_sweep_slug(args: argparse.Namespace, max_len: int) -> str:
    """Sweep suffix from CLI: sample count, seed, each varied dimension and its bounds."""
    tokens: List[str] = [f"n{args.num_samples}", f"s{args.seed}"]
    if not args.no_vary_mass:
        tokens.append(
            f"m{_fmt_slug_float(args.mass_min_kg)}-{_fmt_slug_float(args.mass_max_kg)}"
        )
    for _, lo, hi, slug_tok in _active_scale_variations(args):
        tokens.append(f"{slug_tok}{_fmt_slug_float(lo)}-{_fmt_slug_float(hi)}")
    if args.vary_use_dem:
        tokens.append("dem")
    if args.vary_apophis_only:
        tokens.append("ao")
    slug = "_".join(tokens)
    if len(slug) <= max_len:
        return slug
    h = hashlib.sha256(canonical_sweep_descriptor(args).encode()).hexdigest()[:8]
    keep = max_len - (len(h) + 1)
    if keep < 1:
        return h[:max_len]
    return f"{slug[:keep]}_{h}"


def sanitize_batch_label(label: str) -> str:
    label = label.strip()
    if not label:
        raise ValueError("--batch-label cannot be empty")
    s = _SLUG_SAFE_RE.sub("_", label)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("--batch-label has no valid characters after sanitization")
    return s


def build_batch_directory_basename(args: argparse.Namespace, timestamp: str) -> str:
    """
    Batch folder basename: <prefix>_<timestamp>_<sweep_suffix>.
    Per-run folders remain run_XXXX inside this directory.
    """
    max_len = max(17, int(args.batch_slug_max_len))
    if args.batch_label is not None:
        slug = sanitize_batch_label(args.batch_label)
        if len(slug) > max_len:
            h = hashlib.sha256(slug.encode()).hexdigest()[:8]
            keep = max(1, max_len - (len(h) + 1))
            slug = f"{slug[:keep]}_{h}"
    else:
        slug = build_auto_batch_sweep_slug(args, max_len=max_len)
    return f"{args.prefix}_{timestamp}_{slug}"


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


def _sink_ev_dump_sort_key(path: Path) -> Tuple[int, str]:
    """Sort sink `.ev` dumps by trailing `N<number>` before extension (latest dump last)."""
    m = re.search(r"N(\d+)\.ev$", path.name, re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name)
    return (-1, path.name)


def extract_closest_approach(
    run_dir: Path, prefix: str, earth_sink_id: int, apophis_sink_id: int
) -> Tuple[float, float]:
    earth_candidates = sorted(
        run_dir.glob(f"{prefix}Sink{earth_sink_id:04d}N*.ev"),
        key=_sink_ev_dump_sort_key,
    )
    apophis_candidates = sorted(
        run_dir.glob(f"{prefix}Sink{apophis_sink_id:04d}N*.ev"),
        key=_sink_ev_dump_sort_key,
    )

    if not earth_candidates or not apophis_candidates:
        raise RuntimeError(
            "Could not find Earth/Apophis sink files "
            f"(expected IDs {earth_sink_id} and {apophis_sink_id})."
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


def write_samples_csv(path: Path, samples: List[RunSample], column_order: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["run_id"] + column_order
        writer.writerow(header)
        for idx, s in enumerate(samples, start=1):
            row: List[Any] = [idx]
            for col in column_order:
                if col == "mass_input_kg":
                    row.append(f"{s.mass_kg:.12g}" if s.mass_kg is not None else "")
                elif col == "use_dem":
                    row.append("T" if s.use_dem else "F" if s.use_dem is not None else "")
                elif col == "apophis_only":
                    row.append("T" if s.apophis_only else "F" if s.apophis_only is not None else "")
                else:
                    v = getattr(s, col, None)
                    row.append(f"{float(v):.12g}" if v is not None else "")
            writer.writerow(row)


def summary_secondary_columns(col_order: List[str]) -> List[str]:
    """Columns after mass_input_kg in the summary CSV (mass is its own field)."""
    return [c for c in col_order if c != "mass_input_kg"]


def append_summary(
    path: Path, row: RunRecord, write_header: bool, param_column_order: List[str]
) -> None:
    secondary = summary_secondary_columns(param_column_order)
    base_cols = [
        "run_id",
        "mass_input_kg",
        *secondary,
        "run_dir",
        "status",
        "closest_approach_km",
        "closest_approach_au",
        "error",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(base_cols)
        extra = [row.param_columns.get(c, "") for c in secondary]
        writer.writerow(
            [
                row.run_id,
                f"{row.mass_input_kg:.12g}" if math.isfinite(row.mass_input_kg) else "",
                *extra,
                row.run_dir,
                row.status,
                f"{row.closest_approach_km:.12g}" if not math.isnan(row.closest_approach_km) else "",
                f"{row.closest_approach_au:.12g}" if not math.isnan(row.closest_approach_au) else "",
                row.error,
            ]
        )


def skip_closest_approach(sample: RunSample) -> bool:
    """Earth is absent when apophis_only is True; default sink IDs do not apply."""
    return sample.apophis_only is True


def run_one_case(
    run_id: int,
    sample: RunSample,
    base_setup: Path,
    base_input: Path,
    output_root: Path,
    prefix: str,
    phantomsetup_bin: Path,
    phantom_bin: Path,
    mass_unit: str,
    dry_run: bool,
    earth_sink_id: int,
    apophis_sink_id: int,
) -> RunRecord:
    run_dir = output_root / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_setup = run_dir / f"{prefix}.setup"
    run_input = run_dir / f"{prefix}.in"
    shutil.copy2(base_setup, run_setup)
    shutil.copy2(base_input, run_input)
    param_columns = apply_run_sample_to_setup(run_setup, sample, mass_unit)

    mass_for_record = float(sample.mass_kg) if sample.mass_kg is not None else float("nan")

    if dry_run:
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_for_record,
            run_dir=str(run_dir),
            status="prepared_only",
            closest_approach_km=float("nan"),
            closest_approach_au=float("nan"),
            error="",
            param_columns=param_columns,
        )

    try:
        run_command([str(phantomsetup_bin), prefix], cwd=run_dir, log_path=run_dir / "setup.log")
        run_command([str(phantom_bin), f"{prefix}.in"], cwd=run_dir, log_path=run_dir / "phantom.log")
        if skip_closest_approach(sample):
            return RunRecord(
                run_id=run_id,
                mass_input_kg=mass_for_record,
                run_dir=str(run_dir),
                status="ok",
                closest_approach_km=float("nan"),
                closest_approach_au=float("nan"),
                error="",
                param_columns=param_columns,
            )
        closest_km, closest_au = extract_closest_approach(
            run_dir, prefix, earth_sink_id, apophis_sink_id
        )
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_for_record,
            run_dir=str(run_dir),
            status="ok",
            closest_approach_km=closest_km,
            closest_approach_au=closest_au,
            error="",
            param_columns=param_columns,
        )
    except Exception as exc:  # pragma: no cover - runtime path
        return RunRecord(
            run_id=run_id,
            mass_input_kg=mass_for_record,
            run_dir=str(run_dir),
            status="failed",
            closest_approach_km=float("nan"),
            closest_approach_au=float("nan"),
            error=str(exc),
            param_columns=param_columns,
        )


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        validate_args(args)
        batch_basename = build_batch_directory_basename(args, timestamp)
        output_root = Path(args.output_root).resolve() / batch_basename
        samples = build_run_samples(args.num_samples, args)
        base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(args, base_dir, output_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    col_order = sample_column_order(args)
    samples_csv = output_root / "sobol_mass_samples.csv"
    summary_csv = output_root / "sobol_mass_outputs.csv"
    write_samples_csv(samples_csv, samples, col_order)
    print(f"[INFO] Wrote Sobol sample table: {samples_csv}")
    print(f"[INFO] Varying dimensions ({count_dimensions(args)}): {', '.join(col_order)}")

    for idx, sample in enumerate(samples, start=1):
        mass_str = f"{sample.mass_kg:.6e} kg" if sample.mass_kg is not None else "(template mass)"
        print(f"[INFO] Run {idx}/{len(samples)} mass={mass_str}")
        result = run_one_case(
            run_id=idx,
            sample=sample,
            base_setup=base_setup,
            base_input=base_input,
            output_root=output_root,
            prefix=args.prefix,
            phantomsetup_bin=phantomsetup_bin,
            phantom_bin=phantom_bin,
            mass_unit=args.mass_unit,
            dry_run=args.dry_run,
            earth_sink_id=args.sink_earth_id,
            apophis_sink_id=args.sink_apophis_id,
        )
        append_summary(summary_csv, result, write_header=(idx == 1), param_column_order=col_order)
        print(f"[INFO]   status={result.status} run_dir={result.run_dir}")

    print(f"[INFO] Summary written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
