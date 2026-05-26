#!/usr/bin/env python3
"""Sobol parameter sweep runner for PHANTOM solarsystem-style setups (mass + optional scales).

Runs PHANTOM once per sample; use ``--jobs N`` to run independent cases in parallel processes."""

from __future__ import annotations

import argparse
import concurrent.futures
from concurrent.futures import as_completed
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple


AU_IN_KM = 149_597_870.7
EARTH_SINK_ID_DEFAULT = 4
APOPHIS_SINK_ID_DEFAULT = 11

# MAXPTMASS is a compile-time array bound on the number of sink particles (config.F90).
# The default upstream value is 1000. The close-packed lattice placement routine overshoots
# the requested np_apophis by ~2-3% (observed: np_apophis=1000 -> 1012 actual sinks), so
# any run with np_apophis >= ~975 will exceed MAXPTMASS=1000 and phantomsetup will abort:
#   "ERROR: nptmass=1012 exceeds ptmass array dimensions of 1000"
# Fix: compile with MAXPTMASS=2000 (set in the Makefile 'ifndef MAXPTMASS' block).
_MAXPTMASS_DEFAULT = 1000
_LATTICE_OVERSHOOT_FRACTION = 0.025
_MAXPTMASS_WARN_THRESHOLD = int(_MAXPTMASS_DEFAULT * (1.0 - _LATTICE_OVERSHOOT_FRACTION))  # 975

# Matches the PHANTOM error line emitted when nptmass exceeds the compiled array bound.
_MAXPTMASS_ERR_RE = re.compile(
    r"nptmass\s*=\s*(\d+)\s+exceeds\s+ptmass\s+array\s+dimensions\s+of\s+(\d+)",
    re.IGNORECASE,
)

# Auto-generated batch folder suffix (after "{prefix}_{timestamp}_"); keep paths portable.
_BATCH_SLUG_DEFAULT_MAX_LEN = 120
_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]+")

# Default install root: this directory. Override with PHANTOM_DIR.
_DEFAULT_PHANTOM_DIR = Path(__file__).resolve().parent

_NP_APOPHIS_RE = re.compile(r"^(\s*np_apophis\s*=\s*)\d+", re.MULTILINE | re.IGNORECASE)


def resolve_phantom_executable(phantom_dir: Path, name: str, *, must_exist: bool) -> Path:
    """Prefer ``phantom_dir/bin/<name>``, then ``phantom_dir/<name>`` (flat install)."""
    bin_path = phantom_dir / "bin" / name
    root_path = phantom_dir / name
    for p in (bin_path, root_path):
        if p.is_file():
            return p
    if not must_exist:
        return bin_path
    raise FileNotFoundError(f"Missing PHANTOM binary {name!r}: tried {bin_path}, {root_path}")


def _diagnose_maxptmass_error(log_path: Path) -> Optional[str]:
    """Scan a phantomsetup log for the MAXPTMASS overflow error and return an actionable message.

    Without this, a failed run records only "Command failed (1)" with no indication that
    the binary needs to be recompiled with a larger MAXPTMASS.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _MAXPTMASS_ERR_RE.search(text)
    if not m:
        return None
    actual = int(m.group(1))
    compiled = int(m.group(2))
    needed = int(actual * 1.15) + 1
    return (
        f"phantomsetup: nptmass={actual} exceeds compiled MAXPTMASS={compiled}. "
        f"Rebuild phantom and phantomsetup with MAXPTMASS>={needed}: "
        f"add 'MAXPTMASS={needed}' to the Makefile 'ifndef MAXPTMASS' block, "
        "then run 'make setup && make'."
    )


def _np_apophis_maxptmass_warning(sample: "RunSample") -> Optional[str]:
    """Warn before running when np_apophis is close enough to MAXPTMASS that lattice overshoot
    will push the actual sink count over the compiled limit, causing phantomsetup to abort.
    """
    if sample.np_apophis is None or sample.np_apophis < _MAXPTMASS_WARN_THRESHOLD:
        return None
    return (
        f"np_apophis={sample.np_apophis} is at or above the safe limit ({_MAXPTMASS_WARN_THRESHOLD}) "
        f"for default MAXPTMASS={_MAXPTMASS_DEFAULT}. The close-packed lattice overshoots "
        "np_apophis by ~2-3%, which will cause phantomsetup to abort. "
        "Ensure PHANTOM is compiled with MAXPTMASS>np_apophis "
        "(e.g. set MAXPTMASS=2000 in the Makefile 'ifndef MAXPTMASS' block, then run 'make setup && make')."
    )


# Scale parameters varied via CLI min/max: (RunSample/.setup attribute, argparse lo/hi attrs, batch slug token).
# Order MUST match Sobol dimension ordering used in build_run_samples and CSV columns.
# Spin parameters use the full setup-file key as the RunSample attribute so apply_run_sample_to_setup
# can patch them with a single generic loop (same mechanism as scale_vel etc.).
_SCALE_VARIATION_SPEC: Tuple[Tuple[str, str, str, str], ...] = (
    ("scale_vel", "scale_vel_min", "scale_vel_max", "sv"),
    ("scale_pos", "scale_pos_min", "scale_pos_max", "sp"),
    ("scale_r_apophis", "scale_r_apophis_min", "scale_r_apophis_max", "sra"),
    ("scale_rho", "scale_rho_min", "scale_rho_max", "srho"),
    # Spin parameters — only affect DEM runs (use_dem=T, np_apophis>1); no-op otherwise.
    ("apophis_spin_period",    "spin_period_min",    "spin_period_max",    "sspd"),
    ("apophis_spin_obliquity", "spin_obliquity_min", "spin_obliquity_max", "sobl"),
    ("apophis_spin_azimuth",   "spin_azimuth_min",   "spin_azimuth_max",   "saz"),
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


def _mass_bounds_active(args: argparse.Namespace) -> bool:
    """True when mass is a Sobol dimension: both kg bounds set (same contract as optional scale min/max)."""
    return args.mass_min_kg is not None and args.mass_max_kg is not None


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
    use_shape_crop: Optional[bool] = None
    apophis_only: Optional[bool] = None
    np_apophis: Optional[int] = None
    # Spin: patched into the setup file; Fortran applies them only when use_dem=T, np_apophis>1.
    apophis_spin_period:    Optional[float] = None  # hours
    apophis_spin_obliquity: Optional[float] = None  # degrees from ecliptic north
    apophis_spin_azimuth:   Optional[float] = None  # degrees in ecliptic plane


class RunWorkerPayload(NamedTuple):
    """Picklable bundle for ProcessPoolExecutor workers (paths as str)."""

    run_id: int
    sample: RunSample
    base_setup: str
    base_input: str
    output_root: str
    prefix: str
    phantomsetup_bin: str
    phantom_bin: str
    ref_mass_kg: Optional[float]
    dry_run: bool
    earth_sink_id: int
    apophis_sink_id: int
    ephemeris_cache_dir: Optional[str]
    shape_file: Optional[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Sobol samples over Apophis mass and optional setup_solarsystem.f90 "
            "parameters, then run PHANTOM (one subprocess per sample; use --jobs for parallelism). "
            "Multi-d Sobol uses scipy when installed; otherwise independent 1D sequences per dimension "
            "are used (weaker joint coverage — see pip install -r requirements.txt)."
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
        "--ephemeris-cache-dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, copy every *.txt from this directory into each run folder before phantomsetup. "
            "PHANTOM then skips JPL Horizons download when <object>.txt already exists (see "
            "phantom/src/utils/utils_ephemeris.f90). Files must match the epoch in your .setup."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of Sobol samples (runs) to generate.",
    )
    parser.add_argument(
        "--mass-min-kg",
        type=float,
        default=None,
        help="Lower bound for Apophis mass in kg; omit with --mass-max-kg to leave mass unvaried.",
    )
    parser.add_argument(
        "--mass-max-kg",
        type=float,
        default=None,
        help="Upper bound for Apophis mass in kg; omit with --mass-min-kg to leave mass unvaried.",
    )
    parser.add_argument(
        "--apophis-ref-mass-kg",
        type=float,
        default=None,
        metavar="KG",
        help=(
            "Baseline Apophis mass in kg at scale_rho=1 (i.e. using the template density). "
            "Required when --mass-min-kg and --mass-max-kg are set. "
            "The sampled mass is converted to scale_rho = mass_kg / ref_mass_kg and patched "
            "into the setup file."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Scramble seed for Sobol sequence.",
    )
    # Real scalings and spin parameters — optional Sobol dimensions when both min and max are set.
    # Spin only takes effect in DEM runs (use_dem=T, np_apophis>1); setting bounds without DEM
    # is allowed but a warning is issued at runtime.
    for key, helpt in (
        ("scale-vel", "scale_vel (Apophis velocity scale)"),
        ("scale-pos", "scale_pos (Apophis initial position scale)"),
        ("scale-r-apophis", "scale_r_apophis (Apophis radius scale)"),
        ("scale-rho", "scale_rho (bulk density scale when mass is density-derived)"),
        ("spin-period",    "apophis_spin_period in hours (0 = no spin; DEM runs only)"),
        ("spin-obliquity", "apophis_spin_obliquity in degrees from ecliptic north (DEM runs only)"),
        ("spin-azimuth",   "apophis_spin_azimuth in degrees in ecliptic plane (DEM runs only)"),
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
        "--use-dem-fixed",
        choices=("true", "false"),
        default=None,
        metavar="{true,false}",
        help=(
            "Force use_dem to a fixed value in every run. "
            "Mutually exclusive with --vary-use-dem; omit to leave the template value unchanged."
        ),
    )
    parser.add_argument(
        "--np-apophis",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fix np_apophis to a single value in every run's setup (0=none, 1=sink, N>1=gas/DEM). "
            "Mutually exclusive with --np-apophis-list. Omit to leave the template value unchanged."
        ),
    )
    parser.add_argument(
        "--np-apophis-list",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run one simulation per np_apophis value within a single batch "
            "(e.g. --np-apophis-list 250 500 1000 → run_0001=250, run_0002=500, run_0003=1000). "
            "Mutually exclusive with --np-apophis and --num-samples (run count = len of list). "
            "Other parameters stay at template or fixed values (--use-dem-fixed etc. still apply)."
        ),
    )
    parser.add_argument(
        "--vary-use-shape-crop",
        action="store_true",
        help=(
            "Sample shape cropping with one Sobol dimension (u>=0.5 -> T, writes --shape-file "
            "to obj_file in setup; else blanks obj_file to disable cropping)."
        ),
    )
    parser.add_argument(
        "--use-shape-crop-fixed",
        choices=("true", "false"),
        default=None,
        metavar="{true,false}",
        help=(
            "Force shape cropping to a fixed value in every run: 'true' writes --shape-file to "
            "obj_file; 'false' blanks obj_file to disable cropping. "
            "Mutually exclusive with --vary-use-shape-crop; omit to leave the template obj_file unchanged."
        ),
    )
    parser.add_argument(
        "--shape-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to the shape config file (or .obj) written into obj_file when shape "
            "cropping is enabled (--vary-use-shape-crop or --use-shape-crop-fixed true). "
            "Required when cropping may be enabled."
        ),
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
        default=os.environ.get("PHANTOM_DIR", str(_DEFAULT_PHANTOM_DIR)),
        help=(
            "PHANTOM install root: resolves each binary as <root>/bin/<name> first, "
            "then <root>/<name> (upstream layout vs flat copy)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run directories and setup files without executing PHANTOM.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes for independent PHANTOM runs. "
            "If PHANTOM is built with OpenMP, set OMP_NUM_THREADS=1 (or ensure jobs×threads "
            "does not oversubscribe CPU cores)."
        ),
    )
    parser.add_argument(
        "--saltelli-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use SALib Saltelli/Sobol sample design with base size N (ignores --num-samples for layout). "
            "Approximate PHANTOM runs: N*(D+2), or N*(2*D+2) with --saltelli-calc-second-order (D=varying dims)."
        ),
    )
    parser.add_argument(
        "--saltelli-calc-second-order",
        action="store_true",
        help="Also estimate second-order Sobol indices (requires more runs). For use with Analysis.py.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for each parameter (then run); CLI flags set defaults shown at each prompt.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _strip_interactive_flags(argv: Sequence[str]) -> Tuple[List[str], bool]:
    out: List[str] = []
    interactive = False
    for tok in argv:
        if tok in ("--interactive", "-i"):
            interactive = True
            continue
        out.append(tok)
    return out, interactive


def validate_args(args: argparse.Namespace) -> None:
    if args.jobs < 1:
        raise ValueError("jobs must be >= 1")

    np_list = getattr(args, "np_apophis_list", None)

    # np_apophis_list drives run count directly; num_samples / saltelli_n are irrelevant in that mode.
    if np_list is None:
        if args.saltelli_n is not None:
            if args.saltelli_n < 2:
                raise ValueError("saltelli-n must be >= 2")
        elif args.num_samples < 1:
            raise ValueError("num-samples must be >= 1")

    if args.batch_slug_max_len < 17:
        raise ValueError("batch-slug-max-len must be >= 17 (timestamp prefix is 15 chars + separators)")
    m_lo, m_hi = args.mass_min_kg, args.mass_max_kg
    if (m_lo is None) ^ (m_hi is None):
        raise ValueError("mass bounds: set both --mass-min-kg and --mass-max-kg, or neither")
    if _mass_bounds_active(args):
        if m_lo <= 0 or m_hi <= 0 or m_hi <= m_lo:
            raise ValueError("mass bounds must satisfy 0 < mass-min-kg < mass-max-kg")
        if args.apophis_ref_mass_kg is None:
            raise ValueError("--apophis-ref-mass-kg is required when --mass-min-kg and --mass-max-kg are set")
        if args.apophis_ref_mass_kg <= 0:
            raise ValueError("--apophis-ref-mass-kg must be > 0")
        active_scale = _active_scale_variations(args)
        if any(p == "scale_rho" for p, *_ in active_scale):
            raise ValueError(
                "--mass-min-kg/max-kg and --scale-rho-min/max cannot both be active: "
                "both patch scale_rho in the setup file"
            )
    for param, lo_attr, hi_attr, _ in _SCALE_VARIATION_SPEC:
        lo = getattr(args, lo_attr)
        hi = getattr(args, hi_attr)
        if (lo is None) ^ (hi is None):
            raise ValueError(f"{param}: set both min and max, or neither")
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"{param}: require min < max")

    if args.vary_use_dem and args.use_dem_fixed is not None:
        raise ValueError("--vary-use-dem and --use-dem-fixed are mutually exclusive")

    if args.np_apophis is not None and args.np_apophis < 0:
        raise ValueError("--np-apophis must be >= 0")

    # --np-apophis-list and --np-apophis are mutually exclusive.
    if np_list is not None and args.np_apophis is not None:
        raise ValueError("--np-apophis-list and --np-apophis are mutually exclusive")
    if np_list is not None:
        if len(np_list) < 1:
            raise ValueError("--np-apophis-list requires at least one value")
        if any(n < 0 for n in np_list):
            raise ValueError("all --np-apophis-list values must be >= 0")
        # Warn early for values that risk MAXPTMASS overflow after lattice overshoot.
        high = [n for n in np_list if n >= _MAXPTMASS_WARN_THRESHOLD]
        if high:
            print(
                f"[WARN] --np-apophis-list values {high} >= {_MAXPTMASS_WARN_THRESHOLD}: "
                f"lattice overshoot (~2-3%) may exceed default MAXPTMASS={_MAXPTMASS_DEFAULT}. "
                "Ensure PHANTOM is compiled with MAXPTMASS>max(np_apophis_list) "
                "(e.g. MAXPTMASS=2000 in the Makefile).",
                file=sys.stderr,
            )

    if args.vary_use_shape_crop and args.use_shape_crop_fixed is not None:
        raise ValueError("--vary-use-shape-crop and --use-shape-crop-fixed are mutually exclusive")
    if (args.vary_use_shape_crop or args.use_shape_crop_fixed == "true") and not args.shape_file:
        raise ValueError("--shape-file must be set when shape cropping may be enabled")

    # Warn if spin bounds are active but DEM is not enabled; spin only affects DEM particles.
    _spin_lo_dests = frozenset({"spin_period_min", "spin_obliquity_min", "spin_azimuth_min"})
    if any(getattr(args, d, None) is not None for d in _spin_lo_dests):
        dem_on = args.vary_use_dem or getattr(args, "use_dem_fixed", None) == "true"
        if not dem_on:
            print(
                "[WARN] Spin bounds are set but DEM is not enabled "
                "(--vary-use-dem or --use-dem-fixed true). "
                "Spin velocity kicks only apply when use_dem=T and np_apophis>1 in the setup file.",
                file=sys.stderr,
            )

    # np_apophis_list itself constitutes variation; allow dim=0 in that mode.
    dim = count_dimensions(args)
    if dim == 0 and not np_list:
        raise ValueError(
            "No varying dimensions: set both --mass-min-kg and --mass-max-kg and/or pass scale */ "
            "vary-use-dem / vary-apophis-only / --np-apophis-list."
        )


def count_dimensions(args: argparse.Namespace) -> int:
    n = 0
    if _mass_bounds_active(args):
        n += 1
    n += len(_active_scale_variations(args))
    if args.vary_use_dem:
        n += 1
    if args.vary_use_shape_crop:
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


def apply_run_sample_to_setup(
    setup_path: Path, sample: RunSample, ref_mass_kg: Optional[float] = None, shape_file: Optional[str] = None
) -> Dict[str, str]:
    """Patch setup file; returns map of column_name -> value string for CSV."""
    text = setup_path.read_text(encoding="utf-8")
    columns: Dict[str, str] = {}

    if sample.mass_kg is not None:
        if ref_mass_kg is None:
            raise RuntimeError("ref_mass_kg required to convert mass sample to scale_rho")
        scale_rho_val = sample.mass_kg / ref_mass_kg
        tok = format_real_token(scale_rho_val)
        text = replace_setup_assignment(text, "scale_rho", tok)
        validate_assignment(text, "scale_rho", tok)
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

    if sample.use_shape_crop is not None:
        path_tok = (shape_file or "") if sample.use_shape_crop else ""
        text = replace_setup_assignment(text, "obj_file", path_tok)
        validate_assignment(text, "obj_file", path_tok)
        columns["use_shape_crop"] = "T" if sample.use_shape_crop else "F"

    if sample.apophis_only is not None:
        tok = format_logical_token(sample.apophis_only)
        text = replace_setup_assignment(text, "apophis_only", tok)
        validate_assignment(text, "apophis_only", tok)
        columns["apophis_only"] = "T" if sample.apophis_only else "F"

    if sample.np_apophis is not None:
        if not _NP_APOPHIS_RE.search(text):
            raise RuntimeError("np_apophis assignment not found in .setup file")
        text = _NP_APOPHIS_RE.sub(lambda m: m.group(1) + str(sample.np_apophis), text, count=1)
        columns["np_apophis"] = str(sample.np_apophis)

    setup_path.write_text(text, encoding="utf-8")
    return columns


def build_np_list_samples(args: argparse.Namespace) -> List[RunSample]:
    """One RunSample per value in --np-apophis-list; all other params at template or fixed values.

    This lets a single batch produce run_0001=np1, run_0002=np2, ... without the Sobol dimension
    requirement, replacing the previous approach of spawning one subprocess per count.
    """
    out: List[RunSample] = []
    for n in args.np_apophis_list:
        s = RunSample(np_apophis=n)
        if args.use_dem_fixed is not None:
            s.use_dem = args.use_dem_fixed == "true"
        if args.use_shape_crop_fixed is not None:
            s.use_shape_crop = args.use_shape_crop_fixed == "true"
        out.append(s)
    return out


def build_run_samples(num_samples: int, args: argparse.Namespace) -> List[RunSample]:
    dim = count_dimensions(args)
    unit_rows = sobol_nd_unit_samples(num_samples, dim, args.seed)
    out: List[RunSample] = []
    for row in unit_rows:
        di = 0
        s = RunSample()
        if _mass_bounds_active(args):
            s.mass_kg = args.mass_min_kg + row[di] * (args.mass_max_kg - args.mass_min_kg)
            di += 1
        for param, lo, hi, _ in _active_scale_variations(args):
            setattr(s, param, lo + row[di] * (hi - lo))
            di += 1
        if args.vary_use_dem:
            s.use_dem = row[di] >= 0.5
            di += 1
        elif args.use_dem_fixed is not None:
            s.use_dem = args.use_dem_fixed == "true"
        if args.vary_use_shape_crop:
            s.use_shape_crop = row[di] >= 0.5
            di += 1
        elif args.use_shape_crop_fixed is not None:
            s.use_shape_crop = args.use_shape_crop_fixed == "true"
        if args.vary_apophis_only:
            s.apophis_only = row[di] >= 0.5
            di += 1
        if args.np_apophis is not None:
            s.np_apophis = args.np_apophis
        if di != dim:
            raise RuntimeError("internal error: dimension index mismatch")
        out.append(s)
    return out


def sample_column_order(args: argparse.Namespace) -> List[str]:
    """Stable CSV column order for varied parameters."""
    order: List[str] = []
    if _mass_bounds_active(args):
        order.append("mass_input_kg")
    for param, _, _, _ in _active_scale_variations(args):
        order.append(param)
    if args.vary_use_dem:
        order.append("use_dem")
    if args.vary_use_shape_crop:
        order.append("use_shape_crop")
    if args.vary_apophis_only:
        order.append("apophis_only")
    # np_apophis_list mode: particle count is the varying quantity, record it as a column.
    if getattr(args, "np_apophis_list", None):
        order.append("np_apophis")
    return order


def build_salib_problem(args: argparse.Namespace) -> Dict[str, object]:
    """SALib problem dict; parameter order matches Saltelli rows and RunSample mapping."""
    names: List[str] = []
    bounds: List[List[float]] = []
    if _mass_bounds_active(args):
        names.append("mass_input_kg")
        bounds.append([float(args.mass_min_kg), float(args.mass_max_kg)])
    for param, lo, hi, _ in _active_scale_variations(args):
        names.append(param)
        bounds.append([float(lo), float(hi)])
    if args.vary_use_dem:
        names.append("use_dem")
        bounds.append([0.0, 1.0])
    if args.vary_use_shape_crop:
        names.append("use_shape_crop")
        bounds.append([0.0, 1.0])
    if args.vary_apophis_only:
        names.append("apophis_only")
        bounds.append([0.0, 1.0])
    dim = len(names)
    if dim == 0:
        raise ValueError("Saltelli mode needs at least one varying dimension")
    return {"num_vars": dim, "names": names, "bounds": bounds}


def run_sample_from_salib_row(row: Sequence[float], args: argparse.Namespace) -> RunSample:
    """Build RunSample from one SALib Sobol row (physical bounds)."""
    s = RunSample()
    i = 0
    if _mass_bounds_active(args):
        s.mass_kg = float(row[i])
        i += 1
    for param, _, _, _ in _active_scale_variations(args):
        setattr(s, param, float(row[i]))
        i += 1
    if args.vary_use_dem:
        s.use_dem = float(row[i]) >= 0.5
        i += 1
    elif args.use_dem_fixed is not None:
        s.use_dem = args.use_dem_fixed == "true"
    if args.vary_use_shape_crop:
        s.use_shape_crop = float(row[i]) >= 0.5
        i += 1
    elif args.use_shape_crop_fixed is not None:
        s.use_shape_crop = args.use_shape_crop_fixed == "true"
    if args.vary_apophis_only:
        s.apophis_only = float(row[i]) >= 0.5
        i += 1
    if args.np_apophis is not None:
        s.np_apophis = args.np_apophis
    if i != len(row):
        raise RuntimeError(f"internal error: saltelli row consumed {i} values but width is {len(row)}")
    return s


def expected_saltelli_num_evals(num_vars: int, base_n: int, calc_second_order: bool) -> int:
    """Match SALib.sample.sobol.sample row count."""
    if calc_second_order:
        return base_n * (2 * num_vars + 2)
    return base_n * (num_vars + 2)


def _fmt_slug_float(x: float) -> str:
    """Format a float for use in batch directory names (letters, digits, ., -, _)."""
    return _SLUG_SAFE_RE.sub("", f"{x:.10g}")


def canonical_sweep_descriptor(args: argparse.Namespace) -> str:
    """Stable string for hashing when the auto slug must be truncated."""
    parts = [
        f"num_samples={args.num_samples}",
        f"seed={args.seed}",
        f"saltelli_n={getattr(args, 'saltelli_n', None)}",
        f"saltelli_second={getattr(args, 'saltelli_calc_second_order', False)}",
        f"mass_active={_mass_bounds_active(args)}",
    ]
    if _mass_bounds_active(args):
        parts.append(f"mass_min_kg={args.mass_min_kg}")
        parts.append(f"mass_max_kg={args.mass_max_kg}")
    for param, lo, hi, _ in _active_scale_variations(args):
        parts.append(f"{param}={lo}:{hi}")
    parts.append(f"vary_use_dem={args.vary_use_dem}")
    parts.append(f"vary_use_shape_crop={args.vary_use_shape_crop}")
    parts.append(f"vary_apophis_only={args.vary_apophis_only}")
    # np_apophis_list must be included so sweeps that differ only in particle counts
    # produce distinct hashes when the slug is truncated.
    np_list = getattr(args, "np_apophis_list", None)
    parts.append(f"np_apophis_list={np_list}")
    return "|".join(parts)


def build_auto_batch_sweep_slug(args: argparse.Namespace, max_len: int) -> str:
    """Sweep suffix from CLI: sample count, seed, each varied dimension and its bounds."""
    np_list = getattr(args, "np_apophis_list", None)
    if np_list is not None:
        # np_apophis_list mode: runs are deterministic (fixed list), so seed is meaningless.
        # Encode the particle counts directly so the folder name is self-describing.
        # Short lists (<=5 values): "np250_500_1000"; longer lists: "np3vals_250-1000".
        if len(np_list) <= 5:
            tok_np = "np" + "_".join(str(n) for n in np_list)
        else:
            tok_np = f"np{len(np_list)}vals_{min(np_list)}-{max(np_list)}"
        tokens: List[str] = [tok_np]
    elif args.saltelli_n is not None:
        tok_n = f"salt{args.saltelli_n}"
        if args.saltelli_calc_second_order:
            tok_n += "s2"
        tokens = [tok_n, f"s{args.seed}"]
    else:
        tokens = [f"n{args.num_samples}", f"s{args.seed}"]
    if _mass_bounds_active(args):
        tokens.append(
            f"m{_fmt_slug_float(args.mass_min_kg)}-{_fmt_slug_float(args.mass_max_kg)}"
        )
    for _, lo, hi, slug_tok in _active_scale_variations(args):
        tokens.append(f"{slug_tok}{_fmt_slug_float(lo)}-{_fmt_slug_float(hi)}")
    if args.vary_use_dem:
        tokens.append("dem")
    if args.vary_use_shape_crop:
        tokens.append("shapecrop")
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
    max_len = int(args.batch_slug_max_len)
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


def copy_ephemeris_txt_cache(cache_dir: Path, run_dir: Path) -> int:
    """Copy *.txt ephemeris snippets from cache_dir into run_dir (non-recursive). Returns file count."""
    n = 0
    for src in sorted(cache_dir.glob("*.txt")):
        if src.is_file():
            shutil.copy2(src, run_dir / src.name)
            n += 1
    return n


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


def _sink_times_close(t_a: float, t_b: float) -> bool:
    """Whether two sink row times refer to the same output step (robust float compare)."""
    return math.isclose(t_a, t_b, rel_tol=1e-9, abs_tol=1e-12)


def _pair_sink_rows_by_time(
    t_earth: Sequence[float],
    xyz_earth: Sequence[Tuple[float, float, float]],
    t_apophis: Sequence[float],
    xyz_apophis: Sequence[Tuple[float, float, float]],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Pair Earth and Apophis positions at matching simulation times (sorted merge).

    Rows are matched only when ``t`` agrees within ``_sink_times_close``; index order
    is not assumed, so mis-ordered or unequal-length files still pair correctly when
    they share the same time stamps.
    """
    earth = sorted(zip(t_earth, xyz_earth), key=lambda r: r[0])
    apo = sorted(zip(t_apophis, xyz_apophis), key=lambda r: r[0])
    i, j = 0, 0
    out: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    while i < len(earth) and j < len(apo):
        te, ve = earth[i]
        ta, va = apo[j]
        if _sink_times_close(te, ta):
            out.append((ve, va))
            i += 1
            j += 1
        elif te < ta:
            i += 1
        else:
            j += 1
    return out


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

    pairs = _pair_sink_rows_by_time(t_earth, xyz_earth, t_apophis, xyz_apophis)
    if not pairs:
        raise RuntimeError(
            "No Earth/Apophis sink rows with matching time (after sorting and time alignment). "
            "Check sink `.ev` dumps share the same simulation clock."
        )

    closest_km = float("inf")
    for ve, va in pairs:
        dx = va[0] - ve[0]
        dy = va[1] - ve[1]
        dz = va[2] - ve[2]
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

    phantom_root = Path(args.phantom_dir)
    must_exist = not args.dry_run
    phantomsetup_bin = resolve_phantom_executable(phantom_root, "phantomsetup", must_exist=must_exist)
    phantom_bin = resolve_phantom_executable(phantom_root, "phantom", must_exist=must_exist)

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
                    row.append("T" if s.use_dem is True else "F" if s.use_dem is False else "")
                elif col == "use_shape_crop":
                    row.append("T" if s.use_shape_crop is True else "F" if s.use_shape_crop is False else "")
                elif col == "apophis_only":
                    row.append("T" if s.apophis_only is True else "F" if s.apophis_only is False else "")
                else:
                    v = getattr(s, col, None)
                    row.append(f"{float(v):.12g}" if v is not None else "")
            writer.writerow(row)


def summary_secondary_columns(col_order: List[str]) -> List[str]:
    """Columns after mass_input_kg in the summary CSV (mass is its own field)."""
    return [c for c in col_order if c != "mass_input_kg"]


def write_summary_csv(path: Path, records: List[RunRecord], param_column_order: List[str]) -> None:
    """Write full sobol_mass_outputs.csv: header plus one row per record (sorted by run_id)."""
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
    ordered = sorted(records, key=lambda r: r.run_id)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(base_cols)
        for row in ordered:
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
    ref_mass_kg: Optional[float],
    dry_run: bool,
    earth_sink_id: int,
    apophis_sink_id: int,
    ephemeris_cache_dir: Optional[Path] = None,
    shape_file: Optional[str] = None,
) -> RunRecord:
    run_dir = output_root / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_setup = run_dir / f"{prefix}.setup"
    run_input = run_dir / f"{prefix}.in"
    shutil.copy2(base_setup, run_setup)
    shutil.copy2(base_input, run_input)
    param_columns = apply_run_sample_to_setup(run_setup, sample, ref_mass_kg, shape_file)

    if ephemeris_cache_dir is not None:
        copy_ephemeris_txt_cache(ephemeris_cache_dir, run_dir)

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

    # Warn before starting if np_apophis is high enough that lattice overshoot will likely push
    # the actual sink count above the compiled MAXPTMASS limit (default 1000).
    maxptmass_warn = _np_apophis_maxptmass_warning(sample)
    if maxptmass_warn:
        print(f"[WARN] Run {run_id}: {maxptmass_warn}", file=sys.stderr, flush=True)

    try:
        # phantomsetup is caught separately so we can inspect setup.log for the specific
        # MAXPTMASS overflow message before falling back to a generic "command failed" error.
        try:
            run_command([str(phantomsetup_bin), prefix], cwd=run_dir, log_path=run_dir / "setup.log")
        except RuntimeError:
            diag = _diagnose_maxptmass_error(run_dir / "setup.log")
            raise RuntimeError(
                diag or f"phantomsetup failed (returncode != 0); see {run_dir / 'setup.log'}"
            ) from None
        if sample.use_dem is True:
            in_text = run_input.read_text()
            in_text = replace_setup_assignment(in_text, "nfulldump", f"{1:>10}")
            run_input.write_text(in_text)
            print(f"[INFO] Run {run_id}: DEM enabled — set nfulldump=1 in {run_input.name}", flush=True)
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


def _execute_run_worker(payload: RunWorkerPayload) -> RunRecord:
    return run_one_case(
        payload.run_id,
        payload.sample,
        Path(payload.base_setup),
        Path(payload.base_input),
        Path(payload.output_root),
        payload.prefix,
        Path(payload.phantomsetup_bin),
        Path(payload.phantom_bin),
        payload.ref_mass_kg,
        payload.dry_run,
        payload.earth_sink_id,
        payload.apophis_sink_id,
        Path(payload.ephemeris_cache_dir) if payload.ephemeris_cache_dir else None,
        payload.shape_file,
    )


def _print_run_progress(result: RunRecord, samples: List[RunSample], total: int) -> None:
    idx = result.run_id
    if not (1 <= idx <= len(samples)):
        raise IndexError(f"run_id {idx} out of range for samples list of length {len(samples)}")
    sample = samples[idx - 1]
    mass_str = f"{sample.mass_kg:.6e} kg" if sample.mass_kg is not None else "(template mass)"
    np_str = f" np_apophis={sample.np_apophis}" if sample.np_apophis is not None else ""
    print(f"[INFO] Run {idx}/{total} mass={mass_str}{np_str}", flush=True)
    print(f"[INFO]   status={result.status} run_dir={result.run_dir}", flush=True)


def main() -> int:
    argv_cli, interactive = _strip_interactive_flags(sys.argv[1:])
    if interactive:
        from interactive_run_mass_sobol import run_interactive_wizard

        initial_args = parse_args(argv_cli)
        wizard_argv = run_interactive_wizard(build_parser(), initial_args)
        args = parse_args(wizard_argv)
    else:
        args = parse_args(argv_cli if argv_cli else None)
    base_dir = Path(args.base_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    problem: Optional[Dict[str, object]] = None
    saltelli_meta: Optional[Dict[str, object]] = None
    X_saltelli: Optional[Any] = None
    ephemeris_cache: Optional[Path] = None

    try:
        validate_args(args)
        if args.ephemeris_cache_dir:
            ephemeris_cache = Path(args.ephemeris_cache_dir).expanduser().resolve()
            if not ephemeris_cache.is_dir():
                raise FileNotFoundError(f"Ephemeris cache path is not a directory: {ephemeris_cache}")
            txt_n = sum(1 for p in ephemeris_cache.glob("*.txt") if p.is_file())
            if txt_n == 0:
                print(
                    f"[WARN] No *.txt files in --ephemeris-cache-dir {ephemeris_cache}; "
                    "PHANTOM will attempt live Horizons downloads.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[INFO] Ephemeris cache: {txt_n} *.txt file(s) from {ephemeris_cache}",
                    flush=True,
                )
        batch_basename = build_batch_directory_basename(args, timestamp)
        output_root = Path(args.output_root).resolve() / batch_basename

        if getattr(args, "np_apophis_list", None) is not None:
            # One run per particle count; Sobol sampling is not used in this mode.
            samples = build_np_list_samples(args)
        elif args.saltelli_n is not None:
            problem = build_salib_problem(args)
            try:
                from SALib.sample import sobol as sobol_sample
            except ImportError as exc:
                raise RuntimeError("Saltelli mode requires SALib (pip install -r requirements.txt)") from exc
            X_saltelli = sobol_sample.sample(
                problem,
                args.saltelli_n,
                calc_second_order=args.saltelli_calc_second_order,
                seed=args.seed,
            )
            samples = [run_sample_from_salib_row(X_saltelli[i], args) for i in range(len(X_saltelli))]
            ne = expected_saltelli_num_evals(
                int(problem["num_vars"]), args.saltelli_n, args.saltelli_calc_second_order
            )
            if len(samples) != ne:
                raise RuntimeError(f"internal error: expected {ne} Saltelli rows, got {len(samples)}")
            saltelli_meta = {
                "base_n": args.saltelli_n,
                "calc_second_order": args.saltelli_calc_second_order,
                "num_evals": ne,
                "num_vars": problem["num_vars"],
                "seed": args.seed,
            }
        else:
            samples = build_run_samples(args.num_samples, args)

        base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(args, base_dir, output_root)

        if args.saltelli_n is not None and problem is not None and saltelli_meta is not None:
            (output_root / "saltelli_problem.json").write_text(
                json.dumps(problem, indent=2), encoding="utf-8"
            )
            (output_root / "saltelli_meta.json").write_text(
                json.dumps(saltelli_meta, indent=2), encoding="utf-8"
            )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    col_order = sample_column_order(args)
    samples_csv = output_root / "sobol_mass_samples.csv"
    summary_csv = output_root / "sobol_mass_outputs.csv"
    write_samples_csv(samples_csv, samples, col_order)
    print(f"[INFO] Wrote Sobol sample table: {samples_csv}")
    dim_msg = f"Varying dimensions ({count_dimensions(args)}): {', '.join(col_order)}"
    if args.saltelli_n is not None and saltelli_meta is not None:
        print(
            f"[INFO] Saltelli mode: base_n={args.saltelli_n} evals={saltelli_meta['num_evals']} "
            f"second_order={args.saltelli_calc_second_order}; {dim_msg}"
        )
    else:
        print(f"[INFO] {dim_msg}")
    if args.jobs > 1:
        print(
            f"[INFO] Running {len(samples)} cases with up to {args.jobs} worker process(es) "
            "(set OMP_NUM_THREADS=1 if PHANTOM is OpenMP to limit threads per process)."
        )

    payloads = [
        RunWorkerPayload(
            run_id=idx,
            sample=sample,
            base_setup=str(base_setup),
            base_input=str(base_input),
            output_root=str(output_root),
            prefix=args.prefix,
            phantomsetup_bin=str(phantomsetup_bin),
            phantom_bin=str(phantom_bin),
            ref_mass_kg=args.apophis_ref_mass_kg if _mass_bounds_active(args) else None,
            dry_run=args.dry_run,
            earth_sink_id=args.sink_earth_id,
            apophis_sink_id=args.sink_apophis_id,
            ephemeris_cache_dir=str(ephemeris_cache) if ephemeris_cache is not None else None,
            shape_file=args.shape_file if args.shape_file else None,
        )
        for idx, sample in enumerate(samples, start=1)
    ]
    total_runs = len(samples)
    results: List[RunRecord] = []
    if args.jobs == 1:
        for p in payloads:
            result = _execute_run_worker(p)
            results.append(result)
            _print_run_progress(result, samples, total_runs)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_execute_run_worker, p) for p in payloads]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                _print_run_progress(result, samples, total_runs)

    write_summary_csv(summary_csv, results, col_order)

    sorted_results = sorted(results, key=lambda r: r.run_id)

    saltelli_y_rows: List[Dict[str, object]] = []
    if args.saltelli_n is not None:
        for result in sorted_results:
            saltelli_y_rows.append(
                {
                    "eval_index": result.run_id - 1,
                    "run_id": result.run_id,
                    "closest_approach_km": result.closest_approach_km,
                    "closest_approach_au": result.closest_approach_au,
                    "status": result.status,
                }
            )

    print(f"[INFO] Summary written to: {summary_csv}")

    if args.saltelli_n is not None and X_saltelli is not None and problem is not None:
        num_vars = int(problem["num_vars"])
        if X_saltelli.shape[1] != num_vars:
            raise ValueError(
                f"Saltelli design has {X_saltelli.shape[1]} columns but problem specifies "
                f"num_vars={num_vars}; design matrix and problem dict are inconsistent."
            )
        manifest_path = output_root / "saltelli_eval_manifest.csv"
        names = list(problem["names"])
        with manifest_path.open("w", newline="", encoding="utf-8") as mf:
            mw = csv.writer(mf)
            mw.writerow(["eval_index", "run_id", *names])
            for j in range(len(samples)):
                row_vals = [float(X_saltelli[j, k]) for k in range(num_vars)]
                mw.writerow([j, j + 1, *row_vals])
        print(f"[INFO] Wrote Saltelli eval manifest: {manifest_path}")

        y_path = output_root / "saltelli_Y.csv"
        y_fields = ["eval_index", "run_id", "closest_approach_km", "closest_approach_au", "status"]
        with y_path.open("w", newline="", encoding="utf-8") as yf:
            yw = csv.DictWriter(yf, fieldnames=y_fields)
            yw.writeheader()
            for row in saltelli_y_rows:
                yw.writerow(row)
        print(f"[INFO] Wrote Saltelli model outputs (evaluation order): {y_path}")
        prob_p = (output_root / "saltelli_problem.json").resolve()
        meta_p = (output_root / "saltelli_meta.json").resolve()
        print(
            "[INFO] Sobol indices: python3 Analysis/Analysis.py --method saltelli \\\n"
            f"       --sobol-problem-json {prob_p} \\\n"
            f"       --saltelli-meta-json {meta_p} \\\n"
            f"       --saltelli-y-csv {y_path.resolve()} \\\n"
            "       --saltelli-y-column closest_approach_au"
            + (
                " \\\n       --saltelli-calc-second-order"
                if args.saltelli_calc_second_order
                else ""
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
