"""Boulders: glue a settled body's grains into bonded boulders with phantomclumps, once per run.

Spec: docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md. moddump_clumps.f90 gives grains a clump
ID (part%iclump); dem.f90 bonds grains sharing an ID on the first force evaluation once kb_cgs > 0. The cached
settled body is never modified: each run dir gets its own clumped copy.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

SHAPES = ("round", "ellipsoid", "grown")    # moddump_clumps ishape 0, 1, 2
PLACED = ("none", "binary", "file")         # moddump_clumps iplaced 0, 1, 2
UNCLUMPED = "unclumped"                     # cached body staged as unclumped_00000 + unclumped.in
CLUMPED_DUMP = "clumped_00000"              # moddump appends _00000 to the output name
PLACED_FILE_NAME = "boulders.txt"           # Fortran placed_file is character(len=256): pass a short name
BOND_BINARY_MARKER = b"clump bond tensile strength"  # write_inopt text of bond_sigma_pa (dem.f90)
SETUP_LOG_MARKER = "boulders in body"                # setup_solarsystem.f90, packing_file path
PHANTOM_LOG_MARKER = "DEM clump bonds enabled"       # dem.f90 dem_cohesion_summary, kb_cgs > 0


@dataclass(frozen=True)
class BoulderSpec:
    """phantomclumps answers + bond keys. None = blank answer (Fortran default) / see bond_in_keys."""

    f_boulder: float = 0.3          # fraction of the whole body put into RANDOM boulders
    rb_min: float = 1.5             # grain radii
    rb_max: float = 4.0             # grain radii
    q_slope: float = 3.5            # dN/dR ~ R^-q
    shape: str = "round"
    axis_min: Optional[float] = None    # ellipsoid only
    jagged: Optional[float] = None      # grown only
    placed: str = "none"
    lobe_ratio: Optional[float] = None  # binary only
    placed_file: Optional[str] = None   # file only (absolute path; staged as boulders.txt)
    nmin_clump: Optional[int] = None
    seed: Optional[int] = None
    bond_sigma_pa: float = 0.0
    bond_tau_ratio: float = 1.0
    kb_cgs: Optional[float] = None      # None = kn_cgs of the run .in
    bond_reach: float = 0.1


def _tok(v) -> str:
    if v is None:
        return ""
    return str(v) if isinstance(v, int) else f"{v:.12g}"


def clumps_stdin(spec: BoulderSpec) -> str:
    """Answers in moddump_clumps.f90 prompt order (lines 110-122); conditional prompts only when asked."""
    lines = [_tok(spec.f_boulder), _tok(spec.rb_min), _tok(spec.rb_max), _tok(spec.q_slope),
             str(SHAPES.index(spec.shape))]
    if spec.shape == "ellipsoid":
        lines.append(_tok(spec.axis_min))
    elif spec.shape == "grown":
        lines.append(_tok(spec.jagged))
    lines.append(str(PLACED.index(spec.placed)))
    if spec.placed == "binary":
        lines.append(_tok(spec.lobe_ratio))
    elif spec.placed == "file":
        lines.append(PLACED_FILE_NAME)
    lines += [_tok(spec.bond_reach), _tok(spec.nmin_clump), _tok(spec.seed)]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ClumpResult:
    n_grains: int
    n_placed: int
    n_placed_grains: int
    n_random: int
    n_random_grains: int
    f_dust: float
    n_dropped: int
    ran_out: bool

    @property
    def n_boulders(self) -> int:
        return self.n_placed + self.n_random

    @property
    def n_in_boulders(self) -> int:
        return self.n_placed_grains + self.n_random_grains

    @property
    def f_boulder_actual(self) -> float:
        return self.n_in_boulders / self.n_grains if self.n_grains else 0.0


def _num(pattern: str, text: str, default=None, cast=int):
    m = re.search(pattern, text, re.M)
    return cast(m.group(1)) if m else default


def parse_clumps_log(text: str) -> ClumpResult:
    """Summary block printed by moddump_clumps.f90 (lines 217-230)."""
    n_grains = _num(r"^\s*grains\s+=\s+(\d+)", text)
    if n_grains is None:
        raise RuntimeError("phantomclumps log has no summary block (did it crash before gluing?)")
    placed = re.search(r"^\s*placed boulders\s+=\s+(\d+)\s+holding\s+(\d+)\s+grains", text, re.M)
    return ClumpResult(
        n_grains=n_grains,
        n_placed=int(placed.group(1)) if placed else 0,
        n_placed_grains=int(placed.group(2)) if placed else 0,
        n_random=_num(r"^\s*random boulders\s+=\s+(\d+)", text, 0),
        n_random_grains=_num(r"^\s*grains in random\s+=\s+(\d+)", text, 0),
        f_dust=_num(r"^\s*dust \(free grains\)\s+=\s+([\d.]+)", text, float("nan"), float),
        n_dropped=_num(r"grains cut off by carving, back to dust\s+=\s+(\d+)", text, 0),
        ran_out="ran out of free grains" in text,
    )


def bond_in_keys(spec: BoulderSpec, kn_cgs: float) -> Dict[str, float]:
    """The .in keys that switch the bonds on; bond_reach must match the phantomclumps answer."""
    return {
        "kb_cgs": spec.kb_cgs if spec.kb_cgs is not None else float(kn_cgs),
        "bond_reach": spec.bond_reach,
        "bond_sigma_pa": spec.bond_sigma_pa,
        "bond_tau_pa": spec.bond_tau_ratio * spec.bond_sigma_pa,
    }


def validate_spec(spec: BoulderSpec) -> None:
    """Prompt ranges from moddump_clumps.f90 plus the bond keys; raises ValueError."""
    if not 0.0 <= spec.f_boulder <= 1.0:
        raise ValueError(f"f_boulder must be in [0, 1] (got {spec.f_boulder})")
    if spec.rb_min < 1.0:
        raise ValueError(f"rb_min must be >= 1 grain radius (got {spec.rb_min})")
    if spec.rb_max < spec.rb_min:
        raise ValueError(f"rb_max ({spec.rb_max}) must be >= rb_min ({spec.rb_min})")
    if spec.q_slope < 0.0:
        raise ValueError(f"q_slope must be >= 0 (got {spec.q_slope})")
    if spec.shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES} (got {spec.shape!r})")
    if spec.placed not in PLACED:
        raise ValueError(f"placed must be one of {PLACED} (got {spec.placed!r})")
    if spec.axis_min is not None:
        if spec.shape != "ellipsoid":
            raise ValueError("axis_min only applies to shape ellipsoid")
        if not 0.1 <= spec.axis_min <= 1.0:
            raise ValueError(f"axis_min must be in [0.1, 1] (got {spec.axis_min})")
    if spec.jagged is not None:
        if spec.shape != "grown":
            raise ValueError("jagged only applies to shape grown")
        if not 0.0 <= spec.jagged <= 1.0:
            raise ValueError(f"jagged must be in [0, 1] (got {spec.jagged})")
    if spec.lobe_ratio is not None:
        if spec.placed != "binary":
            raise ValueError("lobe_ratio only applies to placed binary")
        if not 0.1 <= spec.lobe_ratio <= 1.0:
            raise ValueError(f"lobe_ratio must be in [0.1, 1] (got {spec.lobe_ratio})")
    if spec.placed == "file":
        if not spec.placed_file:
            raise ValueError("placed file needs a boulder file (x y z a b c [rx ry rz], km and deg)")
        if not Path(spec.placed_file).is_file():
            raise ValueError(f"boulder file not found: {spec.placed_file}")
    elif spec.placed_file is not None:
        raise ValueError("a boulder file only applies to placed file")
    if spec.nmin_clump is not None and spec.nmin_clump < 2:
        raise ValueError(f"nmin_clump must be >= 2 grains (got {spec.nmin_clump})")
    if spec.bond_sigma_pa < 0.0:
        raise ValueError(f"bond_sigma_pa must be >= 0 (got {spec.bond_sigma_pa})")
    if spec.bond_tau_ratio < 0.0:
        raise ValueError(f"bond_tau_ratio must be >= 0 (got {spec.bond_tau_ratio})")
    if spec.kb_cgs is not None and spec.kb_cgs <= 0.0:
        raise ValueError(f"kb_cgs must be > 0 (got {spec.kb_cgs}); leave it unset for kn_cgs")
    if spec.bond_reach <= 0.0:
        raise ValueError(f"bond_reach must be > 0 (got {spec.bond_reach})")


def fortran_nint(x: float) -> int:
    """Fortran nint for x >= 0: round half away from zero (Python round() is half-to-even)."""
    return int(math.floor(x + 0.5))


def requested(spec: BoulderSpec, n_grains: int) -> bool:
    """Were any boulders asked for? False when nothing is placed and the random target is zero.

    moddump_clumps.f90 sets ntarget = nint(f_boulder * npart); below half a grain it makes no random
    boulders, so a tiny f_boulder on a small body is an all-dust run, not an error.
    """
    return spec.placed != "none" or fortran_nint(spec.f_boulder * n_grains) > 0


def bonds_wanted(run_dir: Path) -> bool:
    """False when run_dir/boulders.json says the run made no boulders (then no bond keys go in the .in,
    so the all-dust baseline matches a non-boulder run). Missing json: True (bonds on)."""
    path = Path(run_dir) / "boulders.json"
    if not path.is_file():
        return True
    info = json.loads(path.read_text(encoding="utf-8"))
    return info["result"]["n_boulders"] > 0


def resolve(sample) -> BoulderSpec:
    """The sample's fixed BoulderSpec with its per-sample axis values (f_boulder, rb_max) filled in."""
    spec = sample.boulders
    return dataclasses.replace(
        spec,
        f_boulder=spec.f_boulder if sample.f_boulder is None else float(sample.f_boulder),
        rb_max=spec.rb_max if sample.rb_max is None else float(sample.rb_max),
    )


def check_bonds_on(log_text: str) -> None:
    if PHANTOM_LOG_MARKER not in log_text:
        raise RuntimeError(f"phantom.log has no '{PHANTOM_LOG_MARKER}': kb_cgs was 0, so the boulders "
                           "ran unbonded")


def _runner():
    try:
        import run_mass_sobol_phantom as mod
    except ImportError:
        from sobol import run_mass_sobol_phantom as mod
    return mod


def clump_body(run_dir: Path, body, spec: BoulderSpec, clumps_bin: Path) -> ClumpResult:
    """Copy the cached body into run_dir and glue it: unclumped_00000 -> clumped_00000."""
    run_dir = Path(run_dir)
    for stale in (run_dir / CLUMPED_DUMP, run_dir / "clumped.in"):
        if stale.exists():
            stale.unlink()  # moddump reads <out>.in as revised defaults
    shutil.copy2(Path(body.dir) / body.dump, run_dir / f"{UNCLUMPED}_00000")
    shutil.copy2(Path(body.dir) / body.infile, run_dir / f"{UNCLUMPED}.in")
    if spec.placed == "file":
        shutil.copy2(Path(spec.placed_file), run_dir / PLACED_FILE_NAME)
    log = run_dir / "clumps.log"
    with open(log, "w", encoding="utf-8") as fh:
        # --maxp: without it phantom_moddump allocates ~6 GB (as in encounter.prepare_hyperbola_run)
        proc = subprocess.run(
            [str(clumps_bin), f"{UNCLUMPED}_00000", "clumped", "0", f"--maxp={max(2000, 4 * body.n_kept)}"],
            cwd=str(run_dir), text=True, input=clumps_stdin(spec), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not (run_dir / CLUMPED_DUMP).is_file():
        raise RuntimeError(f"phantomclumps failed ({proc.returncode}); see {log}")
    return parse_clumps_log(log.read_text(encoding="utf-8", errors="replace"))


def clump_sample(run_dir: Path, body, sample) -> Dict[str, str]:
    """Clump the body for one run; write boulders.json; return the CSV param columns."""
    if not sample.phantomclumps_bin:
        raise RuntimeError("boulder run without phantomclumps (cd sobol && make demtools)")
    spec = resolve(sample)
    res = clump_body(Path(run_dir), body, spec, Path(sample.phantomclumps_bin))
    if requested(spec, res.n_grains) and res.n_boulders == 0:
        raise RuntimeError(f"phantomclumps made no boulders (f_boulder={spec.f_boulder}, "
                           f"placed={spec.placed}); see clumps.log")
    result = {**dataclasses.asdict(res), "n_boulders": res.n_boulders, "n_in_boulders": res.n_in_boulders,
              "f_boulder_actual": res.f_boulder_actual}
    (Path(run_dir) / "boulders.json").write_text(
        json.dumps({"spec": dataclasses.asdict(spec), "result": result}, indent=2), encoding="utf-8")
    return {"f_boulder": f"{spec.f_boulder:.12g}", "rb_max": f"{spec.rb_max:.12g}",
            "f_boulder_actual": f"{res.f_boulder_actual:.6g}", "n_boulders": str(res.n_boulders),
            "boulder_placed": spec.placed, "bond_sigma_pa": f"{spec.bond_sigma_pa:.12g}"}


def check_run_logs(run_dir: Path, *, setup_log: bool) -> None:
    """After phantom returns (before metrics/cleanup): refuse a run whose boulders never bonded."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "boulders.json").read_text(encoding="utf-8"))
    if info["result"]["n_boulders"] == 0:
        return  # all-dust baseline: no bonds expected
    if setup_log and SETUP_LOG_MARKER not in (run_dir / "setup.log").read_text(errors="replace"):
        raise RuntimeError(f"setup.log has no '{SETUP_LOG_MARKER}' line: phantomsetup did not load the clump "
                           "IDs (binary older than the DEMsync-mia clump merge?)")
    check_bonds_on((run_dir / "phantom.log").read_text(errors="replace"))


def attach_tools(samples, phantom_dir: Path, phantom_bin: Path, dry_run: bool) -> None:
    """Resolve phantomclumps and refuse a phantom without clump bonds; sets sample.phantomclumps_bin."""
    wanted = [s for s in samples if s.boulders is not None]
    if not wanted or dry_run:
        return
    r = _runner()
    try:
        clumps = r.resolve_phantom_executable(Path(phantom_dir), "phantomclumps", must_exist=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}. Build it with: cd sobol && make demtools") from None
    if not r._binary_has_string(Path(phantom_bin), BOND_BINARY_MARKER):
        raise RuntimeError(f"{phantom_bin} has no clump bonds (bond_sigma_pa); rebuild from DEMsync-mia "
                           "(cd sobol && make setup && make)")
    for s in wanted:
        s.phantomclumps_bin = str(clumps)


_ARG_FIELDS = (
    ("f_boulder", "f_boulder"), ("rb_max", "rb_max"), ("rb_min", "rb_min"), ("boulder_q", "q_slope"),
    ("boulder_shape", "shape"), ("boulder_axis_min", "axis_min"), ("boulder_jagged", "jagged"),
    ("boulder_placed", "placed"), ("lobe_ratio", "lobe_ratio"), ("boulder_file", "placed_file"),
    ("boulder_nmin", "nmin_clump"), ("bond_sigma_pa", "bond_sigma_pa"), ("bond_tau_ratio", "bond_tau_ratio"),
    ("kb_cgs", "kb_cgs"), ("bond_reach", "bond_reach"),
)
_AXES = (("f_boulder", "f_boulder_min", "f_boulder_max"), ("rb_max", "rb_max_min", "rb_max_max"))
BOULDER_ARG_DESTS = frozenset(
    [d for d, _ in _ARG_FIELDS] + ["boulder_seed"] + [x for _, lo, hi in _AXES for x in (lo, hi)])


def spec_from_args(args) -> BoulderSpec:
    """Fixed BoulderSpec from CLI args (axis values stay on the samples)."""
    kw = {field: getattr(args, dest) for dest, field in _ARG_FIELDS if getattr(args, dest, None) is not None}
    if kw.get("placed_file"):
        kw["placed_file"] = str(Path(kw["placed_file"]).expanduser().resolve())
    kw["seed"] = args.boulder_seed if getattr(args, "boulder_seed", None) is not None else args.seed
    return BoulderSpec(**kw)


def _axis_values(args, param: str, default: float):
    lo, hi = getattr(args, f"{param}_min"), getattr(args, f"{param}_max")
    return (lo, hi) if lo is not None else (default,)


def validate_args(args) -> None:
    """Called from run_mass_sobol_phantom.validate_args after _validate_body_encounter."""
    import sys
    given = sorted(d for d in BOULDER_ARG_DESTS if getattr(args, d, None) is not None)
    if not getattr(args, "boulders", False):
        if given:
            raise ValueError("boulder flags need --boulders: "
                             + ", ".join("--" + d.replace("_", "-") for d in given))
        return
    if args.body_source != "settled":
        raise ValueError("--boulders needs --body-source settled (phantomclumps glues a settled body)")
    if getattr(args, "np_apophis_list", None) and any(getattr(args, lo) is not None for _, lo, _ in _AXES):
        raise ValueError("boulder axes are not sampled in --np-apophis-list mode; pass fixed "
                         "--f-boulder / --rb-max instead")
    for param, lo_attr, hi_attr in _AXES:
        lo, hi = getattr(args, lo_attr), getattr(args, hi_attr)
        flag = "--" + param.replace("_", "-")
        if (lo is None) ^ (hi is None):
            raise ValueError(f"{flag}: set both {flag}-min and {flag}-max, or neither")
        if lo is not None:
            if getattr(args, param) is not None:
                raise ValueError(f"{flag} and {flag}-min/max are mutually exclusive")
            if not lo < hi:
                raise ValueError(f"{flag} bounds must satisfy min < max")
    if args.bond_sigma_pa is None:
        raise ValueError("--boulders needs --bond-sigma-pa (Pa; 0 = stretch-limit breaking only). "
                         "No default: no value has been tested yet")
    base = spec_from_args(args)
    for fb in _axis_values(args, "f_boulder", base.f_boulder):
        for rb in _axis_values(args, "rb_max", base.rb_max):
            validate_spec(dataclasses.replace(base, f_boulder=fb, rb_max=rb))
    if base.bond_sigma_pa == 0.0:
        print("[WARN] --bond-sigma-pa 0: bonds break only on stretch past bond_reach", file=sys.stderr)
