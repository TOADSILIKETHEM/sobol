# Boulders in Sweeps and TUIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let sobol sweeps (CLI, `tui_run.py`, wizard) and single runs (`tui_sim_render.py`) glue the cached settled body into bonded boulders with `phantomclumps`, with `f_boulder` and `rb_max` as Sobol axes.

**Architecture:** A new `sobol/boulders.py` (same style as `encounter.py`) owns the spec, the `phantomclumps` stdin, log parsing, the per-run clump step and the log checks. The runner gains flags, two optional Sobol axes, and a per-sample clump step between "settled body loaded" and "body staged" on both encounter paths. Bond keys reach the run `.in` through the existing `apply_run_sample_to_in()`, which both paths already call.

**Tech Stack:** Python 3, argparse, Textual (TUIs), pytest, PHANTOM moddump binaries (`phantomclumps`, `phantomflyby`).

**Spec:** `sobol/docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md` (read it first).

## Global Constraints

- **No commits.** The user commits only after testing (standing rule). Each task ends with a checkpoint: run the tests, `git -C /home/mboyle/Honours/sobol diff --stat`, and stop for review. The tree already has unrelated uncommitted changes; do not stage, revert or commit them.
- All paths below are relative to `/home/mboyle/Honours/sobol/` unless they start with `../`.
- Test command: `cd /home/mboyle/Honours/sobol && python3 -m pytest <file> -v`. Full suite (~2.5 min): `python3 -m pytest tests`.
- Boulders require `--body-source settled` (which already forces `--dem-model particle` and `--use-dem-fixed true`).
- `bond_sigma_pa` has **no default**; `--boulders` without it is an error.
- `bond_reach` default 0.1; `bond_tau_pa = bond_tau_ratio × bond_sigma_pa`, ratio default 1.0; `kb_cgs` default = the run `.in`'s `kn_cgs`.
- BoulderSpec defaults: `f_boulder 0.3`, `rb_min 1.5`, `rb_max 4.0`, `q_slope 3.5`, shape `round`, placed `none`.
- Batches without `--boulders` must produce byte-identical slugs, column orders and dimension counts to today (regression values in Task 4).
- `phantomclumps` prompt order (`../phantom/src/utils/moddump_clumps.f90:110-122`): `f_boulder`, `rb_min`, `rb_max`, `q_slope`, `ishape`, [`axis_min` if ellipsoid | `jagged` if grown], `iplaced`, [`lobe_ratio` if binary | file name if file], `bond_reach`, `nmin_clump`, seed. A blank line keeps the Fortran default.
- moddump output naming: `phantomclumps unclumped_00000 clumped 0` writes `clumped_00000` and `clumped.in`.
- Log markers: `phantomclumps` summary (`moddump_clumps.f90:217-230`); `setup.log` ` boulders in body ` (`setup_solarsystem.f90:448`); `phantom.log` ` DEM clump bonds enabled` (`dem.f90` `dem_cohesion_summary`, printed when `kb_cgs > 0`). Binary marker for bond support: `b"clump bond tensile strength"`.

## Review Focus

1. **`f_boulder = 0`, placed `none`** (the RQ1 baseline): `phantomclumps` still runs, gives an all-dust body, no error, no bond-log check. Expected: the run succeeds with `n_boulders = 0`. Test in Task 2 (`test_clump_sample_all_dust_is_not_an_error`) and Task 5 (`test_check_run_logs_skips_all_dust`).
2. **Hyperbola `.in` already has `kb_cgs = 0`** (spike: `fly.in`). Expected: replaced by the bond value, not duplicated. Test in Task 5 (`test_apply_in_replaces_present_bond_keys`).
3. **RQ3 pair differs only in `--boulder-placed`.** Expected: different batch slugs (different folders). Test in Task 4 (`test_slug_distinguishes_placed`).
4. **Boulder flags given without `--boulders`** (e.g. TUI fields filled but checkbox off). Expected: an error, not silently ignored. Tests in Task 3 (`test_boulder_flags_need_boulders`) and Task 8 (`test_sweep_tui_boulder_fields_need_checkbox`).
5. **Boulder axes with `--np-apophis-list`** (that mode ignores Sobol dimensions). Expected: rejected. Test in Task 3 (`test_axes_rejected_in_np_list_mode`).

---

### Task 1: `boulders.py` core (spec, stdin, log parsing, bond keys, validation)

**Files:**
- Create: `boulders.py`
- Test: `tests/test_boulders.py`

**Interfaces:**
- Produces:
  - `BoulderSpec` (frozen dataclass; fields and defaults below)
  - `SHAPES = ("round", "ellipsoid", "grown")`, `PLACED = ("none", "binary", "file")`
  - `UNCLUMPED = "unclumped"`, `CLUMPED_DUMP = "clumped_00000"`, `PLACED_FILE_NAME = "boulders.txt"`
  - `BOND_BINARY_MARKER: bytes`, `SETUP_LOG_MARKER: str`, `PHANTOM_LOG_MARKER: str`
  - `clumps_stdin(spec: BoulderSpec) -> str`
  - `ClumpResult` (frozen dataclass) with properties `n_boulders`, `n_in_boulders`, `f_boulder_actual`
  - `parse_clumps_log(text: str) -> ClumpResult`
  - `bond_in_keys(spec: BoulderSpec, kn_cgs: float) -> Dict[str, float]`
  - `validate_spec(spec: BoulderSpec) -> None` (raises `ValueError`)
  - `requested(spec: BoulderSpec) -> bool`
  - `resolve(sample) -> BoulderSpec` (sample = `RunSample` with `.boulders`, `.f_boulder`, `.rb_max`)
  - `check_bonds_on(log_text: str) -> None` (raises `RuntimeError`)

- [ ] **Step 1: Write the failing tests**

`tests/test_boulders.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd

# Summary block of a real run (scratchpad spike 2026-10-06, body np300_srho1_712bb7806311,
# f_boulder=0.3, contact binary, all other prompts default).
CLUMPS_LOG_BINARY = """\
 --- gluing grains into boulders ---
 contact binary along the long axis
 lobe 1:       65 grains, radius  1.072E-01 km
 lobe 2:       64 grains, radius  8.105E-02 km
 grains               =       391
 placed boulders      =         2  holding       129 grains ( 0.330 of the body)
 random boulders      =         8  shape: round
 grains in random     =       142  ( 0.363 of the body, target  0.300)
 grains per random boulder =      6 to     46, mean     17.8
 dust (free grains)   =  0.307
 grains cut off by carving, back to dust =         2
 random boulders under the minimum size, skipped =         1
 now set kb_cgs > 0 (and the same bond_reach) in the .in file and settle again
"""

CLUMPS_LOG_DUST = """\
 --- gluing grains into boulders ---
 grains               =       391
 random boulders      =         0  shape: round
 grains in random     =         0  ( 0.000 of the body, target  0.000)
 dust (free grains)   =  1.000
 grains cut off by carving, back to dust =         0
 random boulders under the minimum size, skipped =         0
"""


def test_stdin_defaults_round_none():
    s = bd.BoulderSpec()
    assert bd.clumps_stdin(s).split("\n") == ["0.3", "1.5", "4", "3.5", "0", "0", "0.1", "", "", ""]


@pytest.mark.parametrize("shape,extra", [("round", []), ("ellipsoid", ["0.5"]), ("grown", ["0.7"])])
@pytest.mark.parametrize("placed,pextra", [("none", []), ("binary", ["0.6"]), ("file", ["boulders.txt"])])
def test_stdin_conditional_prompts(shape, extra, placed, pextra):
    s = bd.BoulderSpec(shape=shape, axis_min=0.5 if shape == "ellipsoid" else None,
                       jagged=0.7 if shape == "grown" else None, placed=placed,
                       lobe_ratio=0.6 if placed == "binary" else None,
                       placed_file="/x/b.txt" if placed == "file" else None, nmin_clump=3, seed=42)
    lines = bd.clumps_stdin(s).rstrip("\n").split("\n")
    expect = (["0.3", "1.5", "4", "3.5", str(bd.SHAPES.index(shape))] + extra
              + [str(bd.PLACED.index(placed))] + pextra + ["0.1", "3", "42"])
    assert lines == expect


def test_parse_binary_log():
    r = bd.parse_clumps_log(CLUMPS_LOG_BINARY)
    assert (r.n_grains, r.n_placed, r.n_placed_grains, r.n_random, r.n_random_grains) == (391, 2, 129, 8, 142)
    assert r.n_dropped == 2 and r.f_dust == pytest.approx(0.307) and not r.ran_out
    assert r.n_boulders == 10 and r.n_in_boulders == 271
    assert r.f_boulder_actual == pytest.approx(271 / 391)


def test_parse_dust_log_has_no_placed_line():
    r = bd.parse_clumps_log(CLUMPS_LOG_DUST)
    assert (r.n_placed, r.n_random, r.n_boulders, r.f_boulder_actual) == (0, 0, 0, 0.0)


def test_parse_flags_ran_out():
    text = CLUMPS_LOG_BINARY + " *** WARNING: ran out of free grains or attempts before the random target ***\n"
    assert bd.parse_clumps_log(text).ran_out


def test_parse_without_summary_raises():
    with pytest.raises(RuntimeError, match="summary"):
        bd.parse_clumps_log("Phantom moddump: crashed\n")


def test_bond_in_keys_defaults():
    k = bd.bond_in_keys(bd.BoulderSpec(bond_sigma_pa=10.0), kn_cgs=1e7)
    assert k == {"kb_cgs": 1e7, "bond_reach": 0.1, "bond_sigma_pa": 10.0, "bond_tau_pa": 10.0}


def test_bond_in_keys_overrides():
    k = bd.bond_in_keys(bd.BoulderSpec(bond_sigma_pa=10.0, bond_tau_ratio=0.5, kb_cgs=3e7, bond_reach=0.05), 1e7)
    assert k == {"kb_cgs": 3e7, "bond_reach": 0.05, "bond_sigma_pa": 10.0, "bond_tau_pa": 5.0}


@pytest.mark.parametrize("kw,match", [
    ({"f_boulder": 1.2}, "f_boulder"),
    ({"rb_min": 0.5}, "rb_min"),
    ({"rb_min": 3.0, "rb_max": 2.0}, "rb_max"),
    ({"q_slope": -1.0}, "q_slope"),
    ({"shape": "cube"}, "shape"),
    ({"axis_min": 0.5}, "ellipsoid"),
    ({"shape": "ellipsoid", "axis_min": 0.05}, "axis_min"),
    ({"jagged": 0.5}, "grown"),
    ({"lobe_ratio": 0.5}, "binary"),
    ({"placed": "binary", "lobe_ratio": 0.05}, "lobe_ratio"),
    ({"placed": "file"}, "file"),
    ({"placed_file": "/tmp/x"}, "file"),
    ({"nmin_clump": 1}, "nmin"),
    ({"bond_sigma_pa": -1.0}, "bond_sigma_pa"),
    ({"bond_tau_ratio": -1.0}, "bond_tau_ratio"),
    ({"kb_cgs": 0.0}, "kb_cgs"),
    ({"bond_reach": 0.0}, "bond_reach"),
])
def test_validate_spec_rejects(kw, match):
    with pytest.raises(ValueError, match=match):
        bd.validate_spec(bd.BoulderSpec(**kw))


def test_validate_spec_file_must_exist(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        bd.validate_spec(bd.BoulderSpec(placed="file", placed_file=str(tmp_path / "nope.txt")))
    f = tmp_path / "b.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    bd.validate_spec(bd.BoulderSpec(placed="file", placed_file=str(f)))


def test_requested():
    assert bd.requested(bd.BoulderSpec())
    assert bd.requested(bd.BoulderSpec(f_boulder=0.0, placed="binary"))
    assert not bd.requested(bd.BoulderSpec(f_boulder=0.0))


def test_resolve_takes_axis_values_from_sample():
    base = bd.BoulderSpec(bond_sigma_pa=10.0)
    s = SimpleNamespace(boulders=base, f_boulder=0.55, rb_max=None)
    r = bd.resolve(s)
    assert (r.f_boulder, r.rb_max, r.bond_sigma_pa) == (0.55, 4.0, 10.0)


def test_check_bonds_on():
    bd.check_bonds_on("...\n DEM clump bonds enabled\n   kb_cgs =   1.0000E+07\n")
    with pytest.raises(RuntimeError, match="unbonded"):
        bd.check_bonds_on("no bonds here\n")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'boulders'`.

- [ ] **Step 3: Write `boulders.py`**

```python
"""Boulders: glue a settled body's grains into bonded boulders with phantomclumps, once per run.

Spec: docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md. moddump_clumps.f90 gives grains a clump
ID (part%iclump); dem.f90 bonds grains sharing an ID on the first force evaluation once kb_cgs > 0. The cached
settled body is never modified: each run dir gets its own clumped copy.
"""
from __future__ import annotations

import dataclasses
import re
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


def requested(spec: BoulderSpec) -> bool:
    """False only for the all-dust baseline (f_boulder = 0, nothing placed)."""
    return spec.f_boulder > 0.0 or spec.placed != "none"


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_boulders.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)**

`git -C /home/mboyle/Honours/sobol status --short boulders.py tests/test_boulders.py` shows both as new. Stop for review.

---

### Task 2: `boulders.py` run side (clump one run, attach tools, log checks)

**Files:**
- Modify: `boulders.py` (append)
- Test: `tests/test_boulders.py` (append)

**Interfaces:**
- Consumes: Task 1 names; `run_mass_sobol_phantom.resolve_phantom_executable(phantom_dir: Path, name: str, *, must_exist: bool) -> Path`; `run_mass_sobol_phantom._binary_has_string(path: Path, needle: bytes) -> bool`; `settled_body.SettledBody` (`.dir`, `.dump`, `.infile`, `.n_kept`).
- Produces:
  - `clump_body(run_dir: Path, body, spec: BoulderSpec, clumps_bin: Path) -> ClumpResult`
  - `clump_sample(run_dir: Path, body, sample) -> Dict[str, str]` (CSV param columns; writes `boulders.json`)
  - `check_run_logs(run_dir: Path, *, setup_log: bool) -> None`
  - `attach_tools(samples, phantom_dir: Path, phantom_bin: Path, dry_run: bool) -> None` (sets `sample.phantomclumps_bin`)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_boulders.py`)

```python
import json
import stat

import settled_body as sb

FAKE_CLUMPS = r"""#!/bin/sh
echo "clumps $*" >> calls.log
[ -e "$2.in" ] && { echo "stale $2.in present"; exit 5; }
cat > clumps_stdin.txt
cp "$1" "$2_00000"
cat "$FAKE_CLUMPS_LOG"
"""


def _fake_clumps(tmp_path, log_text):
    log = tmp_path / "fake_clumps_output.txt"
    log.write_text(log_text)
    p = tmp_path / "phantomclumps"
    p.write_text(FAKE_CLUMPS)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p, log


def _body(tmp_path):
    d = tmp_path / "body"
    d.mkdir()
    (d / "cropped_00002").write_text("dump")
    (d / "cropped.in").write_text("kn_cgs = 1.0E+07\n")
    (d / "body.json").write_text(json.dumps({
        "key": "np300_srho1_abc", "dump": "cropped_00002", "infile": "cropped.in", "n_kept": 391,
        "n_settled": 923, "packing_fraction": 0.83, "utime_s": 2.745e-6}))
    return sb.SettledBody.load(d)


def _run(tmp_path, monkeypatch, log_text, spec):
    clumps, log = _fake_clumps(tmp_path, log_text)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    run = tmp_path / "run_0001"
    run.mkdir()
    s = SimpleNamespace(boulders=spec, f_boulder=None, rb_max=None, phantomclumps_bin=str(clumps))
    return run, bd.clump_sample(run, _body(tmp_path), s)


def test_clump_body_runs_tool_and_stages_body(tmp_path, monkeypatch):
    clumps, log = _fake_clumps(tmp_path, CLUMPS_LOG_BINARY)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "clumped.in").write_text("stale")  # moddump would read it as revised defaults
    spec = bd.BoulderSpec(placed="binary", bond_sigma_pa=10.0)
    r = bd.clump_body(run, _body(tmp_path), spec, clumps)
    assert r.n_boulders == 10
    assert (run / "unclumped_00000").read_text() == "dump"
    assert (run / "unclumped.in").is_file()
    assert (run / bd.CLUMPED_DUMP).is_file()
    assert (run / "clumps_stdin.txt").read_text() == bd.clumps_stdin(spec)
    assert "unclumped_00000 clumped 0 --maxp=2000" in (run / "calls.log").read_text()


def test_clump_body_failure_raises(tmp_path):
    bad = tmp_path / "phantomclumps"
    bad.write_text("#!/bin/sh\nexit 3\n")
    bad.chmod(bad.stat().st_mode | stat.S_IXUSR)
    run = tmp_path / "run_0001"
    run.mkdir()
    with pytest.raises(RuntimeError, match="phantomclumps failed"):
        bd.clump_body(run, _body(tmp_path), bd.BoulderSpec(), bad)


def test_clump_body_stages_placed_file(tmp_path, monkeypatch):
    clumps, log = _fake_clumps(tmp_path, CLUMPS_LOG_BINARY)
    monkeypatch.setenv("FAKE_CLUMPS_LOG", str(log))
    f = tmp_path / "my_boulders.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    run = tmp_path / "run_0001"
    run.mkdir()
    bd.clump_body(run, _body(tmp_path), bd.BoulderSpec(placed="file", placed_file=str(f)), clumps)
    assert (run / bd.PLACED_FILE_NAME).read_text() == f.read_text()


def test_clump_sample_columns_and_json(tmp_path, monkeypatch):
    run, cols = _run(tmp_path, monkeypatch, CLUMPS_LOG_BINARY, bd.BoulderSpec(placed="binary", bond_sigma_pa=10.0))
    assert cols == {"f_boulder": "0.3", "rb_max": "4", "f_boulder_actual": f"{271 / 391:.6g}",
                    "n_boulders": "10", "boulder_placed": "binary", "bond_sigma_pa": "10"}
    info = json.loads((run / "boulders.json").read_text())
    assert info["spec"]["placed"] == "binary" and info["result"]["n_boulders"] == 10


def test_clump_sample_no_boulders_when_requested_raises(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no boulders"):
        _run(tmp_path, monkeypatch, CLUMPS_LOG_DUST, bd.BoulderSpec(f_boulder=0.3, bond_sigma_pa=10.0))


def test_clump_sample_all_dust_is_not_an_error(tmp_path, monkeypatch):
    run, cols = _run(tmp_path, monkeypatch, CLUMPS_LOG_DUST, bd.BoulderSpec(f_boulder=0.0, bond_sigma_pa=10.0))
    assert cols["n_boulders"] == "0"


def test_clump_sample_needs_binary(tmp_path):
    s = SimpleNamespace(boulders=bd.BoulderSpec(), f_boulder=None, rb_max=None, phantomclumps_bin=None)
    with pytest.raises(RuntimeError, match="make demtools"):
        bd.clump_sample(tmp_path, _body(tmp_path), s)


def _write_info(run, n_boulders):
    (run / "boulders.json").write_text(json.dumps({"spec": {}, "result": {"n_boulders": n_boulders}}))


def test_check_run_logs(tmp_path):
    _write_info(tmp_path, 3)
    (tmp_path / "setup.log").write_text(" boulders in body     = 3 (40 grains)\n")
    (tmp_path / "phantom.log").write_text(" DEM clump bonds enabled\n")
    bd.check_run_logs(tmp_path, setup_log=True)
    (tmp_path / "setup.log").write_text("placed 391 pre-built grains\n")
    with pytest.raises(RuntimeError, match="boulders in body"):
        bd.check_run_logs(tmp_path, setup_log=True)
    bd.check_run_logs(tmp_path, setup_log=False)  # hyperbola: no phantomsetup
    (tmp_path / "phantom.log").write_text("nothing\n")
    with pytest.raises(RuntimeError, match="unbonded"):
        bd.check_run_logs(tmp_path, setup_log=False)


def test_check_run_logs_skips_all_dust(tmp_path):
    _write_info(tmp_path, 0)
    bd.check_run_logs(tmp_path, setup_log=True)  # no setup.log / phantom.log needed


def test_attach_tools(tmp_path):
    (tmp_path / "phantomclumps").write_text("x")
    phantom = tmp_path / "phantom"
    phantom.write_bytes(b"\x00" + bd.BOND_BINARY_MARKER + b"\x00")
    on = SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)
    off = SimpleNamespace(boulders=None, phantomclumps_bin=None)
    bd.attach_tools([on, off], tmp_path, phantom, dry_run=False)
    assert on.phantomclumps_bin == str(tmp_path / "phantomclumps") and off.phantomclumps_bin is None


def test_attach_tools_refuses_old_phantom(tmp_path):
    (tmp_path / "phantomclumps").write_text("x")
    phantom = tmp_path / "phantom"
    phantom.write_bytes(b"old build")
    with pytest.raises(RuntimeError, match="clump bonds"):
        bd.attach_tools([SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)],
                        tmp_path, phantom, dry_run=False)


def test_attach_tools_missing_clumps(tmp_path):
    with pytest.raises(FileNotFoundError, match="make demtools"):
        bd.attach_tools([SimpleNamespace(boulders=bd.BoulderSpec(), phantomclumps_bin=None)],
                        tmp_path, tmp_path / "phantom", dry_run=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders.py -v`
Expected: new tests FAIL with `AttributeError: module 'boulders' has no attribute 'clump_body'` (etc.).

- [ ] **Step 3: Implement** (append to `boulders.py`; add `import json, shutil, subprocess` to the imports)

```python
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
    if requested(spec) and res.n_boulders == 0:
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
```

Check `_binary_has_string` on a missing file: if it raises, `test_attach_tools_missing_clumps` still passes (clumps missing raises first). If `_binary_has_string` does not tolerate a missing `phantom_bin`, leave it: `preflight()` has already checked `phantom_bin` exists on every non-dry-run path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_boulders.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 3: Runner CLI flags, `RunSample` fields, validation, fixed overrides

**Files:**
- Modify: `boulders.py` (append `BOULDER_ARG_DESTS`, `spec_from_args`, `validate_args`)
- Modify: `run_mass_sobol_phantom.py`: `RunSample` (~line 326, after `flyby_perturber_earth_masses`); `build_parser()` (after `--body-cache-dir`, ~line 714); `validate_args()` (after `_validate_body_encounter(args)`, line 1094); `_apply_fixed_run_sample_overrides()` (end of function, ~line 1612); new `_boulders_module()` next to `_encounter_module()` (~line 3213)
- Test: `tests/test_boulders_cli.py` (new)

**Interfaces:**
- Consumes: Task 1 `BoulderSpec`, `validate_spec`.
- Produces:
  - `RunSample.f_boulder: Optional[float]`, `RunSample.rb_max: Optional[float]`, `RunSample.boulders: Optional[Any]` (a `BoulderSpec`), `RunSample.phantomclumps_bin: Optional[str]`
  - argparse dests: `boulders` (bool), `f_boulder`, `f_boulder_min`, `f_boulder_max`, `rb_max`, `rb_max_min`, `rb_max_max`, `rb_min`, `boulder_q`, `boulder_shape`, `boulder_axis_min`, `boulder_jagged`, `boulder_placed`, `lobe_ratio`, `boulder_file`, `boulder_nmin`, `boulder_seed`, `bond_sigma_pa`, `bond_tau_ratio`, `kb_cgs`, `bond_reach` (all default `None` except `boulders=False`)
  - `boulders.BOULDER_ARG_DESTS: FrozenSet[str]` (every dest above except `boulders`)
  - `boulders.spec_from_args(args) -> BoulderSpec`, `boulders.validate_args(args) -> None`
  - `run_mass_sobol_phantom._boulders_module()`

- [ ] **Step 1: Write the failing tests** — `tests/test_boulders_cli.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import run_mass_sobol_phantom as runner

# scale_vel axis: the runner rejects a batch with no Sobol dimension before boulder validation runs
SETTLED = ["--use-dem-fixed", "true", "--np-apophis", "300", "--body-source", "settled",
           "--tmax-hours", "12", "--dtmax-hours", "1", "--spin-period-fixed", "30",
           "--scale-vel-min", "0.9", "--scale-vel-max", "1.1"]


def _args(*extra):
    a = runner.parse_args(SETTLED + list(extra))
    runner.validate_args(a)
    return a


def _err(*extra, base=SETTLED):
    a = runner.parse_args(list(base) + list(extra))
    with pytest.raises(ValueError) as ei:
        runner.validate_args(a)
    return str(ei.value)


def test_fixed_boulders_validate_and_reach_samples():
    a = _args("--boulders", "--f-boulder", "0.2", "--rb-max", "5", "--bond-sigma-pa", "10",
              "--boulder-placed", "binary", "--lobe-ratio", "0.6", "--num-samples", "2")
    s = runner.RunSample()
    runner._apply_fixed_run_sample_overrides(s, a)
    assert s.boulders == bd.BoulderSpec(f_boulder=0.2, rb_max=5.0, bond_sigma_pa=10.0, placed="binary",
                                        lobe_ratio=0.6, seed=a.seed)
    assert (s.f_boulder, s.rb_max) == (0.2, 5.0)


def test_no_boulders_leaves_sample_untouched():
    a = _args("--num-samples", "1")
    s = runner.RunSample()
    runner._apply_fixed_run_sample_overrides(s, a)
    assert (s.boulders, s.f_boulder, s.rb_max, s.phantomclumps_bin) == (None, None, None, None)


def test_boulder_flags_need_boulders():
    assert "--boulders" in _err("--f-boulder", "0.2", "--num-samples", "1")


def test_boulders_need_settled():
    lattice = ["--use-dem-fixed", "true", "--np-apophis", "300", "--num-samples", "1",
               "--spin-period-fixed", "30", "--scale-vel-min", "0.9", "--scale-vel-max", "1.1"]
    assert "settled" in _err("--boulders", "--bond-sigma-pa", "10", base=lattice)


def test_boulders_need_sigma():
    assert "--bond-sigma-pa" in _err("--boulders", "--num-samples", "1")


@pytest.mark.parametrize("extra,match", [
    (["--f-boulder-min", "0.1"], "--f-boulder"),                              # one bound only
    (["--f-boulder", "0.2", "--f-boulder-min", "0.1", "--f-boulder-max", "0.3"], "mutually exclusive"),
    (["--rb-max-min", "6", "--rb-max-max", "5"], "min < max"),
    (["--f-boulder-min", "0.1", "--f-boulder-max", "1.5"], "f_boulder"),      # hi bound out of range
    (["--rb-min", "3", "--rb-max-min", "2", "--rb-max-max", "5"], "rb_max"),  # lo bound < rb_min
    (["--lobe-ratio", "0.5"], "binary"),
    (["--boulder-placed", "file"], "file"),
])
def test_bad_boulder_args(extra, match):
    assert match in _err("--boulders", "--bond-sigma-pa", "10", "--num-samples", "4", *extra)


def test_axes_rejected_in_np_list_mode():
    base = ["--use-dem-fixed", "true", "--np-apophis-list", "300", "--body-source", "settled",
            "--tmax-hours", "12", "--dtmax-hours", "1", "--spin-period-fixed", "30"]
    msg = _err("--boulders", "--bond-sigma-pa", "10", "--f-boulder-min", "0.1", "--f-boulder-max", "0.4",
               base=base)
    assert "np-apophis-list" in msg


def test_sigma_zero_warns(capsys):
    _args("--boulders", "--bond-sigma-pa", "0", "--num-samples", "1")
    assert "stretch" in capsys.readouterr().err


def test_boulder_seed_defaults_to_batch_seed():
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--seed", "7")
    assert bd.spec_from_args(a).seed == 7
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--boulder-seed", "99")
    assert bd.spec_from_args(a).seed == 99


def test_boulder_file_is_made_absolute(tmp_path, monkeypatch):
    f = tmp_path / "b.txt"
    f.write_text("0 0 0 0.05 0.05 0.05\n")
    monkeypatch.chdir(tmp_path)
    a = _args("--boulders", "--bond-sigma-pa", "10", "--num-samples", "1", "--boulder-placed", "file",
              "--boulder-file", "b.txt")
    assert bd.spec_from_args(a).placed_file == str(f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders_cli.py -v`
Expected: FAIL, `error: unrecognized arguments: --boulders` (SystemExit) / `AttributeError: 'RunSample' object has no attribute 'boulders'`.

- [ ] **Step 3a: `RunSample` fields** — after `flyby_perturber_earth_masses: Optional[float] = None`:

```python
    # Boulders (phantomclumps; docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md)
    f_boulder: Optional[float] = None   # per-sample axis value; None = boulders.f_boulder
    rb_max: Optional[float] = None      # per-sample axis value; None = boulders.rb_max
    boulders: Optional[Any] = None      # boulders.BoulderSpec (fixed part); None = boulders off
    phantomclumps_bin: Optional[str] = None  # set by boulders.attach_tools before dispatch
```

- [ ] **Step 3b: Flags** — in `build_parser()` right after the `--body-cache-dir` argument:

```python
    parser.add_argument("--boulders", action="store_true",
                        help="Glue the settled body into bonded boulders with phantomclumps before each run "
                             "(needs --body-source settled and --bond-sigma-pa).")
    parser.add_argument("--f-boulder", type=float, default=None, metavar="F",
                        help="Fraction of the body put into RANDOM boulders, 0-1 (default 0.3); placed "
                             "boulders come on top.")
    parser.add_argument("--rb-max", type=float, default=None, metavar="R",
                        help="Largest random boulder radius in grain radii (default 4).")
    for dest_flag, unit in (("f-boulder", "F"), ("rb-max", "R")):
        parser.add_argument(f"--{dest_flag}-min", type=float, default=None, metavar=unit,
                            help=f"Sweep lower bound for --{dest_flag} (boulders; set -max too).")
        parser.add_argument(f"--{dest_flag}-max", type=float, default=None, metavar=unit,
                            help=f"Sweep upper bound for --{dest_flag} (boulders; set -min too).")
    parser.add_argument("--rb-min", type=float, default=None, metavar="R",
                        help="Smallest random boulder radius in grain radii, >= 1 (default 1.5).")
    parser.add_argument("--boulder-q", type=float, default=None, metavar="Q",
                        help="Power-law slope q of the boulder size distribution dN/dR ~ R^-q (default 3.5).")
    parser.add_argument("--boulder-shape", choices=("round", "ellipsoid", "grown"), default=None,
                        help="Random boulder shape (default round).")
    parser.add_argument("--boulder-axis-min", type=float, default=None, metavar="A",
                        help="Smallest ellipsoid axis ratio, 0.1-1 (ellipsoid only; blank = phantomclumps default).")
    parser.add_argument("--boulder-jagged", type=float, default=None, metavar="J",
                        help="Jaggedness of grown boulders, 0-1 (grown only; blank = phantomclumps default).")
    parser.add_argument("--boulder-placed", choices=("none", "binary", "file"), default=None,
                        help="Placed boulders made before the random ones (default none).")
    parser.add_argument("--lobe-ratio", type=float, default=None, metavar="R",
                        help="Contact binary small/large lobe radius, 0.1-1 (binary only).")
    parser.add_argument("--boulder-file", default=None, metavar="PATH",
                        help="Placed boulders, one per line: x y z a b c [rx ry rz] (km, deg; file only).")
    parser.add_argument("--boulder-nmin", type=int, default=None, metavar="N",
                        help="Smallest random boulder kept, in grains (>= 2; blank = phantomclumps default).")
    parser.add_argument("--boulder-seed", type=int, default=None, metavar="N",
                        help="phantomclumps random seed, fixed per batch (default: --seed).")
    parser.add_argument("--bond-sigma-pa", type=float, default=None, metavar="PA",
                        help="Bond tensile strength in Pa, fixed per batch (required with --boulders; "
                             "0 = break on stretch past bond_reach only).")
    parser.add_argument("--bond-tau-ratio", type=float, default=None, metavar="X",
                        help="Bond shear strength as a multiple of --bond-sigma-pa (default 1).")
    parser.add_argument("--kb-cgs", type=float, default=None, metavar="K",
                        help="Bond spring constant g/s^2 per cm (default: the run's kn_cgs).")
    parser.add_argument("--bond-reach", type=float, default=None, metavar="X",
                        help="Max bond surface gap as a fraction of R_i+R_j; written to phantomclumps and "
                             "the .in (default 0.1).")
```

- [ ] **Step 3c: `boulders.py` CLI helpers** (append):

```python
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
```

- [ ] **Step 3d: Hook into the runner**

`_boulders_module()` next to `_encounter_module()`:

```python
def _boulders_module():
    try:
        import boulders as mod
    except ImportError:
        from sobol import boulders as mod
    return mod
```

In `validate_args()`, the line after `_validate_body_encounter(args)`:

```python
    _boulders_module().validate_args(args)
```

At the end of `_apply_fixed_run_sample_overrides()` (after the flyby loop):

```python
    if getattr(args, "boulders", False):
        s.boulders = _boulders_module().spec_from_args(args)
        for dest in ("f_boulder", "rb_max"):
            v = getattr(args, dest, None)
            if v is not None and getattr(s, dest) is None:
                setattr(s, dest, v)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_boulders_cli.py tests/test_boulders.py tests/test_body_encounter_cli.py -v`
Expected: all PASS. If `test_axes_rejected_in_np_list_mode` fails because an existing np-list check fires first with a different message, keep the existing check and assert on its message instead (the input is still rejected, which is what matters).

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 4: Sobol axes plumbing, CSV columns, slug and descriptor

**Files:**
- Modify: `run_mass_sobol_phantom.py`: new `_BOULDER_VARIATION_SPEC` + `_active_boulder_variations()` after `_FLYBY_VARIATION_SPEC` / `_active_flyby_variations()` (~lines 211, 250); `count_dimensions()` (1176); `build_run_samples()` (1655); `sample_column_order()` (1702); `build_salib_problem()`; `run_sample_from_salib_row()` (1770); `canonical_sweep_descriptor()` (1824); `build_auto_batch_sweep_slug()` (1876)
- Test: `tests/test_boulders_cli.py` (append)

**Interfaces:**
- Consumes: Task 3 dests and `RunSample.f_boulder` / `rb_max` / `boulders`.
- Produces: `_BOULDER_VARIATION_SPEC`, `_active_boulder_variations(args) -> List[Tuple[str, float, float, str]]`; sample column names `f_boulder`, `rb_max`, `f_boulder_actual`, `n_boulders`, `boulder_placed`, `bond_sigma_pa`.

Insertion rule for every enumeration site: put the boulder loop **immediately after** the `_active_flyby_variations(args)` loop. Batches without boulders have no boulder dimensions, so their order is unchanged.

- [ ] **Step 1: Write the failing tests** (append):

```python
NP_LIST_SETTLED = ["--use-dem-fixed", "true", "--np-apophis-list", "300", "--body-source", "settled",
                   "--spin-period-fixed", "30", "--tmax-hours", "12", "--dtmax-hours", "1", "--num-samples", "1"]
HYP = ["--use-dem-fixed", "true", "--np-apophis", "300", "--body-source", "settled", "--encounter",
       "hyperbola", "--flyby-rp-km-min", "20000", "--flyby-rp-km-max", "40000", "--flyby-vinf-kms-min", "5",
       "--flyby-vinf-kms-max", "7", "--tmax-hours", "48", "--dtmax-hours", "0.5", "--num-samples", "16"]


def _parsed(argv):
    a = runner.parse_args(argv)
    runner.validate_args(a)
    return a


def test_no_boulder_regression():
    # values captured 2026-10-06 before this change
    a = _parsed(NP_LIST_SETTLED)
    assert runner.build_auto_batch_sweep_slug(a, 200) == "np300_tmx12_dtmx1_settled"
    assert (runner.count_dimensions(a), runner.sample_column_order(a)) == (0, ["np_apophis", "np_kept"])
    assert "boulder" not in runner.canonical_sweep_descriptor(a)
    a = _parsed(HYP)
    assert runner.build_auto_batch_sweep_slug(a, 200) == "n16_s42_rp20000-40000_vinf5-7_tmx48_dtmx0.5_hyp_settled"
    assert (runner.count_dimensions(a), runner.sample_column_order(a)) == (
        2, ["flyby_rp_km", "flyby_vinf_kms", "np_kept"])


def test_boulder_axes_are_dimensions():
    a = _parsed(HYP + ["--boulders", "--bond-sigma-pa", "10", "--f-boulder-min", "0.1", "--f-boulder-max",
                       "0.5", "--rb-max-min", "2", "--rb-max-max", "6"])
    assert runner.count_dimensions(a) == 4
    order = runner.sample_column_order(a)
    assert order[:4] == ["flyby_rp_km", "flyby_vinf_kms", "f_boulder", "rb_max"]
    assert order[-5:] == ["np_kept", "f_boulder_actual", "n_boulders", "boulder_placed", "bond_sigma_pa"]
    samples = runner.build_run_samples(16, a)
    assert all(0.1 <= s.f_boulder <= 0.5 and 2 <= s.rb_max <= 6 for s in samples)
    assert all(s.boulders is not None for s in samples)
    prob = runner.build_salib_problem(a)
    assert prob["names"][2:] == ["f_boulder", "rb_max"] and prob["bounds"][2:] == [[0.1, 0.5], [2.0, 6.0]]
    s = runner.run_sample_from_salib_row([30000, 6, 0.3, 4], a)
    assert (s.f_boulder, s.rb_max) == (0.3, 4.0)


def test_fixed_boulders_add_columns_not_dimensions():
    a = _parsed(NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10", "--f-boulder", "0.2"])
    assert runner.count_dimensions(a) == 0
    assert runner.sample_column_order(a) == ["np_apophis", "np_kept", "f_boulder", "rb_max", "f_boulder_actual",
                                             "n_boulders", "boulder_placed", "bond_sigma_pa"]


def test_slug_distinguishes_placed():
    common = NP_LIST_SETTLED + ["--boulders", "--bond-sigma-pa", "10"]
    none = runner.build_auto_batch_sweep_slug(_parsed(common), 200)
    binary = runner.build_auto_batch_sweep_slug(_parsed(common + ["--boulder-placed", "binary"]), 200)
    assert none != binary and none.endswith("bldnone_sig10") and binary.endswith("bldbinary_sig10")
    d1 = runner.canonical_sweep_descriptor(_parsed(common))
    d2 = runner.canonical_sweep_descriptor(_parsed(common + ["--boulder-placed", "binary"]))
    assert d1 != d2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders_cli.py -v -k "regression or dimensions or columns or slug"`
Expected: `test_no_boulder_regression` PASSES (guards the baseline); the other three FAIL.

- [ ] **Step 3: Implement**

After `_FLYBY_VARIATION_SPEC`:

```python
# Boulders (--boulders only): swept like the other optional dimensions.
_BOULDER_VARIATION_SPEC: Tuple[Tuple[str, str, str, str], ...] = (
    ("f_boulder", "f_boulder_min", "f_boulder_max", "fb"),
    ("rb_max",    "rb_max_min",    "rb_max_max",    "rbx"),
)
```

After `_active_flyby_variations()`:

```python
def _active_boulder_variations(args: argparse.Namespace) -> List[Tuple[str, float, float, str]]:
    """Boulder dimensions that are active (both bounds set), in canonical order."""
    out: List[Tuple[str, float, float, str]] = []
    for param, lo_attr, hi_attr, slug_tok in _BOULDER_VARIATION_SPEC:
        lo = getattr(args, lo_attr, None)
        hi = getattr(args, hi_attr, None)
        if lo is not None and hi is not None:
            out.append((param, lo, hi, slug_tok))
    return out
```

`count_dimensions()`, after `n += len(_active_flyby_variations(args))`:

```python
    n += len(_active_boulder_variations(args))
```

`build_run_samples()`, after the flyby loop:

```python
        for param, lo, hi, _ in _active_boulder_variations(args):
            setattr(s, param, lo + row[di] * (hi - lo))
            di += 1
```

`build_salib_problem()`, after the flyby loop:

```python
    for param, lo, hi, _ in _active_boulder_variations(args):
        names.append(param)
        bounds.append([float(lo), float(hi)])
```

`run_sample_from_salib_row()`, after the flyby loop:

```python
    for param, _, _, _ in _active_boulder_variations(args):
        setattr(s, param, float(row[i]))
        i += 1
```

`sample_column_order()`: after the flyby loop add

```python
    for param, _, _, _ in _active_boulder_variations(args):
        order.append(param)
```

and after the existing `if args.body_source == "settled": order.append("np_kept")` block, at the end:

```python
    if getattr(args, "boulders", False):
        for c in ("f_boulder", "rb_max"):
            if c not in order:
                order.append(c)
        order += ["f_boulder_actual", "n_boulders", "boulder_placed", "bond_sigma_pa"]
```

(`write_samples_csv` writes `getattr(sample, col)` and blanks for `None`, so the output-only columns are blank in `sobol_mass_samples.csv` and filled from `param_columns` in `sobol_mass_outputs.csv`. Check `write_samples_csv` (~line 3105): it calls `float(v)` on non-`None` values; `RunSample` has no `boulder_placed` / `n_boulders` / `f_boulder_actual` / `bond_sigma_pa` attributes, so `getattr` returns `None` and they stay blank.)

`canonical_sweep_descriptor()`: after the `for dest in ("flyby_rp_km", ...)` loop (~line 1861):

```python
    if getattr(args, "boulders", False):
        import dataclasses as _dc
        spec = _boulders_module().spec_from_args(args)
        parts.append(f"boulders={sorted(_dc.asdict(spec).items())}")
        for _, lo, hi, _tok in _active_boulder_variations(args):
            parts.append(f"{_tok}={lo}-{hi}")
```

`build_auto_batch_sweep_slug()`: after the flyby token loop add

```python
    for _, lo, hi, slug_tok in _active_boulder_variations(args):
        tokens.append(f"{slug_tok}{_fmt_slug_float(lo)}-{_fmt_slug_float(hi)}")
```

and after `if args.body_source == "settled": tokens.append("settled")`:

```python
    if getattr(args, "boulders", False):
        bspec = _boulders_module().spec_from_args(args)
        tokens.append(f"bld{bspec.placed}")
        tokens.append(f"sig{_fmt_slug_float(bspec.bond_sigma_pa)}")
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_boulders_cli.py tests/test_body_encounter_cli.py -v`
Expected: all PASS, including `test_no_boulder_regression`.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 5: Run path (clump per run, bond keys into `.in`, log checks, cleanup)

**Files:**
- Modify: `encounter.py`: `apply_settled_body_to_setup()` (line 168) and `prepare_hyperbola_run()` (line 117) get an optional `dump: Optional[Path] = None`
- Modify: `run_mass_sobol_phantom.py`: `_OPTIONAL_IN_KEYS` (line 60); `apply_run_sample_to_in()` (~line 1403); `run_one_case()` ephemeris settled branch (~line 3463) and after the `phantom` run (~line 3508); `_run_hyperbola_case()` (~line 3351); `_cleanup_run_dir()` (~line 3565); `prepare_settled_bodies_and_dem_tools()` (~line 3591)
- Test: `tests/test_boulders_run.py` (new)

**Interfaces:**
- Consumes: Task 2 `clump_sample`, `check_run_logs`, `attach_tools`, `CLUMPED_DUMP`; Task 1 `bond_in_keys`, `resolve`.
- Produces: `encounter.apply_settled_body_to_setup(run_dir, prefix, body, dump=None)`, `encounter.prepare_hyperbola_run(run_dir, prefix, sample, body, phantomflyby_bin, template_setup, dump=None)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_boulders_run.py`:

```python
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import boulders as bd
import run_mass_sobol_phantom as runner
from tests.test_encounter import _body, _flyby, _sample, TEMPLATE, _val
import encounter as enc

TEMPLATE_IN = Path(__file__).parent.parent / "sobol.in"


def _sample_with_boulders(**kw):
    return runner.RunSample(use_dem=True, boulders=bd.BoulderSpec(bond_sigma_pa=10.0, **kw))


def test_apply_in_inserts_bond_keys(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   2.000E+07    ! DEM normal spring constant\n")
    cols = runner.apply_run_sample_to_in(p, _sample_with_boulders())
    t = p.read_text()
    assert float(_val(t, "kb_cgs")) == 2e7          # default = kn_cgs of this .in
    assert float(_val(t, "bond_sigma_pa")) == 10.0 and float(_val(t, "bond_tau_pa")) == 10.0
    assert float(_val(t, "bond_reach")) == 0.1
    assert cols["kb_cgs"] == "20000000"


def test_apply_in_replaces_present_bond_keys(tmp_path):
    p = tmp_path / "fly.in"
    p.write_text("              kn_cgs =   1.000E+07    ! x\n"
                 "              kb_cgs =       0.000    ! x\n"
                 "          bond_reach =       0.100    ! x\n"
                 "       bond_sigma_pa =       0.000    ! x\n"
                 "         bond_tau_pa =       0.000    ! x\n")
    runner.apply_run_sample_to_in(p, _sample_with_boulders(kb_cgs=3e7))
    t = p.read_text()
    assert float(_val(t, "kb_cgs")) == 3e7
    assert len(re.findall(r"^\s*kb_cgs\s*=", t, re.M)) == 1


def test_apply_in_without_boulders_unchanged(tmp_path):
    p = tmp_path / "sobol.in"
    p.write_text("              kn_cgs =   1.000E+07    ! x\n")
    assert runner.apply_run_sample_to_in(p, runner.RunSample(use_dem=True)) == {}
    assert "kb_cgs" not in p.read_text()


def test_settled_setup_stages_given_dump(tmp_path):
    body = _body(tmp_path)
    object.__setattr__(body, "shape_file", str(Path(__file__).parent.parent.parent / "Shapes" / "apophis.shape"))
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "sobol.setup").write_text(TEMPLATE.read_text())
    (run / "clumped_00000").write_text("clumped dump")
    enc.apply_settled_body_to_setup(run, "sobol", body, dump=run / "clumped_00000")
    assert (run / enc.PACKING_FILE_NAME).read_text() == "clumped dump"


def test_hyperbola_flies_given_dump(tmp_path):
    run = tmp_path / "run_0001"
    run.mkdir()
    (run / "clumped_00000").write_text("clumped dump")
    enc.prepare_hyperbola_run(run, "sobol", _sample(), _body(tmp_path), _flyby(tmp_path), TEMPLATE,
                              dump=run / "clumped_00000")
    assert (run / "body_00000").read_text() == "clumped dump"


def test_cleanup_removes_clumped_dump(tmp_path):
    for name in ("clumped_00000", "clumped.in", "unclumped_00000", "boulders.json", "clumps.log", "sobol_00001"):
        (tmp_path / name).write_text("x")
    runner._cleanup_run_dir(tmp_path, "sobol")
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["boulders.json", "clumps.log"]
```

(`SettledBody` is a frozen dataclass; if `object.__setattr__` is awkward, write `"shape_file"` into `body.json` inside the test instead, matching how `tests/test_encounter.py` builds a body for `apply_settled_body_to_setup`, if it has such a test. Look there first and copy its fixture.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders_run.py -v`
Expected: FAIL (`unexpected keyword argument 'dump'`, bond keys missing, `clumped_00000` not removed).

- [ ] **Step 3a: `encounter.py`**

`apply_settled_body_to_setup(run_dir, prefix, body, dump: Optional[Path] = None)`: replace
`shutil.copy2(Path(body.dir) / body.dump, run_dir / PACKING_FILE_NAME)` with

```python
    src = Path(dump) if dump is not None else Path(body.dir) / body.dump  # dump = clumped body (boulders)
    shutil.copy2(src, run_dir / PACKING_FILE_NAME)
```

`prepare_hyperbola_run(..., template_setup: Path, dump: Optional[Path] = None)`: replace
`shutil.copy2(body.dir / body.dump, run_dir / "body_00000")` with

```python
    src = Path(dump) if dump is not None else body.dir / body.dump  # dump = clumped body (boulders)
    shutil.copy2(src, run_dir / "body_00000")
```

Add `from typing import Optional` if it is not already imported.

- [ ] **Step 3b: `.in` keys** — `_OPTIONAL_IN_KEYS`:

```python
_OPTIONAL_IN_KEYS = frozenset({"mu_s", "mu_r", "mu_t", "epsilon_t_dem", "cohesion_pa",
                               "kb_cgs", "bond_reach", "bond_sigma_pa", "bond_tau_pa"})
```

In `apply_run_sample_to_in()`: change `if not active:` to `if not active and sample.boulders is None:`, and right after the `kn_cgs` presence check (`if not re.search(r"^\s*kn_cgs\s*=", ...): return columns`) add:

```python
    if sample.boulders is not None:
        bd = _boulders_module()
        kn_m = re.search(r"^\s*kn_cgs\s*=\s*([^!\s]+)", text, re.MULTILINE)
        kn = sample.kn_cgs if sample.kn_cgs is not None else float(kn_m.group(1))
        active += list(bd.bond_in_keys(bd.resolve(sample), kn).items())
```

- [ ] **Step 3c: `run_one_case()` ephemeris** — replace

```python
            settled = _settled_body_module().SettledBody.load(Path(sample.settled_body_dir))
            param_columns.update(_encounter_module().apply_settled_body_to_setup(run_dir, prefix, settled))
```

with

```python
            settled = _settled_body_module().SettledBody.load(Path(sample.settled_body_dir))
            body_dump = None
            if sample.boulders is not None:
                bd = _boulders_module()
                param_columns.update(bd.clump_sample(run_dir, settled, sample))
                body_dump = run_dir / bd.CLUMPED_DUMP
            param_columns.update(_encounter_module().apply_settled_body_to_setup(
                run_dir, prefix, settled, dump=body_dump))
```

and directly after `run_command([str(phantom_bin), f"{prefix}.in"] + maxp_flag, ...)`:

```python
        if sample.boulders is not None:
            _boulders_module().check_run_logs(run_dir, setup_log=True)
```

- [ ] **Step 3d: `_run_hyperbola_case()`** — replace

```python
        param_columns.update(enc.prepare_hyperbola_run(
            run_dir, prefix, sample, body, Path(phantomflyby_bin), base_setup))
```

with

```python
        body_dump = None
        if sample.boulders is not None:
            bd = _boulders_module()
            param_columns.update(bd.clump_sample(run_dir, body, sample))
            body_dump = run_dir / bd.CLUMPED_DUMP
        param_columns.update(enc.prepare_hyperbola_run(
            run_dir, prefix, sample, body, Path(phantomflyby_bin), base_setup, dump=body_dump))
```

and directly after its `run_command([str(phantom_bin), f"{prefix}.in", *maxp_flag], ...)`:

```python
        if sample.boulders is not None:
            _boulders_module().check_run_logs(run_dir, setup_log=False)
```

- [ ] **Step 3e: Cleanup** — in `_cleanup_run_dir()` extend the pattern tuple and docstring:

```python
    """Delete heavy output files after metrics are safely extracted.

    Removes binary dump files (sobol_NNNNN), per-sink .ev files, phantom.log, and the per-run boulder
    dumps (clumped_*, unclumped_*; the flyby dumps carry iclump). Keeps .in, .setup, setup.log,
    clumps.log, boulders.json and ephemeris .txt files for audit purposes.
    """
    removed = 0
    for pattern in (f"{prefix}_[0-9]*", f"{prefix}Sink*N*.ev", f"{prefix}*.ev", "phantom.log",
                    "clumped_[0-9]*", "clumped.in", "unclumped_[0-9]*", "unclumped.in"):
```

- [ ] **Step 3f: Tools** — in `prepare_settled_bodies_and_dem_tools()`, after the `if args.body_source == "settled": ... print(... settled body spec(s) ready ...)` block:

```python
    _boulders_module().attach_tools(samples, Path(args.phantom_dir).resolve(), phantom_bin, args.dry_run)
```

(Resume calls this same function, so resumed batches get `phantomclumps_bin` too.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_boulders_run.py tests/test_boulders.py tests/test_encounter.py tests/test_particle_dem_runner.py tests/test_dem_contact_keys.py -v`
Expected: all PASS (existing encounter / contact-key tests prove the default paths are unchanged).

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 6: Wizard gate and Analysis inputs

**Files:**
- Modify: `interactive_run_mass_sobol.py`: `DEST_TO_SECTION` (line 83), `INTERACTIVE_BRIEF` (line 191), new `BOULDER_GATE_DESTS` + `_skip_boulder_dest()`, the loop in `run_interactive_wizard()` (line 795), `__all__` (~line 915)
- Modify: `Analysis/Analysis.py`: `INPUT_CANDIDATES` (line 82)
- Test: `tests/test_boulders_cli.py` (append)

**Interfaces:**
- Consumes: `boulders.BOULDER_ARG_DESTS`.
- Produces: `interactive_run_mass_sobol._skip_boulder_dest(dest: str, state) -> bool`.

- [ ] **Step 1: Write the failing tests** (append):

```python
import argparse
import interactive_run_mass_sobol as wiz


def test_wizard_knows_every_boulder_dest():
    for d in bd.BOULDER_ARG_DESTS | {"boulders"}:
        assert wiz.DEST_TO_SECTION[d] == "Boulders"
        assert d in wiz.INTERACTIVE_BRIEF


def test_wizard_boulder_gate():
    lattice = argparse.Namespace(body_source="lattice", boulders=False)
    settled_off = argparse.Namespace(body_source="settled", boulders=False)
    settled_on = argparse.Namespace(body_source="settled", boulders=True)
    assert wiz._skip_boulder_dest("boulders", lattice)
    assert not wiz._skip_boulder_dest("boulders", settled_off)
    assert wiz._skip_boulder_dest("f_boulder", settled_off)
    assert not wiz._skip_boulder_dest("f_boulder", settled_on)
    assert not wiz._skip_boulder_dest("np_apophis", lattice)


def test_analysis_knows_boulder_inputs():
    sys.path.insert(0, str(Path(__file__).parent.parent / "Analysis"))
    import Analysis
    assert {"f_boulder", "rb_max"} <= set(Analysis.INPUT_CANDIDATES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_boulders_cli.py -v -k "wizard or analysis"`
Expected: FAIL (`KeyError: 'boulders'`, missing `_skip_boulder_dest`, missing inputs).

- [ ] **Step 3: Implement**

`DEST_TO_SECTION` additions:

```python
    "boulders": "Boulders",
    "f_boulder": "Boulders", "f_boulder_min": "Boulders", "f_boulder_max": "Boulders",
    "rb_max": "Boulders", "rb_max_min": "Boulders", "rb_max_max": "Boulders",
    "rb_min": "Boulders", "boulder_q": "Boulders", "boulder_shape": "Boulders",
    "boulder_axis_min": "Boulders", "boulder_jagged": "Boulders", "boulder_placed": "Boulders",
    "lobe_ratio": "Boulders", "boulder_file": "Boulders", "boulder_nmin": "Boulders",
    "boulder_seed": "Boulders", "bond_sigma_pa": "Boulders", "bond_tau_ratio": "Boulders",
    "kb_cgs": "Boulders", "bond_reach": "Boulders",
```

`INTERACTIVE_BRIEF` additions (same `(what, valid)` tuple style as the flyby entries):

```python
    "boulders": ("Glue the settled body into bonded boulders with phantomclumps (settled bodies only).", "y/n."),
    "f_boulder": ("Fraction of the body in RANDOM boulders (placed boulders come on top).", "Float 0-1, or Enter (0.3)."),
    "f_boulder_min": ("f_boulder sweep bound (set min and max together).", "Float 0-1, or Enter."),
    "f_boulder_max": ("f_boulder sweep bound (set min and max together).", "Float 0-1, or Enter."),
    "rb_max": ("Largest random boulder radius in grain radii.", "Float >= rb_min, or Enter (4)."),
    "rb_max_min": ("rb_max sweep bound (set min and max together).", "Float >= rb_min, or Enter."),
    "rb_max_max": ("rb_max sweep bound (set min and max together).", "Float >= rb_min, or Enter."),
    "rb_min": ("Smallest random boulder radius in grain radii.", "Float >= 1, or Enter (1.5)."),
    "boulder_q": ("Size distribution slope q, dN/dR ~ R^-q.", "Float >= 0, or Enter (3.5)."),
    "boulder_shape": ("Random boulder shape.", "round / ellipsoid / grown, or Enter (round)."),
    "boulder_axis_min": ("Smallest ellipsoid axis ratio (ellipsoid only).", "Float 0.1-1, or Enter."),
    "boulder_jagged": ("Jaggedness of grown boulders (grown only).", "Float 0-1, or Enter."),
    "boulder_placed": ("Placed boulders made first.", "none / binary / file, or Enter (none)."),
    "lobe_ratio": ("Contact binary small/large lobe radius (binary only).", "Float 0.1-1, or Enter."),
    "boulder_file": ("Placed boulders file: x y z a b c [rx ry rz] per line, km and deg (file only).", "Path, or Enter."),
    "boulder_nmin": ("Smallest random boulder kept, in grains.", "Integer >= 2, or Enter."),
    "boulder_seed": ("phantomclumps random seed (default: --seed).", "Integer, or Enter."),
    "bond_sigma_pa": ("Bond tensile strength in Pa, fixed per batch (required).", "Float >= 0."),
    "bond_tau_ratio": ("Bond shear strength as a multiple of bond_sigma_pa.", "Float >= 0, or Enter (1)."),
    "kb_cgs": ("Bond spring constant g/s^2 per cm.", "Float > 0, or Enter (= kn_cgs)."),
    "bond_reach": ("Max bond surface gap as a fraction of R_i+R_j.", "Float > 0, or Enter (0.1)."),
```

Gate helper (module level, near `MASS_GATE_DESTS`):

```python
# Boulder flags are asked only for settled bodies, and the rest only once --boulders is on.
BOULDER_GATE_DESTS: FrozenSet[str] = frozenset({
    "f_boulder", "f_boulder_min", "f_boulder_max", "rb_max", "rb_max_min", "rb_max_max", "rb_min",
    "boulder_q", "boulder_shape", "boulder_axis_min", "boulder_jagged", "boulder_placed", "lobe_ratio",
    "boulder_file", "boulder_nmin", "boulder_seed", "bond_sigma_pa", "bond_tau_ratio", "kb_cgs", "bond_reach",
})


def _skip_boulder_dest(dest: str, state: argparse.Namespace) -> bool:
    if dest == "boulders":
        return getattr(state, "body_source", "lattice") != "settled"
    return dest in BOULDER_GATE_DESTS and not getattr(state, "boulders", False)
```

In `run_interactive_wizard()`, right after the `if action.nargs == "+": ... continue` block:

```python
        if _skip_boulder_dest(action.dest, state):
            continue
```

Add `"BOULDER_GATE_DESTS"` to `__all__`. `--body-source` is registered before `--boulders`, so `state.body_source` is already answered when the gate is checked.

`Analysis/Analysis.py` `INPUT_CANDIDATES`: before the closing `)` add

```python
    # Boulders (--boulders sweeps).
    "f_boulder",
    "rb_max",
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_boulders_cli.py tests/test_dem_contact_keys.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 7: `tui_run.py` "Body / encounter" section

**Files:**
- Modify: `tui_run.py`: `compose()` (new section after Timeframe, before "Setup toggles", ~line 398); `_build_argv()` (emit after the timeframe flags)
- Test: `tests/test_tui_boulders.py` (new)

**Interfaces:**
- Consumes: existing runner flags `--body-source`, `--encounter`, `--flyby-rp-km[-min/-max]`, `--flyby-vinf-kms[-min/-max]`, `--flyby-start-sep-km`, `--settle-tdyn`, `--relax-tdyn`, `--body-cache-dir`.
- Produces: widget IDs `body-source`, `encounter` (Select); `flyby-rp-km`, `flyby-rp-km-min`, `flyby-rp-km-max`, `flyby-vinf-kms`, `flyby-vinf-kms-min`, `flyby-vinf-kms-max`, `flyby-start-sep-km`, `settle-tdyn`, `relax-tdyn`, `body-cache-dir` (Input).

- [ ] **Step 1: Write the failing tests** — `tests/test_tui_boulders.py`:

```python
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_mass_sobol_phantom as runner
import tui_run as tr
from textual.widgets import Checkbox, Input, Select


def _sweep(cli=(), actions=None):
    async def _scenario():
        app = tr.SobolTUIApp(runner.build_parser().parse_args(list(cli)))
        async with app.run_test() as pilot:
            if actions:
                actions(app)
            await pilot.pause()
            try:
                return app._build_argv(force_dry=True)
            except ValueError as exc:
                return exc
    return asyncio.run(_scenario())


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_default_body_section_emits_nothing():
    argv = _sweep()
    for flag in ("--body-source", "--encounter", "--flyby-rp-km", "--settle-tdyn", "--relax-tdyn"):
        assert flag not in argv


def test_body_section_reaches_argv():
    def act(app):
        app.query_one("#body-source", Select).value = "settled"
        app.query_one("#encounter", Select).value = "hyperbola"
        app.query_one("#flyby-rp-km-min", Input).value = "20000"
        app.query_one("#flyby-rp-km-max", Input).value = "40000"
        app.query_one("#flyby-vinf-kms", Input).value = "6"
        app.query_one("#settle-tdyn", Input).value = "8"
    argv = _sweep(actions=act)
    assert _arg(argv, "--body-source") == "settled" and _arg(argv, "--encounter") == "hyperbola"
    assert _arg(argv, "--flyby-rp-km-min") == "20000" and _arg(argv, "--flyby-rp-km-max") == "40000"
    assert _arg(argv, "--flyby-vinf-kms") == "6" and _arg(argv, "--settle-tdyn") == "8"


def test_body_section_prefills_from_cli():
    cli = ["--body-source", "settled", "--relax-tdyn", "0.3"]
    argv = _sweep(cli=cli)
    assert _arg(argv, "--body-source") == "settled" and _arg(argv, "--relax-tdyn") == "0.3"


def test_bad_number_raises():
    def act(app):
        app.query_one("#flyby-rp-km", Input).value = "far"
    assert isinstance(_sweep(actions=act), ValueError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tui_boulders.py -v`
Expected: FAIL (`NoMatches` for `#body-source`).

- [ ] **Step 3: Implement**

In `compose()`, after the Timeframe loop and before `yield Static("Setup toggles", classes="sec")`:

```python
            # ── Body / encounter (settled bodies, hyperbola) ───────────────
            yield Static("Body / encounter  (settled = settle → crop → relax, cached)", classes="sec")
            yield _Row("body_source",
                       Select([("lattice", "lattice"), ("settled", "settled")],
                              value=getattr(d, "body_source", "lattice") or "lattice",
                              id="body-source", allow_blank=False),
                       "settled body")
            yield _Row("encounter",
                       Select([("ephemeris", "ephemeris"), ("hyperbola", "hyperbola")],
                              value=getattr(d, "encounter", "ephemeris") or "ephemeris",
                              id="encounter", allow_blank=False),
                       "hyperbola = settled")
            for b_id, b_dest, b_hint in (
                ("flyby-rp-km", "flyby_rp_km", "km"),
                ("flyby-rp-km-min", "flyby_rp_km_min", "km sweep"),
                ("flyby-rp-km-max", "flyby_rp_km_max", "km sweep"),
                ("flyby-vinf-kms", "flyby_vinf_kms", "km/s"),
                ("flyby-vinf-kms-min", "flyby_vinf_kms_min", "km/s sweep"),
                ("flyby-vinf-kms-max", "flyby_vinf_kms_max", "km/s sweep"),
            ):
                b_val = getattr(d, b_dest, None)
                yield _Row(f"  {b_dest}", Input("" if b_val is None else str(b_val), id=b_id,
                                                placeholder="hyperbola only"), b_hint)
            for b_id, b_dest, b_default, b_hint in (
                ("flyby-start-sep-km", "flyby_start_sep_km", 4.0e5, "km"),
                ("settle-tdyn", "settle_tdyn", 5.0, "t_dyn"),
                ("relax-tdyn", "relax_tdyn", 0.5, "t_dyn"),
            ):
                b_val = getattr(d, b_dest, b_default)
                yield _Row(f"  {b_dest}", Input("" if b_val in (None, b_default) else str(b_val), id=b_id,
                                                placeholder=f"default {b_default:g}"), b_hint)
            yield _Row("  body_cache_dir", Input(getattr(d, "body_cache_dir", None) or "", id="body-cache-dir",
                                                placeholder="(default <output-root>/settled_bodies)"), "dir")
```

In `_build_argv()`, after the timeframe flags are emitted:

```python
        # Body / encounter: emit only what differs from the runner defaults
        if self._sel("body-source") != "lattice":
            argv.extend(["--body-source", self._sel("body-source")])
        if self._sel("encounter") != "ephemeris":
            argv.extend(["--encounter", self._sel("encounter")])
        for wid in ("flyby-rp-km", "flyby-rp-km-min", "flyby-rp-km-max", "flyby-vinf-kms",
                    "flyby-vinf-kms-min", "flyby-vinf-kms-max", "flyby-start-sep-km", "settle-tdyn", "relax-tdyn"):
            fi(wid, f"--{wid}")
        si("body-cache-dir", "--body-cache-dir")
```

(The runner validates the combination; `tui_run` shows its `ValueError` in the status bar as it does for the other sections.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_tui_boulders.py tests/test_tui_contact_keys.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 8: `tui_run.py` "Boulders" section

**Files:**
- Modify: `tui_run.py`: `compose()` (after Body / encounter); `_build_argv()` (after the body flags)
- Test: `tests/test_tui_boulders.py` (append)

**Interfaces:**
- Consumes: Task 3 flags.
- Produces: widget IDs `boulders` (Checkbox); `boulder-shape`, `boulder-placed` (Select, with a `""` "default" option); Inputs `f-boulder`, `f-boulder-min`, `f-boulder-max`, `rb-max`, `rb-max-min`, `rb-max-max`, `rb-min`, `boulder-q`, `boulder-axis-min`, `boulder-jagged`, `lobe-ratio`, `boulder-file`, `boulder-nmin`, `boulder-seed`, `bond-sigma-pa`, `bond-tau-ratio`, `kb-cgs`, `bond-reach`.

- [ ] **Step 1: Write the failing tests** (append):

```python
def _boulder_act(**fields):
    def act(app):
        app.query_one("#boulders", Checkbox).value = True
        for wid, val in fields.items():
            w = app.query_one(f"#{wid}")
            w.value = val
    return act


def test_boulders_reach_argv():
    argv = _sweep(actions=_boulder_act(**{"f-boulder-min": "0.1", "f-boulder-max": "0.5", "rb-max": "5",
                                          "bond-sigma-pa": "10", "boulder-placed": "binary", "lobe-ratio": "0.6"}))
    assert "--boulders" in argv
    assert _arg(argv, "--f-boulder-min") == "0.1" and _arg(argv, "--f-boulder-max") == "0.5"
    assert _arg(argv, "--rb-max") == "5" and _arg(argv, "--bond-sigma-pa") == "10"
    assert _arg(argv, "--boulder-placed") == "binary" and _arg(argv, "--lobe-ratio") == "0.6"


def test_boulders_off_emits_nothing():
    argv = _sweep()
    assert not any(a.startswith(("--boulder", "--f-boulder", "--rb-", "--bond-", "--kb-cgs", "--lobe"))
                   for a in argv)


def test_sweep_tui_boulder_fields_need_checkbox():
    def act(app):
        app.query_one("#f-boulder", Input).value = "0.2"
    out = _sweep(actions=act)
    assert isinstance(out, ValueError) and "boulders" in str(out)


def test_boulder_prefill_from_cli():
    cli = ["--boulders", "--bond-sigma-pa", "10", "--boulder-shape", "grown"]
    argv = _sweep(cli=cli)
    assert "--boulders" in argv and _arg(argv, "--bond-sigma-pa") == "10"
    assert _arg(argv, "--boulder-shape") == "grown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tui_boulders.py -v -k boulder`
Expected: FAIL (`NoMatches` for `#boulders`).

- [ ] **Step 3: Implement**

Module level in `tui_run.py`, after `_IN_DIMS`:

```python
# Boulders (phantomclumps): (widget id = flag stem, argparse dest, label, hint)
_BOULDER_INPUTS = [
    ("f-boulder",        "f_boulder",        "f_boulder (fixed)",      "0-1, def 0.3"),
    ("f-boulder-min",    "f_boulder_min",    "  f_boulder_min",        "sweep"),
    ("f-boulder-max",    "f_boulder_max",    "  f_boulder_max",        "sweep"),
    ("rb-max",           "rb_max",           "rb_max (fixed)",         "grain radii"),
    ("rb-max-min",       "rb_max_min",       "  rb_max_min",           "sweep"),
    ("rb-max-max",       "rb_max_max",       "  rb_max_max",           "sweep"),
    ("rb-min",           "rb_min",           "rb_min",                 ">= 1, def 1.5"),
    ("boulder-q",        "boulder_q",        "q (dN/dR ~ R^-q)",       "def 3.5"),
    ("boulder-axis-min", "boulder_axis_min", "axis_min",               "ellipsoid"),
    ("boulder-jagged",   "boulder_jagged",   "jagged",                 "grown"),
    ("lobe-ratio",       "lobe_ratio",       "lobe_ratio",             "binary"),
    ("boulder-file",     "boulder_file",     "boulder file",           "file"),
    ("boulder-nmin",     "boulder_nmin",     "nmin (grains)",          ">= 2"),
    ("boulder-seed",     "boulder_seed",     "seed",                   "def --seed"),
    ("bond-sigma-pa",    "bond_sigma_pa",    "bond_sigma_pa",          "Pa, required"),
    ("bond-tau-ratio",   "bond_tau_ratio",   "bond_tau / sigma",       "def 1"),
    ("kb-cgs",           "kb_cgs",           "kb_cgs",                 "def kn_cgs"),
    ("bond-reach",       "bond_reach",       "bond_reach",             "def 0.1"),
]
```

In `compose()` after the Body / encounter section:

```python
            # ── Boulders ───────────────────────────────────────────────────
            yield Static("Boulders  (settled bodies only; f_boulder / rb_max may be Sobol dims)", classes="sec")
            yield _Row("boulders", Checkbox("glue the body into bonded boulders (phantomclumps)",
                                            value=bool(getattr(d, "boulders", False)), id="boulders"))
            yield _Row("  shape", Select([("default (round)", ""), ("round", "round"),
                                          ("ellipsoid", "ellipsoid"), ("grown", "grown")],
                                         value=getattr(d, "boulder_shape", None) or "",
                                         id="boulder-shape", allow_blank=False), "random")
            yield _Row("  placed", Select([("default (none)", ""), ("none", "none"),
                                           ("binary", "binary"), ("file", "file")],
                                          value=getattr(d, "boulder_placed", None) or "",
                                          id="boulder-placed", allow_blank=False), "placed")
            for b_id, b_dest, b_label, b_hint in _BOULDER_INPUTS:
                b_val = getattr(d, b_dest, None)
                yield _Row(f"  {b_label}", Input("" if b_val is None else str(b_val), id=b_id), b_hint)
```

In `_build_argv()` after the body flags:

```python
        # Boulders
        filled = [w for w, *_ in _BOULDER_INPUTS if self._iv(w)] + [
            w for w in ("boulder-shape", "boulder-placed") if self._sel(w)]
        if not self._cb("boulders"):
            if filled:
                raise ValueError("tick boulders, or clear the boulder fields: " + ", ".join(filled))
        else:
            argv.append("--boulders")
            for w in ("boulder-shape", "boulder-placed"):
                if self._sel(w):
                    argv.extend([f"--{w}", self._sel(w)])
            for w, *_ in _BOULDER_INPUTS:
                if w in ("boulder-file",):
                    si(w, f"--{w}")
                elif w in ("boulder-nmin", "boulder-seed"):
                    ii(w, f"--{w}")
                else:
                    fi(w, f"--{w}")
```

Check `self._sel()` (line ~693) returns `""` for the `""` option value; if it returns `str(Select.BLANK)` or similar instead, compare against that.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_tui_boulders.py tests/test_tui_contact_keys.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 9: `tui_sim_render.py` "Boulders" block (one run)

**Files:**
- Modify: `tui_sim_render.py`: `compose()` (after the `flyby_start_sep_km` row, ~line 1289); `_build_sim_params()` (~line 1405); `run_sim_stage()` (~line 281)
- Test: `tests/test_tui_boulders.py` (append)

**Interfaces:**
- Consumes: Task 1 `BoulderSpec`, `validate_spec`; Task 2 `attach_tools`.
- Produces: widget IDs as in Task 8 minus the `-min`/`-max` inputs (`boulders`, `boulder-shape`, `boulder-placed`, `f-boulder`, `rb-max`, `rb-min`, `boulder-q`, `boulder-axis-min`, `boulder-jagged`, `lobe-ratio`, `boulder-file`, `boulder-nmin`, `boulder-seed`, `bond-sigma-pa`, `bond-tau-ratio`, `kb-cgs`, `bond-reach`).

- [ ] **Step 1: Write the failing tests** (append):

```python
import boulders as bd
import tui_sim_render as tsr


def _sim(selects=None, checks=None, **fields):
    async def _scenario():
        app = tsr.SimRenderTUIApp()
        async with app.run_test() as pilot:
            for wid, val in (selects or {}).items():
                app.query_one(f"#{wid}", Select).value = val
            for wid, val in (checks or {}).items():
                app.query_one(f"#{wid}", Checkbox).value = val
            for wid, val in fields.items():
                app.query_one(f"#{wid}", Input).value = val
            await pilot.pause()
            try:
                return app._build_sim_params(False)
            except ValueError as exc:
                return exc
    return asyncio.run(_scenario())


def test_sim_boulders_off_by_default():
    assert _sim().sample.boulders is None


def test_sim_boulders_reach_sample():
    p = _sim(selects={"body-source": "settled", "boulder-placed": "binary"}, checks={"boulders": True},
             **{"f-boulder": "0.2", "bond-sigma-pa": "10", "lobe-ratio": "0.6"})
    assert p.sample.boulders == bd.BoulderSpec(f_boulder=0.2, bond_sigma_pa=10.0, placed="binary", lobe_ratio=0.6)


@pytest.mark.parametrize("selects,fields,match", [
    ({}, {"bond-sigma-pa": "10"}, "settled"),                                     # lattice body
    ({"body-source": "settled"}, {}, "bond_sigma_pa"),                            # sigma missing
    ({"body-source": "settled"}, {"bond-sigma-pa": "10", "f-boulder": "2"}, "f_boulder"),
])
def test_sim_boulders_rejects(selects, fields, match):
    out = _sim(selects=selects, checks={"boulders": True}, **fields)
    assert isinstance(out, ValueError) and match in str(out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tui_boulders.py -v -k sim`
Expected: FAIL (`NoMatches` for `#boulders`).

- [ ] **Step 3: Implement**

`compose()`, right after `yield _Row("flyby_start_sep_km", Input("4e5", id="flyby-sep"), "km")`:

```python
            yield Static("Sim — boulders (settled only)", classes="sec")
            yield _Row("boulders", Checkbox("glue the body into bonded boulders (phantomclumps)",
                                            value=False, id="boulders"), "default: off")
            yield _Row("boulder_shape", Select([("round", "round"), ("ellipsoid", "ellipsoid"), ("grown", "grown")],
                                               value="round", id="boulder-shape", allow_blank=False), "random")
            yield _Row("boulder_placed", Select([("none", "none"), ("binary", "binary"), ("file", "file")],
                                                value="none", id="boulder-placed", allow_blank=False), "placed")
            for b_id, b_label, b_hint in (
                ("f-boulder", "f_boulder", "0-1, def 0.3"), ("rb-min", "rb_min", "def 1.5"),
                ("rb-max", "rb_max", "def 4"), ("boulder-q", "q_slope", "def 3.5"),
                ("boulder-axis-min", "axis_min", "ellipsoid"), ("boulder-jagged", "jagged", "grown"),
                ("lobe-ratio", "lobe_ratio", "binary"), ("boulder-file", "boulder_file", "file"),
                ("boulder-nmin", "nmin_clump", ">= 2"), ("boulder-seed", "seed", "int"),
                ("bond-sigma-pa", "bond_sigma_pa", "Pa, required"), ("bond-tau-ratio", "tau/sigma", "def 1"),
                ("kb-cgs", "kb_cgs", "def kn_cgs"), ("bond-reach", "bond_reach", "def 0.1"),
            ):
                yield _Row(b_label, Input("", id=b_id), b_hint)
```

New method on `SimRenderTUIApp`, next to `_apply_contact_fields`:

```python
    def _boulder_spec(self, body_source: str):
        """BoulderSpec from the Boulders block, or None when the checkbox is off."""
        if not self.query_one("#boulders", Checkbox).value:
            return None
        import boulders as bd
        if body_source != "settled":
            raise ValueError("boulders need body_source = settled")

        def num(wid, cast=float):
            raw = self._iv(wid)
            if not raw:
                return None
            try:
                return cast(raw)
            except ValueError:
                raise ValueError(f"{wid} must be a number (got '{raw}')") from None

        sigma = num("bond-sigma-pa")
        if sigma is None:
            raise ValueError("boulders need bond_sigma_pa (Pa; 0 = stretch-limit breaking only)")
        kw = dict(f_boulder=num("f-boulder"), rb_min=num("rb-min"), rb_max=num("rb-max"),
                  q_slope=num("boulder-q"), shape=self.query_one("#boulder-shape", Select).value,
                  axis_min=num("boulder-axis-min"), jagged=num("boulder-jagged"),
                  placed=self.query_one("#boulder-placed", Select).value, lobe_ratio=num("lobe-ratio"),
                  placed_file=(str(Path(self._iv("boulder-file")).expanduser().resolve())
                               if self._iv("boulder-file") else None),
                  nmin_clump=num("boulder-nmin", int), seed=num("boulder-seed", int), bond_sigma_pa=sigma,
                  bond_tau_ratio=num("bond-tau-ratio"), kb_cgs=num("kb-cgs"), bond_reach=num("bond-reach"))
        spec = bd.BoulderSpec(**{k: v for k, v in kw.items() if v is not None})
        bd.validate_spec(spec)
        return spec
```

In `_build_sim_params()`, after `apply_contact_defaults(sample)`:

```python
        sample.boulders = self._boulder_spec(body_source)
```

In `run_sim_stage()`, right after the `attach_settled_bodies(...)` call (inside `if sample.body_source == "settled":`):

```python
        import boulders
        boulders.attach_tools([sample], Path(params.phantom_dir), phantom_bin, params.dry_run)
```

Check `Checkbox` and `Select` are imported in `tui_sim_render.py` (they are used already: `Select` at line 1277, `Checkbox` at 1291).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_tui_boulders.py tests/test_tui_sim_render.py tests/test_tui_contact_keys.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint (no commit)** — stop for review.

---

### Task 10: Docs

**Files:**
- Modify: `CLAUDE.md` (sobol): § "Settled bodies and hyperbola encounters": new **`--boulders`** bullet; "Tools" bullet (phantomclumps now used with `--boulders`); add a boulder example command under "Key commands"
- Modify: `docs/tui_sim_render.md`: Boulders block fields
- Modify: `../docs/MIA_PACKING_WORKFLOW.md` Option 4: replace "Not wired into sobol (`settled_body.py`)." with a pointer to `--boulders`
- Modify: `../docs/METRICS.md`: columns `f_boulder`, `rb_max`, `f_boulder_actual`, `n_boulders`, `boulder_placed`, `bond_sigma_pa` (+ `kb_cgs`, `bond_reach`, `bond_tau_pa` param columns)
- Modify: `../docs/CONTEXT_CHANGELOG.md`: dated 2026-10-06 entry

- [ ] **Step 1: `sobol/CLAUDE.md` bullet** (insert after the `--encounter hyperbola` bullet):

```markdown
- **`--boulders`** (needs `--body-source settled`, `--bond-sigma-pa`): `boulders.py` copies the cached body into each run dir and runs `phantomclumps` (stdin from `clumps_stdin`; `clumps.log`, `boulders.json`), then stages `clumped_00000` instead of the cached dump on either encounter. Bond keys (`kb_cgs` default = `kn_cgs`, `bond_reach` 0.1, `bond_sigma_pa`, `bond_tau_pa` = `--bond-tau-ratio` × σ) go in through `apply_run_sample_to_in()`; on hyperbola this is load-bearing, `fly.in` comes out with `kb_cgs = 0`. Axes: `--f-boulder-min/max`, `--rb-max-min/max`; everything else fixed per batch (seed default `--seed`). Run fails if boulders were asked for but none made, or `phantom.log` lacks ` DEM clump bonds enabled`. `f_boulder = 0` + `--boulder-placed none` = all-dust baseline. Slug gains `bld<placed>_sig<σ>`. Columns: `f_boulder_actual` (placed + random grains / n_kept), `n_boulders`. Spec: `docs/superpowers/specs/2026-10-06-boulders-in-sweeps-design.md`.
```

and change the Tools bullet text `phantomclumps (moddump_clumps.f90, boulders; not used by the runner)` to `phantomclumps (moddump_clumps.f90, boulders; --boulders)`.

Example command (under the hyperbola example):

```markdown
Boulders on the settled body, f_boulder swept, contact binary (np300, 12 hr):
```bash
OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --phantom-dir /home/mboyle/Honours/sobol \
  --base-dir sobol --prefix sobol --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis 300 \
  --body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 --num-samples 8 --jobs 2 \
  --boulders --f-boulder-min 0 --f-boulder-max 0.5 --boulder-placed binary --bond-sigma-pa 10
```
```

(σ = 10 Pa here is an example only; the batch value comes from the σ trial in the spec's Open items.)

- [ ] **Step 2:** `docs/tui_sim_render.md`: add a "Boulders" subsection listing the fields, "settled only", "bond_sigma_pa required".
- [ ] **Step 3:** `../docs/MIA_PACKING_WORKFLOW.md`: replace `Not wired into sobol (\`settled_body.py\`).` with `Wired into sobol 2026-10-06: \`--boulders\` (\`sobol/CLAUDE.md\` § Settled bodies; per-run clump, no re-settle).`
- [ ] **Step 4:** `../docs/METRICS.md`: add rows for the new columns (definitions as in the spec's "Output columns").
- [ ] **Step 5:** `../docs/CONTEXT_CHANGELOG.md`: `2026-10-06 — sobol --boulders: per-run phantomclumps on the cached settled body; f_boulder / rb_max Sobol axes; bond strength fixed per batch (bond_sigma_pa). tui_run gains Body / encounter + Boulders sections; tui_sim_render a Boulders block.`
- [ ] **Step 6: Checkpoint (no commit)** — stop for review.

---

### Task 11: Full suite and integration runs (manual)

**Files:** none (verification only). Output under `../sobol_mass_runs/`.

- [ ] **Step 1: Full test suite**

Run: `cd /home/mboyle/Honours/sobol && python3 -m pytest tests`
Expected: all PASS (Blender tests launch `blender.exe`; ~2.5 min). Report any failure with its output; do not mark done.

- [ ] **Step 2: Rebuild check** — `cd /home/mboyle/Honours/sobol && strings phantom | grep -c "clump bond tensile strength"` → ≥ 1, and `ls bin/phantomclumps phantomclumps`. If missing: `make setup && make && make demtools`.

- [ ] **Step 3: Ephemeris, contact binary, bonds on** (one run, keep dumps):

```bash
cd /home/mboyle/Honours
OMP_NUM_THREADS=4 python3 sobol/run_mass_sobol_phantom.py --phantom-dir /home/mboyle/Honours/sobol \
  --base-dir sobol --prefix sobol --ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis-list 300 \
  --body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 --num-samples 1 --jobs 1 \
  --boulders --f-boulder 0.3 --boulder-placed binary --bond-sigma-pa 10 --no-cleanup
```

Check in the run dir: `clumps.log` (2 placed boulders), `setup.log` contains `boulders in body`, `phantom.log` contains `DEM clump bonds enabled` and `tensile strength =   10`, outputs CSV row `status = ok` with `n_boulders` and `f_boulder_actual` filled. Then:

```bash
python3 -c "import sarracen,glob; d=sorted(glob.glob('<run_dir>/sobol_0*'))[-1]; df=sarracen.read_phantom(d, separate_types='all'); df=df[0] if isinstance(df,(list,tuple)) else df; print([c for c in df.columns if c.startswith(('iclump','ibond'))])"
```

Expected: `iclump` plus `ibond01`… columns read without error (spec "not yet checked" item).

- [ ] **Step 4: Hyperbola with boulders**: same as the sobol/CLAUDE.md hyperbola example but `--num-samples 1`, fixed `--flyby-rp-km 30000 --flyby-vinf-kms 6`, plus `--boulders --f-boulder 0.3 --bond-sigma-pa 10 --no-cleanup`. Check `phantom.log` has `DEM clump bonds enabled` and the run's `sobol.in` has `kb_cgs` = its `kn_cgs`, `idamp = 0`.

- [ ] **Step 5: All-dust baseline vs no boulders**: run Step 3's command twice with `--boulder-placed none --f-boulder 0` and once without any boulder flags (same seed and settled body). Expected: `unbound_fraction`, `size_ratio` and spin columns agree to the precision PHANTOM reproduces a rerun (compare against a plain rerun of the no-boulder command to set the tolerance).

- [ ] **Step 6: Viz on a bonded run**: `python3 Code/viz/viz_preprocess_dump.py` on Step 3's run dir (see `Code/CLAUDE.md` for arguments). Expected: no error reading `iclump`/`ibond*`.

- [ ] **Step 7: Bond pre-load check** (spec Open item): in Step 3's `phantom.log`, compare the first few steps' kinetic energy with a no-boulder run. A sharp jump at step 1 means bonds start pre-loaded; record the finding in the spec's Open items.

- [ ] **Step 8: Report** results (pass/fail per step with log excerpts) to the user. Commits only when the user says.
