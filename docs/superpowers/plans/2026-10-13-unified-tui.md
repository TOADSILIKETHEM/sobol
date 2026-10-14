# Unified TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sobol/tui.py`, a new Textual app (Simulate / Render / Jobs tabs) that launches sweeps, single runs and renders as detached jobs, alongside the old TUIs, which stay untouched.

**Architecture:** `params_schema.py` describes every runner flag once (axes + plain fields) and turns form values into runner argv, validated by the runner's own `validate_args`. `job.py` executes a `job.json` headlessly (runner subprocess, then renders via `render_backend.py`, a lazy loader over the unchanged `tui_sim_render.py`) and writes `job_status.json`; `jobstore.py` owns those files and detached launch. `tui_app/` is the UI only: it reads and writes job files.

**Tech Stack:** Python 3, Textual 8.2.6 (installed), pytest, argparse (runner), `fcntl.flock`, `subprocess.Popen(start_new_session=True)`.

**Spec:** `sobol/docs/superpowers/specs/2026-10-13-unified-tui-design.md`

## Global Constraints

- Old tools untouched until phase 5: no edits to `tui_run.py`, `tui_sim_render.py`, `render_dumps.py`, `interactive_run_mass_sobol.py`, or their tests. New code imports from them.
- Runner and `settled_body.py` changes are additive only: a relaxed 0-dim rule (`--num-samples 1`, no Saltelli), env var `SOBOL_REPORT_BATCH_DIR`, a per-key cache lock. No new parser flag (the `-i` wizard prompts once per flag).
- No change to sim, metric or render behaviour.
- Jobs launch detached (`start_new_session=True`, stdin `/dev/null`, stdout+stderr → `job.log`) so they survive the TUI closing and SSH drops.
- Must run on laptop WSL, an OzSTAR login node (no Blender, no `Code/local_paths.json`), and other students' PCs. No Slurm logic.
- Keep `prefix = sobol` for anything rendered from a sweep (the converter globs `sobol_[0-9]*`).
- Laptop defaults: sweep `OMP_NUM_THREADS=1`, `--jobs 2`; single run `OMP_NUM_THREADS=4`, `--jobs 1`. Warn (never block) when another job runs on the same host.
- **Commits:** Max's standing rule is no commit or push until he has tested and explicitly asked. Each task ends with a checkpoint that runs the tests and lists the files; run its `git commit` only if Max has authorised per-task commits on `feature/unified-tui` (ask once at execution start). Never push.
- Phase 5 (retirement: extract `pipeline.py`, alias then delete old TUIs) is **not** in this plan. It gets its own plan after Task 13's manual checks pass and Max agrees.
- Match surrounding style: module docstrings saying what and why, `from __future__ import annotations`, `sys.path.insert(0, <sobol dir>)` before sibling imports, short comments only where the reason is not obvious.

## Review Focus

1. **Cancel during a Windows `blender.exe` render:** `killpg` kills the WSL side only; the Windows process may keep rendering. Expect the job to say so in its stage message. Test in Task 7 (`test_cancel_during_render_warns_about_blender`).
2. **Relative paths typed in the form** (e.g. `output_root = runs`) when the TUI was started from another directory: jobs and batches must land where the user meant (resolved against the TUI's cwd) and Show command must print absolute paths. Test in Task 10 (`test_relative_paths_become_absolute`).
3. **Filesystem without `flock`** (possible on OzSTAR Lustre/NFS mounts): the settle must still run, with a warning, not crash. Test in Task 4 (`test_lock_falls_back_without_flock`).
4. **Render-after selection naming runs that failed or do not exist** (`ids 3 99` when run 3 failed): render the rest, list the skipped ids, end `partial`, never crash. Test in Task 7 (`test_selection_skips_failed_and_missing_ids`).
5. **Unreadable status file** (half-written by an old version, hand-edited, disk full): the Jobs tab shows `unknown` for that job and keeps listing the others. Test in Task 6 (`test_read_status_tolerates_corrupt_file`).

---

## File map

| File | Status | Responsibility |
|------|--------|----------------|
| `sobol/params_schema.py` | create | Axes, plain fields, sections, gates; `from_argv`, `to_argv`, `apply_mode`, `check`, `show_command` |
| `sobol/run_mass_sobol_phantom.py` | modify (additive) | 0-dim single run; `_report_batch_dir()` via `SOBOL_REPORT_BATCH_DIR` |
| `sobol/settled_body.py` | modify (additive) | `_cache_lock()` around `build_settled_body` |
| `sobol/render_backend.py` | create | `load()`, `is_sobol_run()`, `form_to_dict/form_from_dict`, `render_run()`, `render_dumps_argv()` |
| `sobol/jobstore.py` | create | `JobSpec`, `JobStatus`, `Stage`; job dir, spec/status IO, `launch`, `cancel`, `liveness`, `list_jobs`, `tail`, `batch_progress` |
| `sobol/job.py` | create | Headless executor of one job dir |
| `sobol/tui_app/__init__.py` | create | package marker |
| `sobol/tui_app/simulate.py` | create | `SimulateForm` widget (schema-built rows, gates, mode) |
| `sobol/tui_app/render.py` | create | `RenderSettings`, `RunPicker`, `batches`, `runs_of`, `parse_selection` |
| `sobol/tui_app/jobs_pane.py` | create | `JobsPane` (list, detail, cancel, resume, log) |
| `sobol/tui_app/app.py` | create | `UnifiedApp` (tabs, buttons, launch) |
| `sobol/tui.py` | create | Entry point, default overrides |
| `sobol/docs/tui.md` | create | User doc for the new app |
| `sobol/CLAUDE.md` | modify | One short "Unified TUI (new)" entry |
| `sobol/tests/test_params_schema.py`, `test_runner_single_run.py`, `test_settle_lock.py`, `test_render_backend.py`, `test_jobstore.py`, `test_job.py`, `test_tui_app.py` | create | Tests |

---

### Task 0: Workspace

**Files:** none (git only)

- [ ] **Step 1: Confirm the base with Max.** `feature/portable-viz` has uncommitted slowmo/resume work in `tui_sim_render.py`, `render_dumps.py` and tests (`git -C /home/mboyle/Honours/sobol status --short`). The new app relies on `RenderFormValues.slowmo`/`resume`, so that work must be committed on `feature/portable-viz` first. Ask Max to commit it (or to authorise you to). Do not proceed until `git -C /home/mboyle/Honours/sobol status --short tui_sim_render.py` prints nothing.

- [ ] **Step 2: Create the worktree** as a sibling of `Code/`, so `honours_code.py` finds `../Code` without `HONOURS_CODE_DIR`:

```bash
git -C /home/mboyle/Honours/sobol worktree add -b feature/unified-tui /home/mboyle/Honours/sobol-unified-tui feature/portable-viz
```

All later paths are relative to `/home/mboyle/Honours/sobol-unified-tui` (written `sobol/...` below means that worktree).

- [ ] **Step 3: Baseline tests**

Run: `cd /home/mboyle/Honours/sobol-unified-tui && python3 -m pytest tests -x -q`
Expected: same pass/skip count as the main checkout (~2.5 min). Record the numbers; every phase ends by matching them plus the new tests.

---

### Task 1: Schema — axes, fields, sections, gates

**Files:**
- Create: `sobol/params_schema.py`
- Test: `sobol/tests/test_params_schema.py`

**Interfaces:**
- Consumes: `run_mass_sobol_phantom.build_parser()`; `interactive_run_mass_sobol.DEST_TO_SECTION`, `.INTERACTIVE_BRIEF`, `.BOULDER_GATE_DESTS`.
- Produces: `Axis(stem, label, lo, hi, fixed=None, pin=False)`, `AXES: Tuple[Axis, ...]`, `PlainField(dest, flag, kind, choices, default, section, brief)`, `PLAIN: Tuple[PlainField, ...]`, `HIDDEN: frozenset`, `AXIS_DESTS: frozenset`, `SECTION_ORDER: Tuple[str, ...]`, `section_of(dest) -> str`, `items_in(section) -> List[Union[Axis, PlainField]]`, `flag_of(dest) -> str`, `AxisValue(sweep=False, fixed="", lo="", hi="")`, `FormValues(plain: Dict[str, object], axes: Dict[str, AxisValue])`, `enabled(key, fv) -> bool`, `SINGLE_HIDDEN: frozenset`.

- [ ] **Step 1: Write the failing tests**

```python
"""params_schema: every runner flag described once (completeness, sections, gates)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import params_schema as ps
import run_mass_sobol_phantom as runner


def _dests():
    return {a.dest for a in runner.build_parser()._actions if a.dest != "help"}


def test_every_runner_dest_is_classified():
    # Fails when a flag is added to build_parser() without a schema decision.
    plain = {f.dest for f in ps.PLAIN}
    assert _dests() == plain | ps.AXIS_DESTS | ps.HIDDEN
    assert not plain & ps.AXIS_DESTS
    assert not plain & ps.HIDDEN


def test_axis_dests_exist_and_are_unique():
    seen = []
    for ax in ps.AXES:
        for d in (ax.lo, ax.hi, ax.fixed):
            if d:
                assert d in _dests(), d
                seen.append(d)
    assert len(seen) == len(set(seen))


def test_every_item_has_a_known_section_and_appears_once():
    for f in ps.PLAIN:
        assert f.section in ps.SECTION_ORDER, f.dest
    for ax in ps.AXES:
        assert ps.section_of(ax.lo) in ps.SECTION_ORDER, ax.stem
    items = [i for s in ps.SECTION_ORDER for i in ps.items_in(s)]
    assert len(items) == len(ps.PLAIN) + len(ps.AXES)


def test_widget_kinds_follow_argparse():
    kinds = {f.dest: f.kind for f in ps.PLAIN}
    assert kinds["boulders"] == "check"
    assert kinds["body_source"] == "select"
    assert kinds["np_apophis_list"] == "list"
    assert kinds["prefix"] == "text"


def test_gates():
    fv = ps.FormValues(plain={"body_source": "lattice", "encounter": "ephemeris", "boulders": False}, axes={})
    assert not ps.enabled("boulders", fv)
    assert not ps.enabled("f_boulder", fv)
    assert not ps.enabled("flyby_rp_km", fv)
    assert not ps.enabled("settle_tdyn", fv)
    assert ps.enabled("prefix", fv)
    fv.plain.update(body_source="settled", boulders=True, encounter="hyperbola")
    assert ps.enabled("boulders", fv)
    assert ps.enabled("bond_sigma_pa", fv)
    assert ps.enabled("f_boulder", fv)
    assert ps.enabled("flyby_rp_km", fv)
    assert ps.enabled("settle_tdyn", fv)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/mboyle/Honours/sobol-unified-tui && python3 -m pytest tests/test_params_schema.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'params_schema'`.

- [ ] **Step 3: Implement**

```python
"""One description of the sweep runner's flags for front-ends (tui_app).

Section and help text come from the -i wizard's DEST_TO_SECTION / INTERACTIVE_BRIEF
(imported, not moved, while the old TUIs still exist). Validity is decided only by the
runner's own parse_args + validate_args (see check()); gates here grey widgets out, nothing more.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interactive_run_mass_sobol as _wizard  # noqa: E402
import run_mass_sobol_phantom as runner  # noqa: E402


@dataclass(frozen=True)
class Axis:
    """A quantity that is either fixed or swept between lo and hi."""

    stem: str
    label: str
    lo: str  # dest of the min flag
    hi: str  # dest of the max flag
    fixed: Optional[str] = None  # dest of the fixed flag, if the runner has one
    # No fixed flag: a fixed value is written as lo == hi (runner allows min <= max for these).
    # fixed None and pin False (mass): the fixed side means "template value".
    pin: bool = False


AXES: Tuple[Axis, ...] = (
    Axis("mass", "mass (kg)", "mass_min_kg", "mass_max_kg"),
    Axis("scale_vel", "scale_vel", "scale_vel_min", "scale_vel_max", pin=True),
    Axis("scale_pos", "scale_pos", "scale_pos_min", "scale_pos_max", pin=True),
    Axis("scale_r_apophis", "scale_r_apophis", "scale_r_apophis_min", "scale_r_apophis_max", pin=True),
    Axis("scale_rho", "scale_rho", "scale_rho_min", "scale_rho_max", pin=True),
    Axis("spin_period", "spin_period (hr)", "spin_period_min", "spin_period_max", fixed="spin_period_fixed"),
    Axis("spin_obliquity", "spin_obliquity (deg)", "spin_obliquity_min", "spin_obliquity_max", pin=True),
    Axis("spin_azimuth", "spin_azimuth (deg)", "spin_azimuth_min", "spin_azimuth_max", pin=True),
    Axis("spin_torque_align", "spin_torque_align (deg)", "spin_torque_align_min", "spin_torque_align_max",
         pin=True),
    Axis("kt", "kt_cgs", "kt_min", "kt_max", fixed="kt_fixed"),
    Axis("eps_n", "epsilon_n_dem", "eps_n_min", "eps_n_max", pin=True),
    Axis("kn", "kn_cgs", "kn_min", "kn_max", pin=True),
    Axis("mu_s", "mu_s", "mu_s_min", "mu_s_max", fixed="mu_s"),
    Axis("mu_r", "mu_r", "mu_r_min", "mu_r_max", fixed="mu_r"),
    Axis("mu_t", "mu_t", "mu_t_min", "mu_t_max", fixed="mu_t"),
    Axis("eps_t", "epsilon_t_dem", "eps_t_min", "eps_t_max", fixed="epsilon_t_dem"),
    Axis("coh_pa", "cohesion_pa (Pa)", "coh_pa_min", "coh_pa_max", pin=True),
    Axis("tmax_hours", "tmax (hr)", "tmax_hours_min", "tmax_hours_max", fixed="tmax_hours"),
    Axis("dtmax_hours", "dtmax (hr)", "dtmax_hours_min", "dtmax_hours_max", fixed="dtmax_hours"),
    Axis("flyby_rp_km", "flyby rp (km)", "flyby_rp_km_min", "flyby_rp_km_max", fixed="flyby_rp_km"),
    Axis("flyby_vinf_kms", "flyby v_inf (km/s)", "flyby_vinf_kms_min", "flyby_vinf_kms_max",
         fixed="flyby_vinf_kms"),
    Axis("f_boulder", "f_boulder", "f_boulder_min", "f_boulder_max", fixed="f_boulder"),
    Axis("rb_max", "rb_max (grain radii)", "rb_max_min", "rb_max_max", fixed="rb_max"),
)
AXIS_DESTS = frozenset(d for ax in AXES for d in (ax.lo, ax.hi, ax.fixed) if d)

# Not shown: the wizard switch, deprecated or ignored flags (kc_* map onto kt_* in the runner).
HIDDEN = frozenset({"interactive", "apophis_ref_mass_kg", "kc_fixed", "kc_min", "kc_max", "kc_scale_ref_np"})

SECTION_ORDER: Tuple[str, ...] = (
    "Paths", "Sampling", "Mass & scale", "Body / encounter", "DEM contact", "Boulders", "Spin", "Time",
    "Setup", "Execution",
)
_WIZARD_SECTION = {
    "Paths": "Paths",
    "Sampling": "Sampling",
    "Mass": "Mass & scale",
    "Scale bounds (optional dimensions)": "Mass & scale",
    "Body / encounter": "Body / encounter",
    "DEM Contact (optional dimensions)": "DEM contact",
    "Boulders": "Boulders",
    "Spin (optional dimension)": "Spin",
    "Time (optional override / dimension)": "Time",
    "Setup toggles": "Setup",
    "Sinks / post-processing": "Setup",
    "Batch naming": "Execution",
    "Execution": "Execution",
}
# Dests the wizard maps leave out.
_EXTRA_SECTION = {
    "spin_torque_align_min": "Spin", "spin_torque_align_max": "Spin",
    "spin_period_fixed": "Spin", "spin_period_list": "Spin",
    "kt_fixed": "DEM contact", "kt_scale_ref_np": "DEM contact", "dn_cohes_factor": "DEM contact",
    "coh_gap_max_cgs": "DEM contact",
    "use_dem_fixed": "Setup",
    "no_cleanup": "Execution",
}

_PARSER = runner.build_parser()
_ACTIONS: Dict[str, argparse.Action] = {a.dest: a for a in _PARSER._actions if a.dest != "help"}


def section_of(dest: str) -> str:
    if dest in _EXTRA_SECTION:
        return _EXTRA_SECTION[dest]
    return _WIZARD_SECTION[_wizard.DEST_TO_SECTION[dest]]


def flag_of(dest: str) -> str:
    return _ACTIONS[dest].option_strings[0]


@dataclass(frozen=True)
class PlainField:
    dest: str
    flag: str
    kind: str  # "check" | "select" | "list" | "text"
    choices: Tuple[str, ...]
    default: object
    section: str
    brief: str


def _kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        return "check"
    if action.choices:
        return "select"
    if action.nargs == "+":
        return "list"
    return "text"


def _brief(action: argparse.Action) -> str:
    if action.dest in _wizard.INTERACTIVE_BRIEF:
        return _wizard.INTERACTIVE_BRIEF[action.dest][0]
    return (action.help or "").split(". ")[0]


PLAIN: Tuple[PlainField, ...] = tuple(
    PlainField(a.dest, a.option_strings[0], _kind(a), tuple(a.choices or ()), a.default, section_of(a.dest),
               _brief(a))
    for a in _PARSER._actions
    if a.dest != "help" and a.dest not in AXIS_DESTS and a.dest not in HIDDEN
)
_PLAIN_BY_DEST = {f.dest: f for f in PLAIN}
_AXIS_BY_DEST = {d: ax for ax in AXES for d in (ax.lo, ax.hi, ax.fixed) if d}


def items_in(section: str) -> List[Union[Axis, PlainField]]:
    """Rows of one section in parser order; an axis sits where its first flag is registered."""
    out: List[Union[Axis, PlainField]] = []
    placed = set()
    for a in _PARSER._actions:
        f = _PLAIN_BY_DEST.get(a.dest)
        if f is not None and f.section == section:
            out.append(f)
        ax = _AXIS_BY_DEST.get(a.dest)
        if ax is not None and ax.stem not in placed and section_of(ax.lo) == section:
            placed.add(ax.stem)
            out.append(ax)
    return out


@dataclass
class AxisValue:
    sweep: bool = False
    fixed: str = ""
    lo: str = ""
    hi: str = ""


@dataclass
class FormValues:
    plain: Dict[str, object] = field(default_factory=dict)  # dest -> str | bool | None (select blank)
    axes: Dict[str, AxisValue] = field(default_factory=dict)  # stem -> AxisValue


def _settled(fv: FormValues) -> bool:
    return fv.plain.get("body_source") == "settled"


def _hyperbola(fv: FormValues) -> bool:
    return fv.plain.get("encounter") == "hyperbola"


def _boulders_on(fv: FormValues) -> bool:
    return _settled(fv) and bool(fv.plain.get("boulders"))


# Keyed by plain dest or axis stem. Visual only: the runner still decides validity.
GATES: Dict[str, Callable[[FormValues], bool]] = {
    **{d: _boulders_on for d in _wizard.BOULDER_GATE_DESTS},
    "boulders": _settled,
    "settle_tdyn": _settled, "relax_tdyn": _settled, "settle_with_friction": _settled, "body_cache_dir": _settled,
    "flyby_rp_km": _hyperbola, "flyby_vinf_kms": _hyperbola, "flyby_start_sep_km": _hyperbola,
    "flyby_perturber_earth_masses": _hyperbola,
}


def enabled(key: str, fv: FormValues) -> bool:
    gate = GATES.get(key)
    return True if gate is None else gate(fv)


# Rows hidden in Single run mode (the whole Sampling section is hidden too).
SINGLE_HIDDEN = frozenset({"jobs"})
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_params_schema.py -v`
Expected: 5 passed. If `test_every_runner_dest_is_classified` lists a dest, add it to `_EXTRA_SECTION` (or `HIDDEN` if deprecated) rather than editing the wizard.

- [ ] **Step 5: Checkpoint**

```bash
python3 -m pytest tests/test_params_schema.py -q
git add params_schema.py tests/test_params_schema.py
git commit -m "params_schema: describe every runner flag once (axes, fields, sections, gates)"   # only if Max authorised
```

---

### Task 2: Schema — argv round trip, mode, check, command

**Files:**
- Modify: `sobol/params_schema.py` (append)
- Test: `sobol/tests/test_params_schema.py` (append)

**Interfaces:**
- Consumes: Task 1 names.
- Produces: `parse(argv) -> argparse.Namespace` (raises `ValueError`, never exits), `from_argv(argv) -> FormValues`, `to_argv(fv) -> List[str]`, `apply_mode(fv, mode: "single"|"sweep") -> FormValues` (copy), `Check(error: Optional[str], warnings: str)`, `check(argv) -> Check`, `show_command(argv, omp: int) -> str`, `absolutise(fv, cwd: Path) -> FormValues` (copy; path dests made absolute).

- [ ] **Step 1: Write the failing tests** (append)

```python
import shlex

import pytest

CASES = {
    "mass_scale": "--base-dir sobol --prefix sobol --num-samples 50 --mass-min-kg 1e10 --mass-max-kg 1e11 "
                  "--scale-vel-min 0.9 --scale-vel-max 1.1 --jobs 2",
    "settled_ephemeris": "--base-dir sobol --prefix sobol --phantom-dir /home/mboyle/Honours/sobol "
                         "--ephemeris-cache-dir sobol --use-dem-fixed true --np-apophis-list 300 "
                         "--body-source settled --spin-period-fixed 30 --tmax-hours 12 --dtmax-hours 1 "
                         "--num-samples 1 --jobs 1",
    "hyperbola": "--use-dem-fixed true --np-apophis 300 --body-source settled --encounter hyperbola "
                 "--flyby-rp-km-min 20000 --flyby-rp-km-max 40000 --flyby-vinf-kms-min 5 --flyby-vinf-kms-max 7 "
                 "--tmax-hours 48 --dtmax-hours 0.5 --num-samples 16 --jobs 2",
    "boulders": "--use-dem-fixed true --np-apophis 300 --body-source settled --spin-period-fixed 30 "
                "--tmax-hours 12 --dtmax-hours 1 --num-samples 8 --jobs 2 --boulders --f-boulder-min 0 "
                "--f-boulder-max 0.5 --boulder-placed binary --bond-sigma-pa 10",
    "contact": "--use-dem-fixed true --np-apophis 300 --mu-s-min 0.2 --mu-s-max 0.8 --mu-r 0.1 --eps-t 0.5 "
               "--coh-pa-min 1 --coh-pa-max 10",
    "pinned": "--use-dem-fixed true --np-apophis 300 --spin-torque-align-min 45 --spin-torque-align-max 45",
    "lists": "--use-dem-fixed true --np-apophis-list 100 300 --spin-period-list 3 30 --no-cleanup",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_matches_hand_written_command(name):
    argv = shlex.split(CASES[name])
    assert vars(ps.parse(ps.to_argv(ps.from_argv(argv)))) == vars(ps.parse(argv))


def test_pinned_axis_reads_back_as_fixed():
    fv = ps.from_argv(shlex.split(CASES["pinned"]))
    assert fv.axes["spin_torque_align"] == ps.AxisValue(sweep=False, fixed="45.0")
    argv = ps.to_argv(fv)
    assert argv[argv.index("--spin-torque-align-min") + 1] == "45.0"
    assert argv[argv.index("--spin-torque-align-max") + 1] == "45.0"


def test_defaults_are_not_emitted():
    assert ps.to_argv(ps.from_argv([])) == []


def test_apply_mode_single_forces_one_run_and_no_sweeps():
    fv = ps.from_argv(shlex.split(CASES["hyperbola"]))
    single = ps.apply_mode(fv, "single")
    assert single.plain["num_samples"] == "1" and single.plain["jobs"] == "1"
    assert not any(av.sweep for av in single.axes.values())
    assert fv.axes["flyby_rp_km"].sweep  # input not mutated


def test_check_reports_runner_errors_without_exiting():
    assert "mass bounds" in ps.check(["--mass-min-kg", "1e10"]).error
    assert "invalid int value" in ps.check(["--num-samples", "x"]).error
    assert ps.check(shlex.split(CASES["mass_scale"])).error is None


def test_show_command():
    cmd = ps.show_command(["--jobs", "2", "--batch-label", "a b"], 1)
    assert cmd == "OMP_NUM_THREADS=1 python3 sobol/run_mass_sobol_phantom.py --jobs 2 --batch-label 'a b'"


def test_relative_paths_become_absolute(tmp_path):
    fv = ps.from_argv(["--output-root", "runs", "--base-dir", "/abs/sobol"])
    out = ps.absolutise(fv, tmp_path)
    assert out.plain["output_root"] == str(tmp_path / "runs")
    assert out.plain["base_dir"] == "/abs/sobol"
    assert fv.plain["output_root"] == "runs"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_params_schema.py -v`
Expected: new tests FAIL with `AttributeError: module 'params_schema' has no attribute 'parse'`.

- [ ] **Step 3: Implement** (append to `params_schema.py`; add `import contextlib`, `import copy`, `import io`, `import shlex` to the imports)

```python
def parse(argv) -> argparse.Namespace:
    """The runner's parser, raising ValueError instead of printing usage and exiting."""
    parser = runner.build_parser()

    def _fail(message: str):
        raise ValueError(message)

    parser.error = _fail
    return parser.parse_args(list(argv))


def _num(x) -> str:
    return "" if x is None else str(x)


def from_argv(argv) -> FormValues:
    ns = parse(argv)
    plain: Dict[str, object] = {}
    for f in PLAIN:
        v = getattr(ns, f.dest)
        if f.kind == "check":
            plain[f.dest] = bool(v)
        elif f.kind == "select":
            plain[f.dest] = v
        elif f.kind == "list":
            plain[f.dest] = " ".join(str(x) for x in v) if v else ""
        else:
            plain[f.dest] = _num(v)
    axes: Dict[str, AxisValue] = {}
    for ax in AXES:
        lo, hi = getattr(ns, ax.lo), getattr(ns, ax.hi)
        fixed = getattr(ns, ax.fixed) if ax.fixed else None
        if ax.pin and lo is not None and lo == hi:
            axes[ax.stem] = AxisValue(fixed=_num(lo))
        elif lo is not None or hi is not None:
            axes[ax.stem] = AxisValue(sweep=True, lo=_num(lo), hi=_num(hi))
        else:
            axes[ax.stem] = AxisValue(fixed=_num(fixed))
    return FormValues(plain, axes)


def _text_default(default) -> str:
    return "" if default is None else str(default)


def to_argv(fv: FormValues) -> List[str]:
    """Form values -> runner argv. Defaults are left out; bad text is passed through for check() to report."""
    out: List[str] = []
    for f in PLAIN:
        v = fv.plain.get(f.dest)
        if f.kind == "check":
            if v:
                out.append(f.flag)
        elif f.kind == "select":
            if v not in (None, "") and v != f.default:
                out += [f.flag, str(v)]
        elif f.kind == "list":
            tokens = str(v or "").split()
            if tokens:
                out += [f.flag, *tokens]
        else:
            text = str(v or "").strip()
            if text and text != _text_default(f.default):
                out += [f.flag, text]
    for ax in AXES:
        av = fv.axes.get(ax.stem, AxisValue())
        if av.sweep:
            for dest, text in ((ax.lo, av.lo), (ax.hi, av.hi)):
                if text.strip():
                    out += [flag_of(dest), text.strip()]
            continue
        text = av.fixed.strip()
        if not text:
            continue
        if ax.fixed:
            out += [flag_of(ax.fixed), text]
        elif ax.pin:
            out += [flag_of(ax.lo), text, flag_of(ax.hi), text]
    return out


def apply_mode(fv: FormValues, mode: str) -> FormValues:
    """Single run = one sample, one job, every axis fixed. Returns a copy."""
    out = copy.deepcopy(fv)
    if mode == "single":
        out.plain.update(num_samples="1", jobs="1", saltelli_n="", saltelli_calc_second_order=False)
        for av in out.axes.values():
            av.sweep = False
    return out


_PATH_DESTS = ("base_dir", "output_root", "phantom_dir", "ephemeris_cache_dir", "body_cache_dir", "shape_file",
               "boulder_file")


def absolutise(fv: FormValues, cwd: Path) -> FormValues:
    """Resolve relative path fields against the directory the TUI was started from. Returns a copy."""
    out = copy.deepcopy(fv)
    for dest in _PATH_DESTS:
        text = str(out.plain.get(dest) or "").strip()
        if text and not Path(text).expanduser().is_absolute():
            out.plain[dest] = str(Path(cwd) / Path(text).expanduser())
    return out


@dataclass
class Check:
    error: Optional[str]
    warnings: str  # whatever validate_args printed


def check(argv) -> Check:
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = parse(argv)
            runner._normalize_kt_cli_aliases(args)
            runner.validate_args(args)
    except ValueError as exc:
        return Check(str(exc), (out.getvalue() + err.getvalue()).strip())
    except Exception as exc:  # e.g. a missing shape file: report it, never kill the UI
        return Check(f"{type(exc).__name__}: {exc}", (out.getvalue() + err.getvalue()).strip())
    return Check(None, (out.getvalue() + err.getvalue()).strip())


def show_command(argv, omp: int) -> str:
    return f"OMP_NUM_THREADS={omp} python3 sobol/run_mass_sobol_phantom.py {shlex.join(list(argv))}".rstrip()
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_params_schema.py -v`
Expected: all pass (5 + 7 parametrised + 6).

- [ ] **Step 5: Checkpoint**

```bash
git add params_schema.py tests/test_params_schema.py
git commit -m "params_schema: argv round trip, single mode, runner-backed check"   # only if Max authorised
```

---

### Task 3: Runner — single all-fixed run and batch-dir report

**Files:**
- Modify: `sobol/run_mass_sobol_phantom.py` (the 0-dim check near line 1191; `main()` right after `output_root = ...`, near line 3870)
- Test: `sobol/tests/test_runner_single_run.py`

**Interfaces:**
- Produces: `runner.REPORT_BATCH_DIR_ENV = "SOBOL_REPORT_BATCH_DIR"`, `runner._report_batch_dir(output_root: Path) -> None`. `validate_args` accepts 0 dims when `num_samples == 1` and `saltelli_n is None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Runner: one all-fixed run (num_samples 1) and SOBOL_REPORT_BATCH_DIR."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import params_schema as ps
import run_mass_sobol_phantom as runner

FIXED = ["--use-dem-fixed", "true", "--np-apophis", "300", "--spin-period-fixed", "30"]


def test_single_run_needs_no_dimension():
    runner.validate_args(runner.parse_args(FIXED + ["--num-samples", "1"]))


def test_zero_dims_still_rejected_for_sweeps():
    with pytest.raises(ValueError, match="No varying dimensions"):
        runner.validate_args(runner.parse_args(FIXED))  # num_samples defaults to 8


def test_zero_dims_still_rejected_with_saltelli():
    with pytest.raises(ValueError):
        runner.validate_args(runner.parse_args(FIXED + ["--num-samples", "1", "--saltelli-n", "4"]))


def test_single_mode_form_passes_check():
    fv = ps.from_argv(FIXED)
    assert ps.check(ps.to_argv(ps.apply_mode(fv, "single"))).error is None


def test_report_batch_dir(tmp_path, monkeypatch):
    report = tmp_path / "batch_dir.txt"
    monkeypatch.setenv(runner.REPORT_BATCH_DIR_ENV, str(report))
    monkeypatch.setattr(sys, "argv", ["run_mass_sobol_phantom.py", "--output-root", str(tmp_path / "out"),
                                      *FIXED, "--num-samples", "1"])

    def stop(*_a, **_k):
        raise RuntimeError("stop after report")

    monkeypatch.setattr(runner, "preflight", stop)
    assert runner.main() == 1
    batch = Path(report.read_text(encoding="utf-8").strip())
    assert batch.parent == (tmp_path / "out").resolve()
    assert batch.name.startswith("sobol_")


def test_no_report_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.REPORT_BATCH_DIR_ENV, raising=False)
    runner._report_batch_dir(tmp_path / "x")  # no file named anywhere, no error
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_runner_single_run.py -v`
Expected: `test_single_run_needs_no_dimension` and `test_single_mode_form_passes_check` FAIL with "No varying dimensions"; report tests FAIL with `AttributeError: ... REPORT_BATCH_DIR_ENV`.

- [ ] **Step 3: Implement.** In `validate_args`, replace the 0-dim check:

```python
    # np_apophis_list itself constitutes variation; allow dim=0 in that mode, and for one fixed run.
    dim = count_dimensions(args)
    single_fixed_run = args.num_samples == 1 and args.saltelli_n is None
    if dim == 0 and not np_list and not single_fixed_run:
        raise ValueError(
            "No varying dimensions: set both --mass-min-kg and --mass-max-kg and/or pass scale */ "
            "vary-use-dem / vary-apophis-only / --np-apophis-list (or --num-samples 1 for one fixed run)."
        )
```

Add near the other module constants (after `_DEFAULT_PHANTOM_DIR`):

```python
# Set by sobol/job.py to a file path; main() writes the batch directory there once it is known.
# An env var, not a flag, so the -i wizard (one prompt per flag) is unchanged.
REPORT_BATCH_DIR_ENV = "SOBOL_REPORT_BATCH_DIR"
```

Add before `def main()`:

```python
def _report_batch_dir(output_root: Path) -> None:
    target = os.environ.get(REPORT_BATCH_DIR_ENV)
    if target:
        Path(target).write_text(f"{output_root}\n", encoding="utf-8")
```

In `main()`, directly after `output_root = Path(args.output_root).resolve() / batch_basename`:

```python
        _report_batch_dir(output_root)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_runner_single_run.py tests/test_body_encounter_cli.py tests/test_dem_contact_keys.py -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint**

```bash
git add run_mass_sobol_phantom.py tests/test_runner_single_run.py
git commit -m "runner: one all-fixed run with --num-samples 1; SOBOL_REPORT_BATCH_DIR"   # only if Max authorised
```

---

### Task 4: Settled-body cache lock

**Files:**
- Modify: `sobol/settled_body.py` (`build_settled_body`, line ~218; add `import fcntl`, `import sys`, `from contextlib import contextmanager`)
- Test: `sobol/tests/test_settle_lock.py`

**Interfaces:**
- Produces: `settled_body._cache_lock(cache_root: Path, key: str)` (context manager). `build_settled_body` signature unchanged.

- [ ] **Step 1: Write the failing tests**

```python
"""Two processes building one settled-body key: the second waits and reuses the body."""
import fcntl
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
import settled_body as sb


def _build(tmp_path):
    tools = SimpleNamespace(moddump=tmp_path / "moddump")
    return sb.build_settled_body(None, tmp_path, tmp_path / "t.setup", tmp_path / "ps", tmp_path / "p", tools, None)


def test_waiting_builder_reuses_body(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "settle_key", lambda spec, bins: "k")
    loaded = []
    monkeypatch.setattr(sb.SettledBody, "load", staticmethod(lambda p: loaded.append(Path(p)) or "BODY"))
    holder = open(tmp_path / "k.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)  # another builder is mid-settle
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("body", _build(tmp_path)))
    th.start()
    th.join(0.5)
    assert th.is_alive()  # blocked on the lock, not settling a second copy
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "body.json").write_text("{}")
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    th.join(5)
    assert out["body"] == "BODY"
    assert loaded == [tmp_path / "k"]


def test_lock_falls_back_without_flock(tmp_path, monkeypatch, capsys):
    def no_flock(*_a):
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(sb.fcntl, "flock", no_flock)
    with sb._cache_lock(tmp_path, "k"):
        pass
    assert "without a lock" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_settle_lock.py -v`
Expected: first test FAILS (the thread does not block; it tries to settle and errors on the missing template), second FAILS with `AttributeError: ... _cache_lock`.

- [ ] **Step 3: Implement.** Add the helper above `build_settled_body`:

```python
@contextmanager
def _cache_lock(cache_root: Path, key: str):
    """Exclusive per-key lock so two launches (CLI, TUI jobs) never settle the same body at once.
    Some cluster filesystems lack flock: then build unlocked, as before, with a warning."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    with open(cache_root / f"{key}.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except OSError as exc:
            print(f"[WARN] settled-body cache {cache_root}: no flock ({exc}); building without a lock",
                  file=sys.stderr)
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
```

In `build_settled_body`, change the docstring's last sentence to "Safe to call concurrently for the same spec: a per-key lock makes the second caller wait and reuse the body." Then wrap everything from `work = Path(cache_root) / f"{key}.partial"` to the final `return` in the lock, re-checking the cache inside it:

```python
    if (final / "body.json").is_file():
        return SettledBody.load(final)
    with _cache_lock(Path(cache_root), key):
        if (final / "body.json").is_file():  # built by another process while we waited
            return SettledBody.load(final)
        work = Path(cache_root) / f"{key}.partial"
        # ... the existing body of the function, unchanged, indented one level ...
        os.replace(work, final)  # atomic: a dir without body.json is never loaded
        return SettledBody.load(final)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_settle_lock.py tests/test_settled_body.py -q`
Expected: all pass.

- [ ] **Step 5: Phase check, then checkpoint**

Run: `python3 -m pytest tests -q` and compare with the Task 0 baseline plus the new tests.

```bash
git add settled_body.py tests/test_settle_lock.py
git commit -m "settled_body: per-key cache lock (closes the concurrent-build gap)"   # only if Max authorised
```

---

### Task 5: Render backend

**Files:**
- Create: `sobol/render_backend.py`
- Test: `sobol/tests/test_render_backend.py`

**Interfaces:**
- Consumes: `tui_sim_render` (unchanged): `RenderFormValues`, `RunRecord`, `PipelineResult`, `PathContext`, `RENDER_PATHS`, `RENDER_PATH_NAMES`, `NPZ_PATHS`, `DATA_DIR`, `check_blender_exe`, `validate_render_form`, `needs_npz`, `count_matching`, `run_convert_stage`, `ConvertError`, `_grains_and_bodies_dirs`, `_viz_dir`, `build_dump_preprocess_command`, `run_dump_preprocess_stage`, `run_render_path`, `run_compare_for_paths`, `format_path_result`, `_cleanup_run_dir`.
- Produces: `load() -> Union[ModuleType, str]`, `is_sobol_run(run_dir) -> bool`, `form_to_dict(form) -> dict`, `form_from_dict(t, d) -> RenderFormValues`, `render_run(t, run_dir, form, keep_dumps=False, base_output_dir=None, on_line=None) -> PipelineResult`, `render_dumps_argv(run_dir, form, *, name=None, prefix="sobol", dem_model=None, np_apophis=None, apophis_only=False, encounter=None) -> List[str]`.

- [ ] **Step 1: Write the failing tests**

```python
"""render_backend: lazy tui_sim_render import and render_run on an existing sobol run dir."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import render_backend as rb

T = rb.load()
needs_t = pytest.mark.skipif(isinstance(T, str), reason=f"render stages unavailable here: {T}")


def test_load_turns_import_exit_into_reason(monkeypatch):
    def boom(name):
        raise SystemExit("Code/local_paths.json missing")

    monkeypatch.setattr(importlib, "import_module", boom)
    reason = rb.load()
    assert isinstance(reason, str) and "local_paths.json" in reason


def test_is_sobol_run(tmp_path):
    assert not rb.is_sobol_run(tmp_path)
    (tmp_path / "sobol.setup").write_text("")
    (tmp_path / "sobol_00000").write_text("")
    assert rb.is_sobol_run(tmp_path)


@needs_t
def test_form_dict_round_trip():
    form = T.RenderFormValues(paths=("composite", "instance_grains"), slowmo=1, compare=True)
    assert rb.form_from_dict(T, rb.form_to_dict(form)) == form


@needs_t
def test_render_run_dump_route(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0003"
    run.mkdir(parents=True)
    (run / "sobol.setup").write_text("")
    for i in range(3):
        (run / f"sobol_{i:05d}").write_text("")
    calls = []
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: calls.append((spec.name, ctx.n_expected_frames))
                        or T.PathResult(spec.name, True, "done", "ok", n_frames=3))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: calls.append(("cleanup", d)))
    res = rb.render_run(T, run, T.RenderFormValues(paths=("instance_grains",)), base_output_dir=tmp_path / "out")
    assert res.ok and res.stage == "done"
    assert calls == [("instance_grains", 3), ("cleanup", run)]


@needs_t
def test_render_run_keeps_dumps_when_asked(tmp_path, monkeypatch):
    run = tmp_path / "batch" / "run_0001"
    run.mkdir(parents=True)
    (run / "sobol_00000").write_text("")
    monkeypatch.setattr(T, "run_dump_preprocess_stage", lambda cmd, targets, on_line=None: {})
    monkeypatch.setattr(T, "build_dump_preprocess_command", lambda ctx, run_dir, targets: ["pre"])
    monkeypatch.setattr(T, "run_render_path", lambda spec, ctx, **k: T.PathResult(spec.name, True, "done", "ok"))
    monkeypatch.setattr(T, "_cleanup_run_dir", lambda d, p: pytest.fail("dumps deleted"))
    assert rb.render_run(T, run, T.RenderFormValues(), keep_dumps=True, base_output_dir=tmp_path).ok


@needs_t
def test_render_run_without_dumps_fails_cleanly(tmp_path):
    run = tmp_path / "batch" / "run_0001"
    run.mkdir(parents=True)
    res = rb.render_run(T, run, T.RenderFormValues(), base_output_dir=tmp_path)
    assert not res.ok and res.stage == "convert" and "no sobol_" in res.message


@needs_t
def test_render_dumps_argv():
    form = T.RenderFormValues(paths=("instance_grains",), samples=64, encode_video=True, slowmo=4, resume=True)
    argv = rb.render_dumps_argv(Path("/r/fly"), form, name="mia", prefix="fly", dem_model="particle",
                                np_apophis=40000, apophis_only=True, encounter="hyperbola")
    assert argv[:1] == ["/r/fly"]
    for pair in (["--prefix", "fly"], ["--name", "mia"], ["--np-apophis", "40000"], ["--samples", "64"],
                 ["--slowmo", "4"], ["--encounter", "hyperbola"]):
        i = argv.index(pair[0])
        assert argv[i:i + 2] == pair
    assert "--video" in argv and "--resume" in argv and "--apophis-only" in argv


@needs_t
def test_render_dumps_argv_refuses_npz_paths():
    with pytest.raises(ValueError, match="instance_grains / composite"):
        rb.render_dumps_argv(Path("/r"), T.RenderFormValues(paths=("per_sphere",)))
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_render_backend.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'render_backend'`.

- [ ] **Step 3: Implement**

```python
"""Render stages for job.py and tui_app, borrowed from tui_sim_render.py (unchanged).

tui_sim_render sys.exits at import when Code/local_paths.json is missing (OzSTAR, a fresh PC),
so it is imported here on demand and a failure becomes a reason string instead of killing the
caller. render_run() is the post-sim half of tui_sim_render._run_pipeline_stages for a run dir
that already exists (a sweep run); phase 5 moves both into pipeline.py and drops this copy.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import math
import re
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Optional, Union


def load() -> Union[ModuleType, str]:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            t = importlib.import_module("tui_sim_render")
    except SystemExit as exc:
        detail = exc.code if isinstance(exc.code, str) else err.getvalue().strip()
        return f"render unavailable: {detail or 'tui_sim_render exited at import'}"
    except ImportError as exc:
        return f"render unavailable: {exc}"
    try:
        t.check_blender_exe()
    except FileNotFoundError as exc:
        return f"render unavailable: {exc}"
    return t


def is_sobol_run(run_dir: Path) -> bool:
    """A runner run dir the converter can read (it globs sobol_[0-9]*)."""
    run_dir = Path(run_dir)
    return (run_dir / "sobol.setup").is_file() and any(run_dir.glob("sobol_[0-9]*"))


def form_to_dict(form) -> dict:
    d = asdict(form)
    d["paths"] = list(form.paths)
    return d


def form_from_dict(t: ModuleType, d: dict):
    return t.RenderFormValues(**{**d, "paths": tuple(d["paths"])})


def _run_id(run_dir: Path) -> int:
    m = re.fullmatch(r"run_(\d+)", run_dir.name)
    return int(m.group(1)) if m else 1


def render_run(t: ModuleType, run_dir: Path, form, keep_dumps: bool = False,
               base_output_dir: Optional[Path] = None, on_line: Optional[Callable[[str], None]] = None):
    """[convert | dump preprocess] -> each ticked path -> [compare] for one finished sobol run dir."""
    if not form.paths:
        raise ValueError("at least one render path must be selected")
    if form.compare and len(form.paths) < 2:
        raise ValueError("compare needs at least two render paths")
    t.validate_render_form(form)
    base = Path(base_output_dir) if base_output_dir is not None else t.DATA_DIR / "DEMCSVs"
    run_dir = Path(run_dir)
    record = t.RunRecord(run_id=_run_id(run_dir), mass_input_kg=math.nan, run_dir=str(run_dir), status="ok",
                         closest_approach_km=math.nan, closest_approach_au=math.nan, error="")
    # DEMDumpConvert names outputs <run dir's parent>/<run dir> (see tui_sim_render._grains_and_bodies_dirs)
    grains_dir, bodies_dir = t._grains_and_bodies_dirs(run_dir, base, run_dir.parent)
    n_dumps = t.count_matching(run_dir, "sobol_[0-9]*")
    if n_dumps == 0:
        return t.PipelineResult(stage="convert", ok=False, record=record,
                                message=f"no sobol_[0-9]* dumps in {run_dir} (cleaned up? rerun with --no-cleanup)")
    dump_route = not t.needs_npz(form.paths)
    if dump_route:
        n_frames, source = n_dumps, f"read {n_dumps} dump(s) directly (convert skipped)"
    else:
        try:
            t.run_convert_stage(record, base)
        except t.ConvertError as exc:
            return t.PipelineResult(stage="convert", ok=False, message=str(exc), record=record)
        n_npz = t.count_matching(grains_dir, "*.npz")
        if n_npz == 0:
            return t.PipelineResult(stage="convert", ok=False, record=record,
                                    message=f"no grain npz files found in {grains_dir}")
        if not keep_dumps:
            t._cleanup_run_dir(run_dir, "sobol")
        n_frames, source = n_npz, f"converted {n_npz} frame(s)"
    n_expected = n_frames if form.max_frames is None else min(n_frames, form.max_frames)
    ctx = t.PathContext(grains_dir=grains_dir, bodies_dir=bodies_dir, batch_dir=base / run_dir.parent.name,
                        run_name=run_dir.name, render_form=form, n_expected_frames=n_expected)
    specs = [t.RENDER_PATHS[n] for n in t.RENDER_PATH_NAMES if n in form.paths]
    if dump_route:
        targets = {s.name: t._viz_dir(ctx, s) for s in specs}
        ctx.preprocessed = t.run_dump_preprocess_stage(
            t.build_dump_preprocess_command(ctx, run_dir, targets), targets, on_line=on_line)
    results = [t.run_render_path(s, ctx, index=i, total=len(specs), on_line=on_line)
               for i, s in enumerate(specs, start=1)]
    all_ok = all(r.ok for r in results)
    message = source + "; " + "; ".join(t.format_path_result(r) for r in results)
    if dump_route and not keep_dumps:
        # same rule as the old TUI: keep the dumps while any path may need a retry from them
        pre_failed = [r.name for r in results if not r.ok and r.stage == "preprocess"]
        if not any(r.ok for r in results) or pre_failed:
            message += f"; dumps kept in {run_dir}"
        else:
            t._cleanup_run_dir(run_dir, "sobol")
    compare = None
    if form.compare:
        compare, compare_message = t.run_compare_for_paths(ctx, results)
        message += "; " + compare_message
        all_ok = all_ok and compare is not None
    return t.PipelineResult(stage="done" if all_ok else "partial", ok=all_ok, message=message, record=record,
                            paths=results, compare=compare)


_DUMP_ROUTE = ("instance_grains", "composite")


def render_dumps_argv(run_dir: Path, form, *, name: Optional[str] = None, prefix: str = "sobol",
                      dem_model: Optional[str] = None, np_apophis: Optional[int] = None, apophis_only: bool = False,
                      encounter: Optional[str] = None) -> List[str]:
    """render_dumps.py argv for a run dir that is not a sobol run (e.g. someone else's phantomflyby run)."""
    if any(p not in _DUMP_ROUTE for p in form.paths):
        raise ValueError("render_dumps.py renders instance_grains / composite only")
    if form.compare:
        raise ValueError("render_dumps.py has no compare; tick one path or render a sobol run")
    argv = [str(run_dir), "--prefix", prefix]
    if name:
        argv += ["--name", name]
    if dem_model:
        argv += ["--dem-model", dem_model]
    if np_apophis is not None:
        argv += ["--np-apophis", str(np_apophis)]
    if apophis_only:
        argv.append("--apophis-only")
    if encounter:
        argv += ["--encounter", encounter]
    argv += ["--paths", *form.paths, "--samples", str(form.samples), "--fps", str(form.fps),
             "--resolution", form.resolution, "--slowmo", str(form.slowmo)]
    if form.max_frames is not None:
        argv += ["--max-frames", str(form.max_frames)]
    if form.encode_video:
        argv.append("--video")
    if form.resume:
        argv.append("--resume")
    return argv
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_render_backend.py -v`
Expected: all pass on the laptop (on a machine without `local_paths.json` the `needs_t` tests skip).

- [ ] **Step 5: Checkpoint**

```bash
git add render_backend.py tests/test_render_backend.py
git commit -m "render_backend: lazy tui_sim_render loader; render an existing sobol run"   # only if Max authorised
```

---

### Task 6: Job store

**Files:**
- Create: `sobol/jobstore.py`
- Test: `sobol/tests/test_jobstore.py`

**Interfaces:**
- Produces:
  - `SOBOL_DIR: Path`, `JOB_PY: Path`, `JOBS_SUBDIR = "_jobs"`
  - `JobSpec(label, sim_argv=None, resume_batch_dir=None, render=None, run_selection="all", render_dumps_argv=None, keep_dumps=False, omp_num_threads=1, host=<hostname>, created=<iso>)`
  - `Stage(name, state="running", started="", ended="", message="")`
  - `JobStatus(state="running", pid=0, pgid=0, host="", batch_dir=None, stages=[])`
  - `jobs_root(output_root) -> Path`, `new_job_dir(output_root, label, now=None) -> Path`, `write_spec`, `read_spec`, `write_status` (atomic), `read_status -> Optional[JobStatus]` (None if missing or unreadable), `list_jobs(output_root) -> List[Path]` (newest first), `pid_alive(pid) -> bool`, `liveness(status) -> str`, `running_on_host(output_root) -> List[Path]`, `launch(job_dir, spec, script=JOB_PY) -> int`, `cancel(status) -> None`, `tail(path, n) -> List[str]`, `batch_progress(batch_dir) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
"""jobstore: job files, detached launch, liveness, cancel."""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import jobstore as js


def test_new_job_dir_is_unique(tmp_path):
    now = datetime(2026, 10, 13, 12, 0, 0)
    a = js.new_job_dir(tmp_path, "np300 single", now=now)
    b = js.new_job_dir(tmp_path, "np300 single", now=now)
    assert a.name == "20261013_120000_np300_single"
    assert b.name == "20261013_120000_np300_single_2"
    assert a.parent == tmp_path / "_jobs"


def test_spec_and_status_round_trip(tmp_path):
    spec = js.JobSpec(label="x", sim_argv=["--jobs", "2"], run_selection={"ids": [1, 3]}, omp_num_threads=4)
    js.write_spec(tmp_path, spec)
    assert js.read_spec(tmp_path) == spec
    st = js.JobStatus(pid=5, pgid=5, host="h", stages=[js.Stage("sim", "done", "a", "b", "ok")])
    js.write_status(tmp_path, st)
    assert js.read_status(tmp_path) == st
    assert not list(tmp_path.glob("*.tmp"))


def test_read_status_tolerates_corrupt_file(tmp_path):
    assert js.read_status(tmp_path) is None
    (tmp_path / "job_status.json").write_text("{\"state\": \"runn")
    assert js.read_status(tmp_path) is None


def test_list_jobs_newest_first(tmp_path):
    a = js.new_job_dir(tmp_path, "a", now=datetime(2026, 1, 1))
    b = js.new_job_dir(tmp_path, "b", now=datetime(2026, 1, 2))
    assert js.list_jobs(tmp_path) == [b, a]


def test_liveness():
    me = js.JobStatus(pid=os.getpid(), host=js.socket.gethostname())
    assert js.liveness(me) == "running"
    p = subprocess.Popen(["true"])
    p.wait()
    assert js.liveness(js.JobStatus(pid=p.pid, host=me.host)) == "died"
    assert js.liveness(js.JobStatus(pid=1, host="elsewhere")) == "running on elsewhere"
    assert js.liveness(js.JobStatus(state="done", pid=p.pid, host=me.host)) == "done"


def test_launch_detaches_and_cancel_kills_group(tmp_path):
    script = tmp_path / "fake_job.py"
    script.write_text(
        "import os, sys, time, pathlib\n"
        "pathlib.Path(sys.argv[1], 'sid').write_text(str(os.getsid(0)))\n"
        "pathlib.Path(sys.argv[1], 'omp').write_text(os.environ['OMP_NUM_THREADS'])\n"
        "time.sleep(60)\n")
    pid = js.launch(tmp_path, js.JobSpec(label="t", omp_num_threads=3), script=script)
    for _ in range(50):
        if (tmp_path / "omp").exists():
            break
        time.sleep(0.1)
    assert int((tmp_path / "sid").read_text()) == pid != os.getsid(0)  # own session: survives SSH drop
    assert (tmp_path / "omp").read_text() == "3"
    js.cancel(js.JobStatus(pid=pid, pgid=pid))
    for _ in range(50):
        if not js.pid_alive(pid):
            break
        time.sleep(0.1)
    assert not js.pid_alive(pid)


def test_batch_progress(tmp_path):
    (tmp_path / "sobol_mass_samples.csv").write_text("run_id,x\n1,a\n2,b\n3,c\n")
    (tmp_path / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n1,r1,ok\n2,r2,failed\n")
    assert js.batch_progress(tmp_path) == "2/3 runs finished: 1 ok, 1 failed"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_jobstore.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'jobstore'`.

- [ ] **Step 3: Implement**

```python
"""Job files under <output_root>/_jobs/<job_id>/ and detached launch. Shared by tui_app and job.py.

job.json (JobSpec) is written once by the TUI; job_status.json (JobStatus) is rewritten atomically
by job.py; job.log holds everything the job printed. The TUI only reads these, so closing it or
losing SSH never touches a running job.
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SOBOL_DIR = Path(__file__).resolve().parent
JOB_PY = SOBOL_DIR / "job.py"
JOBS_SUBDIR = "_jobs"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class JobSpec:
    label: str
    sim_argv: Optional[List[str]] = None  # runner argv; None = render-only job
    resume_batch_dir: Optional[str] = None  # set: run resume_batch.py on this batch instead of a new one
    render: Optional[Dict[str, object]] = None  # RenderFormValues as a dict (render_backend.form_to_dict)
    run_selection: object = "all"  # "all" | {"first": N} | {"ids": [...]} | {"paths": [...]}
    render_dumps_argv: Optional[List[str]] = None  # non-sobol run dir: render_dumps.py argv
    keep_dumps: bool = False  # after rendering
    omp_num_threads: int = 1
    host: str = field(default_factory=socket.gethostname)
    created: str = field(default_factory=_now_iso)


@dataclass
class Stage:
    name: str
    state: str = "running"  # running | done | failed | skipped
    started: str = ""
    ended: str = ""
    message: str = ""


@dataclass
class JobStatus:
    state: str = "running"  # running | done | partial | failed | cancelled
    pid: int = 0
    pgid: int = 0
    host: str = ""
    batch_dir: Optional[str] = None
    stages: List[Stage] = field(default_factory=list)


def jobs_root(output_root: Path) -> Path:
    return Path(output_root) / JOBS_SUBDIR


def new_job_dir(output_root: Path, label: str, now: Optional[datetime] = None) -> Path:
    stem = (now or datetime.now()).strftime("%Y%m%d_%H%M%S") + "_" + (
        re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "job")
    root = jobs_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        d = root / (stem if n == 1 else f"{stem}_{n}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"no free job dir for {stem} in {root}")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_spec(job_dir: Path, spec: JobSpec) -> None:
    _write_json_atomic(Path(job_dir) / "job.json", asdict(spec))


def read_spec(job_dir: Path) -> JobSpec:
    return JobSpec(**json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8")))


def write_status(job_dir: Path, status: JobStatus) -> None:
    _write_json_atomic(Path(job_dir) / "job_status.json", asdict(status))


def read_status(job_dir: Path) -> Optional[JobStatus]:
    try:
        d = json.loads((Path(job_dir) / "job_status.json").read_text(encoding="utf-8"))
        d["stages"] = [Stage(**s) for s in d.get("stages", [])]
        return JobStatus(**d)
    except (OSError, ValueError, TypeError):
        return None


def list_jobs(output_root: Path) -> List[Path]:
    root = jobs_root(output_root)
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / "job.json").is_file()), key=lambda d: d.name, reverse=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child of this process still answers kill(0); reap it if it is ours
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def liveness(status: JobStatus) -> str:
    if status.state != "running":
        return status.state
    if status.host and status.host != socket.gethostname():
        return f"running on {status.host}"
    return "running" if pid_alive(status.pid) else "died"


def running_on_host(output_root: Path) -> List[Path]:
    out = []
    for d in list_jobs(output_root):
        st = read_status(d)
        if st is not None and liveness(st) == "running":
            out.append(d)
    return out


def launch(job_dir: Path, spec: JobSpec, script: Path = JOB_PY) -> int:
    """Start script on job_dir in its own session (no controlling terminal, so no SIGHUP on SSH drop)."""
    env = dict(os.environ, OMP_NUM_THREADS=str(spec.omp_num_threads))
    with open(Path(job_dir) / "job.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, str(script), str(job_dir)], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc.pid


def cancel(status: JobStatus) -> None:
    os.killpg(status.pgid or status.pid, signal.SIGTERM)


def tail(path: Path, n: int = 20) -> List[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def batch_progress(batch_dir: Path) -> str:
    total = len(_csv_rows(Path(batch_dir) / "sobol_mass_samples.csv"))
    rows = _csv_rows(Path(batch_dir) / "sobol_mass_outputs.csv")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return f"{len(rows)}/{total} runs finished: {ok} ok, {len(rows) - ok} failed"
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_jobstore.py -v`
Expected: all pass.

- [ ] **Step 5: Checkpoint**

```bash
git add jobstore.py tests/test_jobstore.py
git commit -m "jobstore: job files, detached launch, liveness, cancel"   # only if Max authorised
```

---

### Task 7: Job executor

**Files:**
- Create: `sobol/job.py`
- Test: `sobol/tests/test_job.py`

**Interfaces:**
- Consumes: `jobstore.*` (Task 6), `render_backend.load/form_from_dict/render_run/is_sobol_run` (Task 5), runner env `SOBOL_REPORT_BATCH_DIR` (Task 3).
- Produces: `job.main(argv: List[str]) -> int` (argv = `[job_dir]`), `job.select_runs(ok: List[Tuple[int, Path]], selection) -> Tuple[List[Path], List[str]]`, `job.Cancelled`. Env `SOBOL_JOB_RUNNER` overrides the runner script path (test hook only).

- [ ] **Step 1: Write the failing tests**

```python
"""job.py against a stub runner: done / partial / failed / cancelled, run selection, renders."""
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import job
import jobstore as js

STUB = textwrap.dedent("""
    import os, pathlib, sys, time
    batch = pathlib.Path(os.environ["STUB_BATCH"])
    batch.mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["SOBOL_REPORT_BATCH_DIR"]).write_text(str(batch) + "\\n")
    rows = [r.split(":") for r in os.environ.get("STUB_ROWS", "").split()]
    (batch / "sobol_mass_samples.csv").write_text("run_id\\n" + "".join(f"{i}\\n" for i, _ in rows))
    (batch / "sobol_mass_outputs.csv").write_text(
        "run_id,run_dir,status\\n" + "".join(f"{i},{batch}/run_{int(i):04d},{s}\\n" for i, s in rows))
    time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
    sys.exit(int(os.environ.get("STUB_RC", "0")))
""")


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "stub_runner.py"
    script.write_text(STUB)
    monkeypatch.setenv("SOBOL_JOB_RUNNER", str(script))
    monkeypatch.setenv("STUB_BATCH", str(tmp_path / "batch"))
    return tmp_path


def _job(tmp_path, **spec):
    d = js.new_job_dir(tmp_path / "out", "t")
    js.write_spec(d, js.JobSpec(label="t", sim_argv=spec.pop("sim_argv", ["--jobs", "1"]), **spec))
    return d


def _fake_backend(monkeypatch, fail_ids=()):
    rendered = []
    t = SimpleNamespace(RenderFormValues=lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(job.render_backend, "load", lambda: t)
    monkeypatch.setattr(job.render_backend, "form_from_dict", lambda t, d: d)
    monkeypatch.setattr(job.render_backend, "is_sobol_run", lambda p: True)

    def render_run(t, run_dir, form, keep_dumps=False, **k):
        rendered.append(Path(run_dir).name)
        ok = Path(run_dir).name not in fail_ids
        return SimpleNamespace(ok=ok, stage="done" if ok else "partial", message="m",
                               paths=[SimpleNamespace(name="instance_grains", n_frames=3, elapsed_s=12.4)])

    monkeypatch.setattr(job.render_backend, "render_run", render_run)
    return rendered


def test_sim_only_done(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok")
    d = _job(stub)
    assert job.main([str(d)]) == 0
    st = js.read_status(d)
    assert st.state == "done" and st.batch_dir == str(stub / "batch")
    assert [s.name for s in st.stages] == ["sim"]


def test_sim_with_no_ok_rows_fails(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:failed")
    monkeypatch.setenv("STUB_RC", "1")
    d = _job(stub, render={"paths": ["instance_grains"]})
    assert job.main([str(d)]) == 1
    st = js.read_status(d)
    assert st.state == "failed" and [s.name for s in st.stages] == ["sim"]


def test_render_after_selected_runs(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:ok")
    rendered = _fake_backend(monkeypatch)
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"first": 2})
    assert job.main([str(d)]) == 0
    assert rendered == ["run_0001", "run_0002"]
    st = js.read_status(d)
    assert st.state == "done"
    assert "instance_grains 3 frames 12 s" in st.stages[-1].message  # per-path timing shown in the Jobs tab


def test_selection_skips_failed_and_missing_ids(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok 3:failed")
    rendered = _fake_backend(monkeypatch)
    d = _job(stub, render={"paths": ["instance_grains"]}, run_selection={"ids": [2, 3, 99]})
    job.main([str(d)])
    st = js.read_status(d)
    assert rendered == ["run_0002"]
    assert st.state == "partial"
    select = next(s for s in st.stages if s.name == "select")
    assert "3" in select.message and "99" in select.message


def test_render_failure_is_partial(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok 2:ok")
    _fake_backend(monkeypatch, fail_ids={"run_0002"})
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    assert js.read_status(d).state == "partial"


def test_render_unavailable_is_partial(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    monkeypatch.setattr(job.render_backend, "load", lambda: "render unavailable: no Blender")
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    st = js.read_status(d)
    assert st.state == "partial" and "no Blender" in st.stages[-1].message


def test_dry_run_is_done_without_render(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:prepared_only")
    d = _job(stub, sim_argv=["--dry-run"], render={"paths": ["instance_grains"]})
    assert job.main([str(d)]) == 0
    assert js.read_status(d).state == "done"


def test_render_only_paths(tmp_path, monkeypatch):
    rendered = _fake_backend(monkeypatch)
    d = js.new_job_dir(tmp_path, "r")
    js.write_spec(d, js.JobSpec(label="r", render={"paths": ["instance_grains"]},
                                run_selection={"paths": [str(tmp_path / "b" / "run_0004")]}))
    assert job.main([str(d)]) == 0
    assert rendered == ["run_0004"]


def test_cancel_marks_cancelled(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    monkeypatch.setenv("STUB_SLEEP", "60")
    d = _job(stub)
    pid = js.launch(d, js.read_spec(d))
    for _ in range(100):
        st = js.read_status(d)
        if st and st.stages:
            break
        time.sleep(0.1)
    js.cancel(st)
    for _ in range(100):
        if js.read_status(d).state != "running":
            break
        time.sleep(0.1)
    assert js.read_status(d).state == "cancelled"


def test_cancel_during_render_warns_about_blender(stub, monkeypatch):
    monkeypatch.setenv("STUB_ROWS", "1:ok")
    _fake_backend(monkeypatch)

    def cancelled(*_a, **_k):
        raise job.Cancelled()

    monkeypatch.setattr(job.render_backend, "render_run", cancelled)
    d = _job(stub, render={"paths": ["instance_grains"]})
    job.main([str(d)])
    st = js.read_status(d)
    assert st.state == "cancelled" and "blender.exe" in st.stages[-1].message


def test_select_runs():
    ok = [(1, Path("r1")), (2, Path("r2")), (3, Path("r3"))]
    assert job.select_runs(ok, "all") == ([Path("r1"), Path("r2"), Path("r3")], [])
    assert job.select_runs(ok, {"first": 1}) == ([Path("r1")], [])
    assert job.select_runs(ok, {"ids": [3, 7]}) == ([Path("r3")], ["7"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_job.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'job'`.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
"""Run one job headlessly: sweep/single run (runner or resume_batch.py), then renders.

    python3 sobol/job.py <job_dir>     # launched detached by jobstore.launch(); stdout -> job.log

Reads <job_dir>/job.json, writes <job_dir>/job_status.json after every stage change. SIGTERM
(Cancel in the TUI, sent to the whole process group) marks the job cancelled.
"""
from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobstore  # noqa: E402
import render_backend  # noqa: E402

SOBOL_DIR = jobstore.SOBOL_DIR
RESUME = SOBOL_DIR / "resume_batch.py"
RENDER_DUMPS = SOBOL_DIR / "render_dumps.py"
REPORT_ENV = "SOBOL_REPORT_BATCH_DIR"  # = run_mass_sobol_phantom.REPORT_BATCH_DIR_ENV (not imported: heavy)
BLENDER_NOTE = "cancelled; a Windows blender.exe started by this job may still be running (check Task Manager)"


class Cancelled(Exception):
    pass


def _on_sigterm(_signum, _frame):
    raise Cancelled()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _runner_script() -> Path:
    return Path(os.environ.get("SOBOL_JOB_RUNNER", SOBOL_DIR / "run_mass_sobol_phantom.py"))


class _Job:
    def __init__(self, job_dir: Path) -> None:
        self.dir = job_dir
        self.spec = jobstore.read_spec(job_dir)
        self.status = jobstore.JobStatus(pid=os.getpid(), pgid=os.getpgid(0), host=jobstore.socket.gethostname())

    def save(self) -> None:
        try:
            jobstore.write_status(self.dir, self.status)
        except OSError as exc:  # status is a view; never stop the job over it
            print(f"[job] status not written: {exc}", flush=True)

    def begin(self, name: str) -> jobstore.Stage:
        stage = jobstore.Stage(name, started=_now())
        self.status.stages.append(stage)
        self.save()
        print(f"[job] {name}", flush=True)
        return stage

    def end(self, stage: jobstore.Stage, state: str, message: str = "") -> None:
        stage.state, stage.ended, stage.message = state, _now(), message
        self.save()
        print(f"[job] {stage.name}: {state} {message}", flush=True)


def _ok_runs(batch_dir: Path) -> Tuple[List[Tuple[int, Path]], int]:
    """(run_id, run_dir) of ok rows, and the number of rows that were not ok."""
    try:
        with open(batch_dir / "sobol_mass_outputs.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], 0
    ok = [(int(r["run_id"]), Path(r["run_dir"])) for r in rows if r.get("status") == "ok"]
    return sorted(ok), len(rows) - len(ok)


def select_runs(ok: List[Tuple[int, Path]], selection) -> Tuple[List[Path], List[str]]:
    """Run dirs to render, and run ids asked for that are not ok (failed or missing)."""
    if selection == "all":
        return [p for _, p in ok], []
    if isinstance(selection, dict) and "first" in selection:
        return [p for _, p in ok[: int(selection["first"])]], []
    if isinstance(selection, dict) and "ids" in selection:
        by_id = dict(ok)
        return ([by_id[i] for i in selection["ids"] if i in by_id],
                [str(i) for i in selection["ids"] if i not in by_id])
    raise ValueError(f"unknown run selection {selection!r}")


def _sim(job: _Job) -> Tuple[Optional[Path], List[Tuple[int, Path]], bool]:
    """Returns (batch_dir, ok runs, any problem). Raises Cancelled."""
    spec = job.spec
    stage = job.begin("sim")
    report = job.dir / "batch_dir.txt"
    if spec.resume_batch_dir:
        cmd = [sys.executable, str(RESUME), "--batch-dir", spec.resume_batch_dir, *spec.sim_argv]
    else:
        cmd = [sys.executable, str(_runner_script()), *spec.sim_argv]
    rc = subprocess.call(cmd, env=dict(os.environ, **{REPORT_ENV: str(report)}))
    batch = Path(spec.resume_batch_dir) if spec.resume_batch_dir else (
        Path(report.read_text(encoding="utf-8").strip()) if report.is_file() else None)
    job.status.batch_dir = str(batch) if batch else None
    if "--dry-run" in spec.sim_argv:
        job.end(stage, "done" if rc == 0 else "failed", f"dry run, exit {rc}; prepared {batch}")
        return batch, [], rc != 0
    ok, not_ok = _ok_runs(batch) if batch else ([], 0)
    message = f"exit {rc}; {len(ok)} ok, {not_ok} not ok" + (f"; batch {batch}" if batch else "")
    job.end(stage, "done" if ok else "failed", message)
    return batch, ok, not_ok > 0 or rc != 0


def _render(job: _Job, run_dirs: List[Path], skipped: List[str]) -> bool:
    """Render each run dir; True when every one rendered."""
    spec = job.spec
    all_ok = True
    if skipped:
        stage = job.begin("select")
        job.end(stage, "skipped", "not rendered (failed or not in batch): run " + ", ".join(skipped))
        all_ok = False
    backend = render_backend.load()
    if isinstance(backend, str):
        stage = job.begin("render")
        job.end(stage, "failed", backend)
        return False
    form = render_backend.form_from_dict(backend, spec.render)
    for run_dir in run_dirs:
        stage = job.begin(f"render {run_dir.name}")
        try:
            if not render_backend.is_sobol_run(run_dir):
                job.end(stage, "failed", f"{run_dir} is not a sobol run with dumps")
                all_ok = False
                continue
            res = render_backend.render_run(backend, run_dir, form, keep_dumps=spec.keep_dumps)
        except Cancelled:
            job.end(stage, "failed", BLENDER_NOTE)
            raise
        except Exception as exc:  # one bad run must not stop the others
            job.end(stage, "failed", f"{type(exc).__name__}: {exc}")
            all_ok = False
            continue
        timing = "; ".join(f"{p.name} {p.n_frames} frames {p.elapsed_s:.0f} s" for p in res.paths)
        job.end(stage, "done" if res.ok else "failed", res.message + (f" | {timing}" if timing else ""))
        all_ok = all_ok and res.ok
    return all_ok


def _render_dumps(job: _Job) -> bool:
    stage = job.begin("render_dumps")
    try:
        rc = subprocess.call([sys.executable, str(RENDER_DUMPS), *job.spec.render_dumps_argv])
    except Cancelled:
        job.end(stage, "failed", BLENDER_NOTE)
        raise
    job.end(stage, "done" if rc == 0 else "failed", f"exit {rc}")
    return rc == 0


def main(argv: List[str]) -> int:
    job = _Job(Path(argv[0]))
    signal.signal(signal.SIGTERM, _on_sigterm)
    job.save()
    spec = job.spec
    try:
        problem = False
        if spec.render_dumps_argv:
            problem = not _render_dumps(job)
        elif spec.sim_argv is not None:
            batch, ok, problem = _sim(job)
            if not ok and "--dry-run" not in spec.sim_argv:
                job.status.state = "failed"
                job.save()
                return 1
            if spec.render and ok:
                run_dirs, skipped = select_runs(ok, spec.run_selection)
                problem = not _render(job, run_dirs, skipped) or problem
        elif spec.render:
            problem = not _render(job, [Path(p) for p in spec.run_selection["paths"]], [])
        job.status.state = "partial" if problem else "done"
    except Cancelled:
        job.status.state = "cancelled"
    except Exception as exc:
        stage = job.begin("error")
        job.end(stage, "failed", f"{type(exc).__name__}: {exc}")
        job.status.state = "failed"
    job.save()
    return 0 if job.status.state == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Note: in a sim job whose runner reports failed rows but also ok rows, `problem` from `_sim` is true, so the job ends `partial` even if every selected render succeeded. That is intended (spec: "Some runs failed → partial").

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_job.py -v`
Expected: all pass. `test_sim_only_done`: the stub writes 2 ok rows and exits 0 → `done`.

- [ ] **Step 5: Phase check, then checkpoint**

Run: `python3 -m pytest tests -q` (baseline + new tests).

Manual smoke (no PHANTOM time): write a job by hand and launch it detached with a dry run:

```bash
cd /home/mboyle/Honours/sobol-unified-tui && python3 - <<'EOF'
import jobstore as js
from pathlib import Path
root = Path("/home/mboyle/Honours/sobol_mass_runs")
d = js.new_job_dir(root, "smoke_dry")
spec = js.JobSpec(label="smoke_dry", sim_argv=["--base-dir", "/home/mboyle/Honours/sobol", "--phantom-dir",
    "/home/mboyle/Honours/sobol", "--use-dem-fixed", "true", "--np-apophis", "300", "--num-samples", "1",
    "--dry-run", "--output-root", str(root)])
js.write_spec(d, spec); print(d, js.launch(d, spec))
EOF
```

Expected: within seconds `job_status.json` in that dir shows `"state": "done"` and a `sim` stage "dry run, exit 0"; `job.log` holds the runner output.

```bash
git add job.py tests/test_job.py
git commit -m "job.py: headless detached executor (sim, run selection, renders, cancel)"   # only if Max authorised
```

---

### Task 8: Simulate form widget

**Files:**
- Create: `sobol/tui_app/__init__.py` (empty), `sobol/tui_app/simulate.py`
- Test: `sobol/tests/test_tui_app.py`

**Interfaces:**
- Consumes: `params_schema` (Tasks 1–2).
- Produces: `field_id(dest) -> str` (`"f-" + dest with - for _`), `axis_id(stem, part) -> str` (`part` in `sweep|fixed|lo|hi`), `section_id(section) -> str`, `SimulateForm(initial: FormValues, mode: str = "sweep")` with `.mode -> str`, `.omp -> int` (raises `ValueError`), `.read() -> FormValues`, `.write(fv)`, `.sync()`.

- [ ] **Step 1: Write the failing tests**

```python
"""tui_app widgets driven through Textual's pilot."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from textual.app import App
from textual.widgets import Checkbox, Input, RadioButton, Select

import params_schema as ps
from tui_app.simulate import SimulateForm, axis_id, field_id, section_id


class _Host(App):
    def __init__(self, fv, mode="sweep"):
        super().__init__()
        self.fv, self.mode = fv, mode

    def compose(self):
        yield SimulateForm(self.fv, mode=self.mode, id="sim")


def _drive(fv, actions, mode="sweep"):
    async def scenario():
        app = _Host(fv, mode)
        async with app.run_test(size=(160, 60)) as pilot:
            form = app.query_one(SimulateForm)
            out = await actions(app, form, pilot)
            await pilot.pause()
            return out
    return asyncio.run(scenario())


def test_form_reads_back_what_it_was_given():
    fv = ps.from_argv(["--use-dem-fixed", "true", "--np-apophis", "300", "--mu-s-min", "0.2", "--mu-s-max", "0.8",
                       "--spin-torque-align-min", "45", "--spin-torque-align-max", "45"])

    async def act(app, form, pilot):
        return form.read()

    out = _drive(fv, act)
    assert ps.to_argv(out) == ps.to_argv(fv)


def test_single_mode_hides_sampling_and_sweep_toggles():
    async def act(app, form, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        return (app.query_one("#" + section_id("Sampling")).display,
                app.query_one("#row-" + field_id("num_samples")).display,
                app.query_one("#" + axis_id("mu_s", "sweep")).display,
                app.query_one("#omp", Input).value, form.mode)

    assert _drive(ps.from_argv([]), act) == (False, False, False, "4", "single")


def test_boulder_rows_follow_gates():
    async def act(app, form, pilot):
        before = app.query_one("#" + field_id("bond_sigma_pa")).disabled
        app.query_one("#" + field_id("body_source"), Select).value = "settled"
        app.query_one("#" + field_id("boulders"), Checkbox).value = True
        await pilot.pause()
        return before, app.query_one("#" + field_id("bond_sigma_pa")).disabled

    assert _drive(ps.from_argv([]), act) == (True, False)


def test_sweep_toggle_swaps_inputs():
    async def act(app, form, pilot):
        app.query_one("#" + axis_id("tmax_hours", "sweep"), Checkbox).value = True
        await pilot.pause()
        return (app.query_one("#" + axis_id("tmax_hours", "fixed")).display,
                app.query_one("#" + axis_id("tmax_hours", "lo")).display)

    assert _drive(ps.from_argv([]), act) == (False, True)


def test_write_replaces_values():
    target = ps.from_argv(["--body-source", "settled", "--np-apophis", "500", "--tmax-hours-min", "10",
                           "--tmax-hours-max", "20"])

    async def act(app, form, pilot):
        form.write(target)
        await pilot.pause()
        return form.read()

    assert ps.to_argv(_drive(ps.from_argv([]), act)) == ps.to_argv(target)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'tui_app'`.

- [ ] **Step 3: Implement** `sobol/tui_app/__init__.py` as an empty file and `sobol/tui_app/simulate.py`:

```python
"""Simulate tab body: every runner flag as a row, built from params_schema.

Rows: plain flags get one widget; axes get a sweep checkbox, a fixed input and min/max inputs.
Gates (params_schema.enabled) only grey rows out; Validate / Launch ask the runner itself.
"""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Checkbox, Input, RadioButton, RadioSet, Select, Static

import params_schema as ps


def field_id(dest: str) -> str:
    return "f-" + dest.replace("_", "-")


def axis_id(stem: str, part: str) -> str:
    return f"ax-{stem.replace('_', '-')}-{part}"


def section_id(section: str) -> str:
    return "sec-" + "".join(c if c.isalnum() else "-" for c in section.lower())


class Row(Horizontal):
    DEFAULT_CSS = """
    Row { height: auto; }
    Row .rl { width: 30; padding: 1 1 0 0; color: $text-muted; }
    Row Input, Row Select { width: 1fr; }
    Row Checkbox { width: auto; }
    """

    def __init__(self, label: str, *widgets, tip: str = "", **kw) -> None:
        super().__init__(**kw)
        self._label, self._widgets, self._tip = label, widgets, tip

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="rl")
        yield from self._widgets

    def on_mount(self) -> None:
        if self._tip:
            self.tooltip = self._tip


def _plain_widget(f: ps.PlainField, value):
    wid = field_id(f.dest)
    if f.kind == "check":
        return Checkbox(f.dest, value=bool(value), id=wid)
    if f.kind == "select":
        return Select([(c, c) for c in f.choices], value=value if value else Select.BLANK, allow_blank=True, id=wid)
    hint = f"default {f.default}" if f.default not in (None, False) else "blank = template / unset"
    if f.kind == "list":
        hint = "space-separated list"
    return Input(str(value or ""), placeholder=hint, id=wid)


def _axis_row(ax: ps.Axis, av: ps.AxisValue) -> Row:
    fixed_hint = "fixed value" if (ax.fixed or ax.pin) else "(template value; tick sweep to vary)"
    return Row(
        ax.label,
        Checkbox("sweep", value=av.sweep, id=axis_id(ax.stem, "sweep")),
        Input(av.fixed, placeholder=fixed_hint, id=axis_id(ax.stem, "fixed")),
        Input(av.lo, placeholder="min", id=axis_id(ax.stem, "lo")),
        Input(av.hi, placeholder="max", id=axis_id(ax.stem, "hi")),
        id=axis_id(ax.stem, "row"),
    )


class SimulateForm(VerticalScroll):
    DEFAULT_CSS = """
    SimulateForm .sec { color: $accent; text-style: bold; padding: 1 0 0 0; }
    SimulateForm #mode-bar { height: auto; }
    SimulateForm #omp { width: 10; }
    """

    def __init__(self, initial: ps.FormValues, mode: str = "sweep", **kw) -> None:
        super().__init__(**kw)
        self._initial, self._initial_mode = initial, mode

    def compose(self) -> ComposeResult:
        with Horizontal(id="mode-bar"):
            with RadioSet(id="mode"):
                yield RadioButton("Single run", value=self._initial_mode == "single", id="mode-single")
                yield RadioButton("Sweep", value=self._initial_mode != "single", id="mode-sweep")
            yield Static("  OMP_NUM_THREADS ", classes="rl")
            yield Input("4" if self._initial_mode == "single" else "1", id="omp")
        for section in ps.SECTION_ORDER:
            yield Static(section, classes="sec", id=section_id(section))
            for item in ps.items_in(section):
                if isinstance(item, ps.Axis):
                    yield _axis_row(item, self._initial.axes.get(item.stem, ps.AxisValue()))
                else:
                    yield Row(item.dest, _plain_widget(item, self._initial.plain.get(item.dest)), tip=item.brief,
                              id="row-" + field_id(item.dest))

    def on_mount(self) -> None:
        self.sync()

    @property
    def mode(self) -> str:
        return "single" if self.query_one("#mode-single", RadioButton).value else "sweep"

    @property
    def omp(self) -> int:
        text = self.query_one("#omp", Input).value.strip()
        try:
            n = int(text)
        except ValueError:
            raise ValueError(f"OMP_NUM_THREADS must be an integer (got {text!r})") from None
        if n < 1:
            raise ValueError("OMP_NUM_THREADS must be >= 1")
        return n

    def read(self) -> ps.FormValues:
        plain = {}
        for f in ps.PLAIN:
            w = self.query_one("#" + field_id(f.dest))
            if f.kind == "check":
                plain[f.dest] = w.value
            elif f.kind == "select":
                plain[f.dest] = None if w.value is Select.BLANK else str(w.value)
            else:
                plain[f.dest] = w.value.strip()
        axes = {
            ax.stem: ps.AxisValue(
                sweep=self.query_one("#" + axis_id(ax.stem, "sweep"), Checkbox).value,
                fixed=self.query_one("#" + axis_id(ax.stem, "fixed"), Input).value.strip(),
                lo=self.query_one("#" + axis_id(ax.stem, "lo"), Input).value.strip(),
                hi=self.query_one("#" + axis_id(ax.stem, "hi"), Input).value.strip(),
            )
            for ax in ps.AXES
        }
        return ps.FormValues(plain, axes)

    def write(self, fv: ps.FormValues) -> None:
        for f in ps.PLAIN:
            w = self.query_one("#" + field_id(f.dest))
            v = fv.plain.get(f.dest)
            if f.kind == "check":
                w.value = bool(v)
            elif f.kind == "select":
                if v:
                    w.value = v
                else:
                    w.clear()
            else:
                w.value = str(v or "")
        for ax in ps.AXES:
            av = fv.axes.get(ax.stem, ps.AxisValue())
            self.query_one("#" + axis_id(ax.stem, "sweep"), Checkbox).value = av.sweep
            self.query_one("#" + axis_id(ax.stem, "fixed"), Input).value = av.fixed
            self.query_one("#" + axis_id(ax.stem, "lo"), Input).value = av.lo
            self.query_one("#" + axis_id(ax.stem, "hi"), Input).value = av.hi
        self.sync()

    def sync(self) -> None:
        fv = self.read()
        single = self.mode == "single"
        self.query_one("#" + section_id("Sampling")).display = not single
        for f in ps.PLAIN:
            self.query_one("#row-" + field_id(f.dest)).display = not (
                single and (f.section == "Sampling" or f.dest in ps.SINGLE_HIDDEN))
            self.query_one("#" + field_id(f.dest)).disabled = not ps.enabled(f.dest, fv)
        for ax in ps.AXES:
            on_ = ps.enabled(ax.stem, fv)
            sweep = fv.axes[ax.stem].sweep and not single
            toggle = self.query_one("#" + axis_id(ax.stem, "sweep"))
            toggle.display, toggle.disabled = not single, not on_
            fixed = self.query_one("#" + axis_id(ax.stem, "fixed"))
            fixed.display = not sweep
            fixed.disabled = not on_ or not (ax.fixed or ax.pin)
            for part in ("lo", "hi"):
                w = self.query_one("#" + axis_id(ax.stem, part))
                w.display, w.disabled = sweep, not on_

    @on(RadioSet.Changed, "#mode")
    def _mode_changed(self) -> None:
        self.query_one("#omp", Input).value = "4" if self.mode == "single" else "1"
        self.sync()

    @on(Checkbox.Changed)
    def _checkbox_changed(self) -> None:
        self.sync()

    @on(Select.Changed)
    def _select_changed(self) -> None:
        self.sync()
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: 5 passed.

- [ ] **Step 5: Checkpoint**

```bash
git add tui_app/__init__.py tui_app/simulate.py tests/test_tui_app.py
git commit -m "tui_app: schema-built Simulate form (mode, axes, gates)"   # only if Max authorised
```

---

### Task 9: Render tab widgets

**Files:**
- Create: `sobol/tui_app/render.py`
- Test: `sobol/tests/test_tui_app.py` (append)

**Interfaces:**
- Consumes: `render_backend` (Task 5), `jobstore` (Task 6), `tui_sim_render` module object `t` (passed in, never imported here).
- Produces: `RunRow(run_id: int, status: str, run_dir: Path, has_dumps: bool)`, `batches(output_root) -> List[Path]`, `runs_of(batch) -> List[RunRow]`, `parse_selection(kind: str, value: str) -> object`, `RenderSettings(t)` with `.read() -> RenderFormValues` (raises `ValueError`), `RunPicker(output_root: Path)` with `.chosen_runs() -> List[Path]`, `.other_run_argv(form) -> Optional[List[str]]`, `.delete_dumps -> bool`, `.reload(output_root)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import pytest

import render_backend
from tui_app import render as tr_render

T = render_backend.load()
needs_t = pytest.mark.skipif(isinstance(T, str), reason="render stages unavailable here")


def test_parse_selection():
    assert tr_render.parse_selection("all", "") == "all"
    assert tr_render.parse_selection("first", "3") == {"first": 3}
    assert tr_render.parse_selection("ids", "1, 4 7") == {"ids": [1, 4, 7]}
    with pytest.raises(ValueError):
        tr_render.parse_selection("first", "0")
    with pytest.raises(ValueError):
        tr_render.parse_selection("ids", "")


def test_batches_and_runs(tmp_path):
    b = tmp_path / "sobol_20261013_120000_n2"
    (b / "run_0001").mkdir(parents=True)
    (b / "run_0001" / "sobol_00000").write_text("")
    (b / "sobol_mass_outputs.csv").write_text(
        f"run_id,run_dir,status\n1,{b}/run_0001,ok\n2,{b}/run_0002,failed\n")
    (tmp_path / "_jobs").mkdir()
    assert tr_render.batches(tmp_path) == [b]
    rows = tr_render.runs_of(b)
    assert [(r.run_id, r.status, r.has_dumps) for r in rows] == [(1, "ok", True), (2, "failed", False)]


class _RenderHost(App):
    def compose(self):
        yield tr_render.RenderSettings(T, id="settings")


@needs_t
def test_render_settings_defaults_and_errors():
    async def scenario():
        app = _RenderHost()
        async with app.run_test(size=(160, 60)) as pilot:
            s = app.query_one(tr_render.RenderSettings)
            form = s.read()
            app.query_one("#r-fps", Input).value = "x"
            await pilot.pause()
            try:
                s.read()
            except ValueError as exc:
                return form, str(exc)
    form, err = asyncio.run(scenario())
    assert form.paths == ("instance_grains",) and form.fps == 24 and form.samples == T.DEFAULT_SAMPLES
    assert "fps" in err
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'render' from 'tui_app'`.

- [ ] **Step 3: Implement** `sobol/tui_app/render.py`:

```python
"""Render tab body: choose runs (from a batch, or any run dir) and render settings.

`t` is the tui_sim_render module from render_backend.load(); this file never imports it, so the
app still starts where rendering is unavailable (OzSTAR).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Set

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Collapsible, DataTable, Input, Select, Static

import render_backend
from tui_app.simulate import Row


@dataclass
class RunRow:
    run_id: int
    status: str
    run_dir: Path
    has_dumps: bool


def batches(output_root: Path) -> List[Path]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = [d for d in root.iterdir() if d.is_dir() and (d / "sobol_mass_outputs.csv").is_file()]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def runs_of(batch: Path) -> List[RunRow]:
    with open(Path(batch) / "sobol_mass_outputs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        run_dir = Path(r["run_dir"])
        out.append(RunRow(int(r["run_id"]), r.get("status", ""), run_dir, any(run_dir.glob("sobol_[0-9]*"))))
    return sorted(out, key=lambda r: r.run_id)


def parse_selection(kind: str, value: str) -> object:
    if kind == "all":
        return "all"
    if kind == "first":
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"first N needs a whole number (got {value!r})") from None
        if n < 1:
            raise ValueError("first N must be >= 1")
        return {"first": n}
    if kind == "ids":
        try:
            ids = [int(x) for x in value.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"run ids must be whole numbers (got {value!r})") from None
        if not ids:
            raise ValueError("list at least one run id")
        return {"ids": ids}
    raise ValueError(f"unknown selection {kind!r}")


_RES_RE = re.compile(r"^\d+x\d+$")


def _int(name: str, text: str, optional: bool = False) -> Optional[int]:
    text = text.strip()
    if not text and optional:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name} must be a whole number (got {text!r})") from None


class RenderSettings(Vertical):
    DEFAULT_CSS = "RenderSettings { height: auto; }"

    def __init__(self, t: ModuleType, **kw) -> None:
        super().__init__(**kw)
        self.t = t

    def compose(self) -> ComposeResult:
        t = self.t
        yield Row("resolution", Input("1920x1080", id="r-resolution"))
        yield Row("samples", Input(str(t.DEFAULT_SAMPLES), id="r-samples"))
        yield Row("fps", Input("24", id="r-fps"))
        yield Row("max_frames", Input("", placeholder="blank = all", id="r-max-frames"))
        yield Row("camera_mode", Select([("auto", "auto"), ("grain_only", "grain_only")], value="auto",
                                        allow_blank=False, id="r-camera-mode"))
        yield Row("paths", *[Checkbox(n, value=(n == "instance_grains"), id=f"r-path-{n.replace('_', '-')}")
                             for n in t.RENDER_PATH_NAMES])
        yield Row("envelope_method", Select([("hull", "hull"), ("sdf", "sdf")], value="hull", allow_blank=False,
                                            id="r-envelope"), tip="composite only")
        yield Row("placeholder_obj", Input(str(t.DEFAULT_PLACEHOLDER_OBJ), id="r-placeholder"),
                  tip="instance_static only")
        yield Row("n_points", Input("1000000", id="r-n-points"), tip="instance_static only")
        yield Row("options", Checkbox("compare", id="r-compare"), Checkbox("encode_video", id="r-video"),
                  Checkbox("resume", id="r-resume"))
        yield Row("video_fps", Input("", placeholder="blank = fps", id="r-video-fps"))
        yield Row("slowmo", Input("1", id="r-slowmo"), tip="instance paths only; keep video_fps 24")

    def _v(self, wid: str) -> str:
        return self.query_one("#" + wid, Input).value

    def _c(self, wid: str) -> bool:
        return self.query_one("#" + wid, Checkbox).value

    def read(self):
        t = self.t
        resolution = self._v("r-resolution").strip()
        if not _RES_RE.match(resolution):
            raise ValueError(f"resolution must look like 1920x1080 (got {resolution!r})")
        paths = tuple(n for n in t.RENDER_PATH_NAMES if self._c(f"r-path-{n.replace('_', '-')}"))
        if not paths:
            raise ValueError("tick at least one render path")
        form = t.RenderFormValues(
            resolution=resolution,
            samples=_int("samples", self._v("r-samples")),
            fps=_int("fps", self._v("r-fps")),
            max_frames=_int("max_frames", self._v("r-max-frames"), optional=True),
            camera_mode=str(self.query_one("#r-camera-mode", Select).value),
            encode_video=self._c("r-video"),
            video_fps=_int("video_fps", self._v("r-video-fps"), optional=True),
            paths=paths,
            envelope_method=str(self.query_one("#r-envelope", Select).value),
            placeholder_obj=self._v("r-placeholder").strip(),
            n_points=_int("n_points", self._v("r-n-points")),
            compare=self._c("r-compare"),
            slowmo=_int("slowmo", self._v("r-slowmo")),
            resume=self._c("r-resume"),
        )
        if form.compare and len(paths) < 2:
            raise ValueError("compare needs at least two render paths")
        t.validate_render_form(form)
        return form


class RunPicker(Vertical):
    """A batch's runs (Enter toggles a row), or one other run dir rendered via render_dumps.py."""

    DEFAULT_CSS = "RunPicker { height: auto; } RunPicker DataTable { height: 12; }"

    def __init__(self, output_root: Path, **kw) -> None:
        super().__init__(**kw)
        self._root = Path(output_root)
        self._rows: List[RunRow] = []
        self._chosen: Set[int] = set()

    def compose(self) -> ComposeResult:
        yield Row("batch", Select([], allow_blank=True, id="r-batch"))
        yield DataTable(id="r-runs", cursor_type="row")
        yield Static("Enter toggles a run; runs without dumps (cleaned up) cannot be rendered.", classes="rl")
        with Collapsible(title="Other run dir (render_dumps.py: instance_grains / composite)", collapsed=True):
            yield Row("run_dir", Input("", placeholder="blank = use the batch above", id="r-other-dir"))
            yield Row("name", Input("", placeholder="output folder name", id="r-other-name"))
            yield Row("prefix", Input("sobol", id="r-other-prefix"))
            yield Row("dem_model", Select([("particle", "particle"), ("sink", "sink")], allow_blank=True,
                                          id="r-other-dem"))
            yield Row("np_apophis", Input("", id="r-other-np"))
            yield Row("apophis_only", Checkbox("dumps hold no Earth/Moon/Sun sinks", id="r-other-apophis-only"))
            yield Row("encounter", Select([("hyperbola", "hyperbola"), ("ephemeris", "ephemeris")],
                                          allow_blank=True, id="r-other-encounter"))
        yield Row("cleanup", Checkbox("delete dumps after a successful render", value=False, id="r-delete-dumps"))

    def on_mount(self) -> None:
        table = self.query_one("#r-runs", DataTable)
        table.add_columns("✓", "run", "status", "dumps", "run_dir")
        self.reload(self._root)

    def reload(self, output_root: Path) -> None:
        self._root = Path(output_root)
        sel = self.query_one("#r-batch", Select)
        sel.set_options([(b.name, str(b)) for b in batches(self._root)])

    @on(Select.Changed, "#r-batch")
    def _batch_changed(self, event: Select.Changed) -> None:
        self._chosen.clear()
        self._rows = [] if event.value is Select.BLANK else runs_of(Path(str(event.value)))
        table = self.query_one("#r-runs", DataTable)
        table.clear()
        for r in self._rows:
            table.add_row("", str(r.run_id), r.status, "yes" if r.has_dumps else "no", str(r.run_dir),
                          key=str(r.run_id))

    @on(DataTable.RowSelected, "#r-runs")
    def _toggle(self, event: DataTable.RowSelected) -> None:
        run_id = int(event.row_key.value)
        row = next(r for r in self._rows if r.run_id == run_id)
        if row.status != "ok" or not row.has_dumps:
            self.app.bell()
            return
        self._chosen ^= {run_id}
        self.query_one("#r-runs", DataTable).update_cell(event.row_key, self.query_one(
            "#r-runs", DataTable).ordered_columns[0].key, "✓" if run_id in self._chosen else "")

    def chosen_runs(self) -> List[Path]:
        return [r.run_dir for r in self._rows if r.run_id in self._chosen]

    @property
    def delete_dumps(self) -> bool:
        return self.query_one("#r-delete-dumps", Checkbox).value

    def other_run_argv(self, form) -> Optional[List[str]]:
        run_dir = self.query_one("#r-other-dir", Input).value.strip()
        if not run_dir:
            return None
        np_text = self.query_one("#r-other-np", Input).value.strip()
        dem = self.query_one("#r-other-dem", Select).value
        enc = self.query_one("#r-other-encounter", Select).value
        return render_backend.render_dumps_argv(
            Path(run_dir).expanduser().resolve(), form,
            name=self.query_one("#r-other-name", Input).value.strip() or None,
            prefix=self.query_one("#r-other-prefix", Input).value.strip() or "sobol",
            dem_model=None if dem is Select.BLANK else str(dem),
            np_apophis=_int("np_apophis", np_text) if np_text else None,
            apophis_only=self.query_one("#r-other-apophis-only", Checkbox).value,
            encounter=None if enc is Select.BLANK else str(enc),
        )
```

Note on `delete_dumps`: render-only jobs default to keeping dumps (`keep_dumps = not delete_dumps`), because the dumps may be the only copy of someone's run.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: all pass.

- [ ] **Step 5: Checkpoint**

```bash
git add tui_app/render.py tests/test_tui_app.py
git commit -m "tui_app: Render tab widgets (run picker, render settings)"   # only if Max authorised
```

---

### Task 10: App shell, Simulate actions, entry point

**Files:**
- Create: `sobol/tui_app/app.py`, `sobol/tui.py`
- Test: `sobol/tests/test_tui_app.py` (append)

**Interfaces:**
- Consumes: Tasks 2, 6, 8, 9; `JobsPane` from Task 11 (this task creates a placeholder `Static` in the Jobs tab; Task 11 swaps it in).
- Produces: `UnifiedApp(initial: FormValues, backend: Union[ModuleType, str], mode="sweep", cwd: Optional[Path]=None)` with `.build_sim_job() -> Tuple[JobSpec, Path]` (raises `ValueError`), `.build_render_job() -> Tuple[JobSpec, Path]` (raises `ValueError`), `.launch_job(spec, output_root) -> Path`, `.set_status(msg, error=False)`; `tui.initial_form(argv) -> FormValues`, `tui.DEFAULT_OVERRIDES`, `tui.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import jobstore as js
import tui
from textual.widgets import Static  # noqa: F401  (used by later tests in this file)
from tui_app.app import UnifiedApp


def _app_run(app, actions):
    async def scenario():
        async with app.run_test(size=(180, 60)) as pilot:
            await pilot.pause()
            return await actions(app, pilot)
    return asyncio.run(scenario())


def test_initial_form_overrides_only_unset_paths():
    fv = tui.initial_form(["--output-root", "/x/runs"])
    assert fv.plain["output_root"] == "/x/runs"
    assert fv.plain["base_dir"] == tui.DEFAULT_OVERRIDES["base_dir"]


def test_launch_sim_writes_job_and_launches(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append((d, spec)) or 4242)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis", "300",
                           "--spin-period-fixed", "30"])

    async def act(app, pilot):
        app.query_one("#mode-single", RadioButton).value = True
        await pilot.pause()
        app.query_one("#launch").press()
        await pilot.pause()
        return launched

    out = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    (job_dir, spec), = out
    assert job_dir.parent == tmp_path / "_jobs"
    assert js.read_spec(job_dir) == spec
    assert spec.omp_num_threads == 4 and spec.render is None
    assert spec.sim_argv[spec.sim_argv.index("--num-samples") + 1] == "1"


def test_invalid_form_creates_no_job(tmp_path, monkeypatch):
    monkeypatch.setattr(js, "launch", lambda *a, **k: pytest.fail("launched"))
    fv = tui.initial_form(["--output-root", str(tmp_path), "--mass-min-kg", "1e10"])

    async def act(app, pilot):
        app.query_one("#launch").press()
        await pilot.pause()
        return app.last_status

    assert "mass bounds" in _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert not (tmp_path / "_jobs").exists()


def test_relative_paths_become_absolute_in_job(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", "runs", "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        app.query_one("#launch").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test", cwd=tmp_path), act)
    argv = launched[0].sim_argv
    assert argv[argv.index("--output-root") + 1] == str(tmp_path / "runs")
    assert (tmp_path / "runs" / "_jobs").is_dir()


def test_render_controls_disabled_without_backend():
    async def act(app, pilot):
        return (len(app.query("#render-unavailable")), len(app.query("#render-settings")),
                app.query_one("#render-after", Checkbox).disabled)

    assert _app_run(UnifiedApp(ps.from_argv([]), "render unavailable: no Blender"), act) == (1, 0, True)


def test_clone_fills_form_from_past_job(tmp_path):
    d = js.new_job_dir(tmp_path, "old")
    js.write_spec(d, js.JobSpec(label="old", sim_argv=["--np-apophis", "777", "--tmax-hours-min", "10",
                                                       "--tmax-hours-max", "20"]))
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app._refresh_clone_list()
        app.query_one("#clone-from", Select).value = str(d)
        app.query_one("#clone").press()
        await pilot.pause()
        return app.query_one(SimulateForm).read()

    out = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert out.plain["np_apophis"] == "777" and out.axes["tmax_hours"].sweep


@needs_t
def test_render_after_forces_no_cleanup_but_keeps_choice(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    fv = tui.initial_form(["--output-root", str(tmp_path), "--use-dem-fixed", "true", "--np-apophis-list", "300"])

    async def act(app, pilot):
        app.query_one("#render-after", Checkbox).value = True
        await pilot.pause()
        app.query_one("#launch").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, T), act)
    spec = launched[0]
    assert "--no-cleanup" in spec.sim_argv and spec.keep_dumps is False
    assert spec.render["paths"] == ["instance_grains"] and spec.run_selection == "all"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: new tests FAIL with `ModuleNotFoundError: No module named 'tui'`.

- [ ] **Step 3: Implement** `sobol/tui_app/app.py`:

```python
"""UnifiedApp: Simulate / Render / Jobs tabs. Builds job.json files and launches job.py detached."""
from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Union

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (Button, Checkbox, Collapsible, Footer, Header, Input, Select, Static, TabbedContent,
                             TabPane)

import jobstore as js
import params_schema as ps
import render_backend
from tui_app.render import RenderSettings, RunPicker, parse_selection
from tui_app.simulate import Row, SimulateForm


class UnifiedApp(App[None]):
    TITLE = "PHANTOM · sweep / single run / render"
    BINDINGS = [Binding("ctrl+l", "launch", "Launch"), Binding("ctrl+q", "quit", "Quit")]
    DEFAULT_CSS = """
    #status { height: auto; min-height: 3; padding: 0 2; background: $surface-darken-1; }
    .err { color: $error; }
    .bar { height: auto; }
    Button { margin: 0 1; }
    """

    def __init__(self, initial: ps.FormValues, backend: Union[ModuleType, str], mode: str = "sweep",
                 cwd: Optional[Path] = None) -> None:
        super().__init__()
        self._initial, self._mode = initial, mode
        self.backend = backend
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.last_status = ""

    @property
    def render_ok(self) -> bool:
        return not isinstance(self.backend, str)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-simulate", id="tabs"):
            with TabPane("Simulate", id="tab-simulate"):
                with Vertical():
                    yield SimulateForm(self._initial, mode=self._mode, id="sim")
                    with Collapsible(title="Render after the sim (settings on the Render tab)", collapsed=True):
                        yield Row("render after", Checkbox("render the selected runs when the sim finishes",
                                                           value=self._mode == "single" and self.render_ok,
                                                           disabled=not self.render_ok, id="render-after"))
                        yield Row("runs", Select([("all ok runs", "all"), ("first N", "first"), ("run ids", "ids")],
                                                 value="all", allow_blank=False, id="sel-kind"),
                                  Input("", placeholder="N, or ids like 1 4 7", id="sel-value"))
                    with Horizontal(classes="bar"):
                        yield Button("Validate", id="validate")
                        yield Button("Show command", id="show-command")
                        yield Select([], allow_blank=True, prompt="clone a past job", id="clone-from")
                        yield Button("Clone", id="clone")
                        yield Button("Launch", variant="primary", id="launch")
            with TabPane("Render", id="tab-render"):
                if self.render_ok:
                    with Vertical():
                        yield RunPicker(self._output_root_from(self._initial), id="picker")
                        yield RenderSettings(self.backend, id="render-settings")
                        yield Button("Render", variant="primary", id="render")
                else:
                    yield Static(Text(self.backend), id="render-unavailable")
            with TabPane("Jobs", id="tab-jobs"):
                yield Static("Jobs list arrives in Task 11", id="jobs-placeholder")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_clone_list()

    # ── helpers ──────────────────────────────────────────────────────────────

    def set_status(self, msg: str, error: bool = False) -> None:
        self.last_status = msg  # tests read this instead of the widget's rendering
        status = self.query_one("#status", Static)
        status.set_class(error, "err")
        status.update(Text(msg))

    def _output_root_from(self, fv: ps.FormValues) -> Path:
        text = str(fv.plain.get("output_root") or "sobol_mass_runs")
        return Path(ps.absolutise(ps.FormValues({"output_root": text}, {}), self.cwd).plain["output_root"])

    def _sim_values(self) -> Tuple[ps.FormValues, str]:
        form = self.query_one(SimulateForm)
        return ps.absolutise(ps.apply_mode(form.read(), form.mode), self.cwd), form.mode

    def _refresh_clone_list(self) -> None:
        root = self._output_root_from(self.query_one(SimulateForm).read())
        options = []
        for d in js.list_jobs(root):
            try:
                if js.read_spec(d).sim_argv is not None:
                    options.append((d.name, str(d)))
            except (OSError, ValueError, TypeError):
                continue
        self.query_one("#clone-from", Select).set_options(options)

    # ── job building ─────────────────────────────────────────────────────────

    def build_sim_job(self) -> Tuple[js.JobSpec, Path]:
        fv, mode = self._sim_values()
        keep_dumps = bool(fv.plain.get("no_cleanup"))
        render = selection = None
        if self.query_one("#render-after", Checkbox).value:
            if not self.render_ok:
                raise ValueError(self.backend)
            render = render_backend.form_to_dict(self.query_one(RenderSettings).read())
            selection = parse_selection(str(self.query_one("#sel-kind", Select).value),
                                        self.query_one("#sel-value", Input).value)
            fv.plain["no_cleanup"] = True  # renders need the dumps; keep_dumps decides after the render
        argv = ps.to_argv(fv)
        result = ps.check(argv)
        if result.error:
            raise ValueError(result.error)
        label = str(fv.plain.get("batch_label") or ("single" if mode == "single" else "sweep"))
        spec = js.JobSpec(label=label, sim_argv=argv, render=render, run_selection=selection or "all",
                          keep_dumps=keep_dumps, omp_num_threads=self.query_one(SimulateForm).omp)
        return spec, self._output_root_from(fv)

    def build_render_job(self) -> Tuple[js.JobSpec, Path]:
        form = self.query_one(RenderSettings).read()
        picker = self.query_one(RunPicker)
        root = self._output_root_from(self.query_one(SimulateForm).read())
        other = picker.other_run_argv(form)
        if other is not None:
            return js.JobSpec(label="render_dumps", render_dumps_argv=other, omp_num_threads=1), root
        runs = picker.chosen_runs()
        if not runs:
            raise ValueError("pick a batch and press Enter on the runs to render, or fill in Other run dir")
        return js.JobSpec(label="render", render=render_backend.form_to_dict(form),
                          run_selection={"paths": [str(p) for p in runs]}, keep_dumps=not picker.delete_dumps,
                          omp_num_threads=1), root

    def launch_job(self, spec: js.JobSpec, output_root: Path) -> Path:
        busy = js.running_on_host(output_root)
        job_dir = js.new_job_dir(output_root, spec.label)
        js.write_spec(job_dir, spec)
        pid = js.launch(job_dir, spec)
        warn = (f"  WARNING: {len(busy)} other job(s) running on this host (laptop throttles above 2)"
                if busy else "")
        self.set_status(f"launched {job_dir.name} (pid {pid}); log: {job_dir / 'job.log'}{warn}")
        self._refresh_clone_list()
        return job_dir

    # ── buttons ──────────────────────────────────────────────────────────────

    @on(Button.Pressed, "#validate")
    def _validate(self) -> None:
        fv, _ = self._sim_values()
        result = ps.check(ps.to_argv(fv))
        if result.error:
            self.set_status(result.error, error=True)
        else:
            self.set_status("OK" + (f"\n{result.warnings}" if result.warnings else ""))

    @on(Button.Pressed, "#show-command")
    def _show_command(self) -> None:
        try:
            spec, _ = self.build_sim_job()
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.set_status(ps.show_command(spec.sim_argv, spec.omp_num_threads))

    @on(Button.Pressed, "#clone")
    def _clone(self) -> None:
        value = self.query_one("#clone-from", Select).value
        if value is Select.BLANK:
            self.set_status("choose a past job to clone", error=True)
            return
        spec = js.read_spec(Path(str(value)))
        self.query_one(SimulateForm).write(ps.from_argv(spec.sim_argv))
        self.set_status(f"form filled from {Path(str(value)).name}")

    @on(Button.Pressed, "#launch")
    def action_launch(self) -> None:
        try:
            spec, root = self.build_sim_job()
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.launch_job(spec, root)

    @on(Button.Pressed, "#render")
    def _render(self) -> None:
        try:
            spec, root = self.build_render_job()
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        self.launch_job(spec, root)
```

And `sobol/tui.py`:

```python
#!/usr/bin/env python3
"""Unified TUI: sweeps, single runs and renders as detached jobs (new; tui_run.py and
tui_sim_render.py still work and are unchanged).

    python3 sobol/tui.py [runner flags to prefill the Simulate form]
    textual serve "python3 sobol/tui.py"   # same app in a browser (needs textual-dev)

Jobs live in <output_root>/_jobs/<job_id>/ and keep running if the TUI or SSH closes.
Doc: sobol/docs/tui.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

SOBOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOBOL_DIR))

try:
    import textual  # noqa: F401
except ImportError:
    sys.exit("textual is not installed: pip install --user --break-system-packages -r sobol/requirements.txt")

import params_schema as ps  # noqa: E402
import render_backend  # noqa: E402
from tui_app.app import UnifiedApp  # noqa: E402

# Launch-location-independent defaults (as tui_sim_render.py): templates and cached ephemeris in sobol/,
# batches in Honours/sobol_mass_runs. Only applied when the flag is not on the command line.
DEFAULT_OVERRIDES = {
    "base_dir": str(SOBOL_DIR),
    "output_root": str(SOBOL_DIR.parent / "sobol_mass_runs"),
    "ephemeris_cache_dir": str(SOBOL_DIR),
}


def initial_form(argv: List[str]) -> ps.FormValues:
    fv = ps.from_argv(argv)
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    for dest, value in DEFAULT_OVERRIDES.items():
        if ps.flag_of(dest) not in given:
            fv.plain[dest] = value
    return fv


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        fv = initial_form(argv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    UnifiedApp(fv, render_backend.load()).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: all pass.

- [ ] **Step 5: Checkpoint**

```bash
git add tui_app/app.py tui.py tests/test_tui_app.py
git commit -m "tui.py: Simulate and Render tabs launch detached jobs"   # only if Max authorised
```

---

### Task 11: Jobs tab

**Files:**
- Create: `sobol/tui_app/jobs_pane.py`
- Modify: `sobol/tui_app/app.py` (replace the `jobs-placeholder` Static with `JobsPane`; switch to the Jobs tab after a launch)
- Test: `sobol/tests/test_tui_app.py` (append)

**Interfaces:**
- Consumes: `jobstore` (Task 6), `UnifiedApp.launch_job`, `UnifiedApp._output_root_from`, `SimulateForm.read` (Task 10).
- Produces: `JobsPane()` with `.refresh_jobs()`, `.current -> Optional[Path]`; `describe(job_dir) -> str`; `elapsed(spec, status) -> str`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from tui_app import jobs_pane


def _fake_job(root, label, state, pid=0, stages=()):
    d = js.new_job_dir(root, label)
    js.write_spec(d, js.JobSpec(label=label, sim_argv=["--jobs", "1"], created="2026-10-13T12:00:00"))
    js.write_status(d, js.JobStatus(state=state, pid=pid, host=js.socket.gethostname(), stages=list(stages)))
    return d


def test_describe_lists_stages_and_progress(tmp_path):
    batch = tmp_path / "b"
    batch.mkdir()
    (batch / "sobol_mass_samples.csv").write_text("run_id\n1\n2\n")
    (batch / "sobol_mass_outputs.csv").write_text("run_id,run_dir,status\n1,r,ok\n")
    d = _fake_job(tmp_path, "a", "running", stages=[js.Stage("sim", "running", "2026-10-13T12:00:00")])
    st = js.read_status(d)
    st.batch_dir = str(batch)
    js.write_status(d, st)
    (d / "job.log").write_text("first\nlast line\n")
    text = jobs_pane.describe(d)
    assert "sim" in text and "1/2 runs finished" in text and "last line" in text


def test_jobs_tab_lists_jobs_including_corrupt_status(tmp_path):
    good = _fake_job(tmp_path, "good", "done")
    bad = _fake_job(tmp_path, "bad", "done")
    (bad / "job_status.json").write_text("{not json")
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        app.query_one("#tabs", TabbedContent).active = "tab-jobs"
        pane = app.query_one(jobs_pane.JobsPane)
        pane.refresh_jobs()
        await pilot.pause()
        table = app.query_one("#jobs", DataTable)
        return {table.get_row(k)[0]: table.get_row(k)[1] for k in table.rows}

    rows = _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert rows == {good.name: "done", bad.name: "unknown"}


def test_resume_launches_resume_job(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(js, "launch", lambda d, spec, script=js.JOB_PY: launched.append(spec) or 1)
    d = _fake_job(tmp_path, "sweep", "partial")
    st = js.read_status(d)
    st.batch_dir = str(tmp_path / "batch")
    js.write_status(d, st)
    fv = tui.initial_form(["--output-root", str(tmp_path)])

    async def act(app, pilot):
        pane = app.query_one(jobs_pane.JobsPane)
        pane.current = d
        app.query_one("#job-resume").press()
        await pilot.pause()

    _app_run(UnifiedApp(fv, "render unavailable: test"), act)
    assert launched[0].resume_batch_dir == str(tmp_path / "batch")
    assert launched[0].sim_argv == ["--jobs", "1"]
```

Add `from textual.widgets import DataTable, Static, TabbedContent` to the test imports.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: FAIL, `ImportError: cannot import name 'jobs_pane'`.

- [ ] **Step 3: Implement** `sobol/tui_app/jobs_pane.py`:

```python
"""Jobs tab: every job under <output_root>/_jobs, its stages, progress and log; cancel and resume."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Log, Static

import jobstore as js


def elapsed(spec: js.JobSpec, status: Optional[js.JobStatus]) -> str:
    try:
        start = datetime.fromisoformat(spec.created)
    except ValueError:
        return "?"
    end = datetime.now()
    if status is not None and status.state != "running" and status.stages and status.stages[-1].ended:
        end = datetime.fromisoformat(status.stages[-1].ended)
    secs = int((end - start).total_seconds())
    return f"{secs // 3600}h{secs % 3600 // 60:02d}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"


def describe(job_dir: Path) -> str:
    status = js.read_status(job_dir)
    lines = [str(job_dir)]
    if status is None:
        lines.append("status: unknown (job_status.json missing or unreadable)")
    else:
        lines.append(f"state: {js.liveness(status)}   pid {status.pid} on {status.host}")
        for s in status.stages:
            lines.append(f"  {s.name:<18} {s.state:<8} {s.message}")
        if status.batch_dir:
            lines.append(f"batch: {status.batch_dir}")
            lines.append("  " + js.batch_progress(Path(status.batch_dir)))
    last = js.tail(job_dir / "job.log", 1)
    if last:
        lines.append(f"log: {last[0]}")
    return "\n".join(lines)


class JobsPane(Vertical):
    DEFAULT_CSS = "JobsPane DataTable { height: 12; } JobsPane Log { height: 1fr; } JobsPane #job-detail { height: auto; }"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.current: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="jobs", cursor_type="row")
        yield Static("", id="job-detail")
        with Horizontal(classes="bar"):
            yield Button("Cancel", variant="error", id="job-cancel")
            yield Button("Resume", id="job-resume")
            yield Button("Show log", id="job-log")
        yield Log(id="job-log-view", max_lines=400)

    def on_mount(self) -> None:
        self.query_one("#jobs", DataTable).add_columns("job", "state", "elapsed", "label")
        self.refresh_jobs()
        self.set_interval(2, self.refresh_jobs)

    def _root(self) -> Path:
        return self.app._output_root_from(self.app.query_one("#sim").read())

    def refresh_jobs(self) -> None:
        table = self.query_one("#jobs", DataTable)
        table.clear()
        for d in js.list_jobs(self._root()):
            try:
                spec = js.read_spec(d)
            except (OSError, ValueError, TypeError):
                continue
            status = js.read_status(d)
            state = js.liveness(status) if status is not None else "unknown"
            table.add_row(d.name, state, elapsed(spec, status), spec.label, key=str(d))
        if self.current is not None:
            self.query_one("#job-detail", Static).update(Text(describe(self.current)))

    @on(DataTable.RowHighlighted, "#jobs")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.current = Path(event.row_key.value)
            self.query_one("#job-detail", Static).update(Text(describe(self.current)))

    @on(Button.Pressed, "#job-cancel")
    def _cancel(self) -> None:
        status = js.read_status(self.current) if self.current else None
        if status is None or js.liveness(status) != "running":
            self.app.set_status("only a job running on this host can be cancelled", error=True)
            return
        js.cancel(status)
        self.app.set_status(f"cancel sent to {self.current.name}")

    @on(Button.Pressed, "#job-resume")
    def _resume(self) -> None:
        if self.current is None:
            self.app.set_status("select a job first", error=True)
            return
        spec, status = js.read_spec(self.current), js.read_status(self.current)
        if spec.sim_argv is None or status is None or not status.batch_dir:
            self.app.set_status("only a sim job with a batch dir can be resumed", error=True)
            return
        if js.liveness(status) == "running":
            self.app.set_status("job is still running; cancel it first", error=True)
            return
        resumed = js.JobSpec(label=spec.label + "_resume", sim_argv=spec.sim_argv, resume_batch_dir=status.batch_dir,
                             render=spec.render, run_selection=spec.run_selection, keep_dumps=spec.keep_dumps,
                             omp_num_threads=spec.omp_num_threads)
        self.app.launch_job(resumed, self._root())

    @on(Button.Pressed, "#job-log")
    def _log(self) -> None:
        if self.current is None:
            return
        view = self.query_one("#job-log-view", Log)
        view.clear()
        view.write_lines(js.tail(self.current / "job.log", 400))
```

In `tui_app/app.py`:
- add `from tui_app.jobs_pane import JobsPane`;
- replace `yield Static("Jobs list arrives in Task 11", id="jobs-placeholder")` with `yield JobsPane(id="jobs-pane")`;
- at the end of `launch_job`, before `return job_dir`, add:

```python
        self.query_one("#tabs", TabbedContent).active = "tab-jobs"
        self.query_one(JobsPane).current = job_dir
        self.query_one(JobsPane).refresh_jobs()
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_tui_app.py -v`
Expected: all pass.

- [ ] **Step 5: Phase check, then checkpoint**

Run: `python3 -m pytest tests -q` (baseline + all new tests).

```bash
git add tui_app/jobs_pane.py tui_app/app.py tests/test_tui_app.py
git commit -m "tui_app: Jobs tab (list, detail, cancel, resume, log)"   # only if Max authorised
```

---

### Task 12: Docs

**Files:**
- Create: `sobol/docs/tui.md`
- Modify: `sobol/CLAUDE.md` (one entry under "Sim + render pipeline TUI (summary)")

- [ ] **Step 1: Write `sobol/docs/tui.md`** with these sections, in this order, using exact names from the code above:
  1. **What it is:** `sobol/tui.py`, Simulate / Render / Jobs; new, alongside `tui_run.py` / `tui_sim_render.py` (unchanged until retirement).
  2. **Launch:** `python3 sobol/tui.py [runner flags]`; browser: `textual serve "python3 sobol/tui.py"` (needs `textual-dev`), and through VS Code port forwarding from the desktop. Defaults `DEFAULT_OVERRIDES` (base_dir and ephemeris cache = `sobol/`, output_root = `Honours/sobol_mass_runs`).
  3. **Simulate:** Single run vs Sweep (what each sets: `--num-samples 1`, `--jobs 1`, OMP 4 / OMP 1); axis rows (sweep / fixed; pinned axes write min = max); gates; Validate (runner's own `validate_args`), Show command, Clone, Render after (forces `--no-cleanup`; `no_cleanup` box decides whether dumps go after the render).
  4. **Render:** batch picker (Enter toggles; runs without dumps are refused), Other run dir (render_dumps.py, instance_grains / composite only), settings, "delete dumps after a successful render" (off by default). Unavailable reason on OzSTAR.
  5. **Jobs:** per-run render timing is in each `render run_NNNN` stage (frames and seconds per path); `<output_root>/_jobs/<job_id>/` (`job.json`, `job_status.json`, `job.log`, `batch_dir.txt`); states `running|done|partial|failed|cancelled`, plus `died`, `running on <host>`, `unknown`; Cancel (process group; a Windows `blender.exe` may survive), Resume (`resume_batch.py` on the batch), Show log. Detached: survives closing the TUI and SSH drops.
  6. **Runner additions:** `--num-samples 1` with no swept axis is allowed; `SOBOL_REPORT_BATCH_DIR`; settled-body cache lock (`<cache-root>/<key>.lock`; no-flock fallback warns).
  7. **Extending:** a new runner flag fails `tests/test_params_schema.py::test_every_runner_dest_is_classified` until it is an axis in `AXES`, a plain field (needs a section: wizard `DEST_TO_SECTION` or `_EXTRA_SECTION`), or `HIDDEN`; gates live in `params_schema.GATES`.
  8. **Known limits:** no mid-render progress bar (stages only); render-only jobs run one run at a time; OzSTAR = Simulate + Jobs only.

- [ ] **Step 2: Add to `sobol/CLAUDE.md`**, after the "Sim + render pipeline TUI (summary)" paragraph:

```markdown
**Unified TUI (new, Oct 2026; old TUIs unchanged until retirement):** `python3 sobol/tui.py`: Simulate (single run or sweep, form built from `params_schema.py`, validated by the runner's own `validate_args`) / Render (sweep runs or any run dir) / Jobs. Every launch is a detached job (`job.py`, `<output_root>/_jobs/<id>/`, survives SSH drops; Cancel, Resume). New runner flags must be classified in `params_schema.py` (test fails otherwise). Detail: `docs/tui.md`. Spec/plan: `docs/superpowers/{specs,plans}/2026-10-13-unified-tui*`.
```

- [ ] **Step 3: Checkpoint**

```bash
git add docs/tui.md CLAUDE.md
git commit -m "docs: unified TUI"   # only if Max authorised
```

---

### Task 13: Manual verification (Max present)

**Files:** none. Record results in the session; fix failures in the task that owns the code, rerun its tests, then repeat the check.

The worktree has no PHANTOM binaries or cached ephemeris (untracked). Launch with the main checkout's:

```bash
cd /home/mboyle/Honours && python3 sobol-unified-tui/tui.py --phantom-dir /home/mboyle/Honours/sobol \
  --ephemeris-cache-dir /home/mboyle/Honours/sobol --base-dir /home/mboyle/Honours/sobol
```

- [ ] **Check 1: single run + render parity.** Simulate → Single run; `use_dem_fixed true`, `np_apophis 300`, `body_source settled`, `spin_period` fixed 30, `tmax` 12, `dtmax` 1; Render tab: `instance_grains` only; Render after ticked; Launch. Expected: Jobs tab shows `sim` then `render run_0001`, state `done`. Run the same inputs through `python3 sobol/tui_sim_render.py` (main checkout). Compare: the `sobol_mass_outputs.csv` row metrics (`closest_approach_km`, spin columns) match, and both `run_0001_render_instance_grains/` folders hold the same number of `frame_*.png`.
- [ ] **Check 2: sweep dry run.** Sweep mode, `mu_s` sweep 0.2–0.8, `num_samples 4`, `dry_run` ticked; Show command prints a runnable command; Launch. Expected: job `done` with "dry run, exit 0", batch dir holds `run_0001..0004` with patched `.setup`.
- [ ] **Check 3: detach.** Launch a short real sweep (`num_samples 2`, `tmax 2`), close the TUI (ctrl+q), wait a minute, reopen: the job is still listed `running` then `done`. Then disconnect Remote-SSH mid-run and reconnect: same.
- [ ] **Check 4: cancel and resume.** Launch a 4-run sweep, Cancel after run 1 finishes: state `cancelled`, no `phantom` processes left (`pgrep -f phantom`). Resume: a `_resume` job finishes the missing runs in the same batch.
- [ ] **Check 5: browser.** Ask Max before installing: `pip install --user --break-system-packages textual-dev`. Then `textual serve "python3 /home/mboyle/Honours/sobol-unified-tui/tui.py"`, forward the port in VS Code, open it on the desktop browser, launch a dry run.
- [ ] **Check 6: OzSTAR** (read-only rule: ask Max before installing or copying anything to the cluster). In a terminal on the login node: the app starts, the Render tab shows the "render unavailable" reason, Simulate → Validate works.
- [ ] **Check 7: old tools untouched.** `git -C /home/mboyle/Honours/sobol-unified-tui diff feature/portable-viz --stat -- tui_run.py tui_sim_render.py render_dumps.py interactive_run_mass_sobol.py` prints nothing; `python3 -m pytest tests -q` matches the Task 0 baseline plus the new tests.

When all seven pass, tell Max and ask whether to (a) merge `feature/unified-tui`, (b) start the phase-5 retirement plan. Do neither without his answer.
