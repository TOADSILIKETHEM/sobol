"""One description of the sweep runner's flags for front-ends (tui_app).

Section and help text come from the -i wizard's DEST_TO_SECTION / INTERACTIVE_BRIEF
(imported, not moved, while the old TUIs still exist). Validity is decided only by the
runner's own parse_args + validate_args (see check()); gates here grey widgets out, nothing more.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import shlex
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


# Values filled into blank fields when a mode is picked. The runner itself keeps refusing to guess
# (no silent default); here every fill is visible in the form and reported on the status line.
SETTLED_DEFAULTS = {"use_dem_fixed": "true", "np_apophis": "2000"}
HYPERBOLA_DEFAULTS = {"flyby_rp_km": "38000", "flyby_vinf_kms": "5.9", "dtmax_hours": "0.5"}  # real 2029: rp geocentric
BOULDER_PLAIN_DEFAULTS = {"bond_sigma_pa": "1e5"}  # the breakup campaign's value; no sigma trial yet
BOULDER_AXIS_DEFAULTS = {"f_boulder": "0.3"}
TMAX_MARGIN = 1.1  # auto tmax = 2 x time to pericentre x this, rounded up to a whole hour


@dataclass
class Filled:
    fv: FormValues
    notes: List[str]  # one "field value" per fill, for the status line
    auto_tmax: str  # tmax the last fill wrote ("" if none); pass back so it keeps following rp / v_inf


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _axis_range(av: AxisValue) -> Optional[Tuple[float, float]]:
    try:
        if av.sweep:
            return float(av.lo), float(av.hi)
        return float(av.fixed), float(av.fixed)
    except ValueError:
        return None


def _auto_tmax(fv: FormValues) -> str:
    """2 x the longest time to pericentre (+ margin), as the runner's geometry check computes it; "" if unknown."""
    import math

    import encounter
    rp, vinf = _axis_range(fv.axes["flyby_rp_km"]), _axis_range(fv.axes["flyby_vinf_kms"])
    try:
        sep = float(fv.plain.get("flyby_start_sep_km") or _PLAIN_BY_DEST["flyby_start_sep_km"].default)
        pert = float(fv.plain.get("flyby_perturber_earth_masses")
                     or _PLAIN_BY_DEST["flyby_perturber_earth_masses"].default)
    except ValueError:
        return ""
    if rp is None or vinf is None or min(rp + vinf) <= 0 or sep <= 0 or pert <= 0:
        return ""
    t = encounter._max_time_to_pericentre_hr(rp[0], rp[1], vinf[0], vinf[1], sep, pert)
    return str(math.ceil(2.0 * t * TMAX_MARGIN))


def fill_defaults(fv: FormValues, last_auto_tmax: str = "") -> Filled:
    """Fill blank fields the chosen modes need (settled body, hyperbola, boulders). Returns a copy.

    Never overwrites a typed value. The one exception is tmax while it still holds last_auto_tmax:
    it is recomputed so it keeps following rp / v_inf / start_sep. A hyperbola or boulders on a
    lattice body switches body_source to settled (the only body either works with).
    """
    out, notes = copy.deepcopy(fv), []
    p, axes = out.plain, out.axes
    hyper, boulders = p.get("encounter") == "hyperbola", bool(p.get("boulders"))

    def plain(dest, value):
        if _blank(p.get(dest)):
            p[dest] = value
            notes.append(f"{dest} {value}")

    def axis(stem, value):
        av = axes[stem]
        if not av.sweep and _blank(av.fixed):
            av.fixed = value
            notes.append(f"{stem} {value}")

    if (hyper or boulders) and p.get("body_source") != "settled":
        p["body_source"] = "settled"
        notes.append("body_source settled")
    if p.get("body_source") == "settled":
        if not p.get("vary_use_dem"):
            plain("use_dem_fixed", SETTLED_DEFAULTS["use_dem_fixed"])
        if _blank(p.get("np_apophis_list")):
            plain("np_apophis", SETTLED_DEFAULTS["np_apophis"])
        if boulders:
            for dest, value in BOULDER_PLAIN_DEFAULTS.items():
                plain(dest, value)
            for stem, value in BOULDER_AXIS_DEFAULTS.items():
                axis(stem, value)
    auto = ""
    if hyper:
        for stem, value in HYPERBOLA_DEFAULTS.items():
            axis(stem, value)
        tmax = axes["tmax_hours"]
        if not tmax.sweep and (_blank(tmax.fixed) or tmax.fixed == last_auto_tmax):
            auto = _auto_tmax(out)
            if auto and auto != tmax.fixed:
                tmax.fixed = auto
                notes.append(f"tmax_hours {auto} (auto)")
            elif not auto and tmax.fixed == last_auto_tmax:
                tmax.fixed = ""  # rp / v_inf half-typed: wait rather than keep a stale value
    return Filled(out, notes, auto)


def single_mode_error(fv: FormValues) -> Optional[str]:
    """A list flag with several values makes one run per value; a single run takes one."""
    for dest in ("np_apophis_list", "spin_period_list"):
        values = str(fv.plain.get(dest) or "").split()
        if len(values) > 1:
            return (f"single run: {dest} has {len(values)} values and would make {len(values)} runs; "
                    "give one value, or switch to Sweep")
    return None


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
