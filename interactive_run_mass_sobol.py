"""Interactive CLI wizard for run_mass_sobol_phantom.py.

Prompts mirror argparse actions on a shared parser so new flags are picked up automatically.
Optional scale and mass bounds are gated before min/max prompts.
Extend DEST_TO_SECTION for headings; add INTERACTIVE_BRIEF entries for brief help + valid answers;
add WIZARD_CUSTOM_HANDLERS for unusual actions.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

# Logical scale parameters (param name in .setup / RunSample) and argparse dests for min/max.
# Order matches run_mass_sobol_phantom._SCALE_VARIATION_SPEC; extend if new scale pairs are added.
SCALE_BOUND_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("scale_vel", "scale_vel_min", "scale_vel_max"),
    ("scale_pos", "scale_pos_min", "scale_pos_max"),
    ("scale_r_apophis", "scale_r_apophis_min", "scale_r_apophis_max"),
    ("scale_rho", "scale_rho_min", "scale_rho_max"),
    ("np_apophis", "np_apophis_min", "np_apophis_max"),
)
SCALE_BOUND_DESTS: FrozenSet[str] = frozenset(
    dest for _, lo, hi in SCALE_BOUND_PAIRS for dest in (lo, hi)
)

# Mass min/max/unit are skipped when user declines the mass gate (mirrors optional scale pattern).
MASS_GATE_DESTS: FrozenSet[str] = frozenset({"mass_min_kg", "mass_max_kg", "mass_unit"})

# Human-readable labels for the scale gate (match setup_solarsystem / argparse help).
SCALE_PARAM_LABELS: Dict[str, str] = {
    "scale_vel": "scale_vel (Apophis velocity scale)",
    "scale_pos": "scale_pos (Apophis initial position scale)",
    "scale_r_apophis": "scale_r_apophis (Apophis radius scale)",
    "scale_rho": "scale_rho (bulk density scale when mass is density-derived)",
    "np_apophis": "np_apophis (Apophis DEM particle count)",
}

# dest -> section title (unknown dests fall under "Other")
DEST_TO_SECTION: Dict[str, str] = {
    "prefix": "Paths",
    "base_dir": "Paths",
    "ephemeris_cache_dir": "Paths",
    "phantom_dir": "Paths",
    "output_root": "Paths",
    "num_samples": "Sampling",
    "seed": "Sampling",
    "saltelli_n": "Sampling",
    "saltelli_calc_second_order": "Sampling",
    "mass_min_kg": "Mass",
    "mass_max_kg": "Mass",
    "mass_unit": "Mass",
    "scale_vel_min": "Scale bounds (optional dimensions)",
    "scale_vel_max": "Scale bounds (optional dimensions)",
    "scale_pos_min": "Scale bounds (optional dimensions)",
    "scale_pos_max": "Scale bounds (optional dimensions)",
    "scale_r_apophis_min": "Scale bounds (optional dimensions)",
    "scale_r_apophis_max": "Scale bounds (optional dimensions)",
    "scale_rho_min": "Scale bounds (optional dimensions)",
    "scale_rho_max": "Scale bounds (optional dimensions)",
    "np_apophis_min": "Scale bounds (optional dimensions)",
    "np_apophis_max": "Scale bounds (optional dimensions)",
    "vary_use_dem": "Setup toggles",
    "use_dem_fixed": "Setup toggles",
    "vary_apophis_only": "Setup toggles",
    "sink_earth_id": "Sinks / post-processing",
    "sink_apophis_id": "Sinks / post-processing",
    "batch_label": "Batch naming",
    "batch_slug_max_len": "Batch naming",
    "dry_run": "Execution",
    "jobs": "Execution",
}

# dest -> handler(action, state, flag, out) -> None (mutates state and out_argv)
WIZARD_CUSTOM_HANDLERS: Dict[str, Callable[[argparse.Action, argparse.Namespace, str, List[str]], None]] = {}

# Set only via --vary-use-dem wizard block or CLI; no separate interactive prompts.
WIZARD_SKIP_DESTS: FrozenSet[str] = frozenset({"use_dem_fixed"})

# dest -> (brief what-it-does, valid answers / input rules). Used for every prompt; extend when
# adding new CLI flags. Unknown dests use _fallback_explain(action).
INTERACTIVE_BRIEF: Dict[str, tuple[str, str]] = {
    "prefix": (
        "Stem of PHANTOM control files: expects <prefix>.in and <prefix>.setup under --base-dir.",
        "Non-empty text (no spaces recommended), e.g. sobol.",
    ),
    "base_dir": (
        "Directory that contains the template .in and .setup files.",
        "Path to a folder, or . for the current working directory.",
    ),
    "ephemeris_cache_dir": (
        "Optional folder of Horizons *.txt snippets copied into each run before phantomsetup (skips downloads).",
        "Path to a folder with mercury.txt etc., or Enter for none / omit live-download fallback.",
    ),
    "num_samples": (
        "How many Sobol sample runs to generate when Saltelli mode is off (--saltelli-n not set).",
        "Positive integer (>= 1 after validation); Enter keeps the shown default.",
    ),
    "mass_min_kg": (
        "Lower bound on Apophis mass in kg (only if you chose to vary mass in the mass gate above).",
        "Positive float; must be < mass max. Both min and max must be set to vary mass.",
    ),
    "mass_max_kg": (
        "Upper bound on Apophis mass in kg when mass is varied.",
        "Positive float; must be greater than mass min.",
    ),
    "mass_unit": (
        "Unit written to m_apophis_in when mass is varied; sampling is always in kg internally.",
        "Exactly one of: kg, g, msun. (Only used when mass min/max are set.)",
    ),
    "seed": (
        "Random seed for scrambled Sobol draws and for Saltelli when used.",
        "Integer; or Enter to keep the current value.",
    ),
    "scale_vel_min": (
        "Lower bound for scale_vel (only if you chose to vary this scale in the gate above).",
        "Float, or Enter to keep current. Both min and max must be set for this dimension.",
    ),
    "scale_vel_max": (
        "Upper bound for scale_vel; must be set together with min and be greater than min.",
        "Float, or Enter to keep current.",
    ),
    "scale_pos_min": (
        "Lower bound for scale_pos (only if you chose to vary this scale in the gate above).",
        "Float, or Enter to keep current.",
    ),
    "scale_pos_max": (
        "Upper bound for scale_pos; set both min and max or neither.",
        "Float, or Enter to keep current.",
    ),
    "scale_r_apophis_min": (
        "Lower bound for scale_r_apophis (only if you chose to vary this scale in the gate above).",
        "Float, or Enter to keep current.",
    ),
    "scale_r_apophis_max": (
        "Upper bound for scale_r_apophis; set both min and max or neither.",
        "Float, or Enter to keep current.",
    ),
    "scale_rho_min": (
        "Lower bound for scale_rho (only if you chose to vary this scale in the gate above).",
        "Float, or Enter to keep current.",
    ),
    "scale_rho_max": (
        "Upper bound for scale_rho; set both min and max or neither.",
        "Float, or Enter to keep current.",
    ),
    "np_apophis_min": (
        "Lower bound for np_apophis (DEM particle count per Apophis body); must be >= 2.",
        "Integer >= 2; or Enter to keep current.",
    ),
    "np_apophis_max": (
        "Upper bound for np_apophis; must be > min. Each run draws an integer uniformly from [min, max].",
        "Integer > np_apophis_min; or Enter to keep current.",
    ),
    "vary_use_dem": (
        "Interactive: two-step use_dem (fixed preset, then vary). CLI: use --vary-use-dem and/or --use-dem-fixed.",
        "See prompts when using -i.",
    ),
    "vary_apophis_only": (
        "If yes, adds a dimension toggling apophis_only (Earth absent when true; CA metrics may be NaN).",
        "y, n, yes, no, t, f, 1, 0; or Enter to keep the current value.",
    ),
    "sink_earth_id": (
        "Sink index for Earth in .ev filenames when extracting closest approach.",
        "Integer (default matches full solar-system setups).",
    ),
    "sink_apophis_id": (
        "Sink index for Apophis in .ev filenames for the same post-processing.",
        "Integer (default matches full solar-system setups).",
    ),
    "output_root": (
        "Parent folder where timestamped batch directories and summary CSVs are written.",
        "Directory path string; or Enter to keep the current value.",
    ),
    "batch_label": (
        "Optional suffix for the batch folder name; if unset, an auto slug from parameters is used.",
        "Any text, or Enter for none / current (omit flag when same as default).",
    ),
    "batch_slug_max_len": (
        "Maximum length of the batch suffix (auto slug or sanitized label) before truncation + hash.",
        "Integer >= 9 (enforced after the wizard).",
    ),
    "phantom_dir": (
        "PHANTOM installation root; must contain bin/phantomsetup and bin/phantom (unless dry-run).",
        "Directory path; or Enter to keep the current value.",
    ),
    "dry_run": (
        "If yes, only prepares run folders and patched setups; does not execute the PHANTOM binary.",
        "y, n, yes, no, t, f, 1, 0; or Enter to keep the current value.",
    ),
    "jobs": (
        "Number of parallel worker processes, each running one PHANTOM case at a time.",
        "Integer >= 1; or Enter to keep the current value.",
    ),
    "saltelli_n": (
        "Saltelli / SALib base size N: turns on Saltelli layout (run count ~ N*(D+2), more if second-order).",
        "Integer >= 2 to enable Saltelli, or Enter for none (plain Sobol uses --num-samples).",
    ),
    "saltelli_calc_second_order": (
        "If yes with Saltelli, use the larger design that also estimates second-order Sobol indices.",
        "y, n, yes, no, t, f, 1, 0; or Enter to keep the current value.",
    ),
}


def _section_for_dest(dest: str) -> str:
    return DEST_TO_SECTION.get(dest, "Other")


def _read_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        print("\n[ERROR] End of input (EOF); interactive mode needs a TTY or piped answers.", file=sys.stderr)
        raise SystemExit(2) from None


def _prompt_mass_vary_selection(state: argparse.Namespace) -> bool:
    """Ask whether Apophis mass is a Sobol dimension; Enter defaults yes if seed has both bounds set."""
    m_lo = getattr(state, "mass_min_kg", None)
    m_hi = getattr(state, "mass_max_kg", None)
    default_yes = m_lo is not None and m_hi is not None
    yn = "Y/n" if default_yes else "y/N"
    what = (
        "Include Apophis mass as a varied Sobol dimension (you will set min/max and unit next)? "
        "If no, the template m_apophis_in line is left unchanged."
    )
    valid = "y, n, yes, no, t, f, 1, 0; or Enter for the suggested default."
    while True:
        raw = _read_line(
            f"{what}\n  Option: (mass gate) vary mass\n  Valid answers: {valid}\n  [{yn}]: "
        ).strip().lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes", "t", "true", "1"):
            return True
        if raw in ("n", "no", "f", "false", "0"):
            return False
        print("  Please enter y or n (empty = suggested default).", file=sys.stderr)


def _prompt_scale_vary_selection(state: argparse.Namespace) -> FrozenSet[str]:
    """Ask which optional scale_* bounds are active Sobol dimensions; Enter defaults from seed (both set -> yes)."""
    print(
        "Optional setup scales: choose which become Sobol dimensions (each selected one gets min/max prompts next).",
        flush=True,
    )
    vary: List[str] = []
    for param, lo_attr, hi_attr in SCALE_BOUND_PAIRS:
        lo = getattr(state, lo_attr, None)
        hi = getattr(state, hi_attr, None)
        default_yes = lo is not None and hi is not None
        yn = "Y/n" if default_yes else "y/N"
        label = SCALE_PARAM_LABELS.get(param, param)
        what = f"Include {label} as a varied Sobol dimension (you will set min and max next)?"
        valid = "y, n, yes, no, t, f, 1, 0; or Enter for the suggested default."
        while True:
            raw = _read_line(
                f"{what}\n  Option: (scale gate) vary {param}\n  Valid answers: {valid}\n  [{yn}]: "
            ).strip().lower()
            if not raw:
                choice = default_yes
                break
            if raw in ("y", "yes", "t", "true", "1"):
                choice = True
                break
            if raw in ("n", "no", "f", "false", "0"):
                choice = False
                break
            print("  Please enter y or n (empty = suggested default).", file=sys.stderr)
        if choice:
            vary.append(param)
    return frozenset(vary)


def _primary_flag(action: argparse.Action) -> str:
    longs = [s for s in action.option_strings if s.startswith("--")]
    if longs:
        return min(longs, key=len)
    return action.option_strings[0]


def _auto_valid_answers(action: argparse.Action) -> str:
    if action.choices is not None:
        return f"One of: {', '.join(map(str, action.choices))}; or Enter to keep the current value."
    if action.type is int:
        return "Integer; or Enter to keep the current value."
    if action.type is float:
        return "Floating-point number; or Enter to keep the current value."
    if action.type is None:
        return "Text (paths allowed); or Enter to keep the current value."
    return "A value accepted by this option; or Enter to keep the current value."


def _fallback_explain(action: argparse.Action) -> tuple[str, str]:
    raw_help = (action.help or f"Configure {action.dest}.").strip()
    first_line = raw_help.split("\n")[0].strip()
    if len(first_line) > 160:
        first_line = first_line[:157] + "..."
    valid = _auto_valid_answers(action)
    return first_line, valid


def _explain_for(action: argparse.Action) -> tuple[str, str]:
    dest = action.dest
    if dest in INTERACTIVE_BRIEF:
        return INTERACTIVE_BRIEF[dest]
    return _fallback_explain(action)


def _prompt_bool(action: argparse.Action, current: bool) -> bool:
    yn = "Y/n" if current else "y/N"
    what, valid = _explain_for(action)
    flag = _primary_flag(action)
    while True:
        raw = _read_line(
            f"{what}\n  Option: {flag}\n  Valid answers: {valid}\n{flag} [{yn}]: "
        ).strip().lower()
        if not raw:
            return current
        if raw in ("y", "yes", "t", "true", "1"):
            return True
        if raw in ("n", "no", "f", "false", "0"):
            return False
        print("  Please enter y or n (empty = keep default).", file=sys.stderr)


def _prompt_scalar(action: argparse.Action, current: object) -> object:
    what, valid = _explain_for(action)
    flag = _primary_flag(action)
    t = action.type
    choices = action.choices
    cur_s = "" if current is None else str(current)
    while True:
        raw = _read_line(f"{what}\n  Option: {flag}\n  Valid answers: {valid}\n[{cur_s}]: ").strip()
        if not raw:
            return current
        if choices is not None and raw not in choices:
            print(f"  Value must be one of: {list(choices)}", file=sys.stderr)
            continue
        if t is None:
            return raw
        try:
            return t(raw)
        except (TypeError, ValueError) as exc:
            print(f"  Invalid value: {exc}", file=sys.stderr)


def _should_emit(action: argparse.Action, value: object) -> bool:
    if action.default is argparse.SUPPRESS:
        return True
    return value != action.default


def _append_flag(out: List[str], flag: str) -> None:
    out.append(flag)


def _append_kv(out: List[str], flag: str, value: object) -> None:
    out.append(flag)
    out.append(str(value))


def _prompt_use_dem_fixed_preset() -> Optional[str]:
    """Return None (leave base .setup), 'true', or 'false' for --use-dem-fixed when not varying."""
    what = (
        "If you do not vary use_dem next: force it True (y), False (n), or leave the base "
        "<prefix>.setup value unchanged (Enter)."
    )
    valid = "y / n / Enter (leave template)."
    while True:
        raw = _read_line(
            f"{what}\n  Option: --use-dem-fixed (preset when not varying)\n  Valid answers: {valid}\n  [y/n/Enter]: "
        ).strip().lower()
        if not raw:
            return None
        if raw in ("y", "yes", "t", "true", "1"):
            return "true"
        if raw in ("n", "no", "f", "false", "0"):
            return "false"
        print("  Please enter y, n, or Enter for template.", file=sys.stderr)


def _collect_use_dem_interactive(
    _action: argparse.Action, state: argparse.Namespace, flag: str, out: List[str]
) -> None:
    vary_current = bool(getattr(state, "vary_use_dem", False))
    what = "Vary use_dem across Sobol samples (adds one Sobol dimension; u>=0.5 -> T)?"
    yn = "Y/n" if vary_current else "y/N"
    valid = "y, n, yes, no, t, f, 1, 0; or Enter for the suggested default."
    while True:
        raw = _read_line(
            f"{what}\n  Option: {flag}\n  Valid answers: {valid}\n{flag} [{yn}]: "
        ).strip().lower()
        if not raw:
            vary = vary_current
            break
        if raw in ("y", "yes", "t", "true", "1"):
            vary = True
            break
        if raw in ("n", "no", "f", "false", "0"):
            vary = False
            break
        print("  Please enter y or n (empty = suggested default).", file=sys.stderr)

    setattr(state, "vary_use_dem", vary)

    if vary:
        setattr(state, "use_dem_fixed", None)
        _append_flag(out, "--vary-use-dem")
    else:
        fixed_preset = _prompt_use_dem_fixed_preset()
        setattr(state, "use_dem_fixed", fixed_preset)
        if fixed_preset is not None:
            _append_kv(out, "--use-dem-fixed", fixed_preset)


WIZARD_CUSTOM_HANDLERS["vary_use_dem"] = _collect_use_dem_interactive


def _collect_saltelli_second_order_interactive(
    action: argparse.Action, state: argparse.Namespace, _flag: str, out: List[str]
) -> None:
    if getattr(state, "saltelli_n", None) is None:
        return
    _collect_store_true(action, state, out)


WIZARD_CUSTOM_HANDLERS["saltelli_calc_second_order"] = _collect_saltelli_second_order_interactive


def _collect_store_true(action: argparse.Action, state: argparse.Namespace, out: List[str]) -> None:
    current = bool(getattr(state, action.dest))
    final = _prompt_bool(action, current)
    setattr(state, action.dest, final)
    if final:
        _append_flag(out, _primary_flag(action))


def _collect_store(action: argparse.Action, state: argparse.Namespace, out: List[str]) -> None:
    current = getattr(state, action.dest)
    final = _prompt_scalar(action, current)
    setattr(state, action.dest, final)
    if _should_emit(action, final):
        _append_kv(out, _primary_flag(action), final)


def _skip_action(action: argparse.Action) -> bool:
    if action.dest in (None, "help", "interactive"):
        return True
    if isinstance(action, argparse._HelpAction):  # noqa: SLF001
        return True
    return False


def run_interactive_wizard(parser: argparse.ArgumentParser, initial_args: argparse.Namespace) -> List[str]:
    """Prompt for each registered option (except help/interactive); return argv tokens."""
    state = argparse.Namespace(**vars(initial_args))
    out: List[str] = []
    last_section: Optional[str] = None
    mass_gate_done = False
    skipped_mass_dests: set[str] = set()
    scale_gate_done = False
    skipped_scale_dests: set[str] = set()

    for action in parser._actions:  # noqa: SLF001
        if _skip_action(action):
            continue

        if action.dest in WIZARD_SKIP_DESTS:
            continue

        sec = _section_for_dest(action.dest)
        if sec != last_section:
            print(f"\n=== {sec} ===", flush=True)
            last_section = sec

        if action.dest in WIZARD_CUSTOM_HANDLERS:
            WIZARD_CUSTOM_HANDLERS[action.dest](action, state, _primary_flag(action), out)
            continue

        if action.dest == "mass_min_kg" and not mass_gate_done:
            if not _prompt_mass_vary_selection(state):
                setattr(state, "mass_min_kg", None)
                setattr(state, "mass_max_kg", None)
                skipped_mass_dests.update(MASS_GATE_DESTS)
            mass_gate_done = True

        if action.dest in skipped_mass_dests:
            continue

        if action.dest in SCALE_BOUND_DESTS and not scale_gate_done:
            vary_params = _prompt_scale_vary_selection(state)
            for param, lo_attr, hi_attr in SCALE_BOUND_PAIRS:
                if param not in vary_params:
                    setattr(state, lo_attr, None)
                    setattr(state, hi_attr, None)
                    skipped_scale_dests.add(lo_attr)
                    skipped_scale_dests.add(hi_attr)
            scale_gate_done = True

        if action.dest in skipped_scale_dests:
            continue

        if action.__class__.__name__ == "_StoreTrueAction":  # noqa: SLF001
            _collect_store_true(action, state, out)
            continue

        if action.nargs not in (None, 0):
            raise RuntimeError(
                f"Interactive wizard: unsupported nargs={action.nargs!r} for --{action.dest}; "
                "add a WIZARD_CUSTOM_HANDLERS entry."
            )

        _collect_store(action, state, out)

    return out


__all__ = [
    "DEST_TO_SECTION",
    "INTERACTIVE_BRIEF",
    "MASS_GATE_DESTS",
    "SCALE_BOUND_DESTS",
    "SCALE_BOUND_PAIRS",
    "WIZARD_CUSTOM_HANDLERS",
    "WIZARD_SKIP_DESTS",
    "run_interactive_wizard",
]
