"""DEM-oriented interactive driver for Sobol sweeps with per-run ``np_apophis``.

Runs the shared wizard from ``interactive_run_mass_sobol``, then for each
requested Apophis particle count copies ``<prefix>.{in,setup}`` (and ``*.txt``) into a
staging directory, patches ``np_apophis`` in the copy only, and invokes
``run_mass_sobol_phantom.py`` once per count. Use the wizard (or CLI seeds) for
``--use-dem-fixed`` / ``--vary-use-dem``; staging does not patch ``use_dem``.

Usage::

    cd sobol
    python3 inter_DEM_run_mass_sobol.py [--dem-np 20 30 64] [flags passed as wizard seeds...]

``--dem-np`` is consumed by this script only (not passed to the child). If omitted,
you are prompted for space-separated integers after the wizard finishes.

Do not pass ``-i`` / ``--interactive`` to the child process; the wizard already ran here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from interactive_run_mass_sobol import run_interactive_wizard

from run_mass_sobol_phantom import (
    build_parser,
    parse_args,
    sanitize_batch_label,
    _MAXPTMASS_DEFAULT,
    _MAXPTMASS_WARN_THRESHOLD,
)

_SO_DIR = Path(__file__).resolve().parent
_RUNNER = _SO_DIR / "run_mass_sobol_phantom.py"

_NP_APOPHIS_RE = re.compile(r"^(\s*np_apophis\s*=\s*)\d+", re.MULTILINE | re.IGNORECASE)
_USE_DEM_RE = re.compile(r"^(\s*use_dem\s*=\s*)[TFtf]\b", re.MULTILINE | re.IGNORECASE)


def print_dem_checklist() -> None:
    print(
        "\n=== DEM mode (inter_DEM) ===\n"
        "- Each ``np_apophis`` value runs a separate sweep subprocess with its own staged ``--base-dir``.\n"
        "- Counts should be >= 2 for a multi-site Apophis model (see PHANTOM setup_solarsystem).\n"
        "- Optional: ``--dem-np N1 N2 ...`` before other flags to skip the particle-count prompt.\n"
        "- Ephemeris: place ``*.txt`` next to your template ``.setup``; copies follow each staging dir.\n"
        "============================\n",
        flush=True,
    )


def print_use_dem_ca_warning() -> None:
    print(
        "\n[WARN] ``--vary-use-dem`` produces multi-sink Apophis runs. The sweep runner's sink .ev\n"
        "       closest-approach step may fail even if PHANTOM succeeds; interpret CSV errors\n"
        "       accordingly or use ``--dry-run``.\n",
        file=sys.stderr,
        flush=True,
    )


def _read_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        print("\n[ERROR] EOF while reading input.", file=sys.stderr)
        raise SystemExit(2) from None


def parse_dem_np_argv(argv: Sequence[str]) -> Tuple[List[str], Optional[List[int]]]:
    """Remove ``--dem-np`` and following integers; return (remaining_argv, counts or None)."""
    out: List[str] = []
    dem: Optional[List[int]] = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--dem-np":
            i += 1
            vals: List[int] = []
            while i < len(argv) and not argv[i].startswith("--"):
                try:
                    vals.append(int(argv[i], 10))
                except ValueError as exc:
                    print(f"[ERROR] Invalid integer after --dem-np: {argv[i]!r} ({exc})", file=sys.stderr)
                    raise SystemExit(2) from exc
                i += 1
            dem = vals
            continue
        out.append(tok)
        i += 1
    return out, dem


def prompt_np_apophis_list() -> List[int]:
    while True:
        raw = _read_line(
            "Enter np_apophis values (space-separated integers, each >= 2), e.g. 20 30 64:\n> "
        ).strip()
        if not raw:
            print("  Please enter at least one integer.", file=sys.stderr)
            continue
        try:
            vals = [int(x, 10) for x in raw.split()]
        except ValueError as exc:
            print(f"  Invalid input: {exc}", file=sys.stderr)
            continue
        if any(v < 2 for v in vals):
            print("  Each value must be >= 2 for this DEM workflow.", file=sys.stderr)
            continue
        return vals


def dem_preflight_vary_use_dem(np_list: Sequence[int], vary_use_dem: bool) -> None:
    if not vary_use_dem:
        return
    if any(n <= 1 for n in np_list):
        print(
            "[ERROR] With --vary-use-dem, every np_apophis must be > 1 (PHANTOM DEM branch).",
            file=sys.stderr,
        )
        raise SystemExit(2)


def validate_np_list(np_list: Sequence[int]) -> None:
    if not np_list:
        print("[ERROR] np_apophis list is empty.", file=sys.stderr)
        raise SystemExit(2)
    if any(n < 2 for n in np_list):
        print("[ERROR] Each np_apophis must be >= 2 in DEM mode.", file=sys.stderr)
        raise SystemExit(2)
    # Warn early if any requested count will trigger the MAXPTMASS overflow in phantomsetup.
    # The close-packed lattice overshoots np_apophis by ~2-3% (e.g. 1000 -> 1012 actual sinks),
    # so values at or above _MAXPTMASS_WARN_THRESHOLD will exceed the default compiled limit of 1000.
    high = [n for n in np_list if n >= _MAXPTMASS_WARN_THRESHOLD]
    if high:
        print(
            f"[WARN] np_apophis values {high} are at or above {_MAXPTMASS_WARN_THRESHOLD}. "
            f"The close-packed lattice overshoots np_apophis by ~2-3%, which will exceed "
            f"the default compiled MAXPTMASS={_MAXPTMASS_DEFAULT} and cause phantomsetup to abort. "
            "Ensure PHANTOM is compiled with MAXPTMASS>max(np_apophis) "
            "(e.g. set MAXPTMASS=2000 in the Makefile 'ifndef MAXPTMASS' block, then run 'make setup && make').",
            file=sys.stderr,
        )


def patch_np_apophis_in_setup_text(text: str, n: int) -> str:
    if not _NP_APOPHIS_RE.search(text):
        raise ValueError("np_apophis assignment not found in .setup text")
    return _NP_APOPHIS_RE.sub(lambda m: m.group(1) + str(n), text, count=1)


def patch_use_dem_true_in_setup_text(text: str) -> str:
    if not _USE_DEM_RE.search(text):
        raise ValueError("use_dem assignment not found in .setup text")
    return _USE_DEM_RE.sub(lambda m: m.group(1) + "T", text, count=1)


def stage_templates(
    user_base: Path,
    prefix: str,
    np_val: int,
    staging_sub: Path,
) -> Path:
    """Copy ``{prefix}.in``, ``{prefix}.setup``, and ``*.txt`` into ``staging_sub``; patch np_apophis."""
    staging_sub.mkdir(parents=True, exist_ok=True)
    for name in (f"{prefix}.in", f"{prefix}.setup"):
        src = user_base / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing template file: {src}")
        shutil.copy2(src, staging_sub / name)
    for txt in user_base.glob("*.txt"):
        if txt.is_file():
            shutil.copy2(txt, staging_sub / txt.name)
    setup_path = staging_sub / f"{prefix}.setup"
    text = setup_path.read_text(encoding="utf-8")
    text = patch_np_apophis_in_setup_text(text, np_val)
    setup_path.write_text(text, encoding="utf-8")
    return staging_sub


def _strip_interactive_child_flags(argv: List[str]) -> None:
    drop = {"--interactive", "-i"}
    i = 0
    while i < len(argv):
        if argv[i] in drop:
            argv.pop(i)
            continue
        i += 1


def _set_or_replace_flag(argv: List[str], flag: str, value: str) -> None:
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            if i + 1 < len(argv):
                argv[i + 1] = value
            else:
                argv.append(value)
            return
        i += 1
    argv.extend([flag, value])


def build_child_argv(wizard_argv: Sequence[str], stage_dir: Path, batch_label: str) -> List[str]:
    argv = list(wizard_argv)
    _strip_interactive_child_flags(argv)
    _set_or_replace_flag(argv, "--base-dir", str(stage_dir))
    _set_or_replace_flag(argv, "--batch-label", batch_label)
    return argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_in = list(sys.argv[1:] if argv is None else argv)
    argv_for_wizard, dem_np_cli = parse_dem_np_argv(argv_in)

    print_dem_checklist()
    initial_args = parse_args(argv_for_wizard)
    wizard_argv = run_interactive_wizard(build_parser(), initial_args)
    final_args = parse_args(wizard_argv)

    if final_args.vary_use_dem:
        print_use_dem_ca_warning()

    if dem_np_cli is not None and len(dem_np_cli) == 0:
        dem_np_cli = None
    np_list = dem_np_cli if dem_np_cli is not None else prompt_np_apophis_list()
    validate_np_list(np_list)
    dem_preflight_vary_use_dem(np_list, final_args.vary_use_dem)

    user_base = Path(final_args.base_dir).expanduser().resolve()
    prefix = final_args.prefix

    staging_root = Path(tempfile.mkdtemp(prefix="inter_dem_staging_"))
    try:
        for n in np_list:
            stage_dir = staging_root / f"np_{n}"
            stage_templates(user_base, prefix, n, stage_dir)
            base_label = final_args.batch_label
            if base_label:
                raw_l = f"{base_label}_np{n}"
            else:
                raw_l = f"np{n}"
            batch_label = sanitize_batch_label(raw_l)
            child_argv = build_child_argv(wizard_argv, stage_dir, batch_label)
            cmd = [sys.executable, str(_RUNNER), *child_argv]
            print(f"\n[INFO] np_apophis={n}  staged={stage_dir}\n[INFO] Running: {' '.join(cmd)}", flush=True)
            try:
                rc = int(subprocess.call(cmd))
            except OSError as exc:
                print(f"[ERROR] subprocess failed: {exc}", file=sys.stderr)
                return 1
            if rc != 0:
                print(f"[ERROR] Child exited with code {rc} (np_apophis={n}).", file=sys.stderr)
                return rc
        return 0
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


__all__ = [
    "build_child_argv",
    "dem_preflight_vary_use_dem",
    "main",
    "parse_dem_np_argv",
    "patch_np_apophis_in_setup_text",
    "patch_use_dem_true_in_setup_text",
    "print_dem_checklist",
    "print_use_dem_ca_warning",
    "run_interactive_wizard",
    "stage_templates",
    "validate_np_list",
]


if __name__ == "__main__":
    raise SystemExit(main())
