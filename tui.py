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
# batches in Honours/sobol_mass_runs; a settled body (the runner's CLI default stays lattice, which
# existing scripts rely on). Only applied when the flag is not on the command line.
DEFAULT_OVERRIDES = {
    "body_source": "settled",
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
