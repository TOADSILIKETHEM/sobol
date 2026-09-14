#!/usr/bin/env python3
"""Textual TUI: configure one PHANTOM DEM run, run it, convert dumps,
render a per-sphere Blender clip headless. One run per launch — see
sobol/tui_run.py for the Sobol sweep TUI.

Launch:
    python3 sobol/tui_sim_render.py
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from run_mass_sobol_phantom import (  # noqa: E402
    RunSample,
    RunRecord,
    EARTH_SINK_ID_DEFAULT,
    APOPHIS_SINK_ID_DEFAULT,
    DEFAULT_DN_COHES_FACTOR,
    preflight,
    run_one_case,
)

# Windows-side Code repo root, reached over the WSL /mnt/c/ interop mount —
# hardcoded the same way sobol/Analysis/run_demtocsv_batch.py hardcodes
# DEFAULT_DEM_DUMP_CONVERT / DEFAULT_BASE_OUT, not derived.
_REPO_WIN_CODE = Path(
    "/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code"
)
BLENDER_EXE = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"

_MNT_DRIVE_RE = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")


def to_windows_path(p: Path) -> str:
    """WSL /mnt/<drive>/... path -> Windows <DRIVE>:/... path string.

    Deliberately simpler than shelling out to `wslpath -w` -- the mount is
    always /mnt/<single-letter>/... in this environment. Only POSIX-absolute
    inputs (starting with "/") are resolved first (so relative segments like
    "/mnt/c/foo/../bar" normalize before the drive-letter swap); a string
    that doesn't start with "/" -- e.g. an already Windows-style path such as
    "C:/Users/..." -- is returned unchanged, since resolving it would
    incorrectly prefix the WSL cwd.
    """
    s = str(p)
    if s.startswith("/"):
        s = str(Path(s).resolve())
    m = _MNT_DRIVE_RE.match(s)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2) or "/"
        return f"{drive}:{rest}"
    return s


# Windows blender.exe cannot open a /mnt/c path passed to --python, so this
# must be a Windows-style path (unlike BLENDER_EXE, which WSL executes and
# therefore stays a /mnt/c path).
DEM_HEADLESS_RENDER = to_windows_path(_REPO_WIN_CODE / "BlenderConvert" / "DEMHeadlessRender.py")


@dataclass
class SimParams:
    """Fixed-value inputs for one PHANTOM run (no Sobol sampling)."""

    prefix: str = "sobol"
    base_dir: Path = field(default_factory=lambda: Path("."))
    phantom_dir: Path = field(default_factory=lambda: Path("."))
    output_root: Path = field(default_factory=lambda: Path("sobol_mass_runs"))
    ephemeris_cache_dir: Optional[Path] = None
    dry_run: bool = False
    earth_sink_id: int = EARTH_SINK_ID_DEFAULT
    apophis_sink_id: int = APOPHIS_SINK_ID_DEFAULT
    shape_file: Optional[str] = None
    sample: RunSample = field(default_factory=RunSample)


@dataclass
class RenderFormValues:
    """Render-only inputs from the TUI's render form."""

    resolution: str = "1920x1080"
    samples: int = 128
    fps: int = 24
    max_frames: Optional[int] = None
    camera_mode: str = "auto"


@dataclass
class RenderParams:
    """Fully resolved inputs to build_render_command() / run_render_stage()."""

    grains_dir: Path
    bodies_dir: Path
    output_dir: Path
    resolution: str = "1920x1080"
    samples: int = 128
    fps: int = 24
    max_frames: Optional[int] = None
    camera_mode: str = "auto"


@dataclass
class PipelineResult:
    stage: str  # "sim" | "convert" | "render" | "done"
    ok: bool
    message: str
    record: Optional[RunRecord] = None


def run_sim_stage(params: SimParams) -> RunRecord:
    """Run one PHANTOM case from params.sample. Blocking — call from a worker thread."""
    base_setup, base_input, phantomsetup_bin, phantom_bin = preflight(
        argparse.Namespace(
            prefix=params.prefix,
            phantom_dir=str(params.phantom_dir),
            dry_run=params.dry_run,
        ),
        params.base_dir,
        params.output_root,
    )
    return run_one_case(
        run_id=1,
        sample=params.sample,
        base_setup=base_setup,
        base_input=base_input,
        output_root=params.output_root,
        prefix=params.prefix,
        phantomsetup_bin=phantomsetup_bin,
        phantom_bin=phantom_bin,
        ref_mass_kg=None,
        dry_run=params.dry_run,
        earth_sink_id=params.earth_sink_id,
        apophis_sink_id=params.apophis_sink_id,
        ephemeris_cache_dir=params.ephemeris_cache_dir,
        shape_file=params.shape_file,
    )
