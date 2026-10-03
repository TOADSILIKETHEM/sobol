#!/usr/bin/env python3
"""Textual TUI: configure one PHANTOM DEM run, run it, convert dumps,
render it headless through any mix of per-sphere, composite, and
point-cloud (instanced real grains / static placeholder) paths, and
optionally stitch the rendered paths side by side (render_compare.py).
One run per launch — see sobol/tui_run.py for the Sobol sweep TUI.

Launch:
    python3 sobol/tui_sim_render.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Analysis"))

from run_mass_sobol_phantom import (  # noqa: E402
    RunSample,
    RunRecord,
    EARTH_SINK_ID_DEFAULT,
    APOPHIS_SINK_ID_DEFAULT,
    DEFAULT_DN_COHES_FACTOR,
    _cleanup_run_dir,
    attach_settled_bodies,
    preflight,
    resolve_default_shape_file,
    resolve_phantom_executable,
    run_one_case,
    sanitize_batch_label,
    _DEFAULT_PHANTOM_DIR as _RMS_DEFAULT_PHANTOM_DIR,
)
from run_demtocsv_batch import (  # noqa: E402
    run_dem_dump_convert,
    DEFAULT_DEM_DUMP_CONVERT,
    _min_dem_grains_from_setup,
)
from render_compare import (  # noqa: E402
    CompareError,
    CompareResult,
    format_compare_result,
    run_compare_stage,
)

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, ScrollableContainer
    from textual.timer import Timer
    from textual.widgets import Button, Checkbox, Footer, Header, Input, Select, Static
except ImportError:
    sys.exit(
        "textual is not installed.\n"
        "Run:  pip install textual\n"
        "(sobol/tui_run.py already depends on it, so it should already be present.)"
    )

# Code (HonoursWin repo) lives in WSL since 2026-09-25; data and Blender
# assets stayed in OneDrive (reached over /mnt/c) so OneDrive backs them up.
# Hardcoded like sobol/Analysis/run_demtocsv_batch.py, not derived.
CODE_DIR = Path("/home/mboyle/Honours/Code")
DATA_DIR = Path("/mnt/c/Users/22boy/OneDrive/Documents/GC-Max_desktop/Honours/Code")
BLENDER_EXE = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"

_MNT_DRIVE_RE = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")
_RESOLUTION_RE = re.compile(r"^(\d+)x(\d+)$")


def to_windows_path(p: Path) -> str:
    """WSL path -> path string a Windows program can open.

    /mnt/<drive>/... -> <DRIVE>:/...; any other POSIX-absolute path (e.g. the
    code under /home) -> \\\\wsl.localhost\\<distro>\\... . Only POSIX-absolute
    inputs are resolved first (so "/mnt/c/foo/../bar" normalizes before the
    swap); a string that doesn't start with "/" -- e.g. an already
    Windows-style "C:/Users/..." -- is returned unchanged.
    """
    s = str(p)
    if not s.startswith("/"):
        return s
    s = str(Path(s).resolve())
    m = _MNT_DRIVE_RE.match(s)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2) or "/"
        return f"{drive}:{rest}"
    distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu")
    return "\\\\wsl.localhost\\" + distro + s.replace("/", "\\")


# Windows blender.exe cannot open a /mnt/c path passed to --python, so this
# must be a Windows-style path (unlike BLENDER_EXE, which WSL executes and
# therefore stays a /mnt/c path).
DEM_HEADLESS_RENDER = to_windows_path(CODE_DIR / "BlenderConvert" / "DEMHeadlessRender.py")

# Executed directly from WSL like BLENDER_EXE (so a /mnt/c path); runs the
# composite and real-grain preprocess scripts, which need scipy / numpy that
# Blender's bundled Python lacks. The Windows venv stays in the OneDrive data
# folder and runs the WSL preprocess scripts via their UNC path.
WIN_VENV_PYTHON = DATA_DIR / ".venv" / "Scripts" / "python.exe"
# Static placeholder shape model -- opened by Windows blender.exe, so a
# Windows-style path.
DEFAULT_PLACEHOLDER_OBJ = to_windows_path(
    DATA_DIR / "BlenderConvert" / "Shapes"
    / "gbo.ast-apophis.jpl.radar.shape_model_v1.0"
    / "gbo.ast-apophis.jpl.radar.shape_model_v1.0" / "data" / "apophis_v233s7.obj"
)

# Form defaults. tmax/dtmax match sobol/sobol.setup (4.5 days, 30 min) so a
# filled box does not shorten an ephemeris run. 108 hr also clears the
# hyperbola rule tmax >= 2 * time-to-pericentre at the flyby defaults below
# (~35.4 hr when start separation is 4e5 km). Flyby pair is the 2029-like
# geocentric pass (rp 38000 km, v_inf 5.9 km/s).
DEFAULT_TMAX_HOURS = "108"
DEFAULT_DTMAX_HOURS = "0.5"
DEFAULT_FLYBY_RP_KM = "38000"
DEFAULT_FLYBY_VINF_KMS = "5.9"

_WIN_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def from_windows_path(s: str) -> Path:
    """Windows <DRIVE>:/... (or <DRIVE>:\\...) -> WSL /mnt/<drive>/... Path.
    Anything else is returned as a Path unchanged."""
    m = _WIN_DRIVE_RE.match(s)
    if not m:
        return Path(s)
    return Path(f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/"))


def check_blender_exe(path: str = BLENDER_EXE) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"blender.exe not found at {path} -- confirm the Blender 5.2.1 LTS "
            "install path (see CLAUDE.md) or update BLENDER_EXE in tui_sim_render.py."
        )


# Form-default helpers (review fix, Important #1). Computed via __file__ so
# they are correct regardless of the directory this script is launched from
# -- "python3 sobol/tui_sim_render.py" from /home/mboyle/Honours previously
# resolved base_dir to "." (Honours, not sobol), where sobol.setup/sobol.in
# do not live; and "python3 tui_sim_render.py" from sobol/ previously landed
# output under sobol/sobol_mass_runs instead of Honours/sobol_mass_runs.

def _default_base_dir() -> Path:
    """sobol.setup / sobol.in live next to this script -- the sobol/ dir."""
    return Path(__file__).resolve().parent


def _default_phantom_dir() -> str:
    """Mirrors run_mass_sobol_phantom.py's own --phantom-dir CLI default
    expression (that module's argparse setup, `default=os.environ.get(
    "PHANTOM_DIR", str(_DEFAULT_PHANTOM_DIR))`): the PHANTOM_DIR env var if
    set, else that module's own _DEFAULT_PHANTOM_DIR (its own directory,
    which happens to be this same sobol/ dir since both scripts live there).
    """
    return os.environ.get("PHANTOM_DIR", str(_RMS_DEFAULT_PHANTOM_DIR))


def _default_output_root() -> Path:
    """Parent of sobol/ (i.e. Honours/) -- so runs land in
    Honours/sobol_mass_runs regardless of the launch cwd."""
    return Path(__file__).resolve().parent.parent / "sobol_mass_runs"


@dataclass
class SimParams:
    """Fixed-value inputs for one PHANTOM run (no Sobol sampling)."""

    prefix: str = "sobol"
    base_dir: Path = field(default_factory=_default_base_dir)
    phantom_dir: Path = field(default_factory=lambda: Path(_default_phantom_dir()))
    output_root: Path = field(default_factory=_default_output_root)
    ephemeris_cache_dir: Optional[Path] = None
    dry_run: bool = False
    earth_sink_id: int = EARTH_SINK_ID_DEFAULT
    apophis_sink_id: int = APOPHIS_SINK_ID_DEFAULT
    shape_file: Optional[str] = None
    settle_tdyn: float = 5.0
    relax_tdyn: float = 0.5
    body_cache_dir: Optional[Path] = None  # None = <output_root parent>/settled_bodies
    sample: RunSample = field(default_factory=RunSample)
    # False = delete raw dumps/.ev/phantom.log once convert is verified (npz exist)
    keep_dumps: bool = False


@dataclass
class RenderFormValues:
    """Render-only inputs from the TUI's render form."""

    resolution: str = "1920x1080"
    samples: int = 500
    fps: int = 24
    max_frames: Optional[int] = None
    camera_mode: str = "auto"
    encode_video: bool = False
    video_fps: Optional[int] = None
    paths: Tuple[str, ...] = ("instance_grains",)
    envelope_method: str = "hull"
    placeholder_obj: str = DEFAULT_PLACEHOLDER_OBJ
    n_points: int = 1_000_000
    compare: bool = False  # stitch every rendered path side by side (needs >= 2 paths)


@dataclass
class RenderParams:
    """Fully resolved inputs to build_render_command() / run_render_stage()."""

    grains_dir: Path
    bodies_dir: Path
    output_dir: Path
    resolution: str = "1920x1080"
    samples: int = 500
    fps: int = 24
    max_frames: Optional[int] = None
    camera_mode: str = "auto"
    encode_video: bool = False
    video_fps: Optional[int] = None
    viz_path: str = "per_sphere"  # DEMHeadlessRender.py --viz-path
    manifest: Optional[Path] = None  # required unless viz_path == "per_sphere"


@dataclass
class PathResult:
    name: str
    ok: bool
    stage: str  # "preprocess" | "render" | "done"
    message: str
    n_frames: int = 0
    elapsed_s: float = 0.0


@dataclass
class PipelineResult:
    stage: str  # "sim" | "convert" | "done" | "partial" | "error"
    ok: bool
    message: str
    record: Optional[RunRecord] = None
    paths: List[PathResult] = field(default_factory=list)
    compare: Optional[CompareResult] = None


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
    sample = params.sample
    earth_sink_id = params.earth_sink_id
    flyby_bin = None
    if sample.body_source == "settled":
        cache_root = params.body_cache_dir or Path(params.output_root).parent / "settled_bodies"
        attach_settled_bodies(
            [sample], template_setup=base_setup,
            shape_file=Path(params.shape_file) if params.shape_file else resolve_default_shape_file(),
            settle_tdyn=params.settle_tdyn, relax_tdyn=params.relax_tdyn, cache_root=cache_root,
            phantomsetup_bin=phantomsetup_bin, phantom_bin=phantom_bin,
            phantom_dir=Path(params.phantom_dir), ephemeris_cache_dir=params.ephemeris_cache_dir,
            dry_run=params.dry_run)
        if not params.dry_run and sample.encounter == "hyperbola":
            import settled_body
            flyby_bin = settled_body.resolve_dem_tools(Path(params.phantom_dir)).flyby
    if sample.encounter == "hyperbola" and earth_sink_id == EARTH_SINK_ID_DEFAULT:
        earth_sink_id = 1  # phantomflyby: Earth is the only sink
    analysis_bin = resolve_phantom_executable(Path(params.phantom_dir), "phantomanalysis", must_exist=False)
    return run_one_case(
        run_id=1,
        sample=sample,
        base_setup=base_setup,
        base_input=base_input,
        output_root=params.output_root,
        prefix=params.prefix,
        phantomsetup_bin=phantomsetup_bin,
        phantom_bin=phantom_bin,
        ref_mass_kg=None,
        dry_run=params.dry_run,
        earth_sink_id=earth_sink_id,
        apophis_sink_id=params.apophis_sink_id,
        ephemeris_cache_dir=params.ephemeris_cache_dir,
        shape_file=params.shape_file,
        phantomflyby_bin=flyby_bin,
        phantomanalysis_bin=analysis_bin if analysis_bin.is_file() else None,
    )


class ConvertError(RuntimeError):
    pass


def _grains_and_bodies_dirs(run_dir: Path, base_output_dir: Path, output_root: Path):
    """DEMDumpConvert.py's own naming rule (CSVconvert/DEMDumpConvert.py:19-22):
    <sim_name> = INPUT_DIR's parent name, <run_name> = INPUT_DIR's own name.
    """
    sim_name = output_root.name
    run_name = run_dir.name
    grains_dir = base_output_dir / sim_name / f"{run_name}_grains_output"
    bodies_dir = base_output_dir / sim_name / f"{run_name}_bodies_output"
    return grains_dir, bodies_dir


def _render_output_dir(batch_dir: Path, run_name: str, path_name: str) -> Path:
    return batch_dir / f"{run_name}_render_{path_name}"


def _compare_output_dir(batch_dir: Path, run_name: str) -> Path:
    return batch_dir / f"{run_name}_compare"


def run_convert_stage(record: RunRecord, base_output_dir: Path) -> None:
    """Convert one run's dumps to bodies CSV + grains npz. Raises ConvertError on any failure."""
    if record.status != "ok":
        raise ConvertError(f"sim did not complete (status={record.status!r}); skipping convert")
    run_dir = Path(record.run_dir)
    min_grains = _min_dem_grains_from_setup(run_dir)
    try:
        run_dem_dump_convert(
            input_dir=run_dir,
            dump_convert_path=DEFAULT_DEM_DUMP_CONVERT,
            base_output_dir=base_output_dir,
            min_dem_grains=min_grains,
        )
    except Exception as exc:
        raise ConvertError(f"conversion failed: {exc}") from exc


class RenderError(RuntimeError):
    pass


class PreprocessError(RuntimeError):
    pass


def _output_tail(result: subprocess.CompletedProcess, n_lines: int = 40) -> str:
    # Blender / script print() output goes to stdout; an uncaught Python
    # traceback goes to stderr. A failure can land in either stream, so
    # combine both before taking the tail.
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return "\n".join(combined.splitlines()[-n_lines:])


def build_render_command(params: RenderParams) -> List[str]:
    cmd = [
        BLENDER_EXE, "--background", "--python", DEM_HEADLESS_RENDER, "--",
        "--viz-path", params.viz_path,
    ]
    if params.viz_path == "per_sphere":
        cmd += [
            "--grains-dir", to_windows_path(params.grains_dir),
            "--bodies-dir", to_windows_path(params.bodies_dir),
        ]
    else:
        cmd += ["--manifest", to_windows_path(params.manifest)]
    cmd += [
        "--output-dir", to_windows_path(params.output_dir),
        "--resolution", params.resolution,
        "--samples", str(params.samples),
        "--fps", str(params.fps),
        "--camera-mode", params.camera_mode,
    ]
    if params.max_frames is not None:
        cmd += ["--max-frames", str(params.max_frames)]
    if params.encode_video:
        cmd += ["--encode-video"]
        if params.video_fps is not None:
            cmd += ["--video-fps", str(params.video_fps)]
    return cmd


def run_render_stage(params: RenderParams) -> subprocess.CompletedProcess:
    cmd = build_render_command(params)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError as exc:
        raise RenderError(f"could not start {cmd[0]}: {exc}") from exc
    if result.returncode != 0:
        raise RenderError(f"blender exited {result.returncode}:\n{_output_tail(result)}")
    return result


@dataclass
class PathContext:
    """Everything a render path needs once sim + convert have succeeded."""

    grains_dir: Path
    bodies_dir: Path
    batch_dir: Path  # <base_output_dir>/<batch name>
    run_name: str  # e.g. "run_0001"
    render_form: RenderFormValues
    n_expected_frames: int
    # dump route: {path name: manifest Path | PreprocessError} from the one
    # shared viz_preprocess_dump.py call; those paths skip their own preprocess
    preprocessed: Optional[Dict[str, object]] = None


def _viz_dir(ctx: PathContext, spec: "RenderPath") -> Path:
    return ctx.batch_dir / f"{ctx.run_name}{spec.viz_dir_suffix}"


def _max_frames_args(ctx: PathContext) -> List[str]:
    mf = ctx.render_form.max_frames
    return ["--max-frames", str(mf)] if mf is not None else []


def build_composite_preprocess_command(ctx: PathContext, output_dir: Path) -> List[str]:
    return [
        str(WIN_VENV_PYTHON), to_windows_path(CODE_DIR / "viz" / "viz_preprocess.py"),
        "--grains-dir", to_windows_path(ctx.grains_dir),
        "--bodies-dir", to_windows_path(ctx.bodies_dir),
        "--output-dir", to_windows_path(output_dir),
        "--envelope-method", ctx.render_form.envelope_method,
    ] + _max_frames_args(ctx)


def build_instance_grains_preprocess_command(ctx: PathContext, output_dir: Path) -> List[str]:
    return [
        str(WIN_VENV_PYTHON),
        to_windows_path(CODE_DIR / "viz" / "viz_preprocess_grains_instance.py"),
        "--grains-dir", to_windows_path(ctx.grains_dir),
        "--bodies-dir", to_windows_path(ctx.bodies_dir),
        "--output-dir", to_windows_path(output_dir),
    ] + _max_frames_args(ctx)


def build_instance_static_preprocess_command(ctx: PathContext, output_dir: Path) -> List[str]:
    # viz_preprocess_lite.py needs mathutils.bvhtree -> Blender's Python.
    # One static cloud regardless of frame count, so no --max-frames.
    # --python-exit-code 1 makes an uncaught exception inside the script
    # exit non-zero -- without it, blender --background --python exits 0
    # even after a traceback, so run_preprocess_stage() would only ever see
    # "wrote no manifest" and never the real error.
    return [
        BLENDER_EXE, "--background", "--python-exit-code", "1", "--python",
        to_windows_path(CODE_DIR / "viz" / "viz_preprocess_lite.py"), "--",
        "--grains-dir", to_windows_path(ctx.grains_dir),
        "--bodies-dir", to_windows_path(ctx.bodies_dir),
        "--output-dir", to_windows_path(output_dir),
        "--shape-obj", to_windows_path(Path(ctx.render_form.placeholder_obj)),
        "--n-points", str(ctx.render_form.n_points),
    ]


def run_preprocess_stage(cmd: List[str], output_dir: Path) -> Path:
    """Run one preprocess subprocess from the Windows repo root; return its manifest path."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", cwd=str(CODE_DIR),
        )
    except OSError as exc:
        raise PreprocessError(f"could not start {cmd[0]}: {exc}") from exc
    if result.returncode != 0:
        raise PreprocessError(f"exited {result.returncode}:\n{_output_tail(result)}")
    manifest = output_dir / "manifest.json"
    if not manifest.is_file():
        # blender --background --python exits 0 even after an uncaught
        # exception unless --python-exit-code is set (see
        # build_instance_static_preprocess_command) -- for scripts that
        # don't set it, or that exit 0 for some other reason without
        # writing a manifest, surface the captured output here rather than
        # silently discarding the traceback.
        raise PreprocessError(f"wrote no manifest at {output_dir}:\n{_output_tail(result)}")
    return manifest


def build_dump_preprocess_command(ctx: PathContext, run_dir: Path, targets: Dict[str, Path]) -> List[str]:
    # resolve(): to_windows_path only converts absolute paths, and the Windows
    # python cannot see a relative WSL path
    cmd = [
        str(WIN_VENV_PYTHON), to_windows_path(CODE_DIR / "viz" / "viz_preprocess_dump.py"),
        "--run-dir", to_windows_path(Path(run_dir).resolve()),
        "--bodies-dir", to_windows_path(ctx.bodies_dir),
    ]
    for name, viz_dir in targets.items():
        cmd += [DUMP_PREPROCESS_FLAGS[name], to_windows_path(viz_dir)]
    if "composite" in targets:
        cmd += ["--envelope-method", ctx.render_form.envelope_method]
    return cmd + _max_frames_args(ctx)


def run_dump_preprocess_stage(cmd: List[str], targets: Dict[str, Path]) -> Dict[str, object]:
    """Run viz_preprocess_dump.py once; {path name: manifest Path | PreprocessError}."""
    for viz_dir in targets.values():
        # success is judged by the manifest existing afterwards
        (viz_dir / "manifest.json").unlink(missing_ok=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", cwd=str(CODE_DIR),
        )
    except OSError as exc:
        err = PreprocessError(f"could not start {cmd[0]}: {exc}")
        return {name: err for name in targets}
    failed = {}
    for line in (result.stdout or "").splitlines():
        if line.startswith("FAILED "):
            name, _, why = line[len("FAILED "):].partition(": ")
            failed[name] = why
    out: Dict[str, object] = {}
    for name, viz_dir in targets.items():
        manifest = viz_dir / "manifest.json"
        if name in failed:
            # checked first: finish() can die mid-write and leave a truncated manifest
            out[name] = PreprocessError(failed[name])
        elif manifest.is_file():
            out[name] = manifest
        else:
            out[name] = PreprocessError(f"exited {result.returncode}:\n{_output_tail(result)}")
    return out


@dataclass(frozen=True)
class RenderPath:
    name: str
    viz_dir_suffix: Optional[str]  # None = no preprocess stage
    build_preprocess_command: Optional[Callable[[PathContext, Path], List[str]]]
    headless_viz_path: str  # DEMHeadlessRender.py --viz-path


# Insertion order is the fixed run order.
RENDER_PATHS: Dict[str, RenderPath] = {
    "per_sphere": RenderPath("per_sphere", None, None, "per_sphere"),
    "composite": RenderPath("composite", "_viz", build_composite_preprocess_command, "composite"),
    "instance_grains": RenderPath(
        "instance_grains", "_viz_instance", build_instance_grains_preprocess_command, "instance",
    ),
    "instance_static": RenderPath(
        "instance_static", "_viz_instance_static", build_instance_static_preprocess_command, "instance",
    ),
}
RENDER_PATH_NAMES: Tuple[str, ...] = tuple(RENDER_PATHS)

# Paths that read the grains npz: per_sphere inside Blender (no sarracen
# there), instance_static via viz_preprocess_lite.py under blender.exe. With
# none of these ticked, convert is skipped and one viz_preprocess_dump.py call
# reads each PHANTOM dump once for composite and instance_grains together
# (Objective 2 scale).
NPZ_PATHS = frozenset({"per_sphere", "instance_static"})

# viz_preprocess_dump.py output-dir flag per dump-route path
DUMP_PREPROCESS_FLAGS: Dict[str, str] = {
    "composite": "--composite-dir",
    "instance_grains": "--instance-dir",
}


def needs_npz(paths) -> bool:
    return any(p in NPZ_PATHS for p in paths)

PATH_CHECKBOX_IDS: Dict[str, str] = {
    "per_sphere": "path-per-sphere",
    "composite": "path-composite",
    "instance_grains": "path-instance-grains",
    "instance_static": "path-instance-static",
}


def run_render_path(spec: RenderPath, ctx: PathContext, on_stage=None, index: int = 1, total: int = 1) -> PathResult:
    """[preprocess] -> render for one path. Never raises for stage failures."""
    t0 = time.monotonic()

    def notify(stage: str, **info) -> None:
        if on_stage is not None:
            on_stage(stage, path=spec.name, index=index, total=total, **info)

    manifest: Optional[Path] = None
    if ctx.preprocessed is not None and spec.name in ctx.preprocessed:
        got = ctx.preprocessed[spec.name]
        if isinstance(got, PreprocessError):
            return PathResult(spec.name, False, "preprocess", str(got), 0, time.monotonic() - t0)
        manifest = got
    elif spec.build_preprocess_command is not None:
        viz_dir = _viz_dir(ctx, spec)
        notify("preprocess", output_dir=viz_dir)
        try:
            manifest = run_preprocess_stage(spec.build_preprocess_command(ctx, viz_dir), viz_dir)
        except PreprocessError as exc:
            return PathResult(spec.name, False, "preprocess", str(exc), 0, time.monotonic() - t0)

    output_dir = _render_output_dir(ctx.batch_dir, ctx.run_name, spec.name)
    notify("render", n_expected=ctx.n_expected_frames, output_dir=output_dir)
    form = ctx.render_form
    params = RenderParams(
        grains_dir=ctx.grains_dir,
        bodies_dir=ctx.bodies_dir,
        output_dir=output_dir,
        resolution=form.resolution,
        samples=form.samples,
        fps=form.fps,
        max_frames=form.max_frames,
        camera_mode=form.camera_mode,
        encode_video=form.encode_video,
        video_fps=form.video_fps,
        viz_path=spec.headless_viz_path,
        manifest=manifest,
    )
    try:
        run_render_stage(params)
    except RenderError as exc:
        return PathResult(
            spec.name, False, "render", str(exc),
            count_matching(output_dir, "frame_*.png"), time.monotonic() - t0,
        )

    elapsed = time.monotonic() - t0
    n_frames = count_matching(output_dir, "frame_*.png")
    if n_frames == 0:
        return PathResult(spec.name, False, "render", f"render wrote no frames to {output_dir}", 0, elapsed)
    message = f"{n_frames} frame(s) in {elapsed:.0f}s to {output_dir}"
    if form.encode_video:
        video_path = output_dir / "animation.mp4"
        message += (
            f", encoded to {video_path}" if video_path.is_file()
            else f", video encode requested but {video_path} not found"
        )
    return PathResult(spec.name, True, "done", message, n_frames, elapsed)


def format_path_result(r: PathResult) -> str:
    if r.ok:
        return f"{r.name}: ok, {r.message}"
    # r.message can carry a multi-line traceback tail (see run_preprocess_stage /
    # PreprocessError) -- keep that in full for the caller (e.g. PathResult.message,
    # shown per-path), but compact it to one line here so a multi-path summary
    # (run_pipeline's "; ".join(...)) doesn't let one failure's traceback swallow
    # the other paths' results.
    lines = [line for line in r.message.splitlines() if line.strip()]
    first = lines[0] if lines else r.message
    if len(lines) > 1:
        return f"{r.name}: failed at {r.stage}: {first} … {lines[-1]}"
    return f"{r.name}: failed at {r.stage}: {first}"


def run_compare_for_paths(
    ctx: PathContext, results: List[PathResult], on_stage=None,
) -> Tuple[Optional[CompareResult], str]:
    """Stitch every path that rendered, side by side. Never raises for compare failures."""
    ok = [r for r in results if r.ok]
    if len(ok) < 2:
        return None, f"compare: skipped, only {len(ok)} path(s) rendered"
    output_dir = _compare_output_dir(ctx.batch_dir, ctx.run_name)
    if on_stage is not None:
        on_stage("compare", n_expected=ctx.n_expected_frames, output_dir=output_dir)
    form = ctx.render_form
    try:
        result = run_compare_stage(
            [(r.name, _render_output_dir(ctx.batch_dir, ctx.run_name, r.name)) for r in ok],
            output_dir,
            fps=form.video_fps or form.fps,
            encode_video=form.encode_video,
        )
    except CompareError as exc:
        return None, f"compare: failed: {exc}"
    except Exception as exc:  # e.g. OneDrive lock / full disk on unlink or mkdir
        return None, f"compare: failed: {type(exc).__name__}: {exc}"
    return result, format_compare_result(result)


def count_matching(directory: Optional[Path], pattern: str) -> int:
    """Return glob hits under directory, or 0 if it is missing / None."""
    if directory is None or not directory.is_dir():
        return 0
    return len(list(directory.glob(pattern)))


def format_live_progress(
    elapsed_s: float,
    stage: str,
    *,
    n_dumps: int = 0,
    n_npz: int = 0,
    n_png: int = 0,
    n_expected: Optional[int] = None,
    warning: str = "",
    path_label: Optional[str] = None,
) -> str:
    """Status-bar text for the live-progress tick. Pure — no widget I/O."""
    elapsed = f"{elapsed_s:.0f}s elapsed"
    label = f" [{path_label}]" if path_label else ""
    if stage == "convert":
        count = f"{n_npz}/{n_expected} npz" if n_expected is not None else f"{n_npz} npz"
        body = f"Converting... {elapsed}, {count}"
    elif stage == "preprocess":
        body = f"Preprocessing{label}... {elapsed}"
    elif stage == "preprocess_dumps":
        count = f", {n_png}/{n_expected} dump(s)" if n_expected is not None else ""
        body = f"Preprocessing dumps... {elapsed}{count}"
    elif stage == "render":
        count = (
            f"{n_png}/{n_expected} frame(s)"
            if n_expected is not None
            else f"{n_png} frame(s)"
        )
        body = f"Rendering{label}... {elapsed}, {count}"
    elif stage == "compare":
        count = (
            f"{n_png}/{n_expected} frame(s)"
            if n_expected is not None
            else f"{n_png} frame(s)"
        )
        body = f"Comparing... {elapsed}, {count}"
    else:
        body = f"Running sim... {elapsed}, {n_dumps} dump file(s)"
    return f"{warning}{body}"


def run_pipeline(
    sim_params: SimParams,
    render_form: RenderFormValues,
    base_output_dir: Path,
    on_stage=None,
) -> PipelineResult:
    """Sim -> [convert] -> each ticked render path ([preprocess] -> render) -> [compare].

    Convert runs only when a ticked path needs the grains npz (needs_npz);
    otherwise one viz_preprocess_dump.py call reads each dump once for every
    ticked path. Sim or convert failure stops everything. A render-path failure is recorded
    on that path and the remaining paths still run (they are independent).

    on_stage(stage: str, **info) is optional. The TUI uses it to flip the
    status bar between dump / npz / preprocess / frame counts.
    """
    if not render_form.paths:
        raise ValueError("at least one render path must be selected")
    unknown = [p for p in render_form.paths if p not in RENDER_PATHS]
    if unknown:
        raise ValueError(f"unknown render path(s): {', '.join(unknown)}")
    if render_form.compare and len(render_form.paths) < 2:
        raise ValueError("compare needs at least two render paths")

    def notify(stage: str, **info) -> None:
        if on_stage is not None:
            on_stage(stage, **info)

    notify("sim")
    record = run_sim_stage(sim_params)
    if record.status != "ok":
        return PipelineResult(
            stage="sim", ok=False,
            message=record.error or f"sim ended with status={record.status!r}",
            record=record,
        )

    run_dir = Path(record.run_dir)
    grains_dir, bodies_dir = _grains_and_bodies_dirs(run_dir, base_output_dir, sim_params.output_root)
    n_dumps = count_matching(run_dir, f"{sim_params.prefix}_[0-9]*")
    dump_route = not needs_npz(render_form.paths)

    if dump_route:
        if n_dumps == 0:
            return PipelineResult(
                stage="convert", ok=False,
                message=f"no {sim_params.prefix}_[0-9]* dumps in {run_dir} to preprocess",
                record=record,
            )
        # nfulldump=1 -> every dump is a full frame; a mini dump would make this an overestimate
        n_frames = n_dumps
        source_message = f"read {n_dumps} dump(s) directly (convert skipped)"
    else:
        notify("convert", n_expected=n_dumps if n_dumps else None)
        try:
            run_convert_stage(record, base_output_dir)
        except ConvertError as exc:
            return PipelineResult(stage="convert", ok=False, message=str(exc), record=record)

        # DEMDumpConvert.py never raises on a bad dump or an empty run -- it
        # prints and carries on -- so run_convert_stage() "succeeding" is not
        # proof any frame was actually converted. Catch that here rather than
        # let Blender fail minutes/hours later on "no grain npz files found".
        n_npz = count_matching(grains_dir, "*.npz")
        if n_npz == 0:
            return PipelineResult(
                stage="convert", ok=False,
                message=(
                    f"no grain npz files found in {grains_dir} -- DEMDumpConvert.py "
                    "ran without raising but converted nothing; check that the run "
                    f"produced dumps matching its hardcoded \"sobol\" prefix "
                    "(sobol_[0-9]*) with enough DEM grains"
                ),
                record=record,
            )
        if not sim_params.keep_dumps:
            # renders read the converted npz/CSV, never the raw dumps
            _cleanup_run_dir(run_dir, sim_params.prefix)
        n_frames = n_npz
        source_message = f"converted {n_npz} frame(s)"

    n_expected_frames = n_frames
    if render_form.max_frames is not None:
        n_expected_frames = min(n_frames, render_form.max_frames)

    ctx = PathContext(
        grains_dir=grains_dir,
        bodies_dir=bodies_dir,
        batch_dir=base_output_dir / sim_params.output_root.name,
        run_name=run_dir.name,
        render_form=render_form,
        n_expected_frames=n_expected_frames,
    )
    specs = [RENDER_PATHS[name] for name in RENDER_PATH_NAMES if name in render_form.paths]
    if dump_route:
        # every path here is composite and/or instance_grains (needs_npz is False)
        targets = {spec.name: _viz_dir(ctx, spec) for spec in specs}
        # composite writes one camera grains/<frame>.npz per dump read
        cam_dir = targets["composite"] / "grains" if "composite" in targets else None
        notify("preprocess_dumps", output_dir=cam_dir,
               n_expected=n_expected_frames if cam_dir is not None else None)
        ctx.preprocessed = run_dump_preprocess_stage(
            build_dump_preprocess_command(ctx, run_dir, targets), targets,
        )
    results = [
        run_render_path(spec, ctx, on_stage=on_stage, index=i, total=len(specs))
        for i, spec in enumerate(specs, start=1)
    ]
    all_ok = all(r.ok for r in results)
    message = source_message + "; " + "; ".join(format_path_result(r) for r in results)
    if dump_route and not sim_params.keep_dumps:
        # the dumps are this route's only grain source: keep them while any
        # path still needs them for a retry (every path failed, or one failed
        # before writing its assets); a render-stage failure has its assets
        pre_failed = [r.name for r in results if not r.ok and r.stage == "preprocess"]
        if not any(r.ok for r in results):
            message += f"; dumps kept in {run_dir} (every render path failed)"
        elif pre_failed:
            message += f"; dumps kept in {run_dir} ({', '.join(pre_failed)} failed at preprocess)"
        else:
            _cleanup_run_dir(run_dir, sim_params.prefix)
    compare = None
    if render_form.compare:
        compare, compare_message = run_compare_for_paths(ctx, results, on_stage=on_stage)
        message += "; " + compare_message
        all_ok = all_ok and compare is not None
    return PipelineResult(
        stage="done" if all_ok else "partial",
        ok=all_ok,
        message=message,
        record=record,
        paths=results,
        compare=compare,
    )


_PER_SPHERE_GRAIN_WARN_THRESHOLD = 2000


def warn_if_high_grain_count(np_apophis: int) -> Optional[str]:
    """Non-blocking warning: the per-sphere path degrades above ~2000 grains
    (see CLAUDE.md's pipeline-comparison table)."""
    if np_apophis <= _PER_SPHERE_GRAIN_WARN_THRESHOLD:
        return None
    return (
        f"Warning: np_apophis={np_apophis} — per-sphere path is intended for "
        f"≤{_PER_SPHERE_GRAIN_WARN_THRESHOLD} grains; consider the composite or "
        "instance_grains path instead. "
    )


def pipeline_warning(np_apophis: int, paths) -> str:
    """Grain-count warning, only when the per-sphere path is ticked."""
    if "per_sphere" not in paths:
        return ""
    return warn_if_high_grain_count(np_apophis) or ""


class _Row(Horizontal):
    """Label + interactive widget on a single row. Same layout as tui_run.py's _Row."""

    DEFAULT_CSS = """
    _Row { height: 3; }
    _Row .rl { width: 26; content-align: right middle; padding-right: 1; color: $text-muted; }
    _Row .rw { width: 1fr; }
    _Row .rh { width: 15; content-align: left middle; padding-left: 1; color: $text-muted; text-style: italic; }
    """

    def __init__(self, label: str, widget, hint: str = "", **kw) -> None:
        super().__init__(**kw)
        self._label = label
        self._widget = widget
        self._hint = hint

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="rl")
        self._widget.add_class("rw")
        yield self._widget
        if self._hint:
            yield Static(self._hint, classes="rh")


class SimRenderTUIApp(App[None]):
    """Configure one PHANTOM DEM run -> run -> convert -> render, in one screen."""

    TITLE = "PHANTOM · Sim + Render"
    SUB_TITLE = "configure -> run -> convert -> render"

    DEFAULT_CSS = """
    Screen { background: $surface; }
    #scroll { height: 1fr; border: solid $primary; margin: 0 1; padding: 0 2; }
    .sec { color: $accent; text-style: bold; padding: 1 0 0 0; }
    #bar { height: 5; padding: 1 2; background: $surface-darken-1; border-top: solid $primary; align: center middle; }
    #status { width: 1fr; content-align: left middle; padding-left: 2; }
    .err { color: $error; }
    .ok  { color: $success; }
    Button { margin: 0 1; }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Live-progress state (spec Component 1 "Live progress") -- set at
        # launch time, read by the set_interval tick and by the final report.
        self._progress_timer: Optional[Timer] = None
        self._pipeline_start: Optional[float] = None
        self._pipeline_run_dir: Optional[Path] = None
        self._pipeline_grains_dir: Optional[Path] = None
        self._pipeline_render_dir: Optional[Path] = None
        self._pipeline_png_glob: str = "frame_*.png"  # "compare_*.png" during compare
        self._pipeline_prefix: str = ""
        self._pipeline_warning: str = ""
        self._pipeline_stage: str = "sim"
        self._pipeline_n_expected: Optional[int] = None
        self._pipeline_dry_run: bool = False
        self._pipeline_path_label: Optional[str] = None
        # This TUI runs one pipeline at a time -- @work(thread=True) defaults
        # to exclusive=False, so without this guard a second click while a
        # worker is in flight would start a second worker that overwrites
        # the _pipeline_* state above out from under the first one. Named
        # _pipeline_running (not _running) because App itself already uses a
        # private _running attribute internally to back App.is_running --
        # shadowing that broke the guard (it read as already-True as soon as
        # the app's run loop started, before any click).
        self._pipeline_running: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="scroll"):
            yield Static("Sim — paths", classes="sec")
            yield _Row("prefix", Input("sobol", id="prefix"))
            yield _Row(
                "batch_label", Input("sim_render", id="batch-label"),
                "batch dir suffix",
            )
            yield _Row("base_dir", Input(str(_default_base_dir()), id="base-dir"), "dir")
            yield _Row("phantom_dir", Input(_default_phantom_dir(), id="phantom-dir"), "dir")
            yield _Row(
                "output_root", Input(str(_default_output_root()), id="output-root"), "parent dir",
            )
            yield _Row(
                "ephemeris_cache_dir",
                Input("", id="eph-cache", placeholder="(none — skips Horizons download)"),
                "optional dir",
            )

            yield Static("Sim — parameters", classes="sec")
            yield _Row("np_apophis", Input("500", id="np-apophis"), "int")
            yield _Row("spin_period (hr)", Input("", id="spin-period", placeholder="(template default)"), "hr")
            yield _Row("spin_torque_align (deg)", Input("", id="spin-torque-align", placeholder="0-180"), "deg")
            yield _Row(
                "kt_cgs", Input("", id="kt-cgs", placeholder="(template default)"), "dyne/cm",
            )
            yield _Row("dn_cohes_factor", Input(str(DEFAULT_DN_COHES_FACTOR), id="dn-cohes-factor"), "float")
            yield _Row("tmax (hr)", Input(DEFAULT_TMAX_HOURS, id="tmax-hours"), "hr")
            yield _Row("dtmax (hr)", Input(DEFAULT_DTMAX_HOURS, id="dtmax-hours"), "hr")
            yield _Row("sink_earth_id", Input(str(EARTH_SINK_ID_DEFAULT), id="sink-earth"), "int")
            yield _Row("sink_apophis_id", Input(str(APOPHIS_SINK_ID_DEFAULT), id="sink-apophis"), "int")
            yield _Row(
                "shape_file",
                Input("", id="shape-file", placeholder="(none — no shape crop)"),
                "optional path",
            )
            yield Static("Sim — body / encounter", classes="sec")
            yield _Row(
                "body_source",
                Select([("lattice", "lattice"), ("settled", "settled")], value="lattice",
                       id="body-source", allow_blank=False),
                "settled = settle → crop → relax (cached)",
            )
            yield _Row(
                "encounter",
                Select([("ephemeris", "ephemeris"), ("hyperbola", "hyperbola")], value="ephemeris",
                       id="encounter", allow_blank=False),
                "hyperbola needs settled; ephemeris + settled uses packing_file",
            )
            yield _Row("flyby_rp_km", Input(DEFAULT_FLYBY_RP_KM, id="flyby-rp"), "km")
            yield _Row("flyby_vinf_kms", Input(DEFAULT_FLYBY_VINF_KMS, id="flyby-vinf"), "km/s")
            yield _Row("flyby_start_sep_km", Input("4e5", id="flyby-sep"), "km")
            yield _Row(
                "keep_dumps",
                Checkbox("keep raw dumps + .ev after convert", value=False, id="keep-dumps"),
                "default: delete",
            )

            yield Static("Render", classes="sec")
            yield _Row("resolution", Input("1920x1080", id="resolution"), "WxH")
            yield _Row("samples", Input("500", id="samples"), "int")
            yield _Row("fps", Input("24", id="fps"), "int")
            yield _Row("max_frames", Input("", id="max-frames", placeholder="(all frames)"), "int, optional")
            yield _Row(
                "camera_mode",
                Select(
                    [("auto", "auto"), ("grain_only", "grain_only")],
                    value="auto", id="camera-mode", allow_blank=False,
                ),
            )
            yield _Row(
                "encode_video",
                Checkbox("encode to MP4 after render", value=False, id="encode-video"),
                "optional final stage",
            )
            yield _Row(
                "video_fps", Input("", id="video-fps", placeholder="(same as fps)"), "int, optional",
            )

            yield Static("Render — paths (any combination; one sim feeds all)", classes="sec")
            yield _Row(
                "per_sphere",
                Checkbox("one sphere per grain", value=False, id=PATH_CHECKBOX_IDS["per_sphere"]),
                "≤~2000 grains, legacy",
            )
            yield _Row(
                "composite",
                Checkbox("envelope mesh + ejecta (D3)", value=False, id=PATH_CHECKBOX_IDS["composite"]),
                "any N",
            )
            yield _Row(
                "instance_grains",
                Checkbox("real grains, GN-instanced", value=True, id=PATH_CHECKBOX_IDS["instance_grains"]),
                "any N",
            )
            yield _Row(
                "instance_static",
                Checkbox("shape-model placeholder cloud", value=False, id=PATH_CHECKBOX_IDS["instance_static"]),
                "not sim motion",
            )
            yield _Row(
                "compare",
                Checkbox("side-by-side of ticked paths", value=False, id="compare"),
                "needs ≥2 paths",
            )
            yield _Row(
                "envelope_method",
                Select(
                    [("hull", "hull"), ("sdf", "sdf")],
                    value="hull", id="envelope-method", allow_blank=False,
                ),
                "composite",
            )
            yield _Row(
                "placeholder_obj", Input(DEFAULT_PLACEHOLDER_OBJ, id="placeholder-obj"), "instance_static",
            )
            yield _Row("n_points", Input("1000000", id="n-points"), "instance_static")

        with Horizontal(id="bar"):
            yield Button("Run Pipeline", id="btn-run", variant="success")
            yield Button("Dry Run", id="btn-dryrun", variant="warning")
            yield Button("Quit", id="btn-quit", variant="error")
            yield Static("Ready.", id="status")
        yield Footer()

    def on_mount(self) -> None:
        try:
            check_blender_exe()
        except FileNotFoundError as exc:
            self._set_status(str(exc), error=True)

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        s = self.query_one("#status", Static)
        s.update(msg)
        s.remove_class("err", "ok")
        s.add_class("err" if error else "ok")

    def _iv(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def _build_sim_params(self, dry_run: bool) -> SimParams:
        np_apophis = int(self._iv("np-apophis"))
        dn = float(self._iv("dn-cohes-factor") or DEFAULT_DN_COHES_FACTOR)
        kt_cgs_raw = self._iv("kt-cgs")
        kt_cgs = float(kt_cgs_raw) if kt_cgs_raw else None
        # cohesion gap = dn * grain diameter, resolved after phantomsetup (run_one_case)
        dn_cohes_factor = dn if kt_cgs is not None and kt_cgs > 0 else None
        shape_file_raw = self._iv("shape-file")
        body_source = self.query_one("#body-source", Select).value
        encounter = self.query_one("#encounter", Select).value
        hyper = encounter == "hyperbola"
        sample = RunSample(
            use_dem=True,
            dem_model="particle",
            np_apophis=np_apophis,
            apophis_spin_period=float(self._iv("spin-period")) if self._iv("spin-period") else None,
            apophis_spin_torque_align_deg=(
                float(self._iv("spin-torque-align")) if self._iv("spin-torque-align") else None
            ),
            kt_cgs=kt_cgs,
            dn_cohes_factor=dn_cohes_factor,
            tmax_hours=float(self._iv("tmax-hours")) if self._iv("tmax-hours") else None,
            dtmax_hours=float(self._iv("dtmax-hours")) if self._iv("dtmax-hours") else None,
            use_shape_crop=True if shape_file_raw and body_source != "settled" else None,
            body_source=body_source,
            encounter=encounter,
            flyby_rp_km=float(self._iv("flyby-rp")) if hyper and self._iv("flyby-rp") else None,
            flyby_vinf_kms=float(self._iv("flyby-vinf")) if hyper and self._iv("flyby-vinf") else None,
            flyby_start_sep_km=float(self._iv("flyby-sep") or 4e5) if hyper else None,
            flyby_perturber_earth_masses=1.0 if hyper else None,
        )
        if encounter == "hyperbola":
            if body_source != "settled":
                raise ValueError("hyperbola needs body_source = settled")
            if sample.flyby_rp_km is None or sample.flyby_vinf_kms is None:
                raise ValueError("hyperbola needs flyby_rp_km and flyby_vinf_kms")
            if sample.tmax_hours is None or sample.dtmax_hours is None:
                raise ValueError("hyperbola needs tmax (hr) and dtmax (hr)")
            if sample.apophis_spin_period is not None or sample.apophis_spin_torque_align_deg is not None:
                raise ValueError("hyperbola has no spin; clear spin_period and spin_torque_align")
            import encounter as enc
            enc.check_flyby_geometry(
                sample.flyby_rp_km, sample.flyby_rp_km, sample.flyby_vinf_kms, sample.flyby_vinf_kms,
                sample.flyby_start_sep_km, sample.flyby_perturber_earth_masses, sample.tmax_hours)
        eph_cache_raw = self._iv("eph-cache")
        prefix = self._iv("prefix")
        # <sim_name> (spec Component 2) is a timestamped batch dir the TUI
        # creates here, not the bare "output_root" field -- that field is the
        # *parent* directory the batch dir gets created under, same
        # <prefix>_<timestamp>_<slug> convention as run_mass_sobol_phantom.py's
        # build_batch_directory_basename(). A blank batch_label is invalid --
        # sanitize_batch_label() raises ValueError, caught in _launch().
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_slug = sanitize_batch_label(self._iv("batch-label"))
        output_root = (
            Path(self._iv("output-root")).resolve() / f"{prefix}_{timestamp}_{label_slug}"
        )
        return SimParams(
            prefix=prefix,
            base_dir=Path(self._iv("base-dir")).resolve(),
            phantom_dir=Path(self._iv("phantom-dir")).resolve(),
            output_root=output_root,
            ephemeris_cache_dir=Path(eph_cache_raw) if eph_cache_raw else None,
            dry_run=dry_run,
            earth_sink_id=int(self._iv("sink-earth")),
            apophis_sink_id=int(self._iv("sink-apophis")),
            shape_file=shape_file_raw or None,
            sample=sample,
            keep_dumps=self.query_one("#keep-dumps", Checkbox).value,
        )

    def _build_render_form(self) -> RenderFormValues:
        # Validated eagerly (review fix, Minor #4) so a typo is caught in the
        # status bar immediately rather than surfacing as a Blender failure
        # possibly hours later once sim + convert have already run.
        resolution = self._iv("resolution")
        m = _RESOLUTION_RE.match(resolution)
        if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
            raise ValueError(
                f"resolution must look like WIDTHxHEIGHT with both > 0 (got {resolution!r})"
            )
        samples = int(self._iv("samples"))
        if samples <= 0:
            raise ValueError(f"samples must be > 0 (got {samples})")
        fps = int(self._iv("fps"))
        if fps <= 0:
            raise ValueError(f"fps must be > 0 (got {fps})")
        max_frames_raw = self._iv("max-frames")
        max_frames = int(max_frames_raw) if max_frames_raw else None
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be > 0 when set (got {max_frames})")
        camera_mode = self.query_one("#camera-mode", Select).value
        encode_video = self.query_one("#encode-video", Checkbox).value
        video_fps_raw = self._iv("video-fps")
        video_fps = int(video_fps_raw) if video_fps_raw else None
        if video_fps is not None and video_fps <= 0:
            raise ValueError(f"video_fps must be > 0 when set (got {video_fps})")
        paths = tuple(
            name for name in RENDER_PATH_NAMES
            if self.query_one(f"#{PATH_CHECKBOX_IDS[name]}", Checkbox).value
        )
        if not paths:
            raise ValueError("tick at least one render path")
        compare = bool(self.query_one("#compare", Checkbox).value)
        if compare and len(paths) < 2:
            raise ValueError("compare needs at least two ticked render paths")
        envelope_method = str(self.query_one("#envelope-method", Select).value)
        placeholder_obj = self._iv("placeholder-obj")
        n_points = 1_000_000
        # Path-specific fields are only validated when their path is ticked.
        if "instance_static" in paths:
            n_points = int(self._iv("n-points"))
            if n_points <= 0:
                raise ValueError(f"n_points must be > 0 (got {n_points})")
            if not from_windows_path(placeholder_obj).is_absolute():
                raise ValueError(
                    f"placeholder_obj must be an absolute path (got {placeholder_obj!r})"
                )
            if not placeholder_obj or not from_windows_path(placeholder_obj).is_file():
                raise ValueError(f"placeholder_obj not found: {placeholder_obj!r}")
        if {"composite", "instance_grains"} & set(paths) and not Path(WIN_VENV_PYTHON).is_file():
            raise ValueError(
                f"Windows venv python not found at {WIN_VENV_PYTHON} -- needed by the "
                "composite / instance_grains preprocess"
            )
        return RenderFormValues(
            resolution=resolution,
            samples=samples,
            fps=fps,
            max_frames=max_frames,
            camera_mode=str(camera_mode),
            encode_video=bool(encode_video),
            video_fps=video_fps,
            paths=paths,
            envelope_method=envelope_method,
            placeholder_obj=placeholder_obj,
            n_points=n_points,
            compare=compare,
        )

    @on(Button.Pressed, "#btn-run")
    def _on_run(self) -> None:
        self._launch(dry_run=False)

    @on(Button.Pressed, "#btn-dryrun")
    def _on_dryrun(self) -> None:
        self._launch(dry_run=True)

    @on(Button.Pressed, "#btn-quit")
    def _on_quit(self) -> None:
        self.exit(None)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#btn-run", Button).disabled = disabled
        self.query_one("#btn-dryrun", Button).disabled = disabled

    def _launch(self, dry_run: bool) -> None:
        if self._pipeline_running:
            self._set_status("A pipeline is already running.", error=True)
            return
        try:
            sim_params = self._build_sim_params(dry_run)
            render_form = self._build_render_form()
        except ValueError as exc:
            self._set_status(f"Error: {exc}", error=True)
            return
        warning = pipeline_warning(sim_params.sample.np_apophis, render_form.paths)
        self._pipeline_running = True
        self._set_buttons_disabled(True)
        self._pipeline_start = time.monotonic()
        self._pipeline_run_dir = sim_params.output_root / "run_0001"
        demcsvs = DATA_DIR / "DEMCSVs"
        grains_dir, _bodies_dir = _grains_and_bodies_dirs(
            self._pipeline_run_dir, demcsvs, sim_params.output_root,
        )
        self._pipeline_grains_dir = grains_dir
        # Set per path by the "render" stage callback (Task 6).
        self._pipeline_render_dir = None
        self._pipeline_png_glob = "frame_*.png"
        self._pipeline_prefix = sim_params.prefix
        self._pipeline_warning = warning
        self._pipeline_stage = "sim"
        self._pipeline_n_expected = None
        self._pipeline_dry_run = dry_run
        self._pipeline_path_label = None
        if dry_run:
            self._set_status(f"{warning}Dry run...")
        else:
            self._tick_progress()
        if self._progress_timer is not None:
            self._progress_timer.stop()
        self._progress_timer = self.set_interval(2.0, self._tick_progress)
        self._run_pipeline_worker(sim_params, render_form)

    def _set_pipeline_stage(self, stage: str, info: Optional[dict] = None) -> None:
        """Called from the worker via call_from_thread when a stage starts."""
        self._pipeline_stage = stage
        info = info or {}
        if "n_expected" in info:
            self._pipeline_n_expected = info["n_expected"]
        if "path" in info:
            self._pipeline_path_label = f"{info['path']} {info['index']}/{info['total']}"
        if stage == "preprocess_dumps":
            # composite's camera files, one per dump read (None: instance_grains only)
            self._pipeline_render_dir = info.get("output_dir")
            self._pipeline_png_glob = "*.npz"
            self._pipeline_path_label = None
        if stage == "render" and "output_dir" in info:
            self._pipeline_render_dir = info["output_dir"]
            self._pipeline_png_glob = "frame_*.png"
        if stage == "compare":
            self._pipeline_render_dir = info.get("output_dir")
            self._pipeline_png_glob = "compare_*.png"
            self._pipeline_path_label = None
        if self._pipeline_running and not self._pipeline_dry_run:
            self._tick_progress()

    def _tick_progress(self) -> None:
        """Coarse live progress while the worker thread runs.

        Stage is set by run_pipeline's on_stage callback. Sim counts dumps;
        convert counts npz; render counts frame_*.png (compare: compare_*.png) so the bar does not
        freeze on the last dump after PHANTOM exits.
        """
        if self._pipeline_start is None:
            return
        if self._pipeline_dry_run:
            elapsed = time.monotonic() - self._pipeline_start
            self._set_status(f"{self._pipeline_warning}Dry run... {elapsed:.0f}s elapsed")
            return
        elapsed = time.monotonic() - self._pipeline_start
        n_dumps = count_matching(
            self._pipeline_run_dir, f"{self._pipeline_prefix}_[0-9]*",
        )
        n_npz = count_matching(self._pipeline_grains_dir, "*.npz")
        n_png = count_matching(self._pipeline_render_dir, self._pipeline_png_glob)
        self._set_status(
            format_live_progress(
                elapsed,
                self._pipeline_stage,
                n_dumps=n_dumps,
                n_npz=n_npz,
                n_png=n_png,
                n_expected=self._pipeline_n_expected,
                warning=self._pipeline_warning,
                path_label=self._pipeline_path_label,
            )
        )

    def _stop_progress_timer(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer = None

    @work(thread=True)
    def _run_pipeline_worker(self, sim_params: SimParams, render_form: RenderFormValues) -> None:
        def on_stage(stage: str, **info) -> None:
            self.call_from_thread(self._set_pipeline_stage, stage, info)

        try:
            result = run_pipeline(
                sim_params, render_form, DATA_DIR / "DEMCSVs", on_stage=on_stage,
            )
        except Exception as exc:  # an uncaught exception here must never kill the app
            result = PipelineResult(
                stage="error", ok=False, message=f"{type(exc).__name__}: {exc}",
            )
        self.call_from_thread(self._report_result, result)

    def _report_result(self, result: PipelineResult) -> None:
        self._stop_progress_timer()
        # Re-enable on every path -- including the worker-exception
        # (stage="error") path -- so a failed run never leaves the app stuck
        # unable to launch another one.
        self._pipeline_running = False
        self._set_buttons_disabled(False)
        elapsed = (
            time.monotonic() - self._pipeline_start if self._pipeline_start is not None else 0.0
        )
        time_prefix = f"[{elapsed:.0f}s] "
        if (
            result.stage == "sim"
            and result.record is not None
            and result.record.status == "prepared_only"
        ):
            # Spec's error-handling table: prepared_only is a dry run, not a
            # failure -- component 2/3 correctly never ran, so this reads as
            # success-shaped, not an error.
            self._set_status(
                f"{time_prefix}Dry run complete — prepared {result.record.run_dir}; "
                "nothing to convert/render.",
                error=False,
            )
            return
        if result.ok:
            self._set_status(f"{time_prefix}Done ({result.stage}): {result.message}", error=False)
        elif result.stage == "partial":
            self._set_status(f"{time_prefix}Finished with failures: {result.message}", error=True)
        else:
            self._set_status(f"{time_prefix}Failed at {result.stage}: {result.message}", error=True)


def main() -> None:
    SimRenderTUIApp().run()


if __name__ == "__main__":
    main()
