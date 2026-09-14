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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Analysis"))

from run_mass_sobol_phantom import (  # noqa: E402
    RunSample,
    RunRecord,
    EARTH_SINK_ID_DEFAULT,
    APOPHIS_SINK_ID_DEFAULT,
    DEFAULT_DN_COHES_FACTOR,
    coh_gap_max_cgs_from_dn,
    preflight,
    run_one_case,
    sanitize_batch_label,
)
from run_demtocsv_batch import (  # noqa: E402
    run_dem_dump_convert,
    DEFAULT_DEM_DUMP_CONVERT,
    _min_dem_grains_from_setup,
)

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, ScrollableContainer
    from textual.timer import Timer
    from textual.widgets import Button, Footer, Header, Input, Select, Static
except ImportError:
    sys.exit(
        "textual is not installed.\n"
        "Run:  pip install textual\n"
        "(sobol/tui_run.py already depends on it, so it should already be present.)"
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


def check_blender_exe(path: str = BLENDER_EXE) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"blender.exe not found at {path} -- confirm the Blender 5.2.1 LTS "
            "install path (see CLAUDE.md) or update BLENDER_EXE in tui_sim_render.py."
        )


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


def _render_output_dir(run_dir: Path, base_output_dir: Path, output_root: Path) -> Path:
    sim_name = output_root.name
    run_name = run_dir.name
    return base_output_dir / sim_name / f"{run_name}_render"


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


def build_render_command(params: RenderParams) -> List[str]:
    cmd = [
        BLENDER_EXE, "--background", "--python", DEM_HEADLESS_RENDER, "--",
        "--grains-dir", to_windows_path(params.grains_dir),
        "--bodies-dir", to_windows_path(params.bodies_dir),
        "--output-dir", to_windows_path(params.output_dir),
        "--resolution", params.resolution,
        "--samples", str(params.samples),
        "--fps", str(params.fps),
        "--camera-mode", params.camera_mode,
    ]
    if params.max_frames is not None:
        cmd += ["--max-frames", str(params.max_frames)]
    return cmd


def run_render_stage(params: RenderParams) -> subprocess.CompletedProcess:
    cmd = build_render_command(params)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Blender's own print() output (including everything
        # DEMGrainsBlenderEarthCam.py prints while exec()'d) goes to stdout;
        # an uncaught Python traceback goes to stderr. A failure can land in
        # either stream, so combine both before taking the tail.
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        tail = "\n".join(combined.splitlines()[-40:])
        raise RenderError(f"blender exited {result.returncode}:\n{tail}")
    return result


def run_pipeline(sim_params: SimParams, render_form: RenderFormValues, base_output_dir: Path) -> PipelineResult:
    """Sim -> convert -> render, stopping at the first failed stage."""
    record = run_sim_stage(sim_params)
    if record.status != "ok":
        return PipelineResult(
            stage="sim", ok=False,
            message=record.error or f"sim ended with status={record.status!r}",
            record=record,
        )

    try:
        run_convert_stage(record, base_output_dir)
    except ConvertError as exc:
        return PipelineResult(stage="convert", ok=False, message=str(exc), record=record)

    run_dir = Path(record.run_dir)
    grains_dir, bodies_dir = _grains_and_bodies_dirs(run_dir, base_output_dir, sim_params.output_root)
    output_dir = _render_output_dir(run_dir, base_output_dir, sim_params.output_root)
    render_params = RenderParams(
        grains_dir=grains_dir,
        bodies_dir=bodies_dir,
        output_dir=output_dir,
        resolution=render_form.resolution,
        samples=render_form.samples,
        fps=render_form.fps,
        max_frames=render_form.max_frames,
        camera_mode=render_form.camera_mode,
    )
    try:
        run_render_stage(render_params)
    except RenderError as exc:
        return PipelineResult(stage="render", ok=False, message=str(exc), record=record)

    n_frames = len(list(output_dir.glob("frame_*.png")))
    return PipelineResult(
        stage="done", ok=True,
        message=f"rendered {n_frames} frame(s) to {output_dir}",
        record=record,
    )


_PER_SPHERE_GRAIN_WARN_THRESHOLD = 2000


def warn_if_high_grain_count(np_apophis: int) -> Optional[str]:
    """Non-blocking warning: this pipeline is per-sphere only (see CLAUDE.md's
    pipeline-comparison table) -- composite/instancing exist precisely
    because per-sphere degrades above ~2000 grains.
    """
    if np_apophis <= _PER_SPHERE_GRAIN_WARN_THRESHOLD:
        return None
    return (
        f"Warning: np_apophis={np_apophis} — per-sphere path is intended for "
        f"≤{_PER_SPHERE_GRAIN_WARN_THRESHOLD} grains; consider the composite or "
        "instancing path instead. "
    )


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
        self._pipeline_prefix: str = ""
        self._pipeline_warning: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="scroll"):
            yield Static("Sim — paths", classes="sec")
            yield _Row("prefix", Input("sobol", id="prefix"))
            yield _Row(
                "batch_label", Input("sim_render", id="batch-label"),
                "batch dir suffix",
            )
            yield _Row("base_dir", Input(".", id="base-dir"), "dir")
            yield _Row("phantom_dir", Input(".", id="phantom-dir"), "dir")
            yield _Row("output_root", Input("sobol_mass_runs", id="output-root"), "parent dir")
            yield _Row(
                "ephemeris_cache_dir",
                Input("", id="eph-cache", placeholder="(none — skips Horizons download)"),
                "optional dir",
            )

            yield Static("Sim — parameters", classes="sec")
            yield _Row("np_apophis", Input("500", id="np-apophis"), "int")
            yield _Row("spin_period (hr)", Input("", id="spin-period", placeholder="(template default)"), "hr")
            yield _Row("spin_torque_align (deg)", Input("", id="spin-torque-align", placeholder="0-180"), "deg")
            yield _Row("kt_cgs", Input("0", id="kt-cgs"), "dyne/cm")
            yield _Row("dn_cohes_factor", Input(str(DEFAULT_DN_COHES_FACTOR), id="dn-cohes-factor"), "float")
            yield _Row("tmax (hr)", Input("", id="tmax-hours", placeholder="(template default)"), "hr")
            yield _Row("dtmax (hr)", Input("", id="dtmax-hours", placeholder="(template default)"), "hr")
            yield _Row("sink_earth_id", Input(str(EARTH_SINK_ID_DEFAULT), id="sink-earth"), "int")
            yield _Row("sink_apophis_id", Input(str(APOPHIS_SINK_ID_DEFAULT), id="sink-apophis"), "int")
            yield _Row(
                "shape_file",
                Input("", id="shape-file", placeholder="(none — no shape crop)"),
                "optional path",
            )

            yield Static("Render", classes="sec")
            yield _Row("resolution", Input("1920x1080", id="resolution"), "WxH")
            yield _Row("samples", Input("128", id="samples"), "int")
            yield _Row("fps", Input("24", id="fps"), "int")
            yield _Row("max_frames", Input("", id="max-frames", placeholder="(all frames)"), "int, optional")
            yield _Row(
                "camera_mode",
                Select([("auto", "auto"), ("grain_only", "grain_only")], value="auto", id="camera-mode"),
            )

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
        kt_cgs = float(self._iv("kt-cgs") or "0")
        coh_gap_max_cgs = (
            coh_gap_max_cgs_from_dn(dn=dn, np_apophis=np_apophis) if kt_cgs > 0 else None
        )
        shape_file_raw = self._iv("shape-file")
        sample = RunSample(
            use_dem=True,
            np_apophis=np_apophis,
            apophis_spin_period=float(self._iv("spin-period")) if self._iv("spin-period") else None,
            apophis_spin_torque_align_deg=(
                float(self._iv("spin-torque-align")) if self._iv("spin-torque-align") else None
            ),
            kt_cgs=kt_cgs,
            coh_gap_max_cgs=coh_gap_max_cgs,
            tmax_hours=float(self._iv("tmax-hours")) if self._iv("tmax-hours") else None,
            dtmax_hours=float(self._iv("dtmax-hours")) if self._iv("dtmax-hours") else None,
            use_shape_crop=True if shape_file_raw else None,
        )
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
        )

    def _build_render_form(self) -> RenderFormValues:
        max_frames_raw = self._iv("max-frames")
        camera_mode = self.query_one("#camera-mode", Select).value
        return RenderFormValues(
            resolution=self._iv("resolution"),
            samples=int(self._iv("samples")),
            fps=int(self._iv("fps")),
            max_frames=int(max_frames_raw) if max_frames_raw else None,
            camera_mode=str(camera_mode),
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

    def _launch(self, dry_run: bool) -> None:
        try:
            sim_params = self._build_sim_params(dry_run)
            render_form = self._build_render_form()
        except ValueError as exc:
            self._set_status(f"Error: {exc}", error=True)
            return
        warning = warn_if_high_grain_count(sim_params.sample.np_apophis) or ""
        self._pipeline_start = time.monotonic()
        self._pipeline_run_dir = sim_params.output_root / "run_0001"
        self._pipeline_prefix = sim_params.prefix
        self._pipeline_warning = warning
        self._set_status(f"{warning}{'Dry run' if dry_run else 'Running'}...")
        if self._progress_timer is not None:
            self._progress_timer.stop()
        self._progress_timer = self.set_interval(2.0, self._tick_progress)
        self._run_pipeline_worker(sim_params, render_form)

    def _tick_progress(self) -> None:
        """Coarse live progress while the worker thread runs (spec Component 1
        "Live progress"): elapsed time plus a dump-file count under run_0001.
        Deliberately does not parse phantom.log -- out of scope per the spec.
        """
        if self._pipeline_start is None:
            return
        elapsed = time.monotonic() - self._pipeline_start
        n_dumps = 0
        if self._pipeline_run_dir is not None and self._pipeline_run_dir.is_dir():
            n_dumps = len(list(self._pipeline_run_dir.glob(f"{self._pipeline_prefix}_[0-9]*")))
        self._set_status(
            f"{self._pipeline_warning}Running... {elapsed:.0f}s elapsed, {n_dumps} dump file(s)"
        )

    def _stop_progress_timer(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer = None

    @work(thread=True)
    def _run_pipeline_worker(self, sim_params: SimParams, render_form: RenderFormValues) -> None:
        try:
            result = run_pipeline(sim_params, render_form, _REPO_WIN_CODE / "DEMCSVs")
        except Exception as exc:  # an uncaught exception here must never kill the app
            result = PipelineResult(
                stage="error", ok=False, message=f"{type(exc).__name__}: {exc}",
            )
        self.call_from_thread(self._report_result, result)

    def _report_result(self, result: PipelineResult) -> None:
        self._stop_progress_timer()
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
        else:
            self._set_status(f"{time_prefix}Failed at {result.stage}: {result.message}", error=True)


def main() -> None:
    SimRenderTUIApp().run()


if __name__ == "__main__":
    main()
